# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the Python DSA metadata builder.

Validates the faithful port of ``DSAMetadataBuilder`` (core/layers/common/
dsa_metadata_builder.cpp) against the cache-spec and slot-expansion rules the
C++ implementation enforces. These tests are pure Python: they do not load the
compiled NPU operators, so they run anywhere.
"""

from __future__ import annotations

import pytest
import torch

from xllm.python.attention.dsa_metadata import (
    DSA_CACHE_SLIDING_WINDOW,
    DSA_CACHE_TOKEN,
    READ_TABLE_LAYOUT_RIGHT_ALIGNED,
    READ_TABLE_LAYOUT_RING,
    DsaMetadata,
    DsaMetadataBuilder,
    build_cache_specs,
    build_cache_specs_v41,
)


def test_build_cache_specs_groups() -> None:
    """Group 0 is always SWA; TOKEN groups register in first-seen order."""
    compress_ratios = [0, 0, 4, 128, 4, 128, 4, 0]
    caches_info, group_infos = build_cache_specs(compress_ratios, window_size=128, n_layers=8)

    # Three groups: SWA(1,128), TOKEN(4,512), TOKEN(128,16).
    assert len(group_infos) == 3
    assert group_infos[0].cache_type == DSA_CACHE_SLIDING_WINDOW
    assert group_infos[0].ratio == 1
    assert group_infos[1].cache_type == DSA_CACHE_TOKEN
    assert group_infos[1].ratio == 4
    assert group_infos[1].block_size == 512
    assert group_infos[2].cache_type == DSA_CACHE_TOKEN
    assert group_infos[2].ratio == 128
    assert group_infos[2].block_size == 16


def test_build_cache_specs_per_layer_cache_counts() -> None:
    """C1 -> 1 cache, C4 -> 8 caches, C128 -> 4 caches."""
    compress_ratios = [0, 4, 128]
    caches_info, _ = build_cache_specs(compress_ratios, 128, 3)

    assert len(caches_info[0]) == 1  # cr=0 -> normalized to 1
    assert len(caches_info[1]) == 8  # cr=4
    assert len(caches_info[2]) == 4  # cr=128


def test_build_cache_specs_does_not_silently_accept_unknown_ratio() -> None:
    caches_info, group_infos = build_cache_specs([2], 128, 1)

    assert len(group_infos) == 1
    assert caches_info == [[]]


def test_build_cache_specs_real_dsv4_config() -> None:
    """The shipped DeepSeek-V4-Flash config produces 2 C1 + 21 C4 + 20 C128.

    config.json has ``compress_ratios`` of length 44 (3 zeros, 21 fours, 20
    one-twenty-eights) but ``num_hidden_layers=43``; only layers 0..42 are
    built, so the trailing zero (index 43) is ignored and two C1 layers
    remain (indices 0 and 1).
    """
    compress_ratios = (
        [0, 0] + [4, 128] * 20 + [4, 0]  # layer 42 is C4; index 43 (zero) is ignored.
    )
    assert len(compress_ratios) == 44
    caches_info, group_infos = build_cache_specs(compress_ratios, 128, 43)

    assert len(group_infos) == 3
    c1 = sum(1 for layer in caches_info if len(layer) == 1)
    c4 = sum(1 for layer in caches_info if len(layer) == 8)
    c128 = sum(1 for layer in caches_info if len(layer) == 4)
    assert c1 == 2
    assert c4 == 21
    assert c128 == 20
    assert c1 + c4 + c128 == 43


def _make_builder(
    n_layers: int = 4,
    read_table_layout: str = READ_TABLE_LAYOUT_RIGHT_ALIGNED,
) -> tuple[DsaMetadataBuilder, list, list]:
    compress_ratios = [0, 4, 128, 4]
    caches_info, group_infos = build_cache_specs(compress_ratios, 128, n_layers)
    return (
        DsaMetadataBuilder(caches_info, group_infos, read_table_layout=read_table_layout),
        caches_info,
        group_infos,
    )


def test_build_seq_lengths_and_start_pos() -> None:
    """start_pos = kv_len - q_len per sequence."""
    builder, _, _ = _make_builder()
    # batch=2, decode (q_len=1 each).
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[6, 8],
        q_seq_lens=[1, 1],
        positions=torch.tensor([5, 7], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    assert dsa.seq_lens.tolist() == [6, 8]
    assert dsa.seq_lens_q.tolist() == [1, 1]
    assert dsa.start_pos.tolist() == [5, 7]
    # actual_seq_lengths_query is cumsum(q_lens) with a leading zero.
    assert dsa.actual_seq_lengths_query.tolist() == [0, 1, 2]
    assert dsa.kv_cu_seq_lens.tolist() == [0, 6, 14]
    assert dsa.max_seqlen_q.ndim == 0
    assert dsa.max_seqlen_kv.ndim == 0


def test_build_max_lengths_include_attention_metadata_capacity() -> None:
    """C++ takes max(params.meta.max_*, max(host sequence lengths))."""
    builder, _, _ = _make_builder()
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[6, 8],
        q_seq_lens=[1, 2],
        positions=torch.tensor([5, 6, 7], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
        max_query_len=16,
        max_seq_len=32,
    )

    assert dsa.max_query_len == 16
    assert dsa.max_seq_len == 32


def test_build_token_group_slot_committed_rows() -> None:
    """A TOKEN cache commits one row per ratio boundary crossed this step."""
    builder, caches_info, group_infos = _make_builder()
    # group 0 = SWA, group 1 = TOKEN(4). Give each a [batch=1, cols=4] table.
    swa_bt = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21, 22, 23]], dtype=torch.int32)
    # kv_len=8, q_len=1 (decode): prev_ctx_len=7, committed = 8//4 - 7//4 = 2 - 1 = 1.
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[8],
        q_seq_lens=[1],
        positions=torch.tensor([7], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    # Layer 1 (cr=4): cmp cache is caches_info[1][0] -> group 1 (TOKEN4).
    cmp_slot = dsa.slot_mappings[1][0]
    # One committed row: compressed_idx = prev_committed = 7//4 = 1.
    # block_idx = 1 // 512 = 0, block_id = token4_bt[0,0] = 20.
    # slot = 20 * 512 + 1 = 10241.
    assert cmp_slot.numel() >= 1
    assert cmp_slot[0].item() == 20 * 512 + 1


def test_build_token_group_slot_empty_between_boundaries() -> None:
    """Eager decode uses an actual empty tensor when no row is committed."""
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21, 22, 23]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[129],
        q_seq_lens=[1],
        positions=torch.tensor([128], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )

    assert dsa.slot_mappings[1][0].numel() == 0


def test_build_token_group_slot_commits_at_later_boundary() -> None:
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21, 22, 23]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[132],
        q_seq_lens=[1],
        positions=torch.tensor([131], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )

    assert dsa.slot_mappings[1][0].tolist() == [20 * 512 + 32]


def test_build_swa_group_slot_query_tokens_only() -> None:
    """A SWA cache writes only the current forward's query token."""
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21, 22, 23]], dtype=torch.int32)
    # kv_len=8, q_len=1, q_start=7, pos=7, block_idx = 7//128 % 4 = 0,
    # block_id = swa_bt[0,0] = 10, offset = 7 % 128 = 7 -> slot = 10*128+7 = 1287.
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[8],
        q_seq_lens=[1],
        positions=torch.tensor([7], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    # Layer 0 (cr=1): the single SWA cache -> group 0.
    swa_slot = dsa.slot_mappings[0][0]
    assert swa_slot[0].item() == 10 * 128 + 7


def test_build_swa_group_uses_scheduler_resolved_slots_for_expanded_decode() -> None:
    """Expanded MTP decode preserves the scheduler's ring-buffer slot order."""
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10], [11]], dtype=torch.int32)
    token4_bt = torch.tensor([[20], [21]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[8, 8],
        q_seq_lens=[2, 2],
        positions=torch.tensor([6, 7, 6, 7], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
        new_cache_slots=[301, 302, 401, 402],
    )

    assert dsa.slot_mappings[0][0].tolist() == [301, 302, 401, 402]


def test_swa_device_path_routes_through_dst_slots_kwarg() -> None:
    """The device SWA branch must call a single ``_swa_slots_on_device``.

    Regression: a stale second definition without the ``dst_slots`` parameter
    shadowed the current one, so every call on the device branch
    (``swa_device_ctx_lens is not None`` -> ``dst_slots=...``) raised
    ``TypeError: _swa_slots_on_device() got an unexpected keyword argument
    'dst_slots'``. This drives that branch end to end and checks the slot.
    """
    builder, _, _ = _make_builder()
    raw_bt = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    device_ctx_lens = torch.tensor([8], dtype=torch.int32)
    dst_slots = torch.full((1,), -1, dtype=torch.int32)

    _, out_slots = builder._process_swa_group(
        raw_bt=raw_bt,
        block_size=128,
        ctx_lens=[8],
        q_lens=[1],
        batch_size=1,
        graph_slot_capacity=0,
        graph_block_table_capacity_cols=0,
        new_cache_slots=None,
        swa_device_ctx_lens=device_ctx_lens,
        dst_slots=dst_slots,
    )

    assert out_slots is dst_slots
    # kv_len=8, q_len=1 -> q_start=7, block_idx=0, block_id=10 -> 10*128+7.
    assert out_slots[0].item() == 10 * 128 + 7


def test_swa_slots_on_device_static_signature_accepts_dst_slots() -> None:
    """The device slot mapper must expose the ``dst_slots`` parameter."""
    raw_bt = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    out_slots = DsaMetadataBuilder._swa_slots_on_device(
        raw_bt,
        128,
        4,
        1,
        1,
        [1],
        None,
        1,
        torch.tensor([8], dtype=torch.int32),
        dst_slots=torch.full((1,), -1, dtype=torch.int32),
    )

    assert out_slots[0].item() == 10 * 128 + 7


def _swa_read_table(
    raw_bt: torch.Tensor,
    ctx_lens: list[int],
    *,
    block_size: int = 128,
    graph_cols: int = 0,
    layout: str = READ_TABLE_LAYOUT_RING,
) -> torch.Tensor:
    """Drive the real builder's SWA read-side table construction."""
    builder, _, _ = _make_builder(read_table_layout=layout)
    read_bt, _ = builder._process_swa_group(
        raw_bt=raw_bt,
        block_size=block_size,
        ctx_lens=list(ctx_lens),
        q_lens=[1] * len(ctx_lens),
        batch_size=len(ctx_lens),
        graph_slot_capacity=len(ctx_lens) if graph_cols > 0 else 0,
        graph_block_table_capacity_cols=graph_cols,
        new_cache_slots=None,
    )
    return read_bt


@pytest.mark.parametrize("ctx_len", [129, 257, 513])
def test_swa_read_table_rings_every_addressed_column(ctx_len: int) -> None:
    """A non-zero ring block must appear in EVERY column a reader addresses.

    The "ring" layout (requested by DeepSeek-V4.1) fills every column with the
    block for its own logical index, ``min(semantic_cols, ceil(ctx / block_size))``
    right-aligned columns carrying the ring is NOT enough: every read-side
    consumer addresses columns absolutely, so the older half of a multi-block
    window would resolve to block 0. With ``raw_bt = [[8]]`` (one ring block,
    id 8) every addressed column must read 8.
    """
    raw_bt = torch.tensor([[8]], dtype=torch.int32)
    read_bt = _swa_read_table(raw_bt, [ctx_len])
    addressed = (ctx_len + 127) // 128
    assert read_bt.shape == (1, addressed)
    assert read_bt[0, :addressed].tolist() == [8] * addressed


def test_swa_read_table_resolves_each_sequence_ring_block() -> None:
    """batch > 1: each row resolves to its OWN ring block, never block 0."""
    rings = [8, 9, 10, 11]
    raw_bt = torch.tensor([[ring] for ring in rings], dtype=torch.int32)
    read_bt = _swa_read_table(raw_bt, [257] * len(rings))
    addressed = (257 + 127) // 128  # 3
    assert read_bt.shape == (len(rings), addressed)
    for seq, ring in enumerate(rings):
        assert read_bt[seq].tolist() == [ring] * addressed


def test_swa_read_table_modulos_by_semantic_cols() -> None:
    """Several ring blocks per sequence: column ``d`` carries ``raw[s, d % sc]``."""
    raw_bt = torch.tensor([[8, 9]], dtype=torch.int32)  # semantic_cols == 2
    read_bt = _swa_read_table(raw_bt, [385])
    addressed = (385 + 127) // 128  # 4
    assert read_bt.shape == (1, addressed)
    assert read_bt[0].tolist() == [8, 9, 8, 9]


def test_swa_read_table_rings_the_full_graph_storage_width() -> None:
    """Production graph width: all 8192 columns carry the sequence ring block.

    Production runs ``graph_block_table_capacity_cols=8192`` with one ring block
    per sequence, so the read table is 8192 wide and every addressed column must
    equal that single ring block (not just the right-aligned tail).
    """
    raw_bt = torch.tensor([[8]], dtype=torch.int32)
    read_bt = _swa_read_table(raw_bt, [257], graph_cols=8192)
    assert read_bt.shape == (1, 8192)
    assert bool((read_bt[0] == 8).all())


def test_swa_read_table_matches_the_write_path_ring_resolution() -> None:
    """Every position's eager read must resolve the block the writer used.

    The write path (``slot_for_position``) maps position ``p`` to
    ``raw_bt[s, (p // block_size) % semantic_cols]``; the eager reader
    (``_gather_swa_positions``) resolves ``(p // block_size) % read_cols``
    against the read-side table and clamps ``-1`` to 0. Both must agree for
    every position, not only the newest window.
    """
    block_size = 128
    ctx_len = 513
    raw_bt = torch.tensor([[8]], dtype=torch.int32)
    read_bt = _swa_read_table(raw_bt, [ctx_len], block_size=block_size)
    read_cols = int(read_bt.size(1))
    semantic_cols = int(raw_bt.size(1))
    for pos in range(ctx_len):
        write_block = int(raw_bt[0, (pos // block_size) % semantic_cols].item())
        eager_block = int(read_bt[0, (pos // block_size) % read_cols].clamp_min(0).item())
        assert eager_block == write_block, f"pos={pos}"


def test_default_swa_read_table_stays_right_aligned() -> None:
    """The shared (non-V4.1) default keeps the historical right-aligned layout."""
    raw_bt = torch.tensor([[8]], dtype=torch.int32)
    read_bt = _swa_read_table(raw_bt, [257], layout=READ_TABLE_LAYOUT_RIGHT_ALIGNED)
    # dst_lens = ceil(257 / 128) = 3, semantic_cols = 1 -> one retained column at
    # the tail, the leading columns stay -1 (the DSV4 / V4-dspark consumers
    # compensate for this shift via _build_dspark_swa_indices).
    assert read_bt.shape == (1, 3)
    assert read_bt[0].tolist() == [-1, -1, 8]


def test_builder_rejects_unknown_read_table_layout() -> None:
    caches_info, group_infos = build_cache_specs([0], 128, 1)
    with pytest.raises(ValueError, match="unknown SWA read-table layout"):
        DsaMetadataBuilder(caches_info, group_infos, read_table_layout="bogus")


def test_build_block_tables_shared_within_group() -> None:
    """Caches in the same group share the same underlying tensor."""
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21, 22, 23]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[8],
        q_seq_lens=[1],
        positions=torch.tensor([7], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    # Layer 1 (cr=4): caches 0,1,7 are TOKEN4 (group 1) -> same slot tensor.
    assert dsa.slot_mappings[1][0].data_ptr() == dsa.slot_mappings[1][1].data_ptr()
    assert dsa.slot_mappings[1][0].data_ptr() == dsa.slot_mappings[1][7].data_ptr()
    # Caches 2-6 are SWA (group 0) -> same slot tensor.
    assert dsa.slot_mappings[1][2].data_ptr() == dsa.slot_mappings[1][3].data_ptr()


# -- group_limit: a sibling model that owns fewer groups than the request ----


def _make_builder_v41(n_layers: int = 3):
    """A V4.1 cache-spec builder (SWA + one TOKEN group per kv-source ratio)."""
    caches_info, group_infos = build_cache_specs_v41([0, 2, 1], [1, 2], [1, 2], window_size=128, n_layers=n_layers)
    return DsaMetadataBuilder(caches_info, group_infos), caches_info, group_infos


def _make_swa_only_builder(n_layers: int = 3):
    """A pure-SWA model (the DSpark draft): exactly one deduplicated group."""
    caches_info, group_infos = build_cache_specs_v41([0] * n_layers, [], [], window_size=128, n_layers=n_layers)
    assert len(group_infos) == 1
    return DsaMetadataBuilder(caches_info, group_infos), caches_info, group_infos


def _v41_tables() -> list[torch.Tensor]:
    """The request's tables in the fixed export order [SWA, C2, C1]."""
    return [
        torch.tensor([[10, 11, 12, 13]], dtype=torch.int32),
        torch.tensor([[20, 21]], dtype=torch.int32),
        torch.tensor([[30, 31]], dtype=torch.int32),
    ]


def _build_v41(builder: DsaMetadataBuilder, group_limit: int | None = None) -> DsaMetadata:
    """One decode step over the fixed V4.1 tables, optionally group-limited."""
    return builder.build(
        multi_block_tables=_v41_tables(),
        kv_seq_lens=[8],
        q_seq_lens=[1],
        positions=torch.tensor([7], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
        group_limit=group_limit,
    )


def test_manager_count_beyond_the_model_groups_is_rejected() -> None:
    """The invariant the DSpark draft tripped: managers > own deduplicated groups."""
    builder, _, _ = _make_swa_only_builder()
    with pytest.raises(ValueError, match="manager count 3 exceeds group count 1"):
        _build_v41(builder)


def test_group_limit_selects_only_the_leading_managers() -> None:
    """A one-group draft consumes just the request's leading SWA manager."""
    builder, _, _ = _make_swa_only_builder()
    dsa = _build_v41(builder, group_limit=1)
    # Every draft layer owns one SWA cache -> the same processed manager tensor.
    for layer_bt in dsa.block_tables:
        assert len(layer_bt) == 1
        assert layer_bt[0].data_ptr() == dsa.block_tables[0][0].data_ptr()
    for layer_slots in dsa.slot_mappings:
        assert len(layer_slots) == 1
    # Slot math comes from the SWA table (block 10, ring offset 7), never C2/C1.
    assert dsa.slot_mappings[0][0].tolist() == [10 * 128 + 7]
    assert dsa.block_tables[0][0][0, 0].item() == 10


def test_group_limit_equal_to_all_groups_is_the_default() -> None:
    """The owning model (no limit) and an explicit full limit agree exactly."""
    builder, _, _ = _make_builder_v41()
    default = _build_v41(builder)
    explicit = _build_v41(builder, group_limit=3)
    assert len(default.block_tables) == len(explicit.block_tables)
    for layer_default, layer_explicit in zip(default.block_tables, explicit.block_tables, strict=True):
        assert len(layer_default) == len(layer_explicit)
        for a, b in zip(layer_default, layer_explicit, strict=True):
            assert a.shape == b.shape
            assert torch.equal(a, b)


def test_group_limit_partial_owner_leaves_later_groups_unmapped() -> None:
    """A two-group owner reads managers 0..1; layers in group 2 get empty slots."""
    builder, _, group_infos = _make_builder_v41()
    assert len(group_infos) == 3
    dsa = _build_v41(builder, group_limit=2)
    # Layer 1 is the ratio-2 kv-source: entries [C2 key, C2 index, swa, swa, swa].
    assert dsa.block_tables[1][0].data_ptr() == dsa.block_tables[1][1].data_ptr()
    assert dsa.block_tables[1][2].data_ptr() == dsa.block_tables[0][0].data_ptr()
    # Layer 2 is the ratio-1 kv-source: its two TOKEN entries (group 2, beyond
    # the limit) stay unmapped while its SWA entry (group 0) is mapped.
    assert dsa.block_tables[2][0].numel() == 0
    assert dsa.block_tables[2][1].numel() == 0
    assert dsa.block_tables[2][2].data_ptr() == dsa.block_tables[0][0].data_ptr()


def test_group_limit_rejects_a_negative_value() -> None:
    builder, _, _ = _make_swa_only_builder()
    with pytest.raises(ValueError, match="group_limit must be non-negative"):
        _build_v41(builder, group_limit=-1)


def test_batchless_build_is_the_b4_empty_length_signature() -> None:
    """A zero-sequence build is what the idle DP shard used to hand the model.

    The builder itself accepts it; the model's attention then maxes an empty
    per-sequence length list. csa_attention now synthesizes the framework dummy
    row before this state can reach the model, so this test pins the raw
    signature the fix has to avoid.
    """
    builder, _, _ = _make_swa_only_builder()
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[],
        q_seq_lens=None,
        positions=torch.tensor([0], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    assert dsa.seq_lens_q.numel() == 0
    # CPython 3.11: "max() arg is an empty sequence"; 3.12: "iterable argument is empty".
    with pytest.raises(ValueError, match="empty"):
        max(int(value) for value in dsa.seq_lens_q.tolist())
def test_build_c4_pad_positions() -> None:
    """c4_pad_positions records next_pos-4 when (pos+1) % 4 == 0."""
    builder, _, _ = _make_builder()
    # q_len=4, q_start=3 -> positions 3,4,5,6. (pos+1)%4==0 at pos=3 (next=4).
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[7],
        q_seq_lens=[4],
        positions=torch.tensor([3, 4, 5, 6], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=True,
        is_chunked_prefill=False,
    )
    # pos=3 -> next_pos=4 -> 4%4==0 -> record 4-4=0.
    assert 0 in dsa.c4_pad_positions.tolist()


def test_graph_compressed_positions_use_zero_padding() -> None:
    """ACL graph position buffers match C++ vector::resize zero fill."""
    builder, _, _ = _make_builder()
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[7],
        q_seq_lens=[4],
        positions=torch.tensor([3, 4, 5, 6], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=True,
        is_chunked_prefill=False,
        enable_graph=True,
    )

    assert dsa.c4_pad_positions.tolist() == [0, 0, 0, 0]
    assert dsa.c128_pad_positions.tolist() == [0, 0, 0, 0]


def test_empty_batch_preserves_cpp_zero_length_buffers() -> None:
    builder, _, _ = _make_builder()
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[],
        q_seq_lens=[],
        positions=torch.empty(0, dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=True,
        is_chunked_prefill=False,
    )

    assert dsa.actual_seq_lengths_query.tolist() == [0]
    assert dsa.kv_cu_seq_lens.tolist() == [0]
    assert dsa.max_seqlen_q.shape == (1,)
    assert dsa.max_seqlen_kv.shape == (1,)
    assert dsa.max_query_len == 0
    assert dsa.max_seq_len == 0
    assert dsa.c4_pad_positions.numel() == 0
    assert dsa.c128_pad_positions.numel() == 0


def test_build_c128_slot_at_compression_boundary() -> None:
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21]], dtype=torch.int32)
    token128_bt = torch.tensor([[30, 31]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt, token128_bt],
        kv_seq_lens=[128],
        q_seq_lens=[1],
        positions=torch.tensor([127], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )

    # Layer 2 uses TOKEN(128) for cache 0. The first compressed row is offset 0.
    assert dsa.slot_mappings[2][0].tolist() == [30 * 16]
    assert dsa.c128_pad_positions.tolist() == [0]


def test_multi_batch_slots_are_concatenated_by_sequence() -> None:
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11], [12, 13]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21], [22, 23]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[4, 8],
        q_seq_lens=[1, 1],
        positions=torch.tensor([3, 7], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )

    assert dsa.slot_mappings[0][0].tolist() == [10 * 128 + 3, 12 * 128 + 7]
    assert dsa.slot_mappings[1][0].tolist() == [20 * 512, 22 * 512 + 1]


def test_packed_manager_block_table_is_unpacked() -> None:
    builder, _, _ = _make_builder()
    packed = torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[packed],
        kv_seq_lens=[128],
        q_seq_lens=[1],
        positions=torch.tensor([127], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )

    assert dsa.block_tables[0][0].tolist() == [[10, -1]]
    assert dsa.block_tables[1][0].tolist() == [[20, 21]]
    assert dsa.block_tables[2][0].tolist() == [[30, 31]]


def test_graph_slots_and_block_tables_use_bucket_capacity() -> None:
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[8],
        q_seq_lens=[1],
        positions=torch.tensor([7, 0, 0, 0], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
        enable_graph=True,
        graph_block_table_capacity_cols=4,
    )

    assert dsa.slot_mappings[1][0].tolist() == [20 * 512 + 1, -1, -1, -1]
    assert dsa.block_tables[1][0].shape == (1, 4)
    assert dsa.block_tables[1][0].tolist() == [[20, 21, -1, -1]]
    assert dsa.slot_mappings[0][0].tolist() == [10 * 128 + 7, -1, -1, -1]
    assert dsa.block_tables[0][0].shape == (1, 4)


def test_rope_cache_is_split_into_zero_copy_cos_and_sin_tables() -> None:
    builder, _, _ = _make_builder()
    cos_sin = torch.arange(24, dtype=torch.float32).view(3, 8)
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[3],
        q_seq_lens=[3],
        positions=torch.arange(3, dtype=torch.int64),
        dsa_cos_sin=cos_sin,
        is_prefill=True,
        is_chunked_prefill=False,
    )

    assert torch.equal(dsa.cos_table, cos_sin[:, :4])
    assert torch.equal(dsa.sin_table, cos_sin[:, 4:])
    assert not dsa.cos_table.is_contiguous()
    assert not dsa.sin_table.is_contiguous()
    assert dsa.cos_table.data_ptr() == cos_sin.data_ptr()
    assert dsa.sin_table.data_ptr() != cos_sin.data_ptr()


def test_compressed_positions_preserve_position_dtype() -> None:
    builder, _, _ = _make_builder()
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[4],
        q_seq_lens=[4],
        positions=torch.arange(4, dtype=torch.int32),
        dsa_cos_sin=None,
        is_prefill=True,
        is_chunked_prefill=False,
    )

    assert dsa.c4_pad_positions.dtype == torch.int32
    assert dsa.c128_pad_positions.dtype == torch.int32


def test_graph_rejects_block_table_larger_than_bucket_capacity() -> None:
    builder, _, _ = _make_builder()
    block_table = torch.tensor([[10, 11, 12]], dtype=torch.int32)

    with pytest.raises(ValueError, match="exceeds bucket capacity"):
        builder.build(
            multi_block_tables=[block_table],
            kv_seq_lens=[8],
            q_seq_lens=[1],
            positions=torch.tensor([7], dtype=torch.int64),
            dsa_cos_sin=None,
            is_prefill=False,
            is_chunked_prefill=False,
            enable_graph=True,
            graph_block_table_capacity_cols=2,
        )


@pytest.mark.parametrize("column_stride", [1, 2])
@pytest.mark.parametrize(
    "source_rows,source_cols,ctx_lens,capacity",
    [
        (2, 2, [9, 9], 4),
        (2, 3, [17, 17], 8),
        (8, 2, [0, 1, 4, 5, 8, 9, 9, 33], 8),
        (3, 2, [5, 5, 9, 9, 13, 13], 8),
        (3, 0, [0, 4, 9], 4),
        (0, 2, [4, 8], 4),
        (96, 3, [17] * 96, 8192),
    ],
)
def test_right_aligned_swa_row_copies_match_scalar_ring_addresses(
    source_rows: int, source_cols: int, ctx_lens: list[int], capacity: int, column_stride: int
) -> None:
    builder, _, _ = _make_builder()
    source = torch.arange(source_rows * source_cols * column_stride, dtype=torch.int32)
    source = source.reshape(source_rows, source_cols * column_stride)[:, ::column_stride]
    if source.numel():
        source[0, 0] = -1
    batch_size = len(ctx_lens)
    result, _ = builder._process_swa_group(
        source,
        block_size=4,
        ctx_lens=ctx_lens,
        q_lens=[1] * batch_size,
        batch_size=batch_size,
        graph_slot_capacity=batch_size,
        graph_block_table_capacity_cols=capacity,
        new_cache_slots=None,
    )
    lengths = [(max(length, 0) + 3) // 4 for length in ctx_lens]
    expected = torch.full((batch_size, max(capacity, source_cols, max(lengths))), -1, dtype=torch.int32)
    for row in range(min(source_rows, batch_size)):
        retained = min(source_cols, lengths[row])
        for column in range(lengths[row] - retained, lengths[row]):
            expected[row, column] = source[row, column % source_cols]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)

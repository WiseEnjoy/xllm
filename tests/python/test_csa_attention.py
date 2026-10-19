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

"""Unit tests for the DeepSeek-V4 CSA/HCA attention backend.

Pure-Python: does not load compiled operators. Covers DSA metadata build,
decode-graph device refresh, and fused inverse RoPE wiring (see the
per-section notes below).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from xllm.python import kernels
import xllm.python.attention.csa2_attention as csa2
from xllm.python.attention import csa_attention as csa_attention_module
from xllm.python.attention import dsa_metadata as dsa_metadata_module
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.csa_attention import (
    COMPRESSED_SPARSE_ATTENTION,
    HEAVILY_COMPRESSED_ATTENTION,
    SLIDING_ATTENTION,
    CsaAttentionBackend,
    DsaAttentionBackend,
    _attention_type_for_compress_ratio,
    _build_dspark_swa_indices,
    _build_prefill_pa_nd_kv,
    _CompressedAttentionCacheMapping,
    _deepseek_v4_ori_window_left,
    _get_layer_cache_tensor,
    _scatter_by_slot,
)
from xllm.python.attention.dsa_metadata import (
    DsaMetadataBuilder,
    build_cache_specs,
)
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)
from xllm.python.model_executor.v4_cp_context import (
    build_deepseek_v4_cp_context,
)


def _make_backend(device: torch.device | None = None) -> CsaAttentionBackend:
    compress_ratios = [0, 4, 128]
    caches_info, group_infos = build_cache_specs(compress_ratios, 128, 3)
    return CsaAttentionBackend(
        compress_ratios=compress_ratios,
        window_size=128,
        n_layers=3,
        num_heads=8,
        attn_head_dim=512,
        index_topk=512,
        index_n_heads=64,
        index_head_dim=128,
        rope_head_dim=64,
        device=device or torch.device("cpu"),
        dtype=torch.bfloat16,
    )


def _patch_metadata_kernels(
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_a5: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sparse_calls: list[dict[str, object]] = []
    qli_calls: list[dict[str, object]] = []

    def fake_sparse_metadata(**kwargs: object) -> torch.Tensor:
        sparse_calls.append(kwargs)
        return torch.tensor([kwargs["cmp_ratio"]], dtype=torch.int32)

    def fake_qli_metadata(**kwargs: object) -> torch.Tensor:
        qli_calls.append(kwargs)
        return torch.tensor([4], dtype=torch.int32)

    for name in ("sparse_attn_sharedkv_metadata", "sparse_flash_mla_metadata"):
        monkeypatch.setattr(kernels, name, fake_sparse_metadata, raising=False)
    for name in ("quant_lightning_indexer_metadata", "quant_lightning_indexer_v2_metadata"):
        monkeypatch.setattr(kernels, name, fake_qli_metadata, raising=False)
    monkeypatch.setattr(
        csa_attention_module.current_platform,
        "get_ascend_soc_generation",
        lambda: "a5" if use_a5 else "a2",
    )
    return sparse_calls, qli_calls


@pytest.mark.parametrize(
    ("use_zero_based_blocks", "first_block", "packed_blocks"),
    [(False, 1, 3), (True, 0, 2)],
)
def test_prefill_pa_nd_kv_preserves_kernel_block_layout(
    use_zero_based_blocks: bool,
    first_block: int,
    packed_blocks: int,
) -> None:
    kv = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2)
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
    block_table_hint = torch.tensor([[7, 8]], dtype=torch.int32)

    packed_kv, block_table = _build_prefill_pa_nd_kv(
        kv,
        cu_seqlens,
        block_table_hint,
        block_size=2,
        use_zero_based_blocks=use_zero_based_blocks,
    )

    assert packed_kv.shape == (packed_blocks, 2, 1, 2)
    assert block_table.tolist() == [[first_block, first_block + 1]]
    if use_zero_based_blocks:
        torch.testing.assert_close(packed_kv[0], kv[:2])
        torch.testing.assert_close(packed_kv[1, 0], kv[2])
        assert torch.equal(packed_kv[1, 1], torch.zeros(1, 2))
    else:
        assert torch.equal(packed_kv[0], torch.zeros(2, 1, 2))
        torch.testing.assert_close(packed_kv[1], kv[:2])
        torch.testing.assert_close(packed_kv[2, 0], kv[2])
        assert torch.equal(packed_kv[2, 1], torch.zeros(1, 2))


def test_decode_window_compact_eager_gathers_newest_window() -> None:
    backend = CsaAttentionBackend(
        compress_ratios=[1],
        window_size=4,
        n_layers=1,
        num_heads=2,
        attn_head_dim=4,
        index_topk=1,
        index_n_heads=2,
        index_head_dim=4,
        rope_head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    ori_kv = torch.arange(32, dtype=torch.float32).reshape(4, 4, 1, 2)
    ori_bt = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    seq_kv = torch.tensor([4, 3], dtype=torch.int32)

    compact_kv, compact_bt, compact_lens = backend._decode_window_compact_eager(
        ori_kv,
        ori_bt,
        seq_kv,
    )

    assert compact_kv.shape == (2, 4, 1, 2)
    assert compact_bt.tolist() == [[0], [1]]
    assert compact_lens.tolist() == [4, 3]
    torch.testing.assert_close(compact_kv[0, 0, 0], ori_kv[0, 0, 0])
    torch.testing.assert_close(compact_kv[1, 0, 0], ori_kv[2, 0, 0])
    assert torch.equal(compact_kv[1, 3, 0], torch.zeros(2))


def test_decode_window_compact_eager_rejects_missing_block_table_rows() -> None:
    backend = CsaAttentionBackend(
        compress_ratios=[1],
        window_size=4,
        n_layers=1,
        num_heads=2,
        attn_head_dim=4,
        index_topk=1,
        index_n_heads=2,
        index_head_dim=4,
        rope_head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    ori_kv = torch.zeros(2, 4, 1, 2)
    ori_bt = torch.zeros(1, 2, dtype=torch.int32)
    seq_kv = torch.tensor([4, 3], dtype=torch.int32)

    with pytest.raises(ValueError, match="SWA block table has 1 rows"):
        backend._decode_window_compact_eager(ori_kv, ori_bt, seq_kv)


def test_graph_window_compact_matches_eager_semantics() -> None:
    backend = CsaAttentionBackend(
        compress_ratios=[1, 1],
        window_size=4,
        n_layers=2,
        num_heads=2,
        attn_head_dim=4,
        index_topk=1,
        index_n_heads=2,
        index_head_dim=4,
        rope_head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    ori_kv = torch.arange(40, dtype=torch.float32).reshape(5, 4, 1, 2)
    ori_bt = torch.tensor([[0, 1], [2, 3], [0, -1]], dtype=torch.int32)
    seq_kv = torch.tensor([4, 3, 0], dtype=torch.int32)
    compressed_metadata = SimpleNamespace(block_tables=[[ori_bt], [ori_bt]])
    context = ForwardContext(
        attention_backend=backend,
        device=torch.device("cpu"),
        metadata=SimpleNamespace(),
        layer_caches=[],
        execution_state=AclGraphExecutionState(persistent_buffers={}),
    )

    with forward_context(context):
        sources = backend._build_swa_window_sources(compressed_metadata, [4, 3, 0])

    assert sources is not None
    assert sources.shape == (1, 3, 4)
    assert backend._swa_window_row_map == [0, 0]
    assert sources[0, 2].tolist() == [-1, -1, -1, -1]

    graph_kv, graph_bt, graph_lens = backend._graph_window_compact(
        ori_kv,
        seq_kv,
        layer_id=1,
    )
    eager_kv, eager_bt, eager_lens = backend._decode_window_compact_eager(
        ori_kv,
        ori_bt,
        seq_kv,
    )

    assert graph_kv.shape == (3, 4, 1, 2)
    assert graph_bt.tolist() == [[0], [1], [2]]
    assert graph_lens.tolist() == [4, 3, 0]
    assert eager_bt.tolist() == graph_bt.tolist()
    assert eager_lens.tolist() == graph_lens.tolist()
    torch.testing.assert_close(graph_kv, eager_kv)
    assert torch.equal(graph_kv[2], torch.zeros_like(graph_kv[2]))


def test_graph_window_sources_keep_stable_address_on_refresh() -> None:
    backend = CsaAttentionBackend(
        compress_ratios=[1],
        window_size=4,
        n_layers=1,
        num_heads=2,
        attn_head_dim=4,
        index_topk=1,
        index_n_heads=2,
        index_head_dim=4,
        rope_head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    block_table = torch.tensor([[0, 1], [2, 3], [0, -1]], dtype=torch.int32)
    compressed_metadata = SimpleNamespace(block_tables=[[block_table]])
    context = ForwardContext(
        attention_backend=backend,
        device=torch.device("cpu"),
        metadata=SimpleNamespace(),
        layer_caches=[],
        execution_state=AclGraphExecutionState(persistent_buffers={}),
    )

    with forward_context(context):
        first = backend._build_swa_window_sources(compressed_metadata, [4, 3, 0])
        assert first is not None
        address = first.data_ptr()
        initial_values = first.clone()

        second = backend._build_swa_window_sources(compressed_metadata, [2, 1, 0])

    assert second is first
    assert second.data_ptr() == address
    assert not torch.equal(second, initial_values)
    assert second[0, 0].tolist() == [0, 1, -1, -1]
    assert second[0, 2].tolist() == [-1, -1, -1, -1]


def test_a5_graph_execute_uses_window_compact(monkeypatch) -> None:
    monkeypatch.setattr(csa_attention_module, "_uses_a5_sparse_kernels", lambda: True)
    backend = CsaAttentionBackend(
        compress_ratios=[1],
        window_size=4,
        n_layers=1,
        num_heads=2,
        attn_head_dim=4,
        index_topk=1,
        index_n_heads=2,
        index_head_dim=4,
        rope_head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    swa = torch.zeros(1, 4, 1, 4, dtype=torch.float32)
    backend.bind_kv_caches([LayerCache(key=None, value=None, swa=swa)])
    ori_bt = torch.tensor([[0]], dtype=torch.int32)
    seq_kv = torch.tensor([3], dtype=torch.int32)
    compressed_metadata = SimpleNamespace(
        block_tables=[[ori_bt]],
        slot_mappings=[[torch.tensor([2], dtype=torch.int32)]],
        actual_seq_lengths_query=torch.tensor([0, 1], dtype=torch.int32),
        actual_seq_lengths_kv=seq_kv,
        input_positions=torch.tensor([2], dtype=torch.int64),
        cos_table=torch.empty(0),
        sin_table=torch.empty(0),
        c4_cos=torch.empty(0),
        c128_cos=torch.empty(0),
        c1_metadata=torch.zeros(1, dtype=torch.int32),
        sparse_metadata_ori_win_left=3,
    )
    metadata = SimpleNamespace(
        dsa_metadata=compressed_metadata,
        is_prefill=False,
        is_chunked_prefill=False,
        dsa_graph_mode=True,
    )
    backend._metadata = metadata
    context = ForwardContext(
        attention_backend=backend,
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[LayerCache(key=None, value=None, swa=swa)],
        execution_state=AclGraphExecutionState(persistent_buffers={}),
    )
    calls: list[dict] = []

    def fake_sparse_attn(**kwargs):
        calls.append(kwargs)
        return kwargs["q"].clone(), torch.empty(0)

    monkeypatch.setattr(csa_attention_module, "_sparse_attn_sharedkv", fake_sparse_attn)
    q = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    kv = torch.full((1, 1, 4), 7.0)
    layer = SimpleNamespace(layer_id=0, attn_sink=None, attn_sink_loaded=False)

    with forward_context(context):
        assert backend._build_swa_window_sources(compressed_metadata, [3]) is not None
        out = backend.execute(q, kv, kv, layer)

    assert torch.equal(out, q)
    assert len(calls) == 1
    eager_kv, eager_bt, eager_lens = backend._decode_window_compact_eager(
        swa,
        ori_bt,
        seq_kv,
    )
    assert calls[0]["ori_block_table"].tolist() == eager_bt.tolist()
    assert calls[0]["seqused_ori_kv"].tolist() == eager_lens.tolist()
    torch.testing.assert_close(calls[0]["ori_kv"], eager_kv)
    assert calls[0]["ori_kv"][0, 2, 0].tolist() == [7.0, 7.0, 7.0, 7.0]


def test_csa2_backend_skips_v4_swa_window_source_prep(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared A5 graph branch must not drag Csa2 into V4 window sources.

    Regression: the base class built SWA window sources under A5 + graph mode
    unconditionally. ``Csa2AttentionBackend`` does not run the base
    constructor, so the V4-only ``_swa_window_layer_specs`` state was missing
    and V4.1 graph mode on A5 died with AttributeError during the first
    metadata build.
    """

    from xllm.python.attention.csa2_attention import Csa2AttentionBackend

    backend = Csa2AttentionBackend(
        compress_ratios=[1, 4],
        window_size=4,
        n_layers=2,
        num_heads=2,
        attn_head_dim=64,
        index_topk=2,
        index_n_heads=2,
        index_head_dim=64,
        rope_head_dim=64,
        device=torch.device("cpu"),
        dtype=torch.float32,
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0],
        candidate_source_layer_id=0,
        candidate_topk_blocks=2,
        candidate_block_size=4,
        acl_graph_enabled=True,
    )
    # The capability contract cuts both ways: the V4 backend declares the
    # consumption, Csa2 opts out because its captured forward computes the
    # window logic in-graph.
    assert backend.graph_uses_swa_window_sources is False
    assert csa_attention_module.CsaAttentionBackend.graph_uses_swa_window_sources is True

    monkeypatch.setattr(csa_attention_module, "_uses_a5_sparse_kernels", lambda: True)
    window_calls: list[Any] = []
    monkeypatch.setattr(
        backend,
        "_build_swa_window_sources",
        lambda *args, **kwargs: window_calls.append(args),
    )
    metadata = SimpleNamespace(
        multi_block_tables=[
            torch.tensor([[10]], dtype=torch.int32),
            torch.tensor([[20]], dtype=torch.int32),
        ],
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        new_cache_slots_host_values=[1280],
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=False,
        dsa_positions=torch.tensor([0], dtype=torch.int64),
        dsa_cos_sin=None,
        dsa_graph_block_table_cols=0,
        dsa_graph_mode=True,
    )

    # Must not raise (the pre-fix AttributeError) and must skip the V4 prep.
    dsa = backend._build_dsa_metadata_for_forward(metadata)

    assert window_calls == []
    assert dsa is not None
    assert dsa.is_acl_graph is True


def test_dspark_native_sas_window_matches_cpp_contract() -> None:
    assert _deepseek_v4_ori_window_left(128, 5, False) == 127
    assert _deepseek_v4_ori_window_left(128, 0, True) == 127
    assert _deepseek_v4_ori_window_left(128, 5, True) == 132


def test_dspark_native_swa_indices_are_shared_by_query_rows() -> None:
    indices = _build_dspark_swa_indices(
        torch.tensor([[10, 11, 12]], dtype=torch.int32),
        torch.tensor([0, 3], dtype=torch.int32),
        torch.tensor([6], dtype=torch.int32),
        window_size=4,
        dspark_block_size=3,
        cache_block_size=4,
    )

    assert indices.shape == (3, 1, 128)
    expected = torch.tensor([40, 41, 42, 43, 44, 45, -1], dtype=torch.int32)
    for row in range(3):
        torch.testing.assert_close(indices[row, 0, :7], expected)


def test_dspark_native_swa_indices_wrap_ring_buffer() -> None:
    indices = _build_dspark_swa_indices(
        torch.tensor([[20, 21]], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([10], dtype=torch.int32),
        window_size=4,
        dspark_block_size=3,
        cache_block_size=2,
    )

    assert indices.shape == (1, 1, 128)
    expected = torch.tensor([41, 42, 43, 40, 41], dtype=torch.int32)
    torch.testing.assert_close(indices[0, 0, :5], expected)


def test_dspark_native_swa_indices_map_expanded_metadata_ring() -> None:
    backend = CsaAttentionBackend(
        compress_ratios=[1],
        window_size=4,
        n_layers=1,
        num_heads=2,
        attn_head_dim=4,
        index_topk=0,
        index_n_heads=2,
        index_head_dim=4,
        rope_head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        dspark_block_size=3,
        dspark_use_native_sas=True,
    )
    compressed_metadata = backend._builder.build(
        multi_block_tables=[torch.tensor([[20, 21]], dtype=torch.int32)],
        kv_seq_lens=[9],
        q_seq_lens=[2],
        positions=torch.tensor([7, 8], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )

    # ``CsaAttentionBackend`` builds the metadata without overriding
    # ``read_table_layout``, so it inherits the shared right-aligned default
    # (``READ_TABLE_LAYOUT_RIGHT_ALIGNED``): only the retained columns are
    # filled, at ``raw[s, d % semantic_cols]``, leaving the leading columns as
    # ``-1``. Only v41/``csa2`` opts into the ring layout that rings every
    # column (``[[20, 21, 20]]``). ``_build_dspark_swa_indices`` is
    # layout-adaptive, so the resolved slots below are unchanged -- which is
    # what this test actually pins.
    assert compressed_metadata.block_tables[0][0].tolist() == [[-1, 21, 20]]
    backend._build_dspark_swa_metadata(compressed_metadata)

    expected = torch.tensor([83, 84, 85, 86, 87, 80], dtype=torch.int32)
    for row in range(2):
        torch.testing.assert_close(compressed_metadata.explicit_swa_indices[row, 0, :6], expected)


def test_compress_ratio_per_layer() -> None:
    b = _make_backend()
    assert b._layer_compress_ratio(0) == 1
    assert b._layer_compress_ratio(1) == 4
    assert b._layer_compress_ratio(2) == 128


@pytest.mark.parametrize(
    ("compress_ratio", "attention_type"),
    [
        (1, SLIDING_ATTENTION),
        (4, COMPRESSED_SPARSE_ATTENTION),
        (128, HEAVILY_COMPRESSED_ATTENTION),
    ],
)
def test_official_attention_type_names(compress_ratio: int, attention_type: str) -> None:
    assert _attention_type_for_compress_ratio(compress_ratio) == attention_type


def test_unsupported_compression_ratio_has_clear_error() -> None:
    with pytest.raises(ValueError, match="unsupported DeepSeek-V4 compression ratio"):
        _attention_type_for_compress_ratio(8)


def test_resolve_cache_mapping_csa() -> None:
    """CSA layer: compressed/index caches plus sliding-window state caches."""
    b = _make_backend()
    m = b._resolve_cache_mapping(1, 4)
    assert m.cmp_cache_idx == 0
    assert m.index_cache_idx == 1
    assert m.indexer_scale_cache_idx == 7
    assert m.ori_cache_idx == 2
    assert m.kv_state_cache_idx == 3
    assert m.score_state_cache_idx == 4
    assert m.index_kv_state_cache_idx == 5
    assert m.index_score_state_cache_idx == 6


def test_resolve_cache_mapping_hca() -> None:
    """HCA layer: compressed cache and sliding-window state caches."""
    b = _make_backend()
    m = b._resolve_cache_mapping(2, 128)
    assert m.cmp_cache_idx == 0
    assert m.ori_cache_idx == 1
    assert m.kv_state_cache_idx == 2
    assert m.score_state_cache_idx == 3
    assert m.index_cache_idx == -1  # HCA has no indexer
    assert m.indexer_scale_cache_idx == -1


def test_resolve_cache_mapping_sliding_attention() -> None:
    """Sliding-attention layer: one SWA cache and no compressed cache."""
    b = _make_backend()
    m = b._resolve_cache_mapping(0, 1)
    assert m.cmp_cache_idx == -1
    assert m.ori_cache_idx == 0
    assert m.index_cache_idx == -1


def test_get_layer_cache_tensor_bounds() -> None:
    tensors = [[torch.empty(0)], [torch.zeros(2), torch.zeros(3)]]
    assert _get_layer_cache_tensor(tensors, 0, 0).numel() == 0
    assert _get_layer_cache_tensor(tensors, 1, 1).numel() == 3
    assert _get_layer_cache_tensor(tensors, 5, 0) is None  # bad layer
    assert _get_layer_cache_tensor(tensors, 1, 9) is None  # bad cache idx


def test_scatter_by_slot_writes_rows() -> None:
    cache = torch.zeros(4, 3, dtype=torch.float32)
    # slot 0 -> row 0, slot 2 -> row 2, slot -1 skipped.
    slots = torch.tensor([0, -1, 2], dtype=torch.int32)
    value = torch.tensor([[1.0, 1.0, 1.0], [9.0, 9.0, 9.0], [2.0, 2.0, 2.0]])
    _scatter_by_slot(cache, slots, value)
    assert torch.equal(cache[0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(cache[2], torch.tensor([2.0, 2.0, 2.0]))
    # Row 1 untouched (slot -1 skipped); value row 1 dropped.
    assert torch.equal(cache[1], torch.zeros(3))


def test_scatter_by_slot_ignores_all_padded_rows() -> None:
    cache = torch.arange(12, dtype=torch.float32).view(4, 3)
    original = cache.clone()
    slots = torch.full((2,), -1, dtype=torch.int32)
    values = torch.full((2, 3), 99.0)

    _scatter_by_slot(cache, slots, values)

    assert torch.equal(cache, original)


def test_scatter_by_slot_npu_works_without_forward_context(monkeypatch: pytest.MonkeyPatch) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    from xllm.python import kernels

    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    monkeypatch.setattr(csa_attention_module, "get_forward_context_or_none", lambda: None)
    monkeypatch.setattr(
        kernels,
        "scatter_nd_update",
        lambda cache, indices, updates: calls.append((cache, indices, updates)),
        raising=False,
    )

    with FakeTensorMode():
        cache = torch.zeros(4, 3, dtype=torch.float32, device="npu")
        slots = torch.tensor([0, -1, 2], dtype=torch.int32, device="npu")
        values = torch.ones(3, 3, dtype=torch.float32, device="npu")
        _scatter_by_slot(cache, slots, values)

    assert len(calls) == 1
    assert calls[0][1].shape == (3, 1)
    assert calls[0][2].shape == (3, 3)


def test_default_mapping_is_empty() -> None:
    m = _CompressedAttentionCacheMapping()
    assert m.cmp_cache_idx == -1
    assert m.ori_cache_idx == -1


def test_prepare_binds_compressed_metadata_to_current_forward(monkeypatch) -> None:
    backend = _make_backend()
    monkeypatch.setattr(backend, "_move_metadata_to_device", lambda compressed_metadata: None)
    monkeypatch.setattr(
        backend,
        "_build_precomputed_metadata",
        lambda compressed_metadata, metadata: None,
    )

    def make_metadata(kv_len: int, is_prefill: bool) -> SimpleNamespace:
        q_len = kv_len if is_prefill else 1
        return SimpleNamespace(
            multi_block_tables=[],
            kv_seq_lens_host=torch.tensor([kv_len], dtype=torch.int32),
            q_seq_lens_host=torch.tensor([q_len], dtype=torch.int32),
            is_prefill=is_prefill,
            is_chunked_prefill=False,
            dsa_metadata=None,
            dsa_positions=None,
            dsa_cos_sin=None,
            dsa_c4_cos_sin=None,
            dsa_c128_cos_sin=None,
            dsa_graph_block_table_cols=0,
            dsa_graph_mode=False,
        )

    prefill = make_metadata(84, True)
    backend.prepare(prefill)
    backend.prepare_dsa_metadata_for_forward()
    prefill_compressed_metadata = prefill.dsa_metadata
    assert prefill_compressed_metadata.max_query_len == 84

    decode = make_metadata(85, False)
    backend.prepare(decode)
    backend.prepare_dsa_metadata_for_forward()
    assert decode.dsa_metadata is not prefill_compressed_metadata
    assert decode.dsa_metadata.max_query_len == 1
    assert prefill.dsa_metadata is prefill_compressed_metadata


def test_prepare_clamps_target_block_tables_to_draft_groups(monkeypatch) -> None:
    """An MTP draft backend consumes only its registered SWA manager table."""
    backend = CsaAttentionBackend(
        compress_ratios=[1],
        window_size=128,
        n_layers=1,
        num_heads=8,
        attn_head_dim=512,
        index_topk=512,
        index_n_heads=64,
        index_head_dim=128,
        rope_head_dim=64,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    monkeypatch.setattr(backend, "_move_metadata_to_device", lambda _metadata: None)
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    metadata = SimpleNamespace(
        multi_block_tables=[
            torch.tensor([[10]], dtype=torch.int32),
            torch.tensor([[20]], dtype=torch.int32),
            torch.tensor([[30]], dtype=torch.int32),
        ],
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        new_cache_slots_host_values=[1280],
        is_prefill=True,
        is_chunked_prefill=False,
        is_dummy=False,
        dsa_positions=torch.tensor([0], dtype=torch.int64),
        dsa_cos_sin=None,
        dsa_graph_block_table_cols=0,
        dsa_graph_mode=False,
    )

    dsa = backend._build_dsa_metadata_for_forward(metadata)

    assert dsa.block_tables[0][0].tolist() == [[10]]
    assert dsa.slot_mappings[0][0].tolist() == [1280]


@pytest.mark.parametrize("index_topk", [512, 1024, 2048])
@pytest.mark.parametrize("compressed_unit_tokens", [128, 2048])
def test_empty_dp_metadata_uses_safe_lengths_and_cache_tables(
    monkeypatch,
    index_topk: int,
    compressed_unit_tokens: int,
) -> None:
    monkeypatch.setattr(dsa_metadata_module, "DSV4_COMPRESSED_BLOCK_TOKEN_SIZE", compressed_unit_tokens)
    backend = _make_backend()
    backend.index_topk = index_topk
    backend.bind_kv_caches(
        [
            LayerCache(key=None, value=None, swa=torch.empty((2, 128, 1, 1))),
            LayerCache(
                key=torch.empty((4, 128, 1, 1)),
                value=None,
                swa=torch.empty((2, 128, 1, 1)),
            ),
            LayerCache(
                key=torch.empty((1, 128, 1, 1)),
                value=None,
                swa=torch.empty((2, 128, 1, 1)),
            ),
        ]
    )
    monkeypatch.setattr(backend, "_move_metadata_to_device", lambda _metadata: None)
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    metadata = SimpleNamespace(
        multi_block_tables=[],
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        is_prefill=True,
        is_chunked_prefill=False,
        is_dummy=True,
        dsa_metadata=None,
        dsa_positions=torch.tensor([0], dtype=torch.int64),
        dsa_cos_sin=None,
        dsa_c4_cos_sin=None,
        dsa_c128_cos_sin=None,
        dsa_graph_mode=False,
        dsa_graph_block_table_cols=0,
    )

    dsa = backend._build_dsa_metadata_for_forward(metadata)

    assert dsa.seq_lens.tolist() == [index_topk]
    assert dsa.seq_lens_q.tolist() == [1]
    assert dsa.max_seq_len == index_topk
    assert dsa.max_query_len == 1
    c4_group = backend.group_infos[backend.caches_info[1][0].group_id]
    c4_tokens = index_topk // c4_group.ratio
    expected_c4_columns = (c4_tokens + c4_group.block_size - 1) // c4_group.block_size
    assert tuple(dsa.block_tables[1][0].shape) == (1, expected_c4_columns)
    assert dsa.slot_mappings[0][0].tolist() == [127]
    assert dsa.slot_mappings[1][0].tolist() == [(c4_tokens - 1) % c4_group.block_size]
    c128_group = backend.group_infos[backend.caches_info[2][0].group_id]
    assert dsa.slot_mappings[2][0].tolist() == [(index_topk // c128_group.ratio - 1) % c128_group.block_size]


@pytest.mark.parametrize("index_topk", [512, 1024, 2048])
@pytest.mark.parametrize("compressed_unit_tokens", [128, 2048])
def test_empty_dp_graph_metadata_preserves_bucket_rows(
    monkeypatch, index_topk: int, compressed_unit_tokens: int
) -> None:
    monkeypatch.setattr(dsa_metadata_module, "DSV4_COMPRESSED_BLOCK_TOKEN_SIZE", compressed_unit_tokens)
    backend = _make_backend()
    backend.index_topk = index_topk
    monkeypatch.setattr(backend, "_move_metadata_to_device", lambda _metadata: None)
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    graph_block_table_cols = max(8, index_topk // 128)
    metadata = SimpleNamespace(
        multi_block_tables=[
            torch.full((4, graph_block_table_cols), -1, dtype=torch.int32),
            torch.full((4, graph_block_table_cols), -1, dtype=torch.int32),
            torch.full((4, graph_block_table_cols), -1, dtype=torch.int32),
        ],
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=[1, 0, 0, 0],
        q_seq_lens_host=None,
        q_seq_lens=torch.ones(4, dtype=torch.int32),
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=True,
        dsa_metadata=None,
        dsa_positions=torch.zeros(4, dtype=torch.int64),
        dsa_cos_sin=None,
        dsa_c4_cos_sin=None,
        dsa_c128_cos_sin=None,
        dsa_graph_mode=True,
        dsa_graph_block_table_cols=graph_block_table_cols,
    )
    dummy_tables = backend._build_empty_dp_block_tables(
        list(metadata.multi_block_tables),
        4,
        index_topk,
        torch.device("cpu"),
        graph_mode=True,
        graph_block_table_capacity_cols=graph_block_table_cols,
    )

    dsa = backend._build_dsa_metadata_for_forward(metadata)

    assert tuple(dummy_tables[0].shape) == (4, 1)
    assert dsa.seq_lens.tolist() == [index_topk] * 4
    assert dsa.seq_lens_q.tolist() == [1, 1, 1, 1]
    assert tuple(dsa.block_tables[0][0].shape) == (4, graph_block_table_cols)
    expected_swa_column = index_topk // 128 - 1
    assert dsa.block_tables[0][0][:, expected_swa_column].tolist() == [0, 0, 0, 0]
    assert tuple(dsa.block_tables[1][0].shape) == (4, graph_block_table_cols)
    assert dsa.block_tables[1][0][:, 0].tolist() == [0, 0, 0, 0]
    assert dsa.slot_mappings[0][0][:4].tolist() == [127, 127, 127, 127]
    c4_group = backend.group_infos[backend.caches_info[1][0].group_id]
    expected_c4_slot = (index_topk // c4_group.ratio - 1) % c4_group.block_size
    assert dsa.slot_mappings[1][0][:4].tolist() == [expected_c4_slot] * 4
    c128_group = backend.group_infos[backend.caches_info[2][0].group_id]
    expected_c128_slot = (index_topk // c128_group.ratio - 1) % c128_group.block_size
    assert dsa.slot_mappings[2][0][:4].tolist() == [expected_c128_slot] * 4


def test_empty_dp_graph_rejects_insufficient_swa_capacity() -> None:
    backend = _make_backend()
    block_tables = [torch.full((4, 8), -1, dtype=torch.int32) for _ in range(3)]

    with pytest.raises(ValueError, match="SWA block table capacity is too small"):
        backend._build_empty_dp_block_tables(
            block_tables,
            4,
            2048,
            torch.device("cpu"),
            graph_mode=True,
            graph_block_table_capacity_cols=8,
        )


def test_prepare_clears_previous_forward_state() -> None:
    backend = _make_backend()
    metadata = SimpleNamespace(
        multi_block_tables=[],
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        is_prefill=False,
        is_chunked_prefill=False,
        dsa_metadata=object(),
        dsa_positions=torch.ones(1),
        dsa_cos_sin=torch.ones(1),
        dsa_c4_cos_sin=torch.ones(1),
        dsa_c128_cos_sin=torch.ones(1),
        dsa_graph_mode=True,
    )
    backend.attach_compressor(lambda *_args: None)
    backend.attach_indexer(lambda *_args: None)
    backend.reset_forward(metadata)
    assert metadata.dsa_metadata is None
    assert metadata.dsa_positions is None
    assert metadata.dsa_cos_sin is None
    assert metadata.dsa_c4_cos_sin is None
    assert metadata.dsa_c128_cos_sin is None
    assert metadata.dsa_graph_mode is False
    assert not hasattr(backend, "_compressor_fn")
    assert not hasattr(backend, "_indexer_fn")


# -- sequence-length supply: expanded spec-verify + idle-shard dummy ---------


def _sequence_metadata(**overrides) -> SimpleNamespace:
    """The minimal metadata surface the DSA builder reads.

    The three DSA compression levels are named c1/c4/c128 by their ratio
    (1/4/128): ``dsa_cos_sin`` is the full-resolution (c1) table and
    ``dsa_c4_cos_sin``/``dsa_c128_cos_sin`` the compressed ones. Upstream
    numbered naming (c0/c1/c2) maps to c1/c4/c128 here. The AICPU tiling
    fields (``c1_metadata``/``c4_metadata``/``c128_metadata``) are filled by
    the backend on the DSAMetadata object, not read by this builder surface.
    """
    values = {
        "multi_block_tables": [],
        "kv_seq_lens_host": None,
        "q_seq_lens_host": None,
        "q_seq_lens": None,
        "is_prefill": False,
        "is_chunked_prefill": False,
        "dsa_metadata": None,
        "dsa_positions": torch.tensor([0]),
        "dsa_cos_sin": None,
        "dsa_c4_cos_sin": None,
        "dsa_c128_cos_sin": None,
        "dsa_graph_block_table_cols": 0,
        "dsa_graph_mode": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _prepared_metadata(backend, metadata) -> None:
    backend.prepare_dsa_metadata_for_forward(metadata)


def test_idle_shard_without_host_lengths_keeps_one_dummy_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idle DP shard runs the forward with no sequences; stay at batch 1.

    Without the fallback the DSA metadata degenerates to batch_size == 0 and the
    model's attention does a bare ``max(q_lens)`` on an empty list
    (csa2_attention.py), which is the B4 smoke signature.
    """
    backend = _make_backend()
    monkeypatch.setattr(backend, "_move_metadata_to_device", lambda _metadata: None)
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    metadata = _sequence_metadata()  # one fake query row, no sequence lengths

    _prepared_metadata(backend, metadata)

    dsa = metadata.dsa_metadata
    assert dsa.seq_lens_q.tolist() == [1]
    assert dsa.seq_lens.tolist() == [1]
    assert dsa.seq_lens_q.numel() == 1
    assert dsa.block_tables == []
    # The literal B4 raise point: csa2_attention maxes the per-sequence q lens.
    assert max(int(value) for value in dsa.seq_lens_q.tolist()) == 1


def test_idle_shard_dummy_rows_follow_the_query_row_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dummy sequence count stays aligned with the padded query rows."""
    backend = _make_backend()
    monkeypatch.setattr(backend, "_move_metadata_to_device", lambda _metadata: None)
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    metadata = _sequence_metadata(dsa_positions=torch.tensor([0, 1, 2]))

    _prepared_metadata(backend, metadata)

    assert metadata.dsa_metadata.seq_lens_q.tolist() == [1]
    assert metadata.dsa_metadata.max_query_len >= 1


def test_expanded_spec_verify_lengths_supply_the_missing_host_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A packed verify batch keeps its lengths in the expanded view."""
    backend = _make_backend()
    monkeypatch.setattr(backend, "_move_metadata_to_device", lambda _metadata: None)
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    expanded = SimpleNamespace(
        enabled=True,
        kv_seq_lens=torch.tensor([7, 9], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([7, 9], dtype=torch.int32),
        kv_seq_lens_host_values=[7, 9],
        q_seq_lens_host=None,
        q_seq_lens_host_values=None,
    )
    metadata = _sequence_metadata(expanded_decode_metadata=expanded)

    _prepared_metadata(backend, metadata)

    dsa = metadata.dsa_metadata
    assert dsa.seq_lens.tolist() == [7, 9]
    assert dsa.seq_lens_q.tolist() == [1, 1]


def test_real_host_lengths_are_never_replaced_by_the_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal batch keeps the C++ host view verbatim (no dummy, no expanded)."""
    backend = _make_backend()
    monkeypatch.setattr(backend, "_move_metadata_to_device", lambda _metadata: None)
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    expanded = SimpleNamespace(
        enabled=True,
        kv_seq_lens=torch.tensor([99], dtype=torch.int32),
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=[99],
        q_seq_lens_host=None,
        q_seq_lens_host_values=None,
    )
    metadata = _sequence_metadata(
        kv_seq_lens_host=torch.tensor([5, 6], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1, 1], dtype=torch.int32),
        expanded_decode_metadata=expanded,
    )

    _prepared_metadata(backend, metadata)

    assert metadata.dsa_metadata.seq_lens.tolist() == [5, 6]
    assert metadata.dsa_metadata.seq_lens_q.tolist() == [1, 1]


def test_empty_forward_without_positions_stays_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No query rows at all -> no synthetic sequence (nothing to align to)."""
    backend = _make_backend()
    monkeypatch.setattr(backend, "_move_metadata_to_device", lambda _metadata: None)
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    metadata = _sequence_metadata(dsa_positions=torch.empty(0, dtype=torch.int64))

    _prepared_metadata(backend, metadata)

    assert metadata.dsa_metadata.seq_lens_q.numel() == 0


def test_dsa_api_aliases_are_equivalent(monkeypatch) -> None:
    backend = _make_backend()
    monkeypatch.setattr(backend, "_move_metadata_to_device", lambda _metadata: None)
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    metadata = SimpleNamespace(
        multi_block_tables=[],
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        is_prefill=False,
        is_chunked_prefill=False,
        dsa_metadata=None,
        dsa_positions=torch.tensor([0]),
        dsa_cos_sin=None,
        dsa_c4_cos_sin=None,
        dsa_c128_cos_sin=None,
        dsa_graph_mode=False,
    )
    backend.prepare_dsa_metadata_for_forward(metadata)
    canonical = metadata.dsa_metadata
    metadata.dsa_metadata = None
    backend.prepare_csa_metadata_for_forward(metadata)
    aliased = metadata.dsa_metadata
    # The alias must resolve to the canonical DSA build semantics, not
    # merely produce a fresh object: field-by-field equality is the real
    # contract of ``prepare_csa_metadata_for_forward`` (a pure delegation).
    assert aliased is not canonical
    assert aliased.seq_lens.tolist() == canonical.seq_lens.tolist() == [1]
    assert aliased.seq_lens_q.tolist() == canonical.seq_lens_q.tolist() == [1]


def test_graph_mode_is_recorded_on_metadata() -> None:
    backend = _make_backend()
    metadata = SimpleNamespace()
    backend.prepare(metadata, graph_mode=True)
    assert metadata.dsa_graph_mode is True


def test_decode_precomputed_metadata_matches_cpp_contract(monkeypatch) -> None:
    backend = _make_backend()
    sparse_calls, qli_calls = _patch_metadata_kernels(monkeypatch)
    compressed_metadata = SimpleNamespace(
        actual_seq_lengths_query=torch.tensor([0, 1], dtype=torch.int32),
        actual_seq_lengths_kv=torch.tensor([85], dtype=torch.int32),
        seq_lens_q=torch.tensor([1], dtype=torch.int32),
        seq_lens=torch.tensor([85], dtype=torch.int32),
        max_query_len=1,
        max_seq_len=85,
    )
    metadata = SimpleNamespace(
        max_query_len=1,
        max_seq_len=85,
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([85], dtype=torch.int32),
    )

    backend._build_precomputed_metadata(compressed_metadata, metadata)

    assert [call["cmp_ratio"] for call in sparse_calls] == [1, 4, 128]
    assert all(call["head_dim"] == 512 for call in sparse_calls)
    assert all(call["cu_seqlens_q"].tolist() == [0, 1] for call in sparse_calls)
    assert all(call["cu_seqlens_ori_kv"].numel() == 0 for call in sparse_calls)
    assert sparse_calls[1]["cmp_topk"] == 512
    assert qli_calls[0]["actual_seq_lengths_query"].tolist() == [1]
    assert qli_calls[0]["actual_seq_lengths_key"].tolist() == [85]
    assert qli_calls[0]["head_dim"] == 128
    assert compressed_metadata.precomputed_metadata_inputs[0] is compressed_metadata.actual_seq_lengths_query


def test_a5_decode_precomputed_metadata_matches_cpp_contract(monkeypatch) -> None:
    backend = _make_backend()
    sparse_calls, qli_calls = _patch_metadata_kernels(monkeypatch, use_a5=True)
    compressed_metadata = SimpleNamespace(
        actual_seq_lengths_query=torch.tensor([0, 1], dtype=torch.int32),
        actual_seq_lengths_kv=torch.tensor([85], dtype=torch.int32),
        seq_lens_q=torch.tensor([1], dtype=torch.int32),
        seq_lens=torch.tensor([85], dtype=torch.int32),
        max_query_len=1,
        max_seq_len=85,
    )
    metadata = SimpleNamespace(
        max_query_len=1,
        max_seq_len=85,
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([85], dtype=torch.int32),
    )

    backend._build_precomputed_metadata(compressed_metadata, metadata)

    assert [call["cmp_ratio"] for call in sparse_calls] == [1, 4, 128]
    assert all(call["head_dim"] == 512 for call in sparse_calls)
    assert all(call["cu_seqlens_q"].tolist() == [0, 1] for call in sparse_calls)
    assert all(call["cu_seqlens_ori_kv"] is None for call in sparse_calls)
    assert all(call["seqused_ori_kv"].tolist() == [85] for call in sparse_calls)
    assert all(call["layout_kv"] == "PA_BBND" for call in sparse_calls)
    assert sparse_calls[1]["seqused_cmp_kv"].tolist() == [21]
    assert sparse_calls[1]["cmp_residual_kv"].tolist() == [1]
    assert sparse_calls[1]["cmp_topk"] == 512
    assert qli_calls[0]["seqused_k"].tolist() == [21]
    assert qli_calls[0]["cmp_residual_k"].tolist() == [1]
    assert qli_calls[0]["head_dim"] == 128
    assert qli_calls[0]["quant_mode"] == 2
    assert qli_calls[0]["layout_k"] == "PA_BBND"
    assert compressed_metadata.precomputed_metadata_inputs[0] is compressed_metadata.actual_seq_lengths_query


def test_dspark_native_decode_metadata_uses_expanded_window(monkeypatch) -> None:
    backend = CsaAttentionBackend(
        compress_ratios=[1],
        window_size=128,
        n_layers=1,
        num_heads=8,
        attn_head_dim=512,
        index_topk=512,
        index_n_heads=64,
        index_head_dim=128,
        rope_head_dim=64,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        dspark_block_size=5,
        dspark_use_native_sas=True,
    )
    sparse_calls, _ = _patch_metadata_kernels(monkeypatch)
    compressed_metadata = SimpleNamespace(
        actual_seq_lengths_query=torch.tensor([0, 5], dtype=torch.int32),
        actual_seq_lengths_kv=torch.tensor([10], dtype=torch.int32),
        seq_lens_q=torch.tensor([5], dtype=torch.int32),
        seq_lens=torch.tensor([10], dtype=torch.int32),
        block_tables=[[torch.tensor([[0]], dtype=torch.int32)]],
        max_query_len=5,
        max_seq_len=10,
    )
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        max_query_len=5,
        max_seq_len=10,
        q_seq_lens_host=torch.tensor([5], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([10], dtype=torch.int32),
    )

    backend._build_precomputed_metadata(compressed_metadata, metadata)

    assert all(call["cu_seqlens_ori_kv"].numel() == 0 for call in sparse_calls)
    assert all(call["ori_win_left"] == 132 for call in sparse_calls)
    assert compressed_metadata.sparse_metadata_ori_win_left == 132
    assert compressed_metadata.explicit_swa_indices.shape == (5, 1, 256)


def test_dspark_native_attention_uses_explicit_swa_indices(monkeypatch) -> None:
    backend = CsaAttentionBackend(
        compress_ratios=[1],
        window_size=4,
        n_layers=1,
        num_heads=2,
        attn_head_dim=4,
        index_topk=0,
        index_n_heads=2,
        index_head_dim=4,
        rope_head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        dspark_block_size=3,
        dspark_use_native_sas=True,
    )
    swa = torch.zeros(3, 4, 1, 4)
    backend.bind_kv_caches([LayerCache(key=None, value=None, swa=swa)])
    compressed_metadata = backend._builder.build(
        multi_block_tables=[torch.tensor([[0, 1, 2]], dtype=torch.int32)],
        kv_seq_lens=[6],
        q_seq_lens=[3],
        positions=torch.tensor([3, 4, 5], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    backend._build_dspark_swa_metadata(compressed_metadata)
    compressed_metadata.c1_metadata = torch.zeros(1, dtype=torch.int32)
    compressed_metadata.sparse_metadata_ori_win_left = 6
    backend._metadata = SimpleNamespace(
        dsa_metadata=compressed_metadata,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    calls: list[dict] = []

    def fake_sparse_attn(**kwargs):
        calls.append(kwargs)
        return kwargs["q"].clone(), torch.empty(0)

    monkeypatch.setattr(csa_attention_module, "_sparse_attn_sharedkv", fake_sparse_attn)
    layer = SimpleNamespace(layer_id=0, attn_sink=None, attn_sink_loaded=False)
    query = torch.zeros(3, 2, 4)
    kv = torch.arange(12, dtype=torch.float32).view(3, 1, 4)

    backend.execute(query, kv, kv, layer)

    assert len(calls) == 1
    assert calls[0]["ori_sparse_indices"] is compressed_metadata.explicit_swa_indices
    assert calls[0]["ori_sparse_indices"].shape == (3, 1, 128)
    assert calls[0]["ori_win_left"] == 6
    assert calls[0]["cu_seqlens_ori_kv"] is None


@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_dspark_native_mixed_ratio_decode_uses_expanded_window(
    monkeypatch: pytest.MonkeyPatch,
    compress_ratio: int,
) -> None:
    backend = CsaAttentionBackend(
        compress_ratios=[compress_ratio],
        window_size=4,
        n_layers=1,
        num_heads=2,
        attn_head_dim=4,
        index_topk=2,
        index_n_heads=2,
        index_head_dim=4,
        rope_head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        dspark_block_size=3,
        dspark_use_native_sas=True,
    )
    backend.bind_kv_caches([LayerCache(key=None, value=None, swa=torch.zeros(1, 4, 1, 4))])
    compressed_metadata = backend._builder.build(
        multi_block_tables=[torch.tensor([[0]], dtype=torch.int32) for _ in backend.group_infos],
        kv_seq_lens=[1],
        q_seq_lens=[1],
        positions=torch.tensor([0], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    compressed_metadata.cos_table = torch.empty(0)
    compressed_metadata.sin_table = torch.empty(0)
    compressed_metadata.c4_cos = torch.empty(0)
    compressed_metadata.c128_cos = torch.empty(0)
    metadata = SimpleNamespace(
        dsa_metadata=compressed_metadata,
        is_prefill=False,
        is_chunked_prefill=False,
        max_query_len=1,
        max_seq_len=1,
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
    )
    calls: list[dict] = []
    _patch_metadata_kernels(monkeypatch)
    monkeypatch.setattr(
        csa_attention_module,
        "_sparse_attn_sharedkv",
        lambda **kwargs: (calls.append(kwargs) or kwargs["q"].clone(), torch.empty(0)),
    )
    backend._build_precomputed_metadata(compressed_metadata, metadata)
    backend._metadata = metadata

    backend.execute(
        torch.zeros(1, 2, 4),
        torch.zeros(1, 1, 4),
        torch.zeros(1, 1, 4),
        SimpleNamespace(layer_id=0, attn_sink=None, attn_sink_loaded=False),
    )

    assert compressed_metadata.sparse_metadata_ori_win_left == 6
    assert calls[0]["ori_win_left"] == 6
    assert calls[0]["ori_sparse_indices"] is None


def test_csa_execute_requires_model_compressor(monkeypatch) -> None:
    backend = _make_backend()
    empty_cache = LayerCache(key=None, value=None)
    cmp_cache = torch.zeros(2, 128, 1, 512)
    swa_cache = torch.zeros(2, 128, 1, 512)
    backend.bind_kv_caches(
        [
            empty_cache,
            LayerCache(key=cmp_cache, value=None, swa=swa_cache),
            empty_cache,
        ]
    )
    block_tables = [[], [torch.tensor([[1]], dtype=torch.int32) for _ in range(8)], []]
    slot_mappings = [[], [torch.tensor([128], dtype=torch.int32) for _ in range(8)], []]
    compressed_metadata = SimpleNamespace(
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        actual_seq_lengths_query=torch.tensor([0, 1], dtype=torch.int32),
        actual_seq_lengths_kv=torch.tensor([1], dtype=torch.int32),
        input_positions=torch.tensor([0], dtype=torch.int64),
        cos_table=torch.zeros(1, 64),
        sin_table=torch.zeros(1, 64),
        c4_cos=torch.zeros(1, 64),
        c4_sin=torch.zeros(1, 64),
        c128_cos=torch.zeros(1, 64),
        c128_sin=torch.zeros(1, 64),
        c4_metadata=torch.zeros(1, dtype=torch.int32),
    )
    backend._metadata = SimpleNamespace(
        dsa_metadata=compressed_metadata,
        is_prefill=False,
        is_chunked_prefill=False,
    )

    with pytest.raises(RuntimeError, match="compressor is required"):
        backend.execute(
            torch.zeros(1, 8, 512),
            torch.zeros(1, 1, 512),
            torch.zeros(1, 1, 512),
            SimpleNamespace(layer_id=1, attn_sink=None),
        )


def test_forward_rope_state_is_owned_by_each_metadata(monkeypatch) -> None:
    backend = _make_backend()
    monkeypatch.setattr(backend, "_move_metadata_to_device", lambda compressed_metadata: None)
    monkeypatch.setattr(
        backend,
        "_build_precomputed_metadata",
        lambda compressed_metadata, metadata: None,
    )

    def make_metadata(kv_len: int, q_len: int) -> SimpleNamespace:
        return SimpleNamespace(
            multi_block_tables=[],
            kv_seq_lens_host=torch.tensor([kv_len], dtype=torch.int32),
            q_seq_lens_host=torch.tensor([q_len], dtype=torch.int32),
            is_prefill=q_len > 1,
            is_chunked_prefill=False,
            dsa_metadata=None,
            dsa_positions=None,
            dsa_cos_sin=None,
            dsa_c4_cos_sin=None,
            dsa_c128_cos_sin=None,
            dsa_graph_block_table_cols=0,
            dsa_graph_mode=False,
        )

    rope_cache = torch.arange(256 * 8, dtype=torch.float32).view(256, 8)
    prefill = make_metadata(84, 84)
    backend.prepare(prefill)
    backend.attach_rope_tables(
        torch.arange(84),
        rope_cache,
        csa_cos_sin=rope_cache,
        hca_cos_sin=rope_cache,
        metadata=prefill,
    )

    decode = make_metadata(85, 1)
    backend.prepare(decode)
    backend.attach_rope_tables(
        torch.tensor([84]),
        rope_cache,
        csa_cos_sin=rope_cache,
        hca_cos_sin=rope_cache,
        metadata=decode,
    )

    backend.prepare_csa_metadata_for_forward(prefill)

    assert prefill.dsa_metadata.input_positions.numel() == 84
    assert decode.dsa_positions.numel() == 1
    assert prefill.dsa_positions.data_ptr() != decode.dsa_positions.data_ptr()


def test_cp_localization_keeps_runtime_metadata_read_only(monkeypatch) -> None:
    backend = _make_backend()
    dsa = backend._builder.build(
        multi_block_tables=[],
        kv_seq_lens=[4],
        q_seq_lens=[4],
        positions=torch.arange(4, dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=True,
        is_chunked_prefill=False,
    )

    class _ReadOnlyMetadata:
        def __init__(self) -> None:
            self.dsa_metadata = dsa
            self.dp_execution_token_counts = (4,)

        @property
        def q_seq_lens_host(self) -> torch.Tensor:
            return torch.tensor([4], dtype=torch.int32)

        @property
        def kv_seq_lens_host(self) -> torch.Tensor:
            return torch.tensor([4], dtype=torch.int32)

        @property
        def max_query_len(self) -> int:
            return 4

        @property
        def max_seq_len(self) -> int:
            return 4

    captured: dict[str, int] = {}

    def capture_precomputed(
        _dsa,
        _metadata,
        *,
        cu_seqlens_ori_kv_override=None,
        max_query_len_override=None,
        max_seq_len_override=None,
    ) -> None:
        del cu_seqlens_ori_kv_override
        captured["max_query_len"] = max_query_len_override
        captured["max_seq_len"] = max_seq_len_override

    monkeypatch.setattr(
        backend,
        "_build_dsa_rope_metadata",
        lambda *_args: {1: (torch.zeros(2, 1), torch.zeros(2, 1))},
    )
    monkeypatch.setattr(backend, "_build_precomputed_metadata", capture_precomputed)
    metadata = _ReadOnlyMetadata()
    cp_context = build_deepseek_v4_cp_context(
        2,
        0,
        [4],
        [4],
        torch.arange(4, dtype=torch.int64),
    )

    backend.localize_dsa_metadata_for_cp(cp_context, metadata)

    assert dsa.seq_lens_q.tolist() == [2]
    assert dsa.seq_lens.tolist() == [2]
    assert captured == {"max_query_len": 2, "max_seq_len": 2}
    assert metadata.q_seq_lens_host.tolist() == [4]
    assert metadata.kv_seq_lens_host.tolist() == [4]


def test_cp_prefill_localization_clears_dspark_native_swa_indices(monkeypatch) -> None:
    backend = CsaAttentionBackend(
        compress_ratios=[1],
        window_size=4,
        n_layers=1,
        num_heads=2,
        attn_head_dim=4,
        index_topk=0,
        index_n_heads=2,
        index_head_dim=4,
        rope_head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        dspark_block_size=3,
        dspark_use_native_sas=True,
    )
    dsa = backend._builder.build(
        multi_block_tables=[torch.tensor([[0]], dtype=torch.int32)],
        kv_seq_lens=[4],
        q_seq_lens=[4],
        positions=torch.arange(4, dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=True,
        is_chunked_prefill=False,
    )
    backend._build_dspark_swa_metadata(dsa)
    assert dsa.explicit_swa_indices.shape[0] == 4

    monkeypatch.setattr(
        backend,
        "_build_dsa_rope_metadata",
        lambda *_args: {1: (torch.zeros(2, 1), torch.zeros(2, 1))},
    )
    _patch_metadata_kernels(monkeypatch)
    metadata = SimpleNamespace(
        dsa_metadata=dsa,
        dp_execution_token_counts=(4,),
        is_prefill=True,
        is_chunked_prefill=False,
        q_seq_lens_host=torch.tensor([4], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([4], dtype=torch.int32),
        max_query_len=4,
        max_seq_len=4,
    )
    cp_context = build_deepseek_v4_cp_context(
        2,
        0,
        [4],
        [4],
        torch.arange(4, dtype=torch.int64),
    )

    backend.localize_dsa_metadata_for_cp(cp_context, metadata)

    assert dsa.actual_seq_lengths_query.tolist() == [0, 2]
    assert dsa.actual_seq_lengths_kv.tolist() == [2]
    assert dsa.explicit_swa_indices is None


def test_graph_dsa_refresh_preserves_tensor_addresses() -> None:
    def make_metadata(value: int, seq_rows: int) -> SimpleNamespace:
        manager_block_table = torch.full((seq_rows, 2), value, dtype=torch.int32)
        manager_slot_mapping = torch.full((seq_rows,), value, dtype=torch.int32)
        return SimpleNamespace(
            seq_lens=torch.full((seq_rows,), value, dtype=torch.int32),
            block_tables=[[manager_block_table]],
            slot_mappings=[[manager_slot_mapping]],
            manager_block_tables=[manager_block_table],
            manager_slot_mappings=[manager_slot_mapping],
            input_rope_by_ratio={
                1: (
                    torch.full((2, 2), value, dtype=torch.float32),
                    torch.full((2, 2), value, dtype=torch.float32),
                )
            },
            explicit_swa_indices=torch.full((seq_rows, 1, 128), value, dtype=torch.int32),
            sparse_metadata_ori_win_left=value,
            max_query_len=value,
            max_seq_len=value,
            is_acl_graph=True,
            precomputed_metadata_inputs=(),
        )

    persistent = make_metadata(1, 4)
    refreshed = make_metadata(7, 2)
    seq_lens_ptr = persistent.seq_lens.data_ptr()
    block_table_ptr = persistent.block_tables[0][0].data_ptr()
    swa_indices_ptr = persistent.explicit_swa_indices.data_ptr()
    rope_ptr = persistent.input_rope_by_ratio[1][0].data_ptr()

    CsaAttentionBackend._copy_graph_dsa_metadata(persistent, refreshed)

    assert persistent.seq_lens.data_ptr() == seq_lens_ptr
    assert persistent.seq_lens.tolist() == [7, 7, 0, 0]
    assert persistent.block_tables[0][0].data_ptr() == block_table_ptr
    assert persistent.block_tables[0][0].tolist() == [[7, 7], [7, 7], [-1, -1], [-1, -1]]
    assert persistent.slot_mappings[0][0].tolist() == [7, 7, -1, -1]
    assert persistent.input_rope_by_ratio[1][0].data_ptr() == rope_ptr
    assert persistent.input_rope_by_ratio[1][0].tolist() == [[7.0, 7.0], [7.0, 7.0]]
    assert persistent.max_query_len == 7
    assert persistent.max_seq_len == 7


def test_graph_replay_refreshes_qli_expanded_rope_in_place() -> None:
    """The QLI indexer's expanded RoPE cache outlives the eager pre-run.

    Graph capture binds the cached expansion's address (the capture hit the
    identity-keyed cache), so the expansion op never enters the graph: the
    replay refresh must rewrite the cached content from the refreshed base
    tables or every replay rotates the indexer query with the pre-run
    positions.
    """
    backend = csa_attention_module.DsaAttentionBackend(
        compress_ratios=[4],
        window_size=4,
        n_layers=1,
        num_heads=1,
        attn_head_dim=4,
        index_topk=4,
        index_n_heads=1,
        index_head_dim=4,
        rope_head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    def make_dsa(kv_len: int, rope_value: int) -> SimpleNamespace:
        dsa = backend._builder.build(
            multi_block_tables=[
                torch.tensor([[1, 2]], dtype=torch.int32),
                torch.tensor([[1, 2]], dtype=torch.int32),
            ],
            kv_seq_lens=[kv_len],
            q_seq_lens=[1],
            positions=torch.tensor([kv_len - 1]),
            dsa_cos_sin=None,
            is_prefill=False,
            is_chunked_prefill=False,
            new_cache_slots=None,
            enable_graph=True,
            graph_block_table_capacity_cols=2,
        )
        pair = (
            torch.full((1, 1), float(rope_value)),
            torch.full((1, 1), float(rope_value)),
        )
        dsa.input_rope_by_ratio = {4: pair}
        dsa.cos_table, dsa.sin_table = pair
        return dsa

    persistent, refreshed = make_dsa(7, 1), make_dsa(8, 7)
    expanded = (
        persistent.cos_table.repeat_interleave(2, dim=-1),
        persistent.sin_table.repeat_interleave(2, dim=-1),
    )
    rope_key = (id(persistent.cos_table), id(persistent.sin_table), torch.float32)
    persistent.qli_expanded_rope[rope_key] = expanded
    persistent.qli_v2_metadata = (
        persistent.seq_lens_q,
        persistent.actual_seq_lengths_kv // 4,
        persistent.actual_seq_lengths_kv % 4,
        persistent.actual_seq_lengths_query,
    )
    metadata = SimpleNamespace(dsa_metadata=persistent)
    backend._build_dsa_metadata_for_forward = lambda *args, **kwargs: refreshed
    state = AclGraphExecutionState(persistent_buffers={})
    context = ForwardContext(
        attention_backend=backend,
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[],
        execution_state=state,
    )

    with forward_context(context):
        # Pin the entry-owned slot guard the same way the captured forward
        # does, then refresh.
        csa_attention_module._slot_guard_tensors(persistent.manager_slot_mappings[0], 1, torch.device("cpu"))
        backend.refresh_dsa_metadata_for_graph_replay(metadata)
        guards = csa_attention_module._slot_guard_tensors(persistent.manager_slot_mappings[0], 1, torch.device("cpu"))

    assert persistent.cos_table.tolist() == [[7.0]]
    assert expanded[0].tolist() == [[7.0, 7.0]]
    assert expanded[1].tolist() == [[7.0, 7.0]]
    assert expanded[0].data_ptr() == persistent.qli_expanded_rope[rope_key][0].data_ptr()
    assert guards[1].tolist() == persistent.manager_slot_mappings[0].clamp_min(0).tolist()
    # The guard lives on this entry's execution state, not the module-wide
    # fallback registry.
    entry_guards = state.persistent_buffers.get(csa_attention_module._SLOT_GUARD_STATE_KEY)
    assert entry_guards is not None and len(entry_guards) == 1


def test_graph_dsa_build_uses_stable_host_length_values(monkeypatch) -> None:
    backend = _make_backend()
    monkeypatch.setattr(backend, "_move_metadata_to_device", lambda _metadata: None)
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    metadata = SimpleNamespace(
        multi_block_tables=[],
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=[9, 1],
        q_seq_lens_host=None,
        q_seq_lens=None,
        is_prefill=False,
        is_chunked_prefill=False,
        dsa_positions=torch.tensor([8, 0], dtype=torch.int64),
        dsa_cos_sin=None,
        dsa_graph_mode=True,
        dsa_graph_block_table_cols=4,
    )

    dsa = backend._build_dsa_metadata_for_forward(metadata)

    assert dsa.seq_lens.tolist() == [9, 1]
    assert dsa.seq_lens_q.tolist() == [1, 1]
    assert dsa.start_pos.tolist() == [8, 0]


def test_prefill_and_dummy_decode_metadata_stay_in_sync_with_execute(
    monkeypatch,
) -> None:
    backend = CsaAttentionBackend(
        compress_ratios=[1],
        window_size=128,
        n_layers=1,
        num_heads=8,
        attn_head_dim=512,
        index_topk=512,
        index_n_heads=64,
        index_head_dim=128,
        rope_head_dim=64,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        dspark_block_size=5,
        dspark_use_native_sas=True,
    )
    swa = torch.zeros(2, 128, 1, 512, dtype=torch.float32)
    backend.bind_kv_caches([LayerCache(key=None, value=None, swa=swa)])
    block_table = torch.tensor([[1]], dtype=torch.int32)
    layer = SimpleNamespace(
        layer_id=0,
        attn_sink=torch.tensor([1.0], dtype=torch.float32),
        attn_sink_loaded=False,
    )
    calls: list[dict] = []

    def fake_sparse_attn(**kwargs):
        calls.append(kwargs)
        return kwargs["q"].clone(), torch.empty(0)

    monkeypatch.setattr(csa_attention_module, "_sparse_attn_sharedkv", fake_sparse_attn)
    _patch_metadata_kernels(monkeypatch)

    def prepare_step(
        kv_len: int,
        q_len: int,
        is_prefill: bool,
        *,
        is_chunked_prefill: bool = False,
        is_dummy: bool = False,
    ):
        compressed_metadata = backend._builder.build(
            multi_block_tables=[block_table],
            kv_seq_lens=[kv_len],
            q_seq_lens=[q_len],
            positions=torch.arange(kv_len - q_len, kv_len, dtype=torch.int64),
            dsa_cos_sin=None,
            is_prefill=is_prefill,
            is_chunked_prefill=is_chunked_prefill,
        )
        metadata = SimpleNamespace(
            dsa_metadata=compressed_metadata,
            is_prefill=is_prefill,
            is_chunked_prefill=is_chunked_prefill,
            is_dummy=is_dummy,
            max_query_len=q_len,
            max_seq_len=kv_len,
            q_seq_lens_host=torch.tensor([q_len], dtype=torch.int32),
            kv_seq_lens_host=torch.tensor([kv_len], dtype=torch.int32),
        )
        backend._build_precomputed_metadata(compressed_metadata, metadata)
        backend._metadata = metadata
        return compressed_metadata

    prefill_metadata = prepare_step(kv_len=2, q_len=2, is_prefill=True)
    assert prefill_metadata.explicit_swa_indices is None
    assert prefill_metadata.sparse_metadata_ori_win_left == 127
    prefill_kv = torch.arange(2 * 512, dtype=torch.float32).view(2, 1, 512)
    backend.execute(torch.zeros(2, 8, 512), prefill_kv, prefill_kv, layer)
    assert torch.equal(swa[1, :2], prefill_kv)
    assert calls[-1]["ori_sparse_indices"] is None
    assert calls[-1]["ori_win_left"] == 127

    decode_metadata = prepare_step(kv_len=3, q_len=1, is_prefill=False)
    assert decode_metadata.explicit_swa_indices is not None
    assert decode_metadata.sparse_metadata_ori_win_left == 132
    decode_kv = torch.full((1, 1, 512), 7.0)
    backend.execute(torch.zeros(1, 8, 512), decode_kv, decode_kv, layer)
    assert torch.equal(swa[1, 2], decode_kv[0])
    assert calls[-1]["cu_seqlens_ori_kv"] is None
    assert calls[-1]["ori_sparse_indices"] is not None
    assert calls[-1]["ori_win_left"] == 132
    assert calls[-1]["sinks"] is None

    dummy_metadata = prepare_step(kv_len=3, q_len=1, is_prefill=True, is_dummy=True)
    assert dummy_metadata.explicit_swa_indices is not None
    assert dummy_metadata.sparse_metadata_ori_win_left == 132
    backend.execute(torch.zeros(1, 8, 512), decode_kv, decode_kv, layer)
    assert calls[-1]["cu_seqlens_ori_kv"] is None
    assert calls[-1]["ori_sparse_indices"] is not None
    assert calls[-1]["ori_win_left"] == 132

    chunked_dummy_metadata = prepare_step(
        kv_len=3,
        q_len=1,
        is_prefill=False,
        is_chunked_prefill=True,
        is_dummy=True,
    )
    assert chunked_dummy_metadata.explicit_swa_indices is None
    assert chunked_dummy_metadata.sparse_metadata_ori_win_left == 127
    backend.execute(torch.zeros(1, 8, 512), decode_kv, decode_kv, layer)
    assert calls[-1]["cu_seqlens_ori_kv"] is not None
    assert calls[-1]["ori_sparse_indices"] is None
    assert calls[-1]["ori_win_left"] == 127

    layer.attn_sink_loaded = True
    prepare_step(kv_len=3, q_len=1, is_prefill=False)
    backend.execute(torch.zeros(1, 8, 512), decode_kv, decode_kv, layer)
    assert torch.equal(calls[-1]["sinks"], layer.attn_sink)


# ---- graph device refresh ----
# Decode-graph DSA refresh chain (pure Python, CPU): replay alias dedup and
# the in-place refresh mechanism (driven by the builder's ``inplace_target`` /
# ``dst`` parameters directly).

# --- [S4a] Alias dedup ------------------------------------------------------
# Mirrors ``dsa_metadata.py`` block/slot assembly: one manager tensor per group
# is appended into every layer's list, so layers alias the same object.
_N_LAYERS = 4
_GROUPS = 3
_PERSISTENT_ROWS = 4
_REFRESHED_ROWS = 2
_COLS = 3

_TENSOR_FIELDS = (
    "seq_lens",
    "seq_lens_q",
    "actual_seq_lengths_kv",
    "actual_seq_lengths_query",
    "kv_cu_seq_lens",
    "max_seqlen_kv",
    "max_seqlen_q",
    "input_positions",
    "c4_pad_positions",
    "c128_pad_positions",
    "start_pos",
    "c4_cos",
    "c4_sin",
    "c128_cos",
    "c128_sin",
    "c4_input_cos",
    "c4_input_sin",
    "c128_input_cos",
    "c128_input_sin",
)

# --- [Candidate A] In-place refresh ----------------------------------------
_GRAPH_COLS = 8

# Value-derived fields the candidate writes in place on this graph path.
_INPLACE_FIELDS = (
    "seq_lens",
    "seq_lens_q",
    "actual_seq_lengths_kv",
    "actual_seq_lengths_query",
    "kv_cu_seq_lens",
    "max_seqlen_kv",
    "max_seqlen_q",
    "input_positions",
    "c4_pad_positions",
    "c128_pad_positions",
    "start_pos",
)

# --- [SWA device slot-mapping cases] ---------------------------------------
# (raw_bt, block_size, ctx_lens, q_lens, new_cache_slots)
_SWA_CASES = [
    # Plain two-row table; seq1's q_start crosses a block boundary.
    (torch.tensor([[0, 1], [2, 3]], dtype=torch.int32), 4, [5, 9], [1, 1], []),
    # Negative (placeholder) ring blocks must map to -1.
    (torch.tensor([[-1, 1], [2, -1]], dtype=torch.int32), 4, [3, 8], [1, 1], []),
    # q_start at the last position of a block and at position 0.
    (torch.tensor([[0, 1], [2, 3]], dtype=torch.int32), 4, [8, 1], [1, 1], []),
    # Production shape: one ring block per sequence (semantic_cols == 1).
    (torch.tensor([[5], [6]], dtype=torch.int32), 128, [130, 7], [1, 1], []),
    # Scheduler-provided slots short-circuit the ring formula (H2D, no D2H).
    (torch.tensor([[0, 1]], dtype=torch.int32), 4, [5], [1], [42]),
]

# --- [S4a] Alias dedup helpers ---------------------------------------------


def _make_metadata(persistent_seed: int, refreshed_seed: int) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Persistent/refreshed DSA metadata with group-level per-layer aliasing."""

    def persistent_tensor(value: int, dtype: torch.dtype = torch.int32) -> torch.Tensor:
        return torch.full((_PERSISTENT_ROWS,), value, dtype=dtype)

    def refreshed_tensor(value: int, dtype: torch.dtype = torch.int32) -> torch.Tensor:
        return torch.full((_REFRESHED_ROWS,), value, dtype=dtype)

    p_block = [
        torch.full((_PERSISTENT_ROWS, _COLS), persistent_seed * 100 + group, dtype=torch.int32)
        for group in range(_GROUPS)
    ]
    p_slots = [
        torch.full((_PERSISTENT_ROWS,), persistent_seed * 100 + group, dtype=torch.int32) for group in range(_GROUPS)
    ]
    r_block = [
        torch.full((_REFRESHED_ROWS, _COLS), refreshed_seed * 10 + group, dtype=torch.int32) for group in range(_GROUPS)
    ]
    r_slots = [
        torch.full((_REFRESHED_ROWS,), refreshed_seed * 10 + group, dtype=torch.int32) for group in range(_GROUPS)
    ]

    meta = SimpleNamespace(
        explicit_swa_indices=torch.full((_PERSISTENT_ROWS, 1, 128), persistent_seed, dtype=torch.int32),
        # New list objects per layer, but the *same* tensor object per group --
        # exactly the alias relation produced by ``dsa_metadata.py``:553-564.
        block_tables=[[p_block[group] for group in range(_GROUPS)] for _ in range(_N_LAYERS)],
        slot_mappings=[[p_slots[group] for group in range(_GROUPS)] for _ in range(_N_LAYERS)],
        input_rope_by_ratio={
            1: (persistent_tensor(persistent_seed, torch.float32), persistent_tensor(persistent_seed, torch.float32))
        },
        sparse_metadata_ori_win_left=persistent_seed,
        max_query_len=persistent_seed,
        max_seq_len=persistent_seed,
        is_acl_graph=True,
        graph_token_capacity=persistent_seed,
        precomputed_metadata_inputs=(),
    )
    for name in _TENSOR_FIELDS:
        dtype = torch.float32 if "cos" in name or "sin" in name else torch.int32
        setattr(meta, name, persistent_tensor(persistent_seed, dtype))
    # Attach the refreshed side under distinct names for readability; the copy
    # helpers only read the fields listed above, so we stage a second namespace.
    refreshed = SimpleNamespace(
        explicit_swa_indices=torch.full((_REFRESHED_ROWS, 1, 128), refreshed_seed, dtype=torch.int32),
        block_tables=[[r_block[group] for group in range(_GROUPS)] for _ in range(_N_LAYERS)],
        slot_mappings=[[r_slots[group] for group in range(_GROUPS)] for _ in range(_N_LAYERS)],
        input_rope_by_ratio={
            1: (refreshed_tensor(refreshed_seed, torch.float32), refreshed_tensor(refreshed_seed, torch.float32))
        },
        sparse_metadata_ori_win_left=refreshed_seed,
        max_query_len=refreshed_seed,
        max_seq_len=refreshed_seed,
        is_acl_graph=True,
        graph_token_capacity=refreshed_seed,
        precomputed_metadata_inputs=(),
    )
    for name in _TENSOR_FIELDS:
        dtype = torch.float32 if "cos" in name or "sin" in name else torch.int32
        setattr(refreshed, name, refreshed_tensor(refreshed_seed, dtype))
    return meta, refreshed


def _legacy_copy_graph_dsa_metadata(persistent: SimpleNamespace, refreshed: SimpleNamespace) -> None:
    """Pre-change behaviour: per-layer copy with no alias dedup."""

    for field_name in _TENSOR_FIELDS:
        CsaAttentionBackend._copy_graph_tensor(
            getattr(persistent, field_name, None),
            getattr(refreshed, field_name, None),
            field_name,
        )
    CsaAttentionBackend._copy_graph_tensor(
        persistent.explicit_swa_indices,
        refreshed.explicit_swa_indices,
        "explicit_swa_indices",
        pad_value=-1,
    )
    for layer_id, (p_layer, r_layer) in enumerate(zip(persistent.block_tables, refreshed.block_tables)):
        CsaAttentionBackend._copy_graph_tensor_list(p_layer, r_layer, f"block_tables[{layer_id}]", pad_value=-1)
    for layer_id, (p_layer, r_layer) in enumerate(zip(persistent.slot_mappings, refreshed.slot_mappings)):
        CsaAttentionBackend._copy_graph_tensor_list(p_layer, r_layer, f"slot_mappings[{layer_id}]", pad_value=-1)
    for ratio, p_pair in persistent.input_rope_by_ratio.items():
        r_pair = refreshed.input_rope_by_ratio[ratio]
        CsaAttentionBackend._copy_graph_tensor(p_pair[0], r_pair[0], f"input_rope[{ratio}].cos")
        CsaAttentionBackend._copy_graph_tensor(p_pair[1], r_pair[1], f"input_rope[{ratio}].sin")
    persistent.max_query_len = refreshed.max_query_len
    persistent.max_seq_len = refreshed.max_seq_len
    persistent.sparse_metadata_ori_win_left = getattr(refreshed, "sparse_metadata_ori_win_left", -1)
    persistent.is_acl_graph = refreshed.is_acl_graph
    persistent.graph_token_capacity = int(getattr(refreshed, "graph_token_capacity", 0) or 0)
    persistent.precomputed_metadata_inputs = refreshed.precomputed_metadata_inputs


# --- [Candidate A] In-place refresh helpers ---------------------------------


# Separate from plain ``_make_backend``: this variant additionally stubs
# ``_build_precomputed_metadata`` -- the AICPU metadata kernels are not
# available on CPU; the base builder only consumes them later, so a no-op
# keeps the field set under test intact.
def _inplace_make_backend(device: torch.device | None = None) -> CsaAttentionBackend:
    backend = _make_backend(device=device)
    backend._build_precomputed_metadata = lambda *args, **kwargs: None  # noqa: SLF001
    return backend


def _multi_block_tables(batch: int, seed: int) -> tuple[torch.Tensor, ...]:
    """One manager table per group in the fixed export order (SWA first)."""
    tables = (
        torch.full((batch, 2), seed + 1, dtype=torch.int32),
        torch.full((batch, 4), seed + 2, dtype=torch.int32),
        torch.full((batch, 2), seed + 3, dtype=torch.int32),
    )
    return tables


def _inplace_forward_metadata(
    kv_seq_lens: list[int],
    seed: int,
    *,
    positions: torch.Tensor | None = None,
    is_dummy: bool = False,
) -> SimpleNamespace:
    batch = len(kv_seq_lens)
    return SimpleNamespace(
        multi_block_tables=_multi_block_tables(batch, seed),
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=list(kv_seq_lens),
        new_cache_slots_host_values=None,
        dsa_positions=positions if positions is not None else torch.arange(batch, dtype=torch.int64) + seed,
        q_seq_lens_host=None,
        q_seq_lens=None,
        dsa_cos_sin=None,
        dsa_c4_cos_sin=None,
        dsa_c128_cos_sin=None,
        dsa_graph_mode=True,
        dsa_graph_block_table_cols=_GRAPH_COLS,
        dsa_graph_token_capacity=_GRAPH_COLS * 2,
        is_dummy=is_dummy,
        is_prefill=False,
        is_chunked_prefill=False,
        expanded_decode_metadata=None,
    )


def _paired_metadata(seed: int) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Capture/replay metadata with the same graph bucket shapes.

    ``dsa_positions`` is one shared, stable tensor whose contents the runner
    rewrites in place every step (``_fill_graph_dsa_positions``), so the capture
    and the replay observe the same object -- exactly the graph contract.
    """
    batch = 4
    positions = torch.arange(batch, dtype=torch.int64)
    capture = _inplace_forward_metadata([3, 5, 7, 9], seed, positions=positions)
    positions.copy_(torch.arange(batch, dtype=torch.int64) + (seed + 1))
    replay = _inplace_forward_metadata([4, 6, 8, 10], seed + 1, positions=positions)
    return capture, replay


def _tensor_signature(value: torch.Tensor) -> tuple[int, tuple[int, ...], torch.dtype]:
    return (value.data_ptr(), tuple(value.shape), value.dtype)


def _persistent_field_tensors(dsa) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name in _INPLACE_FIELDS:
        tensors[name] = getattr(dsa, name)
    for lid, layer in enumerate(dsa.block_tables):
        for ci, tensor in enumerate(layer):
            if tensor is not None and tensor.numel() > 0:
                tensors[f"block_tables[{lid}][{ci}]"] = tensor
    for lid, layer in enumerate(dsa.slot_mappings):
        for ci, tensor in enumerate(layer):
            if tensor is not None and tensor.numel() > 0:
                tensors[f"slot_mappings[{lid}][{ci}]"] = tensor
    return tensors


def _run_copy(persistent, refreshed, inplace_fields=None) -> None:
    CsaAttentionBackend._copy_graph_dsa_metadata(  # noqa: SLF001 - tested unit
        persistent, refreshed, inplace_fields=inplace_fields
    )


def _assert_fields_equal(left, right) -> None:
    # Scalar attributes.
    for name in ("max_query_len", "max_seq_len", "graph_token_capacity", "is_acl_graph"):
        assert getattr(left, name) == getattr(right, name), name
    # Value-derived tensors.
    for name in _INPLACE_FIELDS:
        a = getattr(left, name)
        b = getattr(right, name)
        if a is None or b is None:
            assert a is None and b is None, name
            continue
        assert torch.equal(a, b), name
    # Per-layer manager tables (alias-aware, compare by storage contents).
    assert len(left.block_tables) == len(right.block_tables)
    seen: set[tuple[int, int]] = set()
    for lid, (ll, rr) in enumerate(zip(left.block_tables, right.block_tables)):
        for ci, (a, b) in enumerate(zip(ll, rr)):
            if a.numel() == 0 and b.numel() == 0:
                continue
            key = (a.data_ptr(), b.data_ptr())
            if key in seen:
                continue
            seen.add(key)
            assert torch.equal(a, b), f"block_tables[{lid}][{ci}]"
    seen = set()
    for lid, (ll, rr) in enumerate(zip(left.slot_mappings, right.slot_mappings)):
        for ci, (a, b) in enumerate(zip(ll, rr)):
            if a.numel() == 0 and b.numel() == 0:
                continue
            key = (a.data_ptr(), b.data_ptr())
            if key in seen:
                continue
            seen.add(key)
            assert torch.equal(a, b), f"slot_mappings[{lid}][{ci}]"


# --- [SWA device refresh helpers] ------------------------------------------


def _builder() -> DsaMetadataBuilder:
    caches_info, group_infos = build_cache_specs([0, 4, 128], 128, 3)
    return DsaMetadataBuilder(caches_info, group_infos)


def _run_swa(
    raw_bt: torch.Tensor,
    block_size: int,
    ctx_lens: list[int],
    q_lens: list[int],
    new_cache_slots: list[int],
    *,
    device_ctx_lens: torch.Tensor | None,
    graph_slot_capacity: int | None = None,
    graph_block_table_capacity_cols: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    builder = _builder()
    return builder._process_swa_group(  # noqa: SLF001 - exercise the unit directly
        raw_bt,
        block_size,
        ctx_lens,
        q_lens,
        batch_size=len(ctx_lens),
        graph_slot_capacity=len(ctx_lens) if graph_slot_capacity is None else graph_slot_capacity,
        graph_block_table_capacity_cols=(
            int(raw_bt.size(1)) if graph_block_table_capacity_cols is None else graph_block_table_capacity_cols
        ),
        new_cache_slots=new_cache_slots,
        swa_device_ctx_lens=device_ctx_lens,
    )


# === [S4a] Alias dedup ===


def test_alias_dedup_matches_legacy_bitwise_and_preserves_addresses() -> None:
    p_new, r_new = _make_metadata(persistent_seed=1, refreshed_seed=7)
    p_legacy, r_legacy = _make_metadata(persistent_seed=1, refreshed_seed=7)

    addresses_before = [[tensor.data_ptr() for tensor in layer] for layer in p_new.block_tables]

    CsaAttentionBackend._copy_graph_dsa_metadata(p_new, r_new)
    _legacy_copy_graph_dsa_metadata(p_legacy, r_legacy)

    # Addresses must be untouched: the dedup copies into the captured storage.
    for layer_id, layer in enumerate(p_new.block_tables):
        for cache_id, tensor in enumerate(layer):
            assert tensor.data_ptr() == addresses_before[layer_id][cache_id]

    # Aliasing preserved: same group across layers is still one storage.
    for cache_id in range(_GROUPS):
        ptr = p_new.block_tables[0][cache_id].data_ptr()
        for layer_id in range(_N_LAYERS):
            assert p_new.block_tables[layer_id][cache_id].data_ptr() == ptr

    for field_name in _TENSOR_FIELDS + ("explicit_swa_indices",):
        assert torch.equal(getattr(p_new, field_name), getattr(p_legacy, field_name)), field_name
    for layer_id in range(_N_LAYERS):
        for cache_id in range(_GROUPS):
            assert torch.equal(p_new.block_tables[layer_id][cache_id], p_legacy.block_tables[layer_id][cache_id])
            assert torch.equal(p_new.slot_mappings[layer_id][cache_id], p_legacy.slot_mappings[layer_id][cache_id])
    for ratio in p_new.input_rope_by_ratio:
        for index in (0, 1):
            assert torch.equal(p_new.input_rope_by_ratio[ratio][index], p_legacy.input_rope_by_ratio[ratio][index])
    assert p_new.max_query_len == p_legacy.max_query_len
    assert p_new.max_seq_len == p_legacy.max_seq_len
    assert p_new.graph_token_capacity == p_legacy.graph_token_capacity

    # Spot-check the prefix-pad semantics survived dedup: rows past the
    # refreshed prefix keep the -1 pad, the prefix carries refreshed values.
    expected_block = torch.full((_PERSISTENT_ROWS, _COLS), 70, dtype=torch.int32)
    expected_block[_REFRESHED_ROWS:] = -1
    assert torch.equal(p_new.block_tables[0][0], expected_block)
    expected_slots = torch.full((_PERSISTENT_ROWS,), 70, dtype=torch.int32)
    expected_slots[_REFRESHED_ROWS:] = -1
    assert torch.equal(p_new.slot_mappings[0][0], expected_slots)


def test_alias_dedup_skips_only_duplicate_pairs(monkeypatch) -> None:
    captured: dict[str, int] = {}
    monkeypatch.setattr(
        csa_attention_module,
        "_log_dsa_alias_dedup_once",
        lambda total, skipped: captured.update(total=total, skipped=skipped),
    )
    persistent, refreshed = _make_metadata(persistent_seed=1, refreshed_seed=7)
    CsaAttentionBackend._copy_graph_dsa_metadata(persistent, refreshed)

    total = _N_LAYERS * _GROUPS * 2  # block tables + slot mappings
    unique_pairs = _GROUPS * 2  # one per (group, list)
    assert captured == {"total": total, "skipped": total - unique_pairs}


def test_alias_dedup_keeps_last_write_for_shared_target_distinct_sources() -> None:
    # One target shared by two *distinct* refreshed sources must still be
    # written by every source in order (last write wins), as before dedup.
    target = torch.zeros((4, 2), dtype=torch.int32)
    source_a = torch.full((2, 2), 11, dtype=torch.int32)
    source_b = torch.full((2, 2), 22, dtype=torch.int32)
    seen: set[tuple[int, int]] = set()
    CsaAttentionBackend._copy_graph_tensor_list(
        [target, target], [source_a, source_b], "block_tables[0]", pad_value=-1, seen=seen
    )
    expected = torch.full((4, 2), -1, dtype=torch.int32)
    expected[:2] = 22
    assert torch.equal(target, expected)
def test_move_metadata_to_device_still_moves_cpu_fields(monkeypatch) -> None:
    # A differing backend device must still trigger a real move for every
    # tensor, so eager (CPU-pinned scheduler tables) keeps working.
    backend = _make_backend(torch.device("meta"))
    metadata = SimpleNamespace(
        block_tables=[[torch.zeros((2, 2), dtype=torch.int32)]],
        slot_mappings=[[torch.zeros((2,), dtype=torch.int32)]],
        seq_lens=torch.zeros((2,), dtype=torch.int32),
    )
    original_to = torch.Tensor.to
    calls: list[str] = []

    def counting_to(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(args[0]) if args else "")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", counting_to)
    backend._move_metadata_to_device(metadata)

    # seq_lens + one block table + one slot mapping.
    assert calls == ["meta", "meta", "meta"]


# === [Candidate A] In-place refresh ===


# -- (a) off path equivalence ------------------------------------------------


def test_off_path_matches_plain_build_and_copy() -> None:
    backend = _inplace_make_backend()
    capture_meta, replay_meta = _paired_metadata(seed=0)

    persistent_ref = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001
    persistent_off = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001

    refreshed_ref = backend._build_dsa_metadata_for_forward(replay_meta)  # noqa: SLF001
    refreshed_off = backend._build_dsa_metadata_for_forward(replay_meta)  # noqa: SLF001

    _run_copy(persistent_ref, refreshed_ref)
    _run_copy(persistent_off, refreshed_off)

    _assert_fields_equal(persistent_ref, persistent_off)
    # The off path never aliases: refreshed tensors are fresh storage.
    assert refreshed_off.seq_lens is not persistent_off.seq_lens
    assert refreshed_off.refresh_inplace_fields == frozenset()


# -- (b) on/off numerical equivalence ---------------------------------------


def test_inplace_refresh_matches_off_path_bitwise() -> None:
    backend = _inplace_make_backend()
    capture_meta, replay_meta = _paired_metadata(seed=0)

    persistent_on = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001
    persistent_off = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001

    refreshed_on = backend._build_dsa_metadata_for_forward(  # noqa: SLF001
        replay_meta, inplace_target=persistent_on
    )
    assert refreshed_on.refresh_inplace_fields == frozenset(_INPLACE_FIELDS)

    refreshed_off = backend._build_dsa_metadata_for_forward(replay_meta)  # noqa: SLF001

    _run_copy(persistent_on, refreshed_on, refreshed_on.refresh_inplace_fields)
    _run_copy(persistent_off, refreshed_off)

    _assert_fields_equal(persistent_on, persistent_off)
    # Legacy values (built independently) must also match byte-for-byte.
    for name in _INPLACE_FIELDS:
        assert torch.equal(getattr(persistent_on, name), getattr(refreshed_off, name)), name


def test_inplace_refresh_matches_capture_values_for_same_metadata() -> None:
    backend = _inplace_make_backend()
    capture_meta, _ = _paired_metadata(seed=0)
    persistent = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001
    snapshot = {name: getattr(persistent, name).clone() for name in _INPLACE_FIELDS}

    refreshed = backend._build_dsa_metadata_for_forward(  # noqa: SLF001
        capture_meta, inplace_target=persistent
    )
    _run_copy(persistent, refreshed, refreshed.refresh_inplace_fields)

    for name in _INPLACE_FIELDS:
        assert torch.equal(getattr(persistent, name), snapshot[name]), name


# -- (c) captured-address stability -----------------------------------------


def test_inplace_refresh_preserves_captured_addresses() -> None:
    backend = _inplace_make_backend()
    capture_meta, replay_meta = _paired_metadata(seed=0)
    persistent = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001

    before = {name: _tensor_signature(t) for name, t in _persistent_field_tensors(persistent).items()}

    refreshed = backend._build_dsa_metadata_for_forward(  # noqa: SLF001
        replay_meta, inplace_target=persistent
    )
    _run_copy(persistent, refreshed, refreshed.refresh_inplace_fields)

    after = {name: _tensor_signature(t) for name, t in _persistent_field_tensors(persistent).items()}
    assert before == after

    # The value-derived fields alias the captured objects, not copies.
    for name in _INPLACE_FIELDS:
        assert getattr(persistent, name) is getattr(refreshed, name), name


# -- (d) no host sync / no new per-field ops --------------------------------


def test_inplace_refresh_does_not_host_sync_or_move_fields(monkeypatch) -> None:
    backend = _inplace_make_backend()
    capture_meta, replay_meta = _paired_metadata(seed=0)
    persistent = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001

    def _forbidden_cpu(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"unexpected host sync .cpu() on {self.shape}")

    original_to = torch.Tensor.to
    to_calls: list[int] = []

    def counting_to(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        to_calls.append(id(self))
        return original_to(self, *args, **kwargs)

    copy_calls: list[int] = []
    original_copy = torch.Tensor.copy_

    def counting_copy(self, src, *args, **kwargs):  # type: ignore[no-untyped-def]
        copy_calls.append(id(self))
        return original_copy(self, src, *args, **kwargs)

    alias_checks: list[str] = []
    original_assert = CsaAttentionBackend._assert_graph_tensor_alias  # noqa: SLF001

    def counting_assert(persistent_t, refreshed_t, name):  # type: ignore[no-untyped-def]
        alias_checks.append(name)
        return original_assert(persistent_t, refreshed_t, name)

    monkeypatch.setattr(torch.Tensor, "cpu", _forbidden_cpu)
    monkeypatch.setattr(torch.Tensor, "to", counting_to)
    monkeypatch.setattr(torch.Tensor, "copy_", counting_copy)
    monkeypatch.setattr(CsaAttentionBackend, "_assert_graph_tensor_alias", staticmethod(counting_assert))

    refreshed = backend._build_dsa_metadata_for_forward(  # noqa: SLF001
        replay_meta, inplace_target=persistent
    )
    inplace_ids = {id(getattr(persistent, name)) for name in _INPLACE_FIELDS}
    moved = [ptr for ptr in to_calls if ptr in inplace_ids]
    assert moved == [], "in-place fields must not be moved with .to()"

    # Only the replay pass matters here: the build itself legitimately writes
    # the captured buffers (one ``copy_`` per in-place field).
    build_copy_count = len(copy_calls)
    _run_copy(persistent, refreshed, refreshed.refresh_inplace_fields)

    copied = [ptr for ptr in copy_calls[build_copy_count:] if ptr in inplace_ids]
    assert copied == [], "in-place fields must not be copied on the replay path"
    assert set(alias_checks) == set(_INPLACE_FIELDS)


def test_inplace_refresh_uses_no_device_d2h_in_device_slot_path() -> None:
    # The SWA device slot build, when handed a captured buffer, must not call
    # ``.tolist()`` (a D2H sync) and must write the target buffer in place.
    raw_bt = torch.tensor([[5], [7]], dtype=torch.int32)
    ctx = torch.tensor([3, 4], dtype=torch.int32)
    target = torch.full((4,), -1, dtype=torch.int32)

    original_tolist = torch.Tensor.tolist

    def _forbidden_tolist(self):  # type: ignore[no-untyped-def]
        raise AssertionError("unexpected .tolist() in the device slot path")

    torch.Tensor.tolist = _forbidden_tolist
    try:
        out = DsaMetadataBuilder._swa_slots_on_device(  # noqa: SLF001
            raw_bt,
            block_size=128,
            semantic_cols=1,
            out_slot_rows=4,
            batch_size=2,
            q_lens=[1, 1],
            new_cache_slots=None,
            query_total_tokens=2,
            device_ctx_lens=ctx,
            dst_slots=target,
        )
    finally:
        torch.Tensor.tolist = original_tolist

    assert out is target
    assert torch.equal(target[:2], torch.tensor([5 * 128 + 2, 7 * 128 + 3], dtype=torch.int32))


# -- (e) ownership conflicts -------------------------------------------------


def test_reset_forward_does_not_touch_captured_tensors() -> None:
    backend = _inplace_make_backend()
    capture_meta, _ = _paired_metadata(seed=0)
    persistent = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001
    before = {name: _tensor_signature(t) for name, t in _persistent_field_tensors(persistent).items()}
    contents = {name: t.clone() for name, t in _persistent_field_tensors(persistent).items()}

    backend.reset_forward(capture_meta)

    assert capture_meta.dsa_metadata is None
    assert capture_meta.dsa_positions is None
    after = {name: _tensor_signature(t) for name, t in _persistent_field_tensors(persistent).items()}
    assert before == after
    for name, value in contents.items():
        assert torch.equal(_persistent_field_tensors(persistent)[name], value), name


def test_shared_metadata_draft_group_limit_is_eager_and_isolated() -> None:
    # The DSpark draft shares the request's multi_block_tables but owns one SWA
    # group and is always eager (no graph target): its build must never write
    # into the target's captured persistent metadata.
    backend = _inplace_make_backend()
    capture_meta, replay_meta = _paired_metadata(seed=0)
    persistent = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001
    before = {name: t.clone() for name, t in _persistent_field_tensors(persistent).items()}

    draft = backend._build_dsa_metadata_for_forward(replay_meta, group_limit=1)  # noqa: SLF001
    assert draft.refresh_inplace_fields == frozenset()
    assert draft is not persistent

    after = _persistent_field_tensors(persistent)
    for name, value in before.items():
        assert torch.equal(after[name], value), name


def test_inplace_falls_back_when_shapes_do_not_match() -> None:
    backend = _inplace_make_backend()
    capture_meta, _ = _paired_metadata(seed=0)
    persistent = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001
    before = _tensor_signature(persistent.seq_lens)

    # A different batch size cannot alias the captured buffer; the builder must
    # allocate fresh and let the copy path handle it, leaving the address alone.
    other_meta = _inplace_forward_metadata([1, 2, 3], seed=9)
    refreshed = backend._build_dsa_metadata_for_forward(  # noqa: SLF001
        other_meta, inplace_target=persistent
    )
    assert refreshed.seq_lens is not persistent.seq_lens
    assert "seq_lens" not in refreshed.refresh_inplace_fields
    assert _tensor_signature(persistent.seq_lens) == before


# -- (f) capture-time gating -------------------------------------------------


def test_capture_path_never_uses_inplace_target() -> None:
    # The capture entry point must build a fresh metadata that never aliases or
    # writes into an existing persistent object; the in-place mechanism is a
    # replay-time concern only.
    backend = _inplace_make_backend()
    capture_meta, _ = _paired_metadata(seed=0)

    backend.prepare_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001 - capture entry
    captured = capture_meta.dsa_metadata
    assert captured is not None
    assert captured.refresh_inplace_fields == frozenset()

    other_meta = _inplace_forward_metadata([3, 5, 7, 9], seed=7)
    fresh = backend._build_dsa_metadata_for_forward(other_meta)  # noqa: SLF001
    assert fresh.refresh_inplace_fields == frozenset()
    for name in _INPLACE_FIELDS:
        assert getattr(fresh, name) is not getattr(captured, name), name


# -- (g) known boundary: host-path .item() D2H -------------------------------


def test_host_path_item_d2h_is_a_known_boundary(monkeypatch) -> None:
    # The candidate removes allocations/copies, not the per-row ``.item()`` on
    # the host SWA/token slot loop (the device slot path is what removes that).
    # This pins the boundary so a later reader does not mis-attribute the win.
    backend = _inplace_make_backend()
    capture_meta, replay_meta = _paired_metadata(seed=0)
    persistent = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001

    original_item = torch.Tensor.item
    calls: list[int] = []

    def counting_item(self):  # type: ignore[no-untyped-def]
        calls.append(1)
        return original_item(self)

    monkeypatch.setattr(torch.Tensor, "item", counting_item)
    backend._build_dsa_metadata_for_forward(replay_meta, inplace_target=persistent)  # noqa: SLF001
    assert calls, "expected the host slot loop to keep its per-row .item()"


def test_replay_copy_guard_skips_alias_and_copies_disjoint(monkeypatch) -> None:
    calls: list[int] = []
    original = torch.Tensor.copy_

    def counting_copy(self, src, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(self.data_ptr())
        return original(self, src, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", counting_copy)
    aliased = torch.tensor([3, 3, 3], dtype=torch.int32)
    # Same storage -> data_ptr equality -> an aliased field needs no write.
    DsaAttentionBackend._copy_graph_tensor(aliased, aliased, "seq_lens")
    assert calls == []

    persistent = torch.tensor([0, 0, 0], dtype=torch.int32)
    refreshed = torch.tensor([1, 2, 3], dtype=torch.int32)
    DsaAttentionBackend._copy_graph_tensor(persistent, refreshed, "seq_lens")
    assert calls == [persistent.data_ptr()]
    assert torch.equal(persistent, refreshed)


# === SWA device refresh mechanism ===


def test_swa_device_slots_bit_identical_to_host_path() -> None:
    for raw_bt, block_size, ctx_lens, q_lens, new_cache_slots in _SWA_CASES:
        device_ctx = torch.tensor(ctx_lens, dtype=torch.int32)
        off_bt, off_slots = _run_swa(
            raw_bt.clone(),
            block_size,
            ctx_lens,
            q_lens,
            new_cache_slots,
            device_ctx_lens=None,
        )
        on_bt, on_slots = _run_swa(
            raw_bt.clone(),
            block_size,
            ctx_lens,
            q_lens,
            new_cache_slots,
            device_ctx_lens=device_ctx,
        )
        assert torch.equal(off_bt, on_bt), (raw_bt.tolist(), ctx_lens)
        assert torch.equal(off_slots, on_slots), (raw_bt.tolist(), ctx_lens)


def test_swa_device_slots_expected_values() -> None:
    raw_bt = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    _bt, slots = _run_swa(
        raw_bt,
        4,
        [5, 9],
        [1, 1],
        [],
        device_ctx_lens=torch.tensor([5, 9], dtype=torch.int32),
    )
    # seq0: pos 4 -> block 1 col 1, offset 0 -> 4; seq1: pos 8 -> block 2 col 0 -> 8.
    assert torch.equal(slots, torch.tensor([4, 8], dtype=torch.int32))


def test_swa_device_refresh_writes_into_captured_storage_in_place() -> None:
    persistent_bt = torch.full((2, 3), -7, dtype=torch.int32)
    persistent_slots = torch.full((2,), -7, dtype=torch.int32)
    bt_ptr = persistent_bt.data_ptr()
    slots_ptr = persistent_slots.data_ptr()

    raw_bt = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    on_bt, on_slots = _run_swa(
        raw_bt,
        4,
        [5, 9],
        [1, 1],
        [],
        device_ctx_lens=torch.tensor([5, 9], dtype=torch.int32),
    )
    CsaAttentionBackend._copy_graph_tensor_list([persistent_bt], [on_bt], "block_tables[0]", pad_value=-1)
    CsaAttentionBackend._copy_graph_tensor_list([persistent_slots], [on_slots], "slot_mappings[0]", pad_value=-1)

    assert persistent_bt.data_ptr() == bt_ptr
    assert persistent_slots.data_ptr() == slots_ptr
    assert torch.equal(persistent_bt, on_bt)
    assert torch.equal(persistent_slots, on_slots)




# ---- rope inverse fusion ----
# R2: fused inverse interleaved RoPE on the V4.1 output de-rotation; a CPU
# stand-in exercises the dispatch wiring and the |x|<=8 tolerance boundary
# (scale-8/16). The fused path engages whenever the op is registered and the
# structural guards pass.

# bf16 acceptance tolerance (matches the repo's fused-op suite).
_TOL = 2**-6


def _build_cache(max_pos: int, rotary_dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    pos = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


def _stub_partial_rotary() -> Any:
    """CPU stand-in for ``npu_inplace_partial_rotary_mul``'s contract.

    Mirrors the wrapper: 3D ``[M, H, head_dim]`` input rotated in place over
    the trailing ``rope_head_dim`` slice with the per-element duplicated
    interleave cos/sin; ``inverse`` conjugates the rotation by negating sin.
    Records ``(inverse, shape, start, head)`` per call so tests can assert the
    dispatcher reached it (and with which sign).
    """
    calls: list[tuple[bool, tuple[int, ...], int, int]] = []

    def op(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_start_dim: int,
        rope_head_dim: int,
        inverse: bool = False,
    ) -> None:
        calls.append((bool(inverse), tuple(x.shape), int(rope_start_dim), int(rope_head_dim)))
        start = int(rope_start_dim)
        end = start + int(rope_head_dim)
        signed_sin = -sin if inverse else sin
        rope = x[..., start:end]
        even = rope[..., 0::2].float()
        odd = rope[..., 1::2].float()
        cos_pair = cos[..., 0::2].float().unsqueeze(1)
        sin_pair = signed_sin[..., 0::2].float().unsqueeze(1)
        rotated = torch.stack(
            (even * cos_pair - odd * sin_pair, even * sin_pair + odd * cos_pair),
            dim=-1,
        ).flatten(-2)
        x[..., start:end] = rotated.to(x.dtype)

    op.calls = calls  # type: ignore[attr-defined]
    return op


def _install_fused_stub(monkeypatch: Any) -> Any:
    op = _stub_partial_rotary()
    monkeypatch.setattr(csa2, "_get_partial_rotary_op", lambda: op)
    monkeypatch.setattr(csa2, "_rope_fused_device_supported", lambda x: True)
    monkeypatch.setattr(csa2, "_log_fused_rope_inverse_active", lambda: None)
    return op


def _eager_reference(
    monkeypatch: Any,
    x: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    inverse: bool,
) -> torch.Tensor:
    """Force the eager path (device guard off) on an identical input."""
    original = csa2._rope_fused_device_supported
    csa2._rope_fused_device_supported = lambda x: False
    try:
        return csa2.apply_interleaved_rope(
            x.clone(), positions, cache, inverse=inverse
        )
    finally:
        csa2._rope_fused_device_supported = original


def test_fused_inverse_matches_eager(monkeypatch: Any) -> None:
    op = _install_fused_stub(monkeypatch)

    cache = _build_cache(256, 64)
    cases = [
        (8, 4, 512),  # 3D q-like, partial [448:512)
        (8, 1, 512),  # 3D single-head
        (5, 128),  # 2D latent-like, partial [64:128)
    ]
    for shape in cases:
        torch.manual_seed(sum(shape) + 11)
        x = (torch.randn(*shape) * 0.9).to(torch.bfloat16)
        positions = torch.randint(0, 200, (shape[0],), dtype=torch.int32)

        fused = csa2.apply_interleaved_rope(x.clone(), positions, cache, inverse=True)
        eager = _eager_reference(monkeypatch, x, positions, cache, inverse=True)

        assert op.calls[-1][0] is True, "the dispatcher must forward inverse=True"
        assert fused.shape == eager.shape
        diff = (fused.float() - eager.float()).abs()
        budget = _TOL + _TOL * eager.float().abs()
        assert (diff <= budget).all(), f"shape={shape} max_abs={diff.max().item():.3e}"
        # The un-rotated prefix must survive the in-place kernel untouched.
        rope_width = cache.size(-1)
        if shape[-1] > rope_width:
            assert torch.equal(fused[..., : shape[-1] - rope_width], eager[..., : shape[-1] - rope_width])


def test_fused_forward_unchanged_by_r2(monkeypatch: Any) -> None:
    op = _install_fused_stub(monkeypatch)

    cache = _build_cache(128, 64)
    torch.manual_seed(5)
    x = (torch.randn(6, 3, 256) * 0.9).to(torch.bfloat16)
    positions = torch.arange(6, dtype=torch.int32)

    fused = csa2.apply_interleaved_rope(x.clone(), positions, cache)
    eager = _eager_reference(monkeypatch, x, positions, cache, inverse=False)
    assert op.calls[-1][0] is False
    diff = (fused.float() - eager.float()).abs()
    assert (diff <= _TOL + _TOL * eager.float().abs()).all()


def test_inverse_guard_matrix(monkeypatch: Any) -> None:
    monkeypatch.setattr(csa2, "_get_partial_rotary_op", lambda: (lambda *a, **k: None))
    monkeypatch.setattr(csa2, "_rope_fused_device_supported", lambda x: True)

    cache = _build_cache(64, 64)
    x = torch.zeros(4, 2, 256, dtype=torch.bfloat16)
    positions = torch.arange(4, dtype=torch.int32)

    assert csa2._rope_fused_supported(x, positions, cache, True) is True
    assert csa2._rope_fused_supported(x, positions, cache, False) is True

    # dtype / rope width / cache dtype / contiguity / rank / device.
    assert csa2._rope_fused_supported(x.to(torch.float16), positions, cache, True) is False
    assert csa2._rope_fused_supported(x, positions, _build_cache(64, 16), True) is False
    assert csa2._rope_fused_supported(x, positions, cache.to(torch.bfloat16), True) is False
    assert csa2._rope_fused_supported(x.transpose(0, 1), positions, cache, True) is False
    assert csa2._rope_fused_supported(torch.zeros(4, 2, 256, 1, dtype=torch.bfloat16), positions, cache, True) is False
    # P2-a: an ODD table width (65 // 2 == 32, so the old 2*half == 64 passed).
    odd_cache = torch.randn(64, 65)
    assert csa2._rope_fused_supported(x, positions, odd_cache, True) is False
    assert csa2._rope_fused_supported(x, positions, odd_cache, False) is False
    monkeypatch.setattr(csa2, "_rope_fused_device_supported", lambda x: False)
    assert csa2._rope_fused_supported(x, positions, cache, True) is False


def test_inverse_falls_back_when_op_missing(monkeypatch: Any) -> None:
    monkeypatch.setattr(csa2, "_rope_fused_device_supported", lambda x: True)
    monkeypatch.setattr(csa2, "_get_partial_rotary_op", lambda: None)
    monkeypatch.setattr(csa2, "_log_fused_rope_fallback", lambda *a, **k: None)

    cache = _build_cache(64, 64)
    x = (torch.randn(4, 2, 256) * 0.9).to(torch.bfloat16)
    positions = torch.arange(4, dtype=torch.int32)

    fallback = csa2.apply_interleaved_rope(x.clone(), positions, cache, inverse=True)
    eager = _eager_reference(monkeypatch, x, positions, cache, inverse=True)
    assert torch.equal(fallback, eager)


# ---------------------------------------------------------------------------
# P1-b: amplitude domain of the mixed tolerance
# ---------------------------------------------------------------------------

# Deterministic hash-like pattern (no RNG) so the scale boundary is stable
# across torch builds; found by the rev-3 stress sweep (scale=8 -> ratio ~0.82,
# scale=16 -> ~1.39).
_AMP_HASH = (7.31, 78.233, 9876.54321)
_AMP_SHAPE = (8, 4, 512)


def _amplitude_input(scale: float) -> torch.Tensor:
    n = 1
    for dim in _AMP_SHAPE:
        n *= dim
    idx = torch.arange(n, dtype=torch.float32)
    a, b, c = _AMP_HASH
    h = torch.sin(idx * a + b) * c
    h = (h - torch.floor(h)) * 2.0 - 1.0
    return (h.reshape(_AMP_SHAPE) * scale).to(torch.bfloat16)


def _amplitude_positions() -> torch.Tensor:
    return (torch.arange(_AMP_SHAPE[0], dtype=torch.int32) * 7) % 200


def _inverse_tolerance_ratio(monkeypatch: Any, scale: float) -> float:
    """Worst ``diff / (2**-6 + 2**-6 * |eager|)`` over the amplitude case.

    Reusable on-device check: the real run should sample ``attn_out`` absmax
    once at the inverse call; if it is materially above ~8, this ratio-based
    re-evaluation (swap the stub for the real op) decides the bound.
    """
    cache = _build_cache(256, 64)
    x = _amplitude_input(scale)
    positions = _amplitude_positions()
    fused = csa2.apply_interleaved_rope(x.clone(), positions, cache, inverse=True)
    eager = _eager_reference(monkeypatch, x, positions, cache, inverse=True)
    diff = (fused.float() - eager.float()).abs()
    bound = _TOL + _TOL * eager.float().abs()
    return (diff / bound).max().item()


def test_inverse_tolerance_holds_at_scale8(monkeypatch: Any) -> None:
    _install_fused_stub(monkeypatch)
    ratio = _inverse_tolerance_ratio(monkeypatch, 8.0)
    assert ratio <= 1.0, f"scale=8 ratio={ratio:.3f} (mixed tolerance must hold)"


@pytest.mark.xfail(
    strict=True,
    reason="P1-b: near-cancellation + scale=16 exceeds the mixed tolerance; "
    "re-evaluate the bound against the real attn_out amplitude domain; "
    "strict=True pins this boundary: if upstream precision improves past the "
    "bound this test turns XPASS and goes red",
)
def test_inverse_tolerance_breaks_at_scale16(monkeypatch: Any) -> None:
    _install_fused_stub(monkeypatch)
    ratio = _inverse_tolerance_ratio(monkeypatch, 16.0)
    assert ratio <= 1.0, f"scale=16 ratio={ratio:.3f} (expected to break the bound)"

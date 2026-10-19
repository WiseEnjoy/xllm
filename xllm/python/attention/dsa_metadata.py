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

"""DeepSeek-V4 DSA metadata construction (faithful Python port of the C++
``DSAMetadataBuilder`` in ``core/layers/common/dsa_metadata_builder.cpp``).

This is the Python-path counterpart of the C++ ``build_dsa_fields`` step: it
turns the framework-allocated ``multi_block_tables`` (per-manager block tables,
exposed through ``AttentionMetadataView``) plus the per-layer
``caches_info`` / ``group_infos`` (rebuilt from ``compress_ratios`` +
``window_size``) into the per-layer ``block_tables`` / ``slot_mappings`` the
DSA attention kernel consumes, along with the c4/c128 compressed positions,
sequence-length metadata, and RoPE tables.

The C++ model forward builds this inside ``DeepseekV4ModelImpl``; under
``--model_impl python`` the C++ forward never runs, so the Python DSA attention
backend builds it here from the same inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from xllm.python.platform import current_platform

# ---------------------------------------------------------------------------
# Cache-type enum (mirrors ``DSACacheType`` in dsa_metadata.h).
# ---------------------------------------------------------------------------
DSA_CACHE_TOKEN = 0
DSA_CACHE_SEQUENCE = 1
DSA_CACHE_SLIDING_WINDOW = 2
DSV4_COMPRESSED_BLOCK_TOKEN_SIZE = 128 if current_platform.is_mlu() else 2048

# ---------------------------------------------------------------------------
# Read-side SWA block-table layout.
# ---------------------------------------------------------------------------
# The shared default is right-aligned; DeepSeek-V4.1 explicitly requests the
# ring layout because its read consumers address columns absolutely and do not
# compensate for the right-aligned shift (see ``_process_swa_group``).
READ_TABLE_LAYOUT_RIGHT_ALIGNED = "right_aligned"
READ_TABLE_LAYOUT_RING = "ring"


@dataclass
class DSACacheInfo:
    """Per-cache descriptor: which group it belongs to and its own shape."""

    group_id: int
    cache_type: int
    ratio: int
    block_size: int


@dataclass
class DSAGroupInfo:
    """Per-group descriptor: one block-manager pool."""

    cache_type: int
    ratio: int
    block_size: int


@dataclass
class DsaMetadata:
    """Per-forward DSA metadata, shared across all layers of one model."""

    # Current layer selected by the model loop. Mirrors C++
    # DSAMetadata::layer_id and is updated immediately before each layer runs.
    layer_id: int

    # Sequence lengths (host, int32).
    seq_lens: torch.Tensor
    seq_lens_q: torch.Tensor
    actual_seq_lengths_kv: torch.Tensor
    actual_seq_lengths_query: torch.Tensor
    kv_cu_seq_lens: torch.Tensor
    max_seqlen_kv: torch.Tensor
    max_seqlen_q: torch.Tensor
    max_query_len: int
    max_seq_len: int

    # Positions.
    input_positions: torch.Tensor
    c4_pad_positions: torch.Tensor
    c128_pad_positions: torch.Tensor
    start_pos: torch.Tensor

    # RoPE base tables.
    cos_table: torch.Tensor | None = None
    sin_table: torch.Tensor | None = None
    # Per-ratio compressed RoPE tables (C++ DeepseekV4RotaryEmbedding c4/c128
    # groups, compress_rope_theta, no mscale). Populated per-request by indexing
    # the compress RoPE cache with c4/c128_pad_positions.
    c4_cos: torch.Tensor | None = None
    c4_sin: torch.Tensor | None = None
    c128_cos: torch.Tensor | None = None
    c128_sin: torch.Tensor | None = None
    # Main q/kv RoPE for C4/C128 layers, built from input_positions against the
    # compressed-theta caches. Mirrors DSAMetadata::c4_input_cos/c4_input_sin
    # and DSAMetadata::c128_input_cos/c128_input_sin. These are independent of
    # the per-compressed-row tables above (c4_cos/c4_sin etc.).
    c4_input_cos: torch.Tensor | None = None
    c4_input_sin: torch.Tensor | None = None
    c128_input_cos: torch.Tensor | None = None
    c128_input_sin: torch.Tensor | None = None
    # Main q/kv RoPE pairs keyed by compression ratio. Keeping the complete map
    # on the request-owned metadata makes every ACL graph entry retain the
    # stable tensor addresses captured by each layer.
    input_rope_by_ratio: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    # Expanded QLI RoPE tables scoped to one forward so compressed layers can
    # share the half-to-full-width conversion.
    qli_expanded_rope: dict[tuple[int, int, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    # Global-position RoPE for the KV / compressor / index-cache writes under
    # prefill CP. ``cos_table`` / ``sin_table`` are localized to this rank's
    # query rows, but the projected-attention KV path runs on the full
    # CP-gathered token set and must use the global table of this layer's ratio.
    # Mirrors DSAMetadata::kv_cos/kv_sin; undefined when CP is off.
    kv_cos: torch.Tensor | None = None
    kv_sin: torch.Tensor | None = None

    # block_tables / slot_mappings: [n_layers][n_caches_per_layer]; caches in the
    # same group share the same underlying tensor (no copy).
    block_tables: list[list[torch.Tensor]] = field(default_factory=list)
    slot_mappings: list[list[torch.Tensor]] = field(default_factory=list)
    # One source table per cache manager. The per-layer grids re-reference
    # these tensors, so data movement can visit each manager once.
    manager_block_tables: list[torch.Tensor] = field(default_factory=list)
    manager_slot_mappings: list[torch.Tensor] = field(default_factory=list)

    # Precomputed AICPU tiling metadata (filled by the backend, not the builder).
    c1_metadata: torch.Tensor | None = None
    c4_metadata: torch.Tensor | None = None
    c128_metadata: torch.Tensor | None = None
    qli_metadata: torch.Tensor | None = None
    # A5 QLI v2 length tensors derived once for this forward. Each compressed
    # layer reuses the same request-level values.
    qli_v2_metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    # Runtime window baked into c1/c4/c128 metadata. Native DSpark expands
    # this window and supplies explicit SWA indices to the attention kernel.
    sparse_metadata_ori_win_left: int = -1
    explicit_swa_indices: torch.Tensor | None = None
    hadamard: torch.Tensor | None = None

    # Keep AICPU metadata-builder inputs alive until all asynchronously
    # enqueued kernels (and, for ACL graph, the captured graph entry) are done.
    # C++ gets this lifetime from the owning DSAMetadata fields; Python creates
    # additional empty optional tensors and device copies while precomputing.
    precomputed_metadata_inputs: tuple[torch.Tensor, ...] = ()

    # Owns the NPU storage for request-shaped metadata packed in prepare().
    # Tensor fields above may be views into this buffer, matching C++
    # DSAMetadata::packed_metadata_buffer.
    packed_metadata_buffer: torch.Tensor | None = None

    is_acl_graph: bool = False

    # Static committed-row bound (token rows) of the ACL-graph capacity gathers,
    # fixed for the lifetime of one captured graph. Zero means "not bucketed":
    # eager forwards use their exact per-forward maximum and graph forwards fall
    # back to the backend's import-time granularity.
    graph_token_capacity: int = 0

    # Per-forward DeepSeek-V4 context-parallel state. This is the Python
    # counterpart of DSAMetadata::v4_cp_context; it is populated only for
    # prefill when cp_size > 1 and must never survive into a later forward.
    v4_cp_context: object | None = None

    # Candidate A: names of the value-derived fields a graph-replay build wrote
    # directly into the captured operands. Populated only on the in-place
    # refresh path; empty otherwise. The replay copy asserts these alias their
    # captured storage instead of copying them.
    refresh_inplace_fields: frozenset[str] = field(default_factory=frozenset)


def _normalize_compress_ratio(ratio: int) -> int:
    """Mirrors ``deepseek_v4_normalize_compress_ratio``."""
    return 1 if ratio <= 1 else ratio


def build_cache_specs(
    compress_ratios: Sequence[int],
    window_size: int,
    n_layers: int,
) -> tuple[list[list[DSACacheInfo]], list[DSAGroupInfo]]:
    """Python port of ``deepseek_v4_build_cache_specs`` (deepseek_v4.h:332).

    Builds the per-layer ``caches_info`` and the deduplicated ``group_infos``
    from ``compress_ratios`` + ``window_size``. Group 0 is always the SWA
    (sliding-window) group; TOKEN groups for ratios {4, 128} are registered in
    the order they first appear.
    """
    compressed_block_sizes = {ratio: DSV4_COMPRESSED_BLOCK_TOKEN_SIZE // ratio for ratio in (4, 128)}
    group_infos: list[DSAGroupInfo] = []
    group_key_map: dict[tuple[int, int, int], int] = {}

    def register_group(cache_type: int, ratio: int, block_size: int) -> int:
        key = (ratio, cache_type, block_size)
        gid = group_key_map.get(key)
        if gid is not None:
            return gid
        gid = len(group_infos)
        group_key_map[key] = gid
        group_infos.append(DSAGroupInfo(cache_type, ratio, block_size))
        return gid

    register_group(DSA_CACHE_SLIDING_WINDOW, 1, window_size)
    for ratio in compress_ratios:
        cr = _normalize_compress_ratio(ratio)
        if cr in (4, 128):
            register_group(DSA_CACHE_TOKEN, cr, compressed_block_sizes[cr])

    caches_info: list[list[DSACacheInfo]] = [[] for _ in range(n_layers)]
    for layer_id in range(n_layers):
        cr = compress_ratios[layer_id] if layer_id < len(compress_ratios) else 1
        cr = _normalize_compress_ratio(cr)

        if cr == 1:
            entries = [(DSA_CACHE_SLIDING_WINDOW, 1, window_size)]
        elif cr == 4:
            # cmp_kv, cmp_index, swa, kv_state, score_state, idx_kv,
            # idx_score, indexer_scale.
            entries = [
                (DSA_CACHE_TOKEN, 4, compressed_block_sizes[4]),
                (DSA_CACHE_TOKEN, 4, compressed_block_sizes[4]),
                (DSA_CACHE_SLIDING_WINDOW, 1, window_size),
                (DSA_CACHE_SLIDING_WINDOW, 1, window_size),
                (DSA_CACHE_SLIDING_WINDOW, 1, window_size),
                (DSA_CACHE_SLIDING_WINDOW, 1, window_size),
                (DSA_CACHE_SLIDING_WINDOW, 1, window_size),
                (DSA_CACHE_TOKEN, 4, compressed_block_sizes[4]),
            ]
        elif cr == 128:
            entries = [
                (DSA_CACHE_TOKEN, 128, compressed_block_sizes[128]),
                (DSA_CACHE_SLIDING_WINDOW, 1, window_size),
                (DSA_CACHE_SLIDING_WINDOW, 1, window_size),
                (DSA_CACHE_SLIDING_WINDOW, 1, window_size),
            ]
        else:
            entries = []

        for cache_type, ratio, block_size in entries:
            gid = register_group(cache_type, ratio, block_size)
            caches_info[layer_id].append(DSACacheInfo(gid, cache_type, ratio, block_size))

    return caches_info, group_infos


def last_ring_writes(slots: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep V4.1's latest valid SWA/Engram ring write; eager chunks only."""
    slots = slots.reshape(-1)
    if slots.numel() != values.size(0):
        raise ValueError("cache slot and value row counts must match")
    order = torch.argsort(slots, stable=True)
    sorted_slots = slots[order]
    keep = sorted_slots >= 0
    keep[:-1] &= sorted_slots[:-1] != sorted_slots[1:]
    rows = order[keep]
    return slots[rows], values[rows]


def build_cache_specs_v41(
    compress_ratios: Sequence[int],
    kv_source_layer_ids: Sequence[int],
    index_source_layer_ids: Sequence[int],
    window_size: int,
    n_layers: int,
) -> tuple[list[list[DSACacheInfo]], list[DSAGroupInfo]]:
    """Cache-spec construction for DeepSeek-V4.1 (adaptation contract section 2).

    Group registration order is exact: group 0 is the SWA ring (block size =
    window), followed by one TOKEN group per distinct kv-source compress ratio
    in first-appearance (ascending kv-source layer id) order -- with the
    standard V4.1 plan always ``[SWA, TOKEN(2), TOKEN(1)]``.

    Per-layer entries (positional order matches the C++ allocation and the
    ``LayerCache`` slots): a kv-source layer with ratio ``r`` owns
    ``[key(TOKEN,r), index(TOKEN,r), swa]`` plus ``[kv_state, score_state]``
    (both in the SWA group) when ``r > 1``; every other layer owns only its
    ``swa`` entry. Reuse and Reindex layers read the source layer's caches
    cross-layer (read-side only).
    """
    base_block_size = 128
    group_infos: list[DSAGroupInfo] = []
    group_key_map: dict[tuple[int, int, int], int] = {}

    def register_group(cache_type: int, ratio: int, block_size: int) -> int:
        key = (ratio, cache_type, block_size)
        gid = group_key_map.get(key)
        if gid is not None:
            return gid
        gid = len(group_infos)
        group_key_map[key] = gid
        group_infos.append(DSAGroupInfo(cache_type, ratio, block_size))
        return gid

    swa_gid = register_group(DSA_CACHE_SLIDING_WINDOW, 1, window_size)
    kv_sources = sorted({int(source) for source in kv_source_layer_ids})
    for source in kv_sources:
        if not 0 <= source < n_layers:
            raise ValueError(f"kv source layer {source} is outside the model's {n_layers} layers")
        ratio = int(compress_ratios[source]) if source < len(compress_ratios) else 0
        if ratio <= 0:
            raise ValueError(f"kv source layer {source} must have a positive compress ratio")
        register_group(DSA_CACHE_TOKEN, ratio, base_block_size)

    # The DSA metadata builder pairs multi_block_tables[m] with group_infos[m]
    # by index, and the worker exports multi_block_tables in the fixed
    # kMultiBlockExportOrder (core/framework/block/block.h): [SWA, C4, C128, C2,
    # C1], i.e. TOKEN groups in ratio order [4, 128, 2, 1]. The registration
    # order above follows ascending kv-source layer id, so a config whose first
    # kv-source carries ratio 1 (before any ratio-2 source) would register
    # [SWA, TOKEN(1), TOKEN(2)] and silently mis-pair every TOKEN block table
    # with the wrong group. Fail loud instead of miscomputing slots.
    export_token_ratios = [4, 128, 2, 1]  # TOKEN groups of kMultiBlockExportOrder, in order.
    actual_token_ratios = [gi.ratio for gi in group_infos if gi.cache_type == DSA_CACHE_TOKEN]
    expected_token_ratios = [ratio for ratio in export_token_ratios if ratio in actual_token_ratios]
    if actual_token_ratios != expected_token_ratios:
        raise ValueError(
            f"V4.1 TOKEN group registration order {actual_token_ratios} does not match the "
            f"worker multi_block_tables export order {expected_token_ratios} "
            "(kMultiBlockExportOrder [SWA, C4, C128, C2, C1]); multi_block_tables[m] would pair "
            "with the wrong group_infos[m]. Ensure a kv-source layer with compress ratio 2 "
            "precedes any kv-source layer with ratio 1 in kv_source_layer_ids."
        )

    caches_info: list[list[DSACacheInfo]] = [[] for _ in range(n_layers)]
    kv_source_set = set(kv_sources)
    for layer_id in range(n_layers):
        ratio = int(compress_ratios[layer_id]) if layer_id < len(compress_ratios) else 0
        entries: list[tuple[int, int, int]]
        if layer_id in kv_source_set and ratio > 0:
            # LayerCache slots: key(0), index(2), swa(5), kv_state(6),
            # score_state(7) -- the states exist only for pooling ratios > 1.
            entries = [
                (DSA_CACHE_TOKEN, ratio, base_block_size),
                (DSA_CACHE_TOKEN, ratio, base_block_size),
                (DSA_CACHE_SLIDING_WINDOW, 1, window_size),
            ]
            if ratio > 1:
                entries.append((DSA_CACHE_SLIDING_WINDOW, 1, window_size))
                entries.append((DSA_CACHE_SLIDING_WINDOW, 1, window_size))
        else:
            entries = [(DSA_CACHE_SLIDING_WINDOW, 1, window_size)]
        for cache_type, entry_ratio, block_size in entries:
            gid = register_group(cache_type, entry_ratio, block_size)
            caches_info[layer_id].append(DSACacheInfo(gid, cache_type, entry_ratio, block_size))
        if layer_id not in kv_source_set:
            # Every layer's first entry addresses the SWA group; assert the
            # registration invariant so a regression cannot silently reorder
            # the manager indices (debug-only; see build_cache_specs_v41 API).
            assert caches_info[layer_id][0].group_id == swa_gid, (
                "V4.1 cache-spec registration lost the SWA group invariant"
            )
    # API symmetry only: Reindex/Reuse layers own only their SWA entry, so the
    # index-source list never influences grouping.
    del index_source_layer_ids
    return caches_info, group_infos


class DsaMetadataBuilder:
    """Faithful Python port of ``DSAMetadataBuilder`` (dsa_metadata_builder.cpp).

    Construct once per model (with the static ``caches_info`` / ``group_infos``),
    then call :meth:`build` every forward to expand the per-manager block tables
    into per-layer ``block_tables`` / ``slot_mappings``.
    """

    def __init__(
        self,
        caches_info: list[list[DSACacheInfo]],
        group_infos: list[DSAGroupInfo],
        read_table_layout: str = READ_TABLE_LAYOUT_RIGHT_ALIGNED,
    ) -> None:
        if read_table_layout not in (READ_TABLE_LAYOUT_RIGHT_ALIGNED, READ_TABLE_LAYOUT_RING):
            raise ValueError(
                f"unknown SWA read-table layout: {read_table_layout!r} "
                f"(expected {READ_TABLE_LAYOUT_RIGHT_ALIGNED!r} or {READ_TABLE_LAYOUT_RING!r})"
            )
        self.caches_info = caches_info
        self.group_infos = group_infos
        # Layout of the SWA read-side block table rebuilt in
        # ``_process_swa_group``. ``"right_aligned"`` keeps the historical
        # shared-path behaviour (DSV4 / V4-dspark); ``"ring"`` is requested by
        # the DeepSeek-V4.1 backend whose readers need every column addressed.
        self.read_table_layout = read_table_layout
        # Ring-layout column index cache for ``_process_swa_group``: the
        # ``arange(width) % semantic_cols`` table is a pure function of its
        # key, so rebuilding it every forward only adds allocations to the
        # metadata-refresh hot path.
        self._ring_cols_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}
        # Whether the latest build() produced any -1 padding lanes in its
        # manager tables (padding rows / columns). Consumers use this to skip
        # the "-1 remap to block 0" masked fill when nothing can be negative.
        self.outputs_have_padding = False

    # -- public API ---------------------------------------------------------

    def build(
        self,
        multi_block_tables: Sequence[torch.Tensor],
        kv_seq_lens: Sequence[int],
        q_seq_lens: Sequence[int] | None,
        positions: torch.Tensor,
        dsa_cos_sin: torch.Tensor | None,
        is_prefill: bool,
        is_chunked_prefill: bool,
        new_cache_slots: Sequence[int] | None = None,
        enable_graph: bool = False,
        graph_block_table_capacity_cols: int = 0,
        max_query_len: int = 0,
        max_seq_len: int = 0,
        group_limit: int | None = None,
        swa_device_ctx_lens: torch.Tensor | None = None,
        dst: DsaMetadata | None = None,
    ) -> DsaMetadata:
        """Build one forward's DSA metadata.

        ``swa_device_ctx_lens`` opts the SWA group's slot mapping into the
        pure-device path: it is the device-resident per-sequence KV length
        truth (``metadata.kv_seq_lens``). ``None`` (the default) keeps the
        historical host-loop build.

        ``dst`` opts the whole build into the in-place refresh path: the
        graph-captured ``DsaMetadata`` whose tensor buffers were baked into
        the ACL graph. When every shape matches, the value-derived fields are
        written straight into those buffers and the returned metadata aliases
        them, so the replay copy chain degenerates to a shape/alias
        assertion. ``None`` (the default and the only value the replay path
        passes) keeps the historical build byte-for-byte.
        """
        batch_size = len(kv_seq_lens)
        if q_seq_lens is None or len(q_seq_lens) != batch_size:
            if is_prefill or is_chunked_prefill:
                q_lens = list(kv_seq_lens)
            else:
                q_lens = [1] * batch_size
        else:
            q_lens = list(q_seq_lens)

        dsa = self._build_seq_lengths(
            kv_seq_lens,
            q_lens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            dst=dst,
        )
        dsa.input_positions = positions
        dsa.is_acl_graph = enable_graph
        if dsa_cos_sin is not None and dsa_cos_sin.numel() > 0:
            # Preserve the source layout instead of copying each half-width
            # RoPE table; NPU kernels accept these request-shaped views.
            cos_sin_chunks = dsa_cos_sin.chunk(2, dim=-1)
            dsa.cos_table = cos_sin_chunks[0]
            dsa.sin_table = cos_sin_chunks[1]
        if positions is not None and positions.numel() > 0:
            self._build_positions(dsa, kv_seq_lens, q_lens, enable_graph, dst=dst)
        if dst is not None and self._start_pos_dst_ok(dst, dsa):
            # ``out=`` writes the captured buffer and leaves the value identical
            # to the allocating ``(a - b).to(int32)`` on int32 operands.
            torch.sub(dsa.actual_seq_lengths_kv, dsa.seq_lens_q, out=dst.start_pos)
            dsa.start_pos = dst.start_pos
        else:
            dsa.start_pos = (dsa.actual_seq_lengths_kv - dsa.seq_lens_q).to(torch.int32)

        self.outputs_have_padding = False
        self._build_block_tables_and_slots(
            multi_block_tables,
            kv_seq_lens,
            q_lens,
            batch_size,
            positions,
            enable_graph,
            graph_block_table_capacity_cols,
            new_cache_slots,
            dsa,
            group_limit=group_limit,
            swa_device_ctx_lens=swa_device_ctx_lens,
            dst=dst,
        )
        return dsa

    # -- in-place refresh destination compatibility (candidate A) --

    @staticmethod
    def _start_pos_dst_ok(dst: DsaMetadata, dsa: DsaMetadata) -> bool:
        target = getattr(dst, "start_pos", None)
        if target is None or target.dtype != torch.int32:
            return False
        if target.shape != dsa.seq_lens_q.shape:
            return False
        return target.device == dsa.seq_lens_q.device == dsa.actual_seq_lengths_kv.device

    # -- step 1: sequence lengths (build_seq_lengths, cpp:574-661) ----------

    def _build_seq_lengths(
        self,
        kv_seq_lens: Sequence[int],
        q_lens: Sequence[int],
        *,
        max_query_len: int,
        max_seq_len: int,
        dst: DsaMetadata | None = None,
    ) -> DsaMetadata:
        device = torch.device("cpu")
        kv = torch.tensor(kv_seq_lens, dtype=torch.int32, device=device)
        q = torch.tensor(q_lens, dtype=torch.int32, device=device)
        zeros_prefix = torch.zeros(1, dtype=torch.int32, device=device)
        actual_seq_lengths_query = torch.cat([zeros_prefix, q.cumsum(0).to(torch.int32)])
        kv_cu = torch.cat([zeros_prefix, kv.cumsum(0).to(torch.int32)])
        max_kv = kv.max().to(torch.int32) if kv.numel() else torch.zeros(1, dtype=torch.int32, device=device)
        max_q = q.max().to(torch.int32) if q.numel() else torch.zeros(1, dtype=torch.int32, device=device)
        max_query_len = max(int(max_query_len), max((int(value) for value in q_lens), default=0))
        max_seq_len = max(
            int(max_seq_len),
            max((int(value) for value in kv_seq_lens), default=0),
        )
        if dst is not None and self._seq_lengths_dst_ok(
            dst,
            kv,
            q,
            actual_seq_lengths_query,
            kv_cu,
            max_kv,
            max_q,
        ):
            # In-place refresh: write the host-derived values straight into the
            # graph-captured buffers (addresses never move); ``seq_lens`` /
            # ``actual_seq_lengths_kv`` already alias one captured storage
            # (checked by ``_seq_lengths_dst_ok``), so one write updates both
            # operands without rebinding either.
            #
            # ``kv_cu_seq_lens`` here carries the HOST cumsum semantics: its
            # padded tail keeps advancing, unlike the C++ op which re-broadcasts
            # the last real value. Harmless today (the decode ACL path never
            # consumes ``kv_cu_seq_lens``) but MUST be revisited if a future
            # consumer reads the cumulative axis across pad lanes.
            dst.seq_lens.copy_(kv, non_blocking=True)
            dst.seq_lens_q.copy_(q, non_blocking=True)
            dst.actual_seq_lengths_query.copy_(actual_seq_lengths_query, non_blocking=True)
            dst.kv_cu_seq_lens.copy_(kv_cu, non_blocking=True)
            dst.max_seqlen_kv.copy_(max_kv, non_blocking=True)
            dst.max_seqlen_q.copy_(max_q, non_blocking=True)
            dst.max_query_len = max_query_len
            dst.max_seq_len = max_seq_len
            dst.layer_id = -1
            return DsaMetadata(
                layer_id=-1,
                seq_lens=dst.seq_lens,
                seq_lens_q=dst.seq_lens_q,
                actual_seq_lengths_kv=dst.seq_lens,
                actual_seq_lengths_query=dst.actual_seq_lengths_query,
                kv_cu_seq_lens=dst.kv_cu_seq_lens,
                max_seqlen_kv=dst.max_seqlen_kv,
                max_seqlen_q=dst.max_seqlen_q,
                max_query_len=max_query_len,
                max_seq_len=max_seq_len,
                input_positions=torch.empty(0),
                c4_pad_positions=torch.empty(0, dtype=torch.int64),
                c128_pad_positions=torch.empty(0, dtype=torch.int64),
                start_pos=torch.empty(0),
            )
        return DsaMetadata(
            layer_id=-1,
            seq_lens=kv,
            seq_lens_q=q,
            actual_seq_lengths_kv=kv,
            actual_seq_lengths_query=actual_seq_lengths_query,
            kv_cu_seq_lens=kv_cu,
            max_seqlen_kv=max_kv,
            max_seqlen_q=max_q,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            input_positions=torch.empty(0),
            c4_pad_positions=torch.empty(0, dtype=torch.int64),
            c128_pad_positions=torch.empty(0, dtype=torch.int64),
            start_pos=torch.empty(0),
        )

    @staticmethod
    def _seq_lengths_dst_ok(
        dst: DsaMetadata,
        kv: torch.Tensor,
        q: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        kv_cu: torch.Tensor,
        max_kv: torch.Tensor,
        max_q: torch.Tensor,
    ) -> bool:
        """Whether the captured buffers can absorb the seq-lens family in place."""
        for name, source in (
            ("seq_lens", kv),
            ("seq_lens_q", q),
            ("actual_seq_lengths_query", actual_seq_lengths_query),
            ("kv_cu_seq_lens", kv_cu),
            ("max_seqlen_kv", max_kv),
            ("max_seqlen_q", max_q),
        ):
            target = getattr(dst, name, None)
            if target is None or target.dtype != torch.int32 or target.shape != source.shape:
                return False
        alias = getattr(dst, "actual_seq_lengths_kv", None)
        if alias is None or alias.data_ptr() != dst.seq_lens.data_ptr():
            # The capture contract aliases seq_lens / actual_seq_lengths_kv onto
            # one storage; an unaliased destination would lose one of the two
            # captured operands, so fall back to the copy path.
            return False
        return True

    # -- step 2: positions (build_positions, cpp:663-777) ------------------

    def _build_positions(
        self,
        dsa: DsaMetadata,
        kv_seq_lens: Sequence[int],
        q_lens: Sequence[int],
        enable_graph: bool,
        dst: DsaMetadata | None = None,
    ) -> None:
        """Collect c4/c128 compressed RoPE positions.

        For each query token at absolute ``pos``, when ``(pos + 1) % ratio == 0``
        the compressed RoPE needs the position ``next_pos - ratio``.
        """
        total_tokens = int(dsa.input_positions.numel())
        # Concatenate the per-sequence position ranges in sequence-major order,
        # then select positions whose successor crosses each ratio boundary.
        ranges = [
            torch.arange(kv_len - min(q_lens[seq], kv_len), kv_len, dtype=torch.int64)
            for seq, kv_len in enumerate(kv_seq_lens)
        ]
        positions = torch.cat(ranges) if ranges else torch.empty(0, dtype=torch.int64)
        next_positions = positions + 1
        c4_positions = next_positions[(next_positions % 4) == 0] - 4
        c128_positions = next_positions[(next_positions % 128) == 0] - 128

        def _pad(values: torch.Tensor, ratio: int) -> torch.Tensor:
            if enable_graph:
                # Graph mode pads to total_tokens so the tensor address is stable
                # across bucket sizes. C++ vector::resize() zero-fills the tail.
                target = total_tokens
            else:
                # Non-graph: resize to min(total_tokens, total_tokens//ratio + batch_size)
                # with 0 padding, matching C++ dsa_metadata_builder.cpp:717
                # (c4_target = min(num_tokens, num_tokens/4 + batch_size)).
                target = min(total_tokens, total_tokens // ratio + len(kv_seq_lens))
            out = torch.zeros(target, dtype=dsa.input_positions.dtype)
            count = min(int(values.numel()), target)
            if count > 0:
                out[:count] = values[:count].to(dsa.input_positions.dtype)
            return out

        dsa.c4_pad_positions = self._write_positions(dst, "c4_pad_positions", _pad(c4_positions, 4))
        dsa.c128_pad_positions = self._write_positions(dst, "c128_pad_positions", _pad(c128_positions, 128))

    @staticmethod
    def _write_positions(dst: DsaMetadata | None, name: str, source: torch.Tensor) -> torch.Tensor:
        """Write a compressed-position vector into its captured buffer if possible."""
        if dst is None:
            return source
        target = getattr(dst, name, None)
        if target is None or target.dtype != source.dtype or target.shape != source.shape:
            return source
        target.copy_(source, non_blocking=True)
        return target

    # -- step 3: block_tables / slot_mappings (build_dsa_fields, cpp:169-264)

    def _build_block_tables_and_slots(
        self,
        multi_block_tables: Sequence[torch.Tensor],
        ctx_lens: Sequence[int],
        q_lens: Sequence[int],
        batch_size: int,
        positions: torch.Tensor,
        enable_graph: bool,
        graph_block_table_capacity_cols: int,
        new_cache_slots: Sequence[int] | None,
        dsa: DsaMetadata,
        group_limit: int | None = None,
        swa_device_ctx_lens: torch.Tensor | None = None,
        dst: DsaMetadata | None = None,
    ) -> None:
        """Expand the request's per-group block tables into per-layer metadata.

        ``group_limit`` is for a model that shares this request's metadata but
        owns fewer cache groups than its owner (the DSpark draft reads the
        target's metadata): the request's tables follow the fixed export order
        (SWA first), so a model's own deduplicated groups are its leading
        entries. ``None`` (the owning model) consumes every table.

        ``dst`` (in-place refresh) supplies the graph-captured per-group tensor
        that each manager's table is written into directly.
        """
        if not multi_block_tables or not self.caches_info:
            return

        active = list(multi_block_tables)
        if group_limit is not None:
            if group_limit < 0:
                raise ValueError(f"group_limit must be non-negative, got {group_limit}")
            active = active[:group_limit]
        manager_num = len(active)
        # Packed [manager, blocks] auto-unpack when batch_size == 1.
        if (
            manager_num == 1
            and batch_size == 1
            and active[0].dim() == 2
            and active[0].size(0) > 1
            and active[0].size(0) <= len(self.group_infos)
        ):
            packed = active[0].contiguous()
            active = [packed[m].unsqueeze(0).contiguous() for m in range(packed.size(0))]
            manager_num = len(active)

        if manager_num > len(self.group_infos):
            raise ValueError(f"manager count {manager_num} exceeds group count {len(self.group_infos)}")
        if enable_graph and graph_block_table_capacity_cols > 0:
            for manager_id, block_table in enumerate(active):
                if block_table.dim() != 2:
                    raise ValueError(
                        f"ACL graph multi_block_tables must be 2-D: manager {manager_id} has rank {block_table.dim()}"
                    )
                if block_table.size(1) > graph_block_table_capacity_cols:
                    raise ValueError(
                        "ACL graph block table exceeds bucket capacity: "
                        f"manager {manager_id} requires {block_table.size(1)} "
                        f"columns, capacity is {graph_block_table_capacity_cols}"
                    )

        graph_slot_capacity = int(positions.numel()) if enable_graph and positions.numel() > 0 else 0
        total_tokens = sum(int(x) for x in ctx_lens)

        dst_bt: list[torch.Tensor | None] = [None] * manager_num
        dst_slots: list[torch.Tensor | None] = [None] * manager_num
        if dst is not None:
            for m in range(manager_num):
                dst_bt[m], dst_slots[m] = self._find_group_dst(dst, m)

        proc_bt: list[torch.Tensor] = [torch.empty(0)] * manager_num
        proc_slots: list[torch.Tensor] = [torch.empty(0)] * manager_num
        for m in range(manager_num):
            gi = self.group_infos[m]
            proc_bt[m], proc_slots[m] = self._process_group(
                active[m],
                gi,
                ctx_lens,
                q_lens,
                batch_size,
                total_tokens,
                graph_slot_capacity,
                graph_block_table_capacity_cols,
                new_cache_slots,
                swa_device_ctx_lens,
                dst_bt=dst_bt[m],
                dst_slots=dst_slots[m],
            )

        n_layers = len(self.caches_info)
        dsa.block_tables = [[] for _ in range(n_layers)]
        dsa.slot_mappings = [[] for _ in range(n_layers)]
        for lid in range(n_layers):
            for ci in range(len(self.caches_info[lid])):
                gid = self.caches_info[lid][ci].group_id
                if gid < manager_num:
                    dsa.block_tables[lid].append(proc_bt[gid])
                    dsa.slot_mappings[lid].append(proc_slots[gid])
                else:
                    dsa.block_tables[lid].append(torch.empty(0))
                    dsa.slot_mappings[lid].append(torch.empty(0))
        dsa.manager_block_tables = list(proc_bt)
        dsa.manager_slot_mappings = list(proc_slots)

    def _find_group_dst(
        self,
        dst: DsaMetadata,
        manager_id: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Locate the captured block-table / slot buffer of one manager group.

        Every layer/cache in a group aliases one tensor, so the first entry for
        ``manager_id`` is the captured buffer the in-place refresh writes into.
        """
        block_tables = getattr(dst, "block_tables", None) or []
        slot_mappings = getattr(dst, "slot_mappings", None) or []
        for lid, layer in enumerate(block_tables):
            if lid >= len(self.caches_info):
                break
            for ci, cache_info in enumerate(self.caches_info[lid]):
                if cache_info.group_id != manager_id or ci >= len(layer):
                    continue
                block = layer[ci]
                slot = None
                if lid < len(slot_mappings) and ci < len(slot_mappings[lid]):
                    slot = slot_mappings[lid][ci]
                return block, slot
        return None, None

    @staticmethod
    def _block_table_dst_ok(
        dst_bt: torch.Tensor | None,
        raw_bt: torch.Tensor,
        batch_size: int,
        cols: int,
    ) -> bool:
        return (
            dst_bt is not None
            and dst_bt.dtype == torch.int32
            and dst_bt.dim() == 2
            and dst_bt.device == raw_bt.device
            and dst_bt.size(0) >= batch_size
            and dst_bt.size(1) == cols
        )

    @staticmethod
    def _slot_dst_ok(dst_slots: torch.Tensor | None, raw_bt: torch.Tensor, rows: int) -> bool:
        return (
            dst_slots is not None
            and dst_slots.dtype == torch.int32
            and dst_slots.device == raw_bt.device
            and dst_slots.numel() == rows
        )

    # -- per-group processing (process_group, cpp:323-362) -----------------

    def _process_group(
        self,
        raw_bt: torch.Tensor,
        gi: DSAGroupInfo,
        ctx_lens: Sequence[int],
        q_lens: Sequence[int],
        batch_size: int,
        total_tokens: int,
        graph_slot_capacity: int,
        graph_block_table_capacity_cols: int,
        new_cache_slots: Sequence[int] | None,
        swa_device_ctx_lens: torch.Tensor | None = None,
        dst_bt: torch.Tensor | None = None,
        dst_slots: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gi.cache_type == DSA_CACHE_TOKEN:
            return self._process_token_group(
                raw_bt,
                gi.ratio,
                gi.block_size,
                ctx_lens,
                q_lens,
                batch_size,
                graph_slot_capacity,
                graph_block_table_capacity_cols,
                dst_bt=dst_bt,
                dst_slots=dst_slots,
            )
        if gi.cache_type == DSA_CACHE_SLIDING_WINDOW:
            return self._process_swa_group(
                raw_bt,
                gi.block_size,
                ctx_lens,
                q_lens,
                batch_size,
                graph_slot_capacity,
                graph_block_table_capacity_cols,
                new_cache_slots,
                swa_device_ctx_lens,
                dst_bt=dst_bt,
                dst_slots=dst_slots,
            )
        # SEQUENCE: expand the whole context.
        return self._expand_blocks_to_slots(raw_bt, gi, ctx_lens, batch_size, total_tokens)

    # -- TOKEN group (process_token_group, cpp:364-464) --------------------
    # Commits only the compressed rows crossed by the current forward step.

    @staticmethod
    def _gather_block_ids(
        raw_bt: torch.Tensor,
        seq_ids: list[int],
        block_idxs: list[int],
    ) -> list[int]:
        """Gather ``raw_bt[seq, block_idx]`` for a whole batch with one device
        op and one host sync.

        The scalar equivalent (``int(raw_bt[s, b].item())`` per slot) costs one
        device synchronization per call, which dominates the metadata rebuild
        once the batch grows (a concurrency-16 refresh gathers ~100 slots per
        manager). Out-of-range requests are reported as ``-1`` so callers can
        treat them as absent slots.
        """
        rows = int(raw_bt.size(0))
        cols = int(raw_bt.size(1))
        out = [-1] * len(seq_ids)
        if not seq_ids or rows <= 0 or cols <= 0:
            return out
        flat = [s * cols + b for s, b in zip(seq_ids, block_idxs) if 0 <= s < rows and 0 <= b < cols]
        keep = [i for i, (s, b) in enumerate(zip(seq_ids, block_idxs)) if 0 <= s < rows and 0 <= b < cols]
        if not keep:
            return out
        flat_idx = torch.tensor(flat, dtype=torch.int64, device=raw_bt.device)
        got = raw_bt.reshape(-1)[flat_idx].tolist()
        for j, i in enumerate(keep):
            out[i] = got[j]
        return out

    def _process_token_group(
        self,
        raw_bt: torch.Tensor,
        ratio: int,
        block_size: int,
        ctx_lens: Sequence[int],
        q_lens: Sequence[int],
        batch_size: int,
        graph_slot_capacity: int,
        graph_block_table_capacity_cols: int,
        dst_bt: torch.Tensor | None = None,
        dst_slots: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        committed_rows = 0
        for seq in range(batch_size):
            ctx_len = int(ctx_lens[seq])
            q_len = max(0, min(int(q_lens[seq]), ctx_len))
            prev_ctx_len = ctx_len - q_len
            committed_rows += ctx_len // ratio - prev_ctx_len // ratio

        out_slot_rows = max(graph_slot_capacity, committed_rows) if graph_slot_capacity > 0 else committed_rows
        use_dst_slots = self._slot_dst_ok(dst_slots, raw_bt, out_slot_rows)

        slots_list = [-1] * out_slot_rows
        seq_ids: list[int] = []
        comp_idxs: list[int] = []
        for seq in range(batch_size):
            ctx_len = int(ctx_lens[seq])
            q_len = max(0, min(int(q_lens[seq]), ctx_len))
            prev_ctx_len = ctx_len - q_len
            prev_committed = prev_ctx_len // ratio
            committed = ctx_len // ratio
            new_committed = committed - prev_committed
            seq_ids.extend([seq] * new_committed)
            comp_idxs.extend(range(prev_committed, committed))
        # Vectorized equivalent of the per-slot ``slot_for_compressed_index``
        # loop (cpp:408-433): one batched gather instead of one ``.item()``
        # device sync per newly committed row.
        block_ids = self._gather_block_ids(
            raw_bt,
            seq_ids,
            [compressed_idx // block_size for compressed_idx in comp_idxs],
        )
        for i, block_id in enumerate(block_ids):
            if block_id >= 0:
                slots_list[i] = block_id * block_size + comp_idxs[i] % block_size
        if use_dst_slots:
            out_slots = dst_slots
            out_slots.copy_(torch.tensor(slots_list, dtype=torch.int32), non_blocking=True)
        else:
            out_slots = torch.tensor(slots_list, dtype=torch.int32, device=raw_bt.device)

        out_bt = raw_bt
        if graph_slot_capacity > 0 and graph_block_table_capacity_cols > 0:
            cap = max(graph_block_table_capacity_cols, int(raw_bt.size(1)))
            if self._block_table_dst_ok(dst_bt, raw_bt, batch_size, cap):
                out_bt = dst_bt
                out_bt.fill_(-1)
                self.outputs_have_padding = True
                rows = min(batch_size, int(raw_bt.size(0)))
                copy_cols = min(int(raw_bt.size(1)), cap)
                if rows > 0 and copy_cols > 0:
                    out_bt[:rows, :copy_cols] = raw_bt[:rows, :copy_cols]
            else:
                if int(raw_bt.size(0)) == batch_size and int(raw_bt.size(1)) >= cap:
                    out_bt = raw_bt.clone()
                else:
                    out_bt = self._pad_block_table(raw_bt, batch_size, cap, -1)
                    self.outputs_have_padding = True
        return out_bt, out_slots

    # -- SWA group (process_swa_group, cpp:466-572) ------------------------
    # Writes only the current forward's query tokens, ring-indexed by position.

    def _process_swa_group(
        self,
        raw_bt: torch.Tensor,
        block_size: int,
        ctx_lens: Sequence[int],
        q_lens: Sequence[int],
        batch_size: int,
        graph_slot_capacity: int,
        graph_block_table_capacity_cols: int,
        new_cache_slots: Sequence[int] | None,
        swa_device_ctx_lens: torch.Tensor | None = None,
        dst_bt: torch.Tensor | None = None,
        dst_slots: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_total_tokens = 0
        for seq in range(batch_size):
            query_total_tokens += max(0, min(int(q_lens[seq]), int(ctx_lens[seq])))

        out_slot_rows = max(graph_slot_capacity, query_total_tokens) if graph_slot_capacity > 0 else query_total_tokens
        semantic_cols = int(raw_bt.size(1))
        use_dst_slots = self._slot_dst_ok(dst_slots, raw_bt, out_slot_rows)

        if swa_device_ctx_lens is not None:
            # Pure-device slot mapping. Uses the device KV-length truth
            # (bit-equal to the host ``ctx_lens`` on the decode graph path) so
            # there is no per-token ``.item()`` / ``.tolist()`` D2H here.
            out_slots = self._swa_slots_on_device(
                raw_bt,
                block_size,
                semantic_cols,
                out_slot_rows,
                batch_size,
                q_lens,
                new_cache_slots,
                query_total_tokens,
                swa_device_ctx_lens,
                dst_slots=dst_slots if use_dst_slots else None,
            )
        else:
            write_idx = 0
            slots_list = [-1] * out_slot_rows
            if new_cache_slots is not None and len(new_cache_slots) == query_total_tokens:
                slots_list[:query_total_tokens] = [int(slot) for slot in new_cache_slots]
                write_idx = query_total_tokens
            else:
                rows = int(raw_bt.size(0))
                seq_ids: list[int] = []
                positions: list[int] = []
                out_indices: list[int] = []
                for seq in range(batch_size):
                    ctx_len = int(ctx_lens[seq])
                    q_len = max(0, min(int(q_lens[seq]), ctx_len))
                    if seq < rows:
                        q_start = ctx_len - q_len
                        seq_ids.extend([seq] * q_len)
                        positions.extend(range(q_start, q_start + q_len))
                        out_indices.extend(range(write_idx, write_idx + q_len))
                    # Sequences whose block-table rows are missing keep their -1
                    # padding at their own output offset; every sequence advances
                    # the cursor either way so gathered tokens land at the output
                    # index their (seq, position) occupies.
                    write_idx += q_len
                # Vectorized equivalent of the per-position ``slot_for_position``
                # loop (cpp:466-572): one batched gather instead of one
                # ``.item()`` device sync per query token.
                block_ids = self._gather_block_ids(
                    raw_bt,
                    seq_ids,
                    [(pos // block_size) % semantic_cols if semantic_cols > 0 else 0 for pos in positions],
                )
                for i, block_id in enumerate(block_ids):
                    if block_id >= 0:
                        slots_list[out_indices[i]] = block_id * block_size + positions[i] % block_size
            if use_dst_slots:
                out_slots = dst_slots
                out_slots.copy_(torch.tensor(slots_list, dtype=torch.int32), non_blocking=True)
            else:
                out_slots = torch.tensor(slots_list, dtype=torch.int32, device=raw_bt.device)

        # Rebuild the read-side block table with the SWA ring semantics.
        #
        # The write path (`slot_for_position`) maps position `p` to
        # `raw_bt[seq, (p // block_size) % semantic_cols]`, so a read consumer
        # must resolve logical column `(p // block_size) % semantic_cols`.
        # Two layouts satisfy that contract:
        #
        #   * "ring" (DeepSeek-V4.1, requested by ``Csa2AttentionBackend``):
        #     every storage column `d` carries the ring block for its own
        #     logical index, `new_bt[s, d] = raw_bt[s, d % semantic_cols]`, for
        #     EVERY column `d`. The three V4.1 readers address the table by
        #     absolute column `(p // block_size) % table_width` and do NOT
        #     compensate for a shifted layout: `csa2_attention.py`'s
        #     `_gather_swa_positions` / `_swa_position_slots` and
        #     `deepseek_v41_engram.py`'s `_cache_source`. In production
        #     `semantic_cols == 1` and the writer only fills column 0, so a
        #     right-aligned table parks the single ring block in the LAST
        #     column: every other addressed column reads `-1`. The eager
        #     readers `clamp_min(0)` that to block 0 -- a foreign KV block
        #     whenever the sequence's ring block is not 0 (e.g. ctx=257 with
        #     `blocks=[0,9]` resolves as block 0 for 127/128 positions) -- and
        #     the fused `sparse_attn_sharedkv` / `sparse_flash_mla` op does not
        #     clamp at all: it reads the `int32_t(-1)` through a `uint64_t`
        #     (`sparse_attn_sharedkv_common.h:149-158`), wraps it to a huge
        #     offset and reads out of bounds.
        #   * "right_aligned" (shared default, DSV4 / V4-dspark): only the last
        #     `min(semantic_cols, dst_lens)` columns are filled and the rest
        #     stay `-1`. Its consumers compensate for the shift:
        #     `csa_attention.py`'s `_build_dspark_swa_indices` is
        #     layout-adaptive (`valid_block_counts` / `first_retained_columns`)
        #     and resolves both layouts identically, so this default keeps the
        #     historical behaviour for those paths.
        #
        # The table width decision below is shared by both layouts.
        dst_lens = [(max(int(ctx_lens[s]), 0) + block_size - 1) // block_size for s in range(batch_size)]
        max_dst_len = max(max(dst_lens) if dst_lens else 0, semantic_cols)
        if graph_slot_capacity > 0 and graph_block_table_capacity_cols > 0:
            storage_cols = max(graph_block_table_capacity_cols, int(raw_bt.size(1)))
            max_dst_len = max(max_dst_len, storage_cols)
        use_dst_bt = self._block_table_dst_ok(dst_bt, raw_bt, batch_size, max_dst_len)
        rows = min(batch_size, int(raw_bt.size(0)))
        ring_fill = (
            self.read_table_layout == READ_TABLE_LAYOUT_RING and rows > 0 and max_dst_len > 0 and semantic_cols > 0
        )
        if ring_fill:
            # The ring gather below overwrites every column of ``[:rows]``, so
            # uninitialized storage plus a padding-row fill replaces a full
            # ``torch.full(-1)`` pass over the whole table.
            new_bt = dst_bt if use_dst_bt else torch.empty(
                (batch_size, max_dst_len),
                dtype=torch.int32,
                device=raw_bt.device,
            )
            if batch_size > rows:
                new_bt[rows:].fill_(-1)
                self.outputs_have_padding = True
            # Vectorized equivalent of a NAIVE per-column ring fill: filling
            # `max_dst_len` columns (up to the graph storage width) with one
            # device op each on every forward would cost ~151x-2141x more.
            cache_key = (max_dst_len, semantic_cols, raw_bt.device)
            cols = self._ring_cols_cache.get(cache_key)
            if cols is None:
                if len(self._ring_cols_cache) >= 64:
                    self._ring_cols_cache.clear()
                cols = torch.arange(max_dst_len, device=raw_bt.device) % semantic_cols
                self._ring_cols_cache[cache_key] = cols
            new_bt[:rows] = raw_bt[:rows, cols]
        else:
            if use_dst_bt:
                new_bt = dst_bt
                new_bt.fill_(-1)
            else:
                new_bt = torch.full(
                    (batch_size, max_dst_len),
                    -1,
                    dtype=torch.int32,
                    device=raw_bt.device,
                )
            self.outputs_have_padding = True
            if self.read_table_layout != READ_TABLE_LAYOUT_RING:
                # Right-aligned tables keep -1 outside the retained columns
                # by design; flag them so consumers preserve the historical
                # remap behaviour.
                self.outputs_have_padding = True
                for s in range(batch_size):
                    if s >= raw_bt.size(0):
                        continue
                    retained_cols = min(semantic_cols, dst_lens[s])
                    start_col = dst_lens[s] - retained_cols
                    for j in range(retained_cols):
                        logical_col = start_col + j
                        physical_col = logical_col % semantic_cols
                        new_bt[s, logical_col] = raw_bt[s, physical_col]
        return new_bt, out_slots

    @staticmethod
    def _swa_slots_on_device(
        raw_bt: torch.Tensor,
        block_size: int,
        semantic_cols: int,
        out_slot_rows: int,
        batch_size: int,
        q_lens: Sequence[int],
        new_cache_slots: Sequence[int] | None,
        query_total_tokens: int,
        device_ctx_lens: torch.Tensor,
        dst_slots: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Device-only form of the SWA slot mapping.

        Vectorized equivalent of the host loop: for every committed token row
        ``slot = block_id * block_size + pos % block_size`` where
        ``block_id = raw_bt[row, (pos // block_size) % semantic_cols]`` and a
        negative block maps to ``-1``. ``pos`` is the row's last query position
        ``ctx_len - q_len``; ``device_ctx_lens`` is the device truth equal to the
        host ``ctx_lens`` on the decode graph path.

        ``new_cache_slots`` (host, from the C++ scheduler) still uploads via a
        one-shot H2D -- it is an H2D, not the D2H sync this path removes. The
        device branch is only selected after the caller proves the ``q == 1``
        decode contract and ``ctx_len >= 1`` for every row, so the host loop's
        monotonic ``write_idx`` degenerates to the row index and the row-aligned
        write below is bit-identical.
        """
        use_dst = (
            dst_slots is not None
            and dst_slots.dtype == torch.int32
            and dst_slots.device == raw_bt.device
            and dst_slots.numel() == out_slot_rows
        )
        if use_dst:
            out_slots = dst_slots
            out_slots.fill_(-1)
        else:
            out_slots = torch.full((out_slot_rows,), -1, dtype=torch.int32, device=raw_bt.device)
        if new_cache_slots is not None and len(new_cache_slots) == query_total_tokens:
            out_slots[:query_total_tokens] = torch.tensor(
                [int(slot) for slot in new_cache_slots],
                dtype=torch.int32,
                device=raw_bt.device,
            )
            return out_slots
        rows = min(batch_size, int(raw_bt.size(0)), int(device_ctx_lens.numel()))
        if rows <= 0 or semantic_cols <= 0 or block_size <= 0:
            return out_slots
        ctx = device_ctx_lens[:rows].to(torch.int32)
        query_lens = torch.tensor(
            [max(0, int(q_lens[seq])) for seq in range(rows)],
            dtype=torch.int32,
            device=raw_bt.device,
        )
        q_eff = torch.minimum(query_lens, ctx.clamp_min(0))
        q_start = ctx - q_eff
        block_idx = ((q_start // block_size) % semantic_cols).to(torch.int64)
        block_id = raw_bt[:rows].gather(1, block_idx.unsqueeze(1)).squeeze(1)
        offset = q_start % block_size
        slot = block_id * block_size + offset
        slot = torch.where(block_id < 0, torch.full_like(slot, -1), slot)
        slot = torch.where(q_eff > 0, slot, torch.full_like(slot, -1))
        out_slots[:rows] = slot.to(torch.int32)
        return out_slots

    # -- SEQUENCE group (expand_blocks_to_slots, cpp:270-307) --------------

    def _expand_blocks_to_slots(
        self,
        raw_bt: torch.Tensor,
        gi: DSAGroupInfo,
        ctx_lens: Sequence[int],
        batch_size: int,
        total_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_size = gi.block_size
        slots = torch.full((total_tokens,), -1, dtype=torch.int32, device=raw_bt.device)
        max_blocks = int(raw_bt.size(1))
        start_idx = 0
        for seq in range(batch_size):
            token_len = int(ctx_lens[seq])
            slot_num = self._compute_slot_num(gi, token_len)
            if seq >= raw_bt.size(0):
                start_idx += token_len
                continue
            filled = 0
            for blk in range(max_blocks):
                if filled >= slot_num:
                    break
                block_id = int(raw_bt[seq, blk].item())
                if block_id < 0:
                    break
                for off in range(block_size):
                    if filled >= slot_num:
                        break
                    slots[start_idx + filled] = block_id * block_size + off
                    filled += 1
            start_idx += token_len
        # Replace -1 padding with 0 (C++ does torch::where(eq(-1), 0, raw)).
        slots = torch.where(slots.eq(-1), torch.zeros_like(slots), slots)
        return raw_bt, slots

    @staticmethod
    def _compute_slot_num(gi: DSAGroupInfo, token_len: int) -> int:
        if gi.cache_type == DSA_CACHE_TOKEN:
            return token_len // gi.ratio
        # SLIDING_WINDOW
        block_size = gi.block_size
        if token_len > block_size:
            return token_len % block_size + block_size
        remainder = token_len % block_size
        return block_size if (remainder == 0 and token_len > 0) else remainder

    @staticmethod
    def _pad_block_table(
        raw_bt: torch.Tensor,
        batch_size: int,
        capacity_cols: int,
        pad_value: int,
    ) -> torch.Tensor:
        cols = max(capacity_cols, int(raw_bt.size(1)))
        rows = min(batch_size, int(raw_bt.size(0)))
        copy_cols = min(int(raw_bt.size(1)), cols)
        if int(raw_bt.size(0)) == batch_size and copy_cols == cols:
            # Shape already matches: one clone instead of a full ``pad_value``
            # pass followed by a copy.
            return raw_bt.clone()
        out = torch.full((batch_size, cols), pad_value, dtype=torch.int32, device=raw_bt.device)
        out[:rows, :copy_cols] = raw_bt[:rows, :copy_cols]
        return out

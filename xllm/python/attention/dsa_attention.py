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

"""DeepSeek-V4 DSA attention backend.

Consumes the :class:`DsaMetadata` built by :mod:`dsa_metadata` and drives the
two-stage sparse attention (``sparse_attn_sharedkv``), the KV compressor, and
the quantized lightning indexer. This is the Python-path counterpart of the
C++ ``DSAttentionImpl`` (core/layers/npu_torch/deepseek_sparse_attention.cpp).

The backend owns no KV storage: caches are bound from the C++ executor's
``LayerCache`` 11-tuple. Per step it (1) builds DSA metadata from the framework
``multi_block_tables``, (2) resolves the per-layer 8-cache mapping, (3) writes
new KV into the SWA cache, (4) runs the compressor into the compressed cache
when ``compress_ratio > 1``, (5) runs the indexer to pick top-k compressed
blocks when ``compress_ratio == 4``, and (6) calls ``sparse_attn_sharedkv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import math
import os

import torch

from xllm.python.attention.backend import (
    AttentionBackend,
    DsaIndexContext,
    LayerCache,
)
from xllm.python.attention.dsa_metadata import (
    DSA_CACHE_SLIDING_WINDOW,
    DSA_CACHE_TOKEN,
    DsaMetadata,
    DsaMetadataBuilder,
    build_cache_specs,
)
from xllm.python.model_executor.forward_context import (
    get_execution_buffer,
    get_forward_context,
)
from xllm.python.platform import current_platform
from xllm.python import dsa_dump
from scripts.logger import logger

# DSA_DRAFT_EXPLICIT_INDICES routes the DSpark draft attention through
# explicit per-row logical-position indices (ori_sparse_indices) instead of
# the prefill encoding over the raw ring table, mirroring the reference
# DSpark implementations: the window->slot resolution happens in index
# construction (torch ops) and the kernel never interprets the ring table.
# Validated end to end (bit-exact greedy outputs, unchanged acceptance,
# perfect verbatim recall) including ring tables up to 4K-token sequences;
# it also makes the draft forward decode-encoded and graph-capturable,
# which removes most of the draft's unoverlapped host dispatch from the
# decode step. Set to 0 to fall back to the eager prefill-encoding draft.
_DRAFT_EXPLICIT_INDICES = os.environ.get(
    "DSA_DRAFT_EXPLICIT_INDICES", "1"
) == "1"

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention
    from xllm.python.attention.backend import AttentionMetadata

# Sparse mask modes used by the C++ DSA attention (rightDownCausal variants).
_MASK_MODE_RIGHT_DOWN_CAUSAL = 3
_MASK_MODE_COMPRESS = 4


@dataclass
class _DsaCacheMapping:
    """Per-layer resolved cache indices (mirrors C++ ``DsaCacheMapping``)."""

    cmp_cache_idx: int = -1
    index_cache_idx: int = -1
    indexer_scale_cache_idx: int = -1
    ori_cache_idx: int = -1
    kv_state_cache_idx: int = -1
    score_state_cache_idx: int = -1
    index_kv_state_cache_idx: int = -1
    index_score_state_cache_idx: int = -1


@dataclass(frozen=True)
class _DsaForwardMeta:
    """Subset of C++ ModelInputParams::meta used by DSV4 metadata builders."""

    q_max_seq_len: int
    kv_max_seq_len: int


class DsaAttentionBackend(AttentionBackend):
    """DSA attention backend for DeepSeek-V4 on NPU.

    The model supplies its config so the backend can rebuild the static
    ``caches_info`` / ``group_infos`` once (mirroring
    ``deepseek_v4_build_cache_specs``) and precompute the per-step RoPE / Hadamard
    tables the indexer needs.
    """

    def __init__(
        self,
        compress_ratios: list[int],
        window_size: int,
        n_layers: int,
        num_heads: int,
        attn_head_dim: int,
        index_topk: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        dspark_block_size: int = 0,
    ) -> None:
        self.caches_info, self.group_infos = build_cache_specs(
            compress_ratios, window_size, n_layers
        )
        self._builder = DsaMetadataBuilder(self.caches_info, self.group_infos)
        self.window_size = window_size
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.num_heads = num_heads
        self.head_dim = attn_head_dim
        self.device = device
        self.dtype = dtype
        self.scale = attn_head_dim ** -0.5
        # DSpark draft block attention: every row of the N-wide query block
        # attends the trailing window_size prefix tokens plus the whole
        # block (non-causal), i.e. [q - (window + block - 1), q] — matching
        # the reference DSpark SAS window (window_size + block_size - 1).
        # Non-draft backends keep the plain [q - (window - 1), q] SWA.
        self._dspark_block_size = dspark_block_size
        self.attn_win_left = (
            window_size + dspark_block_size - 1
            if dspark_block_size > 0
            else max(window_size - 1, 0)
        )
        # Attended ori-side length cap for the metadata tiling (decode rows):
        # the draft block reads window + block tokens through the ring table
        # while a plain decode row reads window tokens through the compact.
        self.attn_win_capacity = (
            window_size + dspark_block_size
            if dspark_block_size > 0
            else window_size
        )

        self._kv_caches: list[LayerCache] = []
        self._metadata: AttentionMetadata | None = None
        self._graph_mode = False
        self._graph_dsa: DsaMetadata | None = None
        self._graph_bt_capacity_cols = 0
        self._rope_stash: dict[str, torch.Tensor | None] = {}

    @property
    def graph_dummy_kv_len(self) -> int:
        """Padding-lane KV length for graph buckets (C++ fill_empty_dp_rank:
        the AICPU DSA metadata kernels reject cmp_topk/window > kv_len)."""
        return max(self.index_topk, self.window_size, 1)

    # -- AttentionBackend interface -----------------------------------------

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        self._kv_caches = kv_caches
        # The C++ side allocates the DSV4 caches with torch::empty, leaving
        # NaN/garbage in never-written blocks. The CANN sparse_flash_mla reads
        # beyond the committed region on the decode tiling path, so the
        # garbage propagates into the attention output as NaN. Zero every
        # cache tensor once at bind time.
        for cache in self._kv_caches:
            for name in (
                "key", "value", "index", "conv", "ssm", "swa",
                "compress_kv_state", "compress_score_state",
                "compress_index_kv_state", "compress_index_score_state",
                "indexer_scale",
            ):
                tensor = getattr(cache, name, None)
                if tensor is not None and tensor.numel() > 0:
                    tensor.zero_()

    def _current_forward_metadata(self) -> AttentionMetadata:
        try:
            return get_forward_context().metadata
        except RuntimeError:
            if self._metadata is None:
                raise
            return self._metadata

    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
    ) -> None:
        self._graph_mode = graph_mode
        self._metadata = metadata
        if graph_mode:
            self._refresh_graph_metadata(metadata)

    def _refresh_graph_metadata(self, metadata: AttentionMetadata) -> None:
        """Rebuild the DSA metadata into persistent buffers, once per replay.

        Runs OUTSIDE the graph (the runner calls ``prepare`` every step before
        replay). Mirrors C++ ``prepare_graph_forward_metadata``: the host-side
        builder re-runs per step, every tensor field lands in an
        execution-state persistent buffer (fixed address, ``copy_`` refresh),
        and the precomputed AICPU metadata kernels are rebuilt off-graph into
        persistent outputs. The captured forward only binds
        ``self._graph_dsa`` (see ``prepare_dsa_metadata_for_forward``).
        """
        multi_block_tables = list(metadata.multi_block_tables)
        kv_host = metadata.kv_seq_lens_host
        kv_values = getattr(metadata, "kv_seq_lens_host_values", None)
        if kv_host is not None and kv_host.numel() > 0:
            kv_seq_lens = kv_host.cpu().tolist()
        elif kv_values is not None:
            kv_seq_lens = list(kv_values)
        else:
            kv_seq_lens = []
        q_host = getattr(metadata, "q_seq_lens_host", None)
        q_seq_lens = (
            q_host.cpu().tolist()
            if q_host is not None and q_host.numel() > 0
            else None
        )
        positions = getattr(metadata, "dsa_positions", None)
        if positions is None:
            positions = torch.empty(0, dtype=torch.int64)
        # Bucket-stable capacities: block tables pad to the widest table
        # (the builder derives slot capacity from the padded positions).
        if self._graph_bt_capacity_cols == 0 and multi_block_tables:
            self._graph_bt_capacity_cols = max(
                (t.size(1) for t in multi_block_tables), default=0
            )
        dsa_metadata = self._builder.build(
            multi_block_tables=multi_block_tables,
            kv_seq_lens=kv_seq_lens,
            q_seq_lens=q_seq_lens,
            positions=positions,
            dsa_cos_sin=self._rope_stash.get("cos_sin"),
            is_prefill=metadata.is_prefill,
            is_chunked_prefill=metadata.is_chunked_prefill,
            enable_graph=True,
            graph_block_table_capacity_cols=self._graph_bt_capacity_cols,
            new_cache_slots=getattr(metadata, "new_cache_slots", None),
        )
        self._populate_dsa_rope(dsa_metadata, metadata)
        # Window indices first, while the builder output is still on CPU
        # (saves per-layer D2H reads of the block tables).
        self._build_window_src_indices(dsa_metadata, kv_seq_lens)
        # Remap padding-lane -1 block-table entries to block 0 on the CPU
        # tensors (before the device move). The CANN compressor/indexer
        # kernels do not guard against -1 block IDs and compute invalid
        # addresses -> device error 507011 under concurrent batches with
        # padding. The window compact's valid mask already zeroes the
        # attention contribution of padded lanes. Builder output shares one
        # tensor per manager across layers, so dedup by object identity to
        # process ~6 unique tensors instead of 258 layer-cache entries.
        _seen_bts: set[int] = set()
        for lid in range(len(dsa_metadata.block_tables)):
            for ci in range(len(dsa_metadata.block_tables[lid])):
                bt = dsa_metadata.block_tables[lid][ci]
                if bt is not None and bt.numel() > 0 and bt.device.type == "cpu":
                    bt_id = id(bt)
                    if bt_id in _seen_bts:
                        continue
                    _seen_bts.add(bt_id)
                    bt[bt < 0] = 0
        self._move_metadata_to_device(dsa_metadata, persistent=True)
        # AICPU tiling metadata is built IN the captured graph (see
        # prepare_dsa_metadata_for_forward), not off-graph here, so the
        # kernels' launch + persist copy move into the replay.
        self._graph_dsa = dsa_metadata
        metadata.dsa_metadata = dsa_metadata
        # Open a dump tick per refresh step: the captured forward's layer-level
        # snaps (kernel wrappers) then land under this tick, so capture-time
        # tensors are comparable against eager decode dumps.
        dsa_dump.start_fwd(
            "decode",
            max_layer=len(self._kv_caches) - 1 if self._kv_caches else -1,
            ntokens=max(len(kv_seq_lens), 1),
            meta={"graph_mode": True, "kv_lens": kv_seq_lens[:8]},
        )
        # Snapshot the refresh inputs (per-manager tables, first manager's
        # slots) so graph-mode inputs are diffable against eager decode dumps.
        for mid, table in enumerate(multi_block_tables[:4]):
            dsa_dump.snap(
                f"graph_bt_{mid}",
                {"bt": table},
                layer=0,
                kind="moe",
                extra={"rows": int(table.size(0)), "cols": int(table.size(1))},
            )
        if dsa_metadata.block_tables and dsa_metadata.slot_mappings:
            dsa_dump.snap(
                "graph_slots_0",
                {
                    "bt": dsa_metadata.block_tables[0][0]
                    if dsa_metadata.block_tables[0] else None,
                    "slots": dsa_metadata.slot_mappings[0][0]
                    if dsa_metadata.slot_mappings[0] else None,
                },
                layer=0,
                kind="moe",
                extra={
                    "seq_lens": kv_seq_lens[:8],
                    "positions": positions[:8].tolist()
                    if positions.numel() else [],
                },
            )
        # Persisted fields the captured graph actually reads: verify their
        # CONTENT (post copy_) against eager decode dumps.
        dsa_dump.snap(
            "graph_meta",
            {
                "seq_q": dsa_metadata.actual_seq_lengths_query,
                "seq_kv": dsa_metadata.actual_seq_lengths_kv,
                "seq_lens": dsa_metadata.seq_lens,
                "positions": dsa_metadata.input_positions,
                "start_pos": dsa_metadata.start_pos,
                "c4_pos": dsa_metadata.c4_pad_positions,
                "c1_meta": dsa_metadata.c1_metadata,
                "c4_meta": dsa_metadata.c4_metadata,
                "qli_meta": dsa_metadata.qli_metadata,
            },
            layer=0,
            kind="moe",
            extra={"c1_len": int(dsa_metadata.c1_metadata.numel()) if dsa_metadata.c1_metadata is not None else 0},
        )
        dsa_dump.end_fwd()

    def reset_forward(self, metadata: AttentionMetadata | None = None) -> None:
        """Drop request-owned DSA state before attaching the next request.

        The model calls this immediately before attaching its current RoPE
        tables, so a forward that errors early cannot leak stale tensors and
        callbacks into the next request. ``_cmp_linear_bases`` is deliberately
        kept: it holds compact-cache pool layout (slot-derived base per layer),
        not request-owned inputs, and the decode fallback path depends on it.
        """
        metadata = self._metadata if metadata is None else metadata
        if metadata is None:
            return
        metadata.dsa_metadata = None
        metadata.dsa_positions = None
        metadata.dsa_cos_sin = None
        metadata.dsa_c4_cos_sin = None
        metadata.dsa_c128_cos_sin = None
        metadata.dsa_graph_mode = False
        for name in (
            "_compressor_fn",
            "_indexer_fn",
            "_current_hidden",
            "_current_kv_hidden",
            "_current_qr",
            "_current_qr_pertoken_scale",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def prepare_dsa_metadata_for_forward(
        self,
        metadata: AttentionMetadata | None = None,
    ) -> None:
        """Build DSA metadata inside model forward, matching C++ ownership/order."""
        metadata = metadata or self._metadata
        assert metadata is not None
        # Graph capture: the runner's per-step prepare (outside the graph)
        # already rebuilt and refreshed the persistent DSA buffers in
        # ``self._graph_dsa``. Binding it here lets the captured forward read
        # address-stable tensors; the eager path below stays untouched.
        if self._graph_mode and self._graph_dsa is not None:
            metadata.dsa_metadata = self._graph_dsa
            # Build the AICPU tiling metadata (c1/c4/c128/qli) inside the
            # captured graph, reading the packed persistent views. Matches C++
            # build_precomputed_metadata in-graph; removes the per-step
            # off-graph launch + persist copy from the refresh.
            self._build_precomputed_metadata(self._graph_dsa, metadata, persistent=True)
            return
        multi_block_tables = list(metadata.multi_block_tables)
        kv_seq_lens_host = metadata.kv_seq_lens_host
        kv_seq_lens = (
            kv_seq_lens_host.cpu().tolist()
            if kv_seq_lens_host is not None and kv_seq_lens_host.numel() > 0
            else []
        )
        q_seq_lens_host = getattr(metadata, "q_seq_lens_host", None)
        q_seq_lens = (
            q_seq_lens_host.cpu().tolist()
            if q_seq_lens_host is not None and q_seq_lens_host.numel() > 0
            else None
        )
        # DSA RoPE tables and positions are model-owned; the backend reads them
        # off the metadata when the model attaches them (see attach_rope_tables).
        positions = getattr(metadata, "dsa_positions", None)
        if positions is None:
            positions = torch.empty(0, dtype=torch.int64)
        dsa_cos_sin = getattr(metadata, "dsa_cos_sin", None)
        dsa_metadata = self._builder.build(
            multi_block_tables=multi_block_tables,
            kv_seq_lens=kv_seq_lens,
            q_seq_lens=q_seq_lens,
            positions=positions,
            dsa_cos_sin=dsa_cos_sin,
            is_prefill=metadata.is_prefill,
            is_chunked_prefill=metadata.is_chunked_prefill,
            enable_graph=False,
            new_cache_slots=getattr(metadata, "new_cache_slots", None),
        )
        self._populate_dsa_rope(dsa_metadata, metadata)
        self._move_metadata_to_device(dsa_metadata)
        self._build_precomputed_metadata(dsa_metadata, metadata)
        metadata.dsa_metadata = dsa_metadata
        dsa_dump.start_fwd(
            "prefill" if metadata.is_prefill else "decode",
            max_layer=len(self._kv_caches) - 1 if self._kv_caches else -1,
            ntokens=int(metadata.max_query_len),
            meta={
                "is_prefill": metadata.is_prefill,
                "is_chunked_prefill": metadata.is_chunked_prefill,
                "max_query_len": metadata.max_query_len,
                "max_seq_len": metadata.max_seq_len,
                "device": str(self.device),
            },
        )

    def select_dsa_layer_rope(
        self,
        layer_id: int,
        cos_sin_cache: torch.Tensor,
        metadata: AttentionMetadata | None = None,
    ) -> None:
        """Select the main q/kv RoPE group for the current DSV4 layer.

        C++ updates ``DSAMetadata::layer_id/cos/sin`` in the model layer loop
        from ``input_rope_by_ratio``. Python keeps the full cache here because
        the model and indexer gather it with the current input positions, but
        the selected group and lifetime are otherwise identical.
        """
        metadata = metadata or self._metadata
        if metadata is None or metadata.dsa_metadata is None:
            raise RuntimeError("DSA metadata must be prepared before selecting layer RoPE")
        dsa = metadata.dsa_metadata
        dsa.layer_id = layer_id
        # Cache the chunk views per table identity: the cos_sin_cache is a
        # persistent model buffer, and re-chunking 43 layers x 2 .contiguous()
        # copies of (max_pos, dim/2) bf16 was pure bandwidth waste (5.4 GB/step
        # measured). The consumers (compressor ratio=1 path, QLI partial RoPE)
        # either index_select specific rows or apply their own .to()/.contiguous()
        # on the gathered result, so the raw strided views suffice.
        cache = getattr(self, "_rope_chunk_cache", None)
        if cache is None or cache[0] is not cos_sin_cache:
            chunks = cos_sin_cache.chunk(2, dim=-1)
            self._rope_chunk_cache = (cos_sin_cache, chunks[0], chunks[1])
            dsa.cos_table = chunks[0]
            dsa.sin_table = chunks[1]
        else:
            dsa.cos_table = cache[1]
            dsa.sin_table = cache[2]

    def _populate_dsa_rope(
        self,
        dsa: DsaMetadata,
        metadata: AttentionMetadata | None = None,
    ) -> None:
        """Build request-shaped RoPE tensors for the current forward."""
        metadata = metadata or self._metadata
        css = (
            getattr(metadata, "dsa_cos_sin", None)
            if metadata is not None
            else None
        )
        if css is None:
            css = self._rope_stash.get("cos_sin")
        if dsa.cos_table is None and css is not None and css.numel() > 0:
            # Strided views suffice (same contract as the metadata builder):
            # no consumer reads the full table; gather-first consumers select
            # rows before any layout requirement applies.
            dsa.cos_table, dsa.sin_table = css.chunk(2, dim=-1)
        c4css = (
            getattr(metadata, "dsa_c4_cos_sin", None)
            if metadata is not None
            else None
        )
        if c4css is None:
            c4css = self._rope_stash.get("c4")
        if c4css is not None and dsa.c4_pad_positions.numel() > 0:
            c4_idx = (
                dsa.c4_pad_positions.clamp(0, c4css.size(0) - 1).long().to(
                    c4css.device
                )
            )
            dsa.c4_cos, dsa.c4_sin = (
                tensor.contiguous()
                for tensor in c4css.index_select(0, c4_idx).chunk(2, dim=-1)
            )
        c128css = (
            getattr(metadata, "dsa_c128_cos_sin", None)
            if metadata is not None
            else None
        )
        if c128css is None:
            c128css = self._rope_stash.get("c128")
        if c128css is not None and dsa.c128_pad_positions.numel() > 0:
            c128_idx = (
                dsa.c128_pad_positions.clamp(0, c128css.size(0) - 1).long().to(
                    c128css.device
                )
            )
            dsa.c128_cos, dsa.c128_sin = (
                tensor.contiguous()
                for tensor in c128css.index_select(0, c128_idx).chunk(2, dim=-1)
            )

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Attention,
    ) -> torch.Tensor:
        """Full DSA attention path for one layer.

        ``q``/``k``/``v`` here are the model-projected, RoPE-applied tensors the
        DeepseekV4 attention layer hands in; the backend only owns cache writes
        and the kernel dispatch.
        """
        metadata = self._current_forward_metadata()
        dsa = getattr(metadata, "dsa_metadata", None)
        assert dsa is not None
        # Late-populate dsa.cos_table/sin_table if prepare ran before the model
        # attached the RoPE tables (prepare is called by the executor before
        # model.forward, so _dsa_cos_sin may have been None at prepare time).
        if dsa.cos_table is None or dsa.sin_table is None:
            self._populate_dsa_rope(dsa, metadata)
        # Late-populate per-ratio compressed RoPE tables: index the c4/c128
        # compress RoPE cache with the per-token compressed positions
        # (c4_pad_positions / c128_pad_positions, built by DsaMetadataBuilder).
        # Mirrors C++ DeepseekV4RotaryEmbedding::build(positions_map) per group.
        # Also late-populate dsa.input_positions (prepare ran before model.forward
        # set self._positions, so it was empty at prepare time).
        if dsa.input_positions.numel() == 0:
            pos = getattr(metadata, "dsa_positions", None)
            if pos is not None and pos.numel() > 0:
                dsa.input_positions = pos
        if dsa.c4_cos is None or dsa.c128_cos is None:
            self._populate_dsa_rope(dsa, metadata)
        layer_id = layer.layer_id
        compress_ratio = self._layer_compress_ratio(layer_id)
        mapping = self._resolve_cache_mapping(layer_id, compress_ratio)
        layer_cache = self._kv_caches[layer_id]
        is_prefill = metadata.is_prefill
        is_chunked_prefill = metadata.is_chunked_prefill
        use_temporary_prefill_kv = is_prefill and not is_chunked_prefill
        # 1) Prepare ori_kv for attention (mirrors C++ :790-816).
        # Full prefill: use kv directly as temporary PA_ND (don't scatter to paged).
        # Decode/chunked: scatter to paged SWA cache.
        ori_kv = layer_cache.swa
        ori_slot = _get_layer_cache_tensor(dsa.slot_mappings, layer_id, mapping.ori_cache_idx)
        ori_block_table = _get_layer_cache_tensor(
            dsa.block_tables, layer_id, mapping.ori_cache_idx
        )
        if use_temporary_prefill_kv:
            # Prefill: build temporary PA_ND cache from kv (mirrors C++
            # build_prefill_pa_nd_kv, deepseek_sparse_attention.cpp:272-368).
            ori_kv_for_attn, ori_block_table_for_attn = _build_prefill_pa_nd_kv(
                k,
                dsa.actual_seq_lengths_query,
                ori_block_table,
                self.window_size,
            )
        else:
            if ori_kv is not None and ori_slot is not None:
                _scatter_by_slot(ori_kv, ori_slot, k)
            ori_kv_for_attn = ori_kv
            ori_block_table_for_attn = ori_block_table

        # 2) Compressor: pool KV into the compressed cache when ratio > 1.
        cmp_kv = layer_cache.key
        cmp_slot = _get_layer_cache_tensor(dsa.slot_mappings, layer_id, mapping.cmp_cache_idx)
        cmp_block_table = _get_layer_cache_tensor(
            dsa.block_tables, layer_id, mapping.cmp_cache_idx
        )
        if (
            compress_ratio > 1
            and cmp_kv is not None
            and cmp_slot is not None
        ):
            compressor_fn = getattr(self, "_compressor_fn", None)
            if compressor_fn is None:
                raise RuntimeError(
                    f"DSA compressor is required for compression ratio {compress_ratio}"
                )
            compressed = compressor_fn(
                layer_id,
                layer_cache,
                dsa,
                mapping,
                cmp_block_table,
                compress_ratio,
            )
            _scatter_by_slot(cmp_kv, cmp_slot, compressed)
            dsa_dump.snap(
                "scatter_cmp_kv",
                {
                    "compressed": compressed,
                    "cmp_kv": cmp_kv,
                    "cmp_slot": cmp_slot,
                    "cmp_block_table": cmp_block_table,
                },
                layer=layer_id,
                kind="moe" if layer_id else "dense",
            )

        # 3) Indexer: select top-k compressed blocks when ratio == 4.
        compress_topk_idxs: torch.Tensor | None = None
        if compress_ratio == 4 and cmp_kv is not None:
            indexer_fn = getattr(self, "_indexer_fn", None)
            if indexer_fn is None:
                raise RuntimeError("DSA indexer is required for compression ratio 4")
            compress_topk_idxs = indexer_fn(
                layer_id,
                layer_cache,
                dsa,
                mapping,
                q,
            )
            # -1 entries are the kernels' invalid-slot marker (QLI v2 pads
            # short candidate lists with -1); sparse_attn_sharedkv /
            # sparse_flash_mla both treat negative indices as fill slots.
            # Do NOT clamp them to 0 -- that would fabricate references to
            # compressed block 0.
            if compress_topk_idxs is None:
                raise RuntimeError("DSA indexer returned no top-k indices")

        # 4) Two-stage sparse attention over original + compressed KV.
        # The metadata tensors live on CPU (DsaMetadataBuilder); move to device
        # for the NPU kernel, matching the C++ H2D transfer of packed metadata.
        if compress_ratio == 1:
            sparse_meta = dsa.c1_metadata
        elif compress_ratio == 4:
            sparse_meta = dsa.c4_metadata
        elif compress_ratio == 128:
            sparse_meta = dsa.c128_metadata
        else:
            sparse_meta = None
        if sparse_meta is None:
            raise RuntimeError(
                "DSA sparse metadata is missing for compression ratio "
                f"{compress_ratio}"
            )
        seq_q = dsa.actual_seq_lengths_query
        seq_kv = dsa.actual_seq_lengths_kv
        sparse_meta_for_kernel = sparse_meta
        ori_block_table_for_kernel = ori_block_table_for_attn
        cmp_block_table_for_kernel = cmp_block_table
        # Match C++ DSAttention's optional contract exactly: prefill and
        # chunked prefill pass query cu-seqlens, while decode leaves
        # cu_seqlens_ori_kv as std::nullopt. A defined empty tensor selects a
        # different ACL optional-input path and causes small decode drift.
        use_prefill_attn = is_prefill or is_chunked_prefill
        cu_seqlens_ori_kv_for_attn = (
            seq_q
            if use_prefill_attn
            else None
        )
        if not use_prefill_attn and not self._graph_mode:
            # Unified eager decode: B-broadcast window gather over the REAL
            # block table (physical addressing, no ring-wrap assumptions)
            # plus cmp passthrough. The legacy host-driven compact bypass
            # assumed batch==1 (single-sequence window, first-sequence
            # newest slot) and corrupts/crashes concurrent decode batches;
            # the CANN kernel also mis-decodes the raw ring table, so the
            # eager path now mirrors the verified graph path.
            (
                ori_kv_for_attn,
                ori_block_table_for_attn,
                seqused_ori_kernel,
            ) = self._decode_window_compact_eager(
                ori_kv_for_attn, ori_block_table_for_attn, seq_kv, layer_id
            )
            if compress_ratio > 1:
                seqused_cmp_kernel = seq_kv // compress_ratio
                cmp_residual_kernel = seq_kv % compress_ratio
            else:
                seqused_cmp_kernel = None
                cmp_residual_kernel = None
            seq_kv_for_kernel = None
        elif self._graph_mode and not use_prefill_attn:
            # Static window compact for the ori (SWA-ring) side. The CANN
            # sparse_flash_mla kernel mis-decodes the real ring block table at
            # decode shapes (verified by bisection: feeding the ring table to
            # the ori side garbles output while the cmp side is fine), so the
            # graph gathers the newest
            # window into a 0-based in-graph buffer with fully device-side
            # ring indexing: no host reads, no data-dependent allocations,
            # bucket-static shapes. Head-aligned layout keeps the eager
            # semantics (compact[j] = token used-k+j, seqused=k). The cmp
            # side keeps the true paged layout and lengths.
            (
                ori_kv_for_attn,
                ori_block_table_for_attn,
                seqused_ori_kernel,
            ) = self._graph_window_compact(
                ori_kv_for_attn, seq_kv, layer_id
            )
            if compress_ratio > 1:
                seqused_cmp_kernel = seq_kv // compress_ratio
                cmp_residual_kernel = seq_kv % compress_ratio
            else:
                seqused_cmp_kernel = None
                cmp_residual_kernel = None
            seq_kv_for_kernel = None
        else:
            seqused_ori_kernel = None
            seqused_cmp_kernel = None
            cmp_residual_kernel = None
            seq_kv_for_kernel = seq_kv
        # DSpark draft (eager, chunked-prefill encoded, SWA-only): route the
        # ori side through explicit per-row slot indices instead. The grid
        # resolves the ring window outside the kernel, so this is the
        # graph-compatible decode encoding the reference implementations use.
        draft_grid = None
        draft_topk_length = None
        if (
            _DRAFT_EXPLICIT_INDICES
            and self._dspark_block_size > 0
            and use_prefill_attn
            and not is_prefill
            and compress_ratio == 1
        ):
            # Both eager and graph mode: the grid build is pure device ops
            # on persistent tensors, so the captured forward replays it
            # against the refreshed table and KV lengths.
            draft_grid, draft_topk_length = self._build_draft_swa_indices(
                ori_block_table, seq_kv
            )
        out, _lse = _sparse_attn_sharedkv(
            _dump_layer=layer,
            q=q,
            ori_kv=ori_kv_for_attn,
            cmp_kv=cmp_kv if compress_ratio > 1 else None,
            ori_sparse_indices=draft_grid,
            ori_topk_length=draft_topk_length,
            cmp_sparse_indices=compress_topk_idxs,
            ori_block_table=ori_block_table_for_attn,
            cmp_block_table=cmp_block_table_for_kernel if compress_ratio > 1 else None,
            cu_seqlens_q=seq_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv_for_attn,
            # C++ passes nullopt for compressed KV cu-seqlens; cmp_kv is PA_ND
            # and addressed through cmp_block_table/topk.
            cu_seqlens_cmp_kv=None,
            seqused_q=None,
            seqused_kv=seq_kv_for_kernel,
            seqused_ori_kv=seqused_ori_kernel,
            seqused_cmp_kv=seqused_cmp_kernel,
            cmp_residual_kv=cmp_residual_kernel,
            # sinks: the attention sink parameter (attn_sink) is required by the
            # sparse_attn_sharedkv kernel (C++ :949 passes attn_sink_ when loaded).
            sinks=layer.attn_sink if hasattr(layer, "attn_sink") else None,
            metadata=sparse_meta_for_kernel,
            softmax_scale=self.scale,
            cmp_ratio=compress_ratio,
            # SWA sparse-ori contract (explicit indices): mask mode 0 with
            # negative window bounds -- the visible set is defined entirely
            # by the index grid, no additional window masking.
            ori_mask_mode=(
                0 if draft_grid is not None else _MASK_MODE_COMPRESS
            ),
            cmp_mask_mode=_MASK_MODE_RIGHT_DOWN_CAUSAL,
            ori_win_left=-1 if draft_grid is not None else self.attn_win_left,
            ori_win_right=-1 if draft_grid is not None else 0,
            layout_q="TND",
            layout_kv="PA_ND",
            return_softmax_lse=False,
        )
        # Full prefill reads a temporary PA_ND cache so attention does not
        # depend on the persistent SWA cache. Match C++ step 8 by writing the
        # projected KV into the persistent cache only after that attention
        # finishes; decode reads this cache on the next forward.
        if use_temporary_prefill_kv and ori_kv is not None and ori_slot is not None:
            _scatter_by_slot(ori_kv, ori_slot, k)
        return out

    def mla_index_context(self, layer: Attention) -> DsaIndexContext:
        """Hand the DSA indexer its paged index cache + block tables + slots."""
        metadata = self._current_forward_metadata()
        dsa = getattr(metadata, "dsa_metadata", None)
        assert dsa is not None
        layer_id = layer.layer_id
        compress_ratio = self._layer_compress_ratio(layer_id)
        mapping = self._resolve_cache_mapping(layer_id, compress_ratio)
        layer_cache = self._kv_caches[layer_id]
        index_slot = _get_layer_cache_tensor(
            dsa.slot_mappings, layer_id, mapping.index_cache_idx
        )
        index_block_table = _get_layer_cache_tensor(
            dsa.block_tables, layer_id, mapping.index_cache_idx
        )
        cmp_block_table = _get_layer_cache_tensor(
            dsa.block_tables, layer_id, mapping.cmp_cache_idx
        )
        return DsaIndexContext(
            index_cache=layer_cache.index if layer_cache.index is not None else torch.empty(0),
            indexer_scale=layer_cache.indexer_scale,
            slot_mapping=index_slot if index_slot is not None else torch.empty(0),
            block_table=index_block_table,
            cmp_block_table=cmp_block_table,
            kv_state=layer_cache.compress_kv_state,
            score_state=layer_cache.compress_score_state,
            kv_block_table=_get_layer_cache_tensor(
                dsa.block_tables, layer_id, mapping.kv_state_cache_idx
            ),
            score_block_table=_get_layer_cache_tensor(
                dsa.block_tables, layer_id, mapping.score_state_cache_idx
            ),
            actual_seq_q=dsa.actual_seq_lengths_query,
            actual_seq_kv=dsa.actual_seq_lengths_kv,
            start_pos=dsa.start_pos,
            qli_metadata=dsa.qli_metadata,
        )

    @property
    def num_kv_blocks(self) -> int:
        if self._kv_caches and self._kv_caches[0].swa is not None:
            return self._kv_caches[0].swa.size(0)
        return 0

    @property
    def page_size(self) -> int:
        if self._kv_caches and self._kv_caches[0].swa is not None and self._kv_caches[0].swa.dim() > 1:
            return self._kv_caches[0].swa.size(1)
        return self.window_size

    # -- model-attached state ----------------------------------------------
    # The DeepseekV4 model owns the RoPE tables, Hadamard matrix, and the
    # compressor/indexer callables. It attaches them to the backend before the
    # first forward so the backend can stage them into DsaMetadata.

    def attach_rope_tables(
        self,
        positions: torch.Tensor,
        dsa_cos_sin: torch.Tensor | None,
        graph_bt_cols: int = 0,
        c4_cos_sin: torch.Tensor | None = None,
        c128_cos_sin: torch.Tensor | None = None,
        metadata: AttentionMetadata | None = None,
    ) -> None:
        # In graph mode the model attaches its (stable, model-owned) tables at
        # every captured forward, but the metadata fields are reset per
        # forward and the positions tensor is an in-graph cast. Stash the
        # tables on the backend instead; the per-replay refresh reads the
        # stash and the runner-owned static positions buffer.
        self._rope_stash = {
            "cos_sin": dsa_cos_sin,
            "c4": c4_cos_sin,
            "c128": c128_cos_sin,
        }
        if self._graph_mode:
            return
        metadata = metadata or self._metadata
        if metadata is not None:
            metadata.dsa_positions = positions
            metadata.dsa_cos_sin = dsa_cos_sin
            metadata.dsa_c4_cos_sin = c4_cos_sin
            metadata.dsa_c128_cos_sin = c128_cos_sin

    def attach_compressor(self, fn) -> None:
        """``fn(layer_id, layer_cache, dsa, mapping, cmp_block_table) -> Tensor``."""
        self._compressor_fn = fn

    def attach_indexer(self, fn) -> None:
        """``fn(layer_id, layer_cache, dsa, mapping, q) -> Tensor`` (topk idxs)."""
        self._indexer_fn = fn

    # -- internals ----------------------------------------------------------

    def _layer_compress_ratio(self, layer_id: int) -> int:
        if layer_id < len(self.caches_info):
            caches = self.caches_info[layer_id]
            for ci in caches:
                if ci.cache_type == DSA_CACHE_TOKEN:
                    return ci.ratio
        return 1

    def _resolve_cache_mapping(
        self, layer_id: int, compress_ratio: int
    ) -> _DsaCacheMapping:
        """Python port of ``resolve_cache_mapping`` (deepseek_sparse_attention.cpp:92)."""
        mapping = _DsaCacheMapping()
        if layer_id < 0 or layer_id >= len(self.caches_info):
            return mapping
        token_ratio_indices: list[int] = []
        swa_indices: list[int] = []
        for cache_idx, ci in enumerate(self.caches_info[layer_id]):
            if ci.cache_type == DSA_CACHE_TOKEN and ci.ratio == compress_ratio:
                token_ratio_indices.append(cache_idx)
            if ci.cache_type == DSA_CACHE_SLIDING_WINDOW:
                swa_indices.append(cache_idx)
        if token_ratio_indices and compress_ratio > 1:
            mapping.cmp_cache_idx = token_ratio_indices[0]
        if len(token_ratio_indices) > 1:
            mapping.index_cache_idx = token_ratio_indices[1]
        if len(token_ratio_indices) > 2:
            mapping.indexer_scale_cache_idx = token_ratio_indices[2]
        if swa_indices:
            mapping.ori_cache_idx = swa_indices[0]
        if len(swa_indices) > 1:
            mapping.kv_state_cache_idx = swa_indices[1]
        if len(swa_indices) > 2:
            mapping.score_state_cache_idx = swa_indices[2]
        if len(swa_indices) > 3:
            mapping.index_kv_state_cache_idx = swa_indices[3]
        if len(swa_indices) > 4:
            mapping.index_score_state_cache_idx = swa_indices[4]
        return mapping

    def _build_precomputed_metadata(
        self,
        dsa: DsaMetadata,
        metadata: AttentionMetadata,
        persistent: bool = False,
    ) -> None:
        """Build the AICPU tiling metadata for each compress ratio present.

        Mirrors the C++ ``build_precomputed_metadata`` step: one
        ``sparse_attn_sharedkv_metadata`` per ratio, plus one
        ``quant_lightning_indexer_metadata`` for the qli path.
        """
        from xllm.python import kernels

        seq_q = dsa.actual_seq_lengths_query
        seq_kv = dsa.actual_seq_lengths_kv
        batch_size = int(max(dsa.actual_seq_lengths_kv.numel(), 1))
        forward_meta = _build_dsa_forward_meta(dsa, metadata)
        max_q = forward_meta.q_max_seq_len
        max_kv = forward_meta.kv_max_seq_len
        is_prefill = max_q > 1
        cu_seqlens_ori_kv = seq_q if is_prefill else None
        cu_seqlens_cmp_kv = None
        seqused_q = None
        seqused_kv = seq_kv
        # Metadata kernels enqueue asynchronously. Retain their tensor inputs
        # on the current forward's DsaMetadata, as C++ DSAMetadata does.
        dsa.precomputed_metadata_inputs = tuple(
            (seq_q, seq_kv, cu_seqlens_ori_kv, cu_seqlens_cmp_kv,
             seqused_q, seqused_kv)
        )
        retained_meta_inputs = []
        for ratio in (1, 4, 128):
            has_cmp = ratio > 1
            cmp_topk = self.index_topk if ratio == 4 else 0
            if has_cmp:
                seqused_cmp_kv = seq_kv // ratio
                cmp_residual_kv = seq_kv % ratio
                retained_meta_inputs += [seqused_cmp_kv, cmp_residual_kv]
                max_seqlen_cmp_kv = int(max_kv) // ratio
            else:
                seqused_cmp_kv = None
                cmp_residual_kv = None
                max_seqlen_cmp_kv = 0
            sparse_metadata = kernels.sparse_flash_mla_metadata(
                num_heads_q=self.num_heads,
                num_heads_kv=1,
                head_dim=self.head_dim,
                cu_seqlens_q=seq_q,
                cu_seqlens_ori_kv=None,
                cu_seqlens_cmp_kv=None,
                seqused_q=None,
                seqused_ori_kv=(
                    # Decode runs the ori side over the clamped window
                    # compact (eager and graph alike); the tiling metadata
                    # must describe the same length the kernel receives, or
                    # the AICPU tiling reads past the window capacity. The
                    # DSpark draft block reads window + block tokens through
                    # the ring table instead of the window compact.
                    seq_kv.clamp(max=self.attn_win_capacity)
                    if not is_prefill
                    else seq_kv
                ),
                seqused_cmp_kv=seqused_cmp_kv,
                cmp_residual_kv=cmp_residual_kv,
                ori_topk_length=(
                    # Sparse-ori contract: the metadata op validates the
                    # per-row valid-entry count alongside ori_topk=K and
                    # mask mode 0 (visible = min(window + block, seq_len)).
                    seq_kv.clamp(max=self.attn_win_capacity)
                    .to(torch.int32)
                    .unsqueeze(1)
                    if (
                        ratio == 1
                        and _DRAFT_EXPLICIT_INDICES
                        and self._dspark_block_size > 0
                        and not is_prefill
                    )
                    else None
                ),
                cmp_topk_length=None,
                batch_size=batch_size,
                max_seqlen_q=max_q,
                max_seqlen_ori_kv=max_kv,
                max_seqlen_cmp_kv=max_seqlen_cmp_kv,
                ori_topk=(
                    # SWA sparse-ori (DSpark draft explicit indices): the
                    # metadata must carry the index grid width K so the
                    # tiling matches the indices the kernel receives.
                    self._draft_index_width()
                    if (
                        ratio == 1
                        and _DRAFT_EXPLICIT_INDICES
                        and self._dspark_block_size > 0
                        and not is_prefill
                    )
                    else 0
                ),
                cmp_topk=cmp_topk,
                cmp_ratio=ratio,
                ori_mask_mode=(
                    0
                    if (
                        ratio == 1
                        and _DRAFT_EXPLICIT_INDICES
                        and self._dspark_block_size > 0
                        and not is_prefill
                    )
                    else _MASK_MODE_COMPRESS
                ),
                cmp_mask_mode=(0 if not has_cmp else _MASK_MODE_RIGHT_DOWN_CAUSAL),
                ori_win_left=(
                    -1
                    if (
                        ratio == 1
                        and _DRAFT_EXPLICIT_INDICES
                        and self._dspark_block_size > 0
                        and not is_prefill
                    )
                    else self.attn_win_left
                ),
                ori_win_right=(
                    -1
                    if (
                        ratio == 1
                        and _DRAFT_EXPLICIT_INDICES
                        and self._dspark_block_size > 0
                        and not is_prefill
                    )
                    else 0
                ),
                layout_q="TND",
                layout_kv="PA_BBND",
                has_ori_kv=True,
                has_cmp_kv=has_cmp,
            )
            if persistent:
                sparse_metadata = self._persist_tensor(
                    ("dsa_sfm_meta", ratio), sparse_metadata
                )
            if ratio == 1:
                dsa.c1_metadata = sparse_metadata
            elif ratio == 4:
                dsa.c4_metadata = sparse_metadata
            elif ratio == 128:
                dsa.c128_metadata = sparse_metadata
        dsa.precomputed_metadata_inputs = dsa.precomputed_metadata_inputs + tuple(
            retained_meta_inputs
        )
        query_lens = (
            seq_q[1:].clone() if seq_q.numel() > 1 else dsa.seq_lens_q
        )
        key_lens = dsa.seq_lens if dsa.seq_lens.numel() else seq_kv
        cmp_ratio = 4
        use_v2 = current_platform.is_ascend950()
        if use_v2:
            # v2 metadata contract (must mirror the main v2 kernel call):
            # actS2SizeOrig = seqused_k * cmp_ratio + cmp_residual_k, so
            # seqused_k is the committed compressed-key count (raw //
            # cmp_ratio) and cmp_residual_k is raw % cmp_ratio.
            # cu_seqlens_q is the (B+1,) prefix sum with index 0 fixed to 0.
            # layout_k must be PA_BBND: the metadata op rejects PA_BSND
            # (EZ0027), and the main kernel's paged-key layout is PA_BBND.
            key_lens_comp = (key_lens // cmp_ratio).to(torch.int32)
            cmp_residual_k = (key_lens % cmp_ratio).to(torch.int32)
            query_len_cumsum = (
                seq_q if seq_q.numel() > 1 else dsa.kv_cu_seq_lens
            )
            dsa.precomputed_metadata_inputs += (
                query_len_cumsum,
                key_lens,
                key_lens_comp,
                cmp_residual_k,
            )
            dsa.qli_metadata = kernels.quant_lightning_indexer_v2_metadata(
                cu_seqlens_q=query_len_cumsum,
                cu_seqlens_k=None,
                seqused_q=None,
                seqused_k=key_lens_comp,
                cmp_residual_k=cmp_residual_k,
                num_heads_q=max(self.index_n_heads, 1),
                num_heads_k=1,
                head_dim=max(self.index_head_dim, 1),
                topk=self.index_topk,
                quant_mode=2,
                batch_size=int(max(key_lens_comp.size(0), 1)),
                max_seqlen_q=max(max_q, 1),
                max_seqlen_k=max(max_kv, 1),
                layout_q="TND",
                layout_k="PA_BBND",
                mask_mode=_MASK_MODE_RIGHT_DOWN_CAUSAL,
                cmp_ratio=cmp_ratio,
                device=str(self.device),
            )
            if persistent:
                dsa.qli_metadata = self._persist_tensor(
                    ("dsa_qli_meta",), dsa.qli_metadata
                )
        else:
            # quant_lightning_indexer 走 CANN 默认版（xllm_ops 已禁止编译该算子），
            # 不再构建 AICPU metadata；deepseek_v4 侧会对空 metadata 兜底。
            dsa.qli_metadata = None
        dsa_dump.snap(
            "quant_lightning_indexer_metadata",
            {
                "cu_seqlens_q": seq_q,
                "seqused_q": query_lens,
                "seqused_k_raw": key_lens,
                "seqused_k_compressed": (key_lens // cmp_ratio).to(torch.int32),
                "cmp_residual_k": (key_lens % cmp_ratio).to(torch.int32),
                "qli_metadata": dsa.qli_metadata,
            },
            extra={"soc": current_platform.get_npu_chip(), "v2": use_v2},
        )

    def _draft_index_width(self) -> int:
        """Index grid last-dim K for the draft explicit-indices path."""
        min_width = self.window_size + max(self._dspark_block_size, 1)
        return ((min_width + 127) // 128) * 128

    def _build_draft_swa_indices(
        self,
        ori_block_table: torch.Tensor | None,
        seq_kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Explicit logical-position indices for the DSpark draft (rows, 1, W).

        Port of the reference DSpark index construction onto the
        sparse_flash_mla SWA-sparse-ori contract: every expanded row of a
        draft block sees the trailing ``window_size`` prefix tokens plus the
        WHOLE draft block (non-causal), i.e. logical positions
        ``[end - (window + block), end)`` with ``end`` the group's shared KV
        length. Entries are ABSOLUTE logical positions; the kernel resolves
        them through the block table. Columns beyond the visible length are
        -1 with the per-row count in the returned topk_length.

        Reads the REBUILT per-manager SWA table (the builder maps logical
        block column j to the ring physical column j % width, so the
        kernel's direct ``bt[pos // 128]`` addressing is correct for the
        trailing draft window). Everything is a device op on persistent
        tensors -- no host reads, no data-dependent shapes -- so the path is
        ACL-graph capturable. Rows whose window fell entirely on invalid
        table entries (graph bucket padding keeps the dummy KV length with
        an all-invalid table) get a zero topk length so the kernel reads
        nothing through them; their outputs are sliced off on replay.
        """
        if ori_block_table is None or ori_block_table.numel() == 0:
            return None, None
        device = seq_kv.device
        rows = int(seq_kv.numel())
        if rows < 1 or ori_block_table.size(0) < rows:
            return None, None
        width = self._draft_index_width()
        min_width = self.window_size + max(self._dspark_block_size, 1)

        row_bt = ori_block_table[:rows].to(device=device, dtype=torch.int64)
        ends = seq_kv.to(device=device, dtype=torch.int64)
        starts = (ends - min_width).clamp(min=0)
        visible = (ends - starts).to(torch.int32)
        cols = torch.arange(width, device=device, dtype=torch.int64)
        pos = starts.unsqueeze(1) + cols.unsqueeze(0)
        col_ok = cols.unsqueeze(0) < (ends - starts).unsqueeze(1)
        # Resolve each column's block entry to detect padding rows (their
        # tables are all-invalid). The clamp only guards the gather against
        # out-of-bounds columns on rows whose positions already fail
        # col_ok; live rows always index inside the rebuilt width.
        table_cols = int(row_bt.size(1))
        blk_col = (pos // 128).clamp(max=table_cols - 1)
        blk = torch.gather(row_bt, 1, blk_col)
        row_ok = ((blk >= 0) | ~col_ok).all(dim=1)
        indices = torch.where(
            col_ok & row_ok.unsqueeze(1), pos, torch.full_like(pos, -1)
        ).to(torch.int32).unsqueeze(1)
        topk_length = torch.where(
            row_ok, visible, torch.zeros_like(visible)
        ).reshape(-1, 1)
        return indices, topk_length

    def _decode_window_compact_eager(
        self,
        ori_kv: torch.Tensor,
        ori_bt: torch.Tensor,
        seq_kv: torch.Tensor,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eager decode window gather (batch-safe, mirrors the graph path).

        Same semantics as the graph's ``_graph_window_compact``: the newest
        ``window`` tokens per sequence are gathered through the REAL block
        table (physical addressing) into a zero-based per-sequence block,
        with the clamped window length as the kernel-side seqused. Works for
        any batch size, unlike the legacy single-sequence compact bypass.
        """
        bs = 128
        win = self.window_size
        n_seqs = seq_kv.size(0)
        head_dim = ori_kv.size(-1)
        device = ori_kv.device

        kv_cpu = seq_kv.detach().to("cpu", torch.int64)
        k = kv_cpu.clamp(max=win)
        token = kv_cpu.unsqueeze(1) - k.unsqueeze(1) + torch.arange(win).unsqueeze(0)
        valid = torch.arange(win).unsqueeze(0) < k.unsqueeze(1)

        # Prefer the cached host copy stashed by _move_metadata_to_device
        # (keyed by the device tensor's data_ptr — the builder dedups per
        # manager so all layers of a group share one device tensor). This
        # avoids 43 D2H syncs per decode step.
        host_cache = getattr(self, "_eager_bt_host", {})
        bt_cpu = host_cache.get(ori_bt.data_ptr())
        if bt_cpu is not None:
            bt_host = bt_cpu.to(torch.int64)
        else:
            bt_host = (
                ori_bt.detach().cpu() if ori_bt.device.type != "cpu"
                else ori_bt.detach()
            ).to(torch.int64)
        # Placeholder-stable ring addressing. The SWA manager releases
        # slid-out leading blocks but keeps them in the table as invalid
        # (-1) placeholders exactly so (pos // block_size) % width keeps
        # resolving each position to its owning column. Mapping through
        # the count of valid columns instead shifts the mapping by the
        # number of released blocks once the first one slides out
        # (cached_tokens >= window + num_spec_tokens + block_size), and
        # the window reads the wrong physical blocks.
        rows = bt_host.size(0)
        width = bt_host.size(1)
        seq_sel = torch.arange(n_seqs).clamp(max=rows - 1)
        row_bt = bt_host[seq_sel]
        blk_pos = (token // bs) % width
        blk_id = torch.gather(row_bt, 1, blk_pos)
        blk_ok = blk_id >= 0
        src = blk_id.clamp(min=0) * bs + (token % bs)
        slot_valid = valid & blk_ok
        src = torch.where(slot_valid, src, torch.zeros_like(src))

        src_idx = src.to(device)
        valid_d = slot_valid.to(device)
        k_d = k.to(device)
        kv_flat = ori_kv.reshape(-1, head_dim)
        gathered = kv_flat[src_idx]
        compact = torch.where(
            valid_d.unsqueeze(-1),
            gathered,
            torch.zeros((), dtype=ori_kv.dtype, device=device),
        )
        compact_kv = compact.reshape(n_seqs, bs, 1, head_dim)
        bt = torch.arange(n_seqs, device=device, dtype=torch.int32).unsqueeze(1)
        seqused = k.to(torch.int32).to(device)
        return compact_kv, bt, seqused

    def _window_src_for_group(
        self,
        dsa_metadata: DsaMetadata,
        layer_id: int,
        ori_idx: int,
        token: torch.Tensor,
        valid: torch.Tensor,
        n_seqs: int,
    ) -> torch.Tensor:
        """Window source slots for one manager group (vectorized, CPU)."""
        bs = 128
        win = self.window_size
        if (
            ori_idx < 0
            or layer_id >= len(dsa_metadata.block_tables)
            or ori_idx >= len(dsa_metadata.block_tables[layer_id])
        ):
            return torch.zeros((n_seqs, win), dtype=torch.int64)
        bt = dsa_metadata.block_tables[layer_id][ori_idx]
        if bt is None or bt.numel() == 0:
            return torch.zeros((n_seqs, win), dtype=torch.int64)
        bt_host = (
            bt.detach().cpu()
            if bt.device.type != "cpu"
            else bt.detach()
        ).to(torch.int64)
        # Placeholder-stable ring addressing (see
        # _decode_window_compact_eager): modulo the FULL table width so
        # released leading blocks kept as -1 placeholders do not shift the
        # position -> column mapping.
        rows = bt_host.size(0)
        width = bt_host.size(1)
        seq_sel = torch.arange(n_seqs).clamp(max=rows - 1)
        row_bt = bt_host[seq_sel]
        blk_pos = (token // bs) % width
        blk_id = torch.gather(row_bt, 1, blk_pos)
        blk_ok = blk_id >= 0
        src = blk_id.clamp(min=0) * bs + (token % bs)
        return torch.where(valid & blk_ok, src, torch.zeros_like(src))

    def _build_window_src_indices(
        self,
        dsa_metadata: DsaMetadata,
        kv_seq_lens: Sequence[int],
    ) -> None:
        """Refresh the window source-slot indices for all layers (host).

        Vectorized rebuild per SWA manager (all layers sharing a group read
        the same block table), landing in ONE merged persist buffer
        (win_src_all: [n_layers, B, window]) so the per-step refresh pays a
        single H2D copy. Physical block-table addressing -- released leading
        blocks keep -1 placeholders in the table, so never assume a dense
        ring wrap; resolve every position through the table instead.
        """
        bs = 128
        win = self.window_size
        n_layers = len(self.caches_info)
        n_seqs = max(len(kv_seq_lens), 1)
        if n_layers == 0:
            return
        kv = torch.tensor(
            list(kv_seq_lens[:n_seqs]) + [0] * max(0, n_seqs - len(kv_seq_lens)),
            dtype=torch.int64,
        )
        k = kv.clamp(max=win)  # (B,)
        used = kv  # (B,)
        token = used.unsqueeze(1) - k.unsqueeze(1) + torch.arange(win).unsqueeze(0)
        # token j (B, W) valid iff j < k
        valid = torch.arange(win).unsqueeze(0) < k.unsqueeze(1)

        merged: list[torch.Tensor] = []
        seen: dict[int, int] = {}
        row_ids: list[int] = []
        for lid in range(n_layers):
            mapping = self._resolve_cache_mapping(
                lid, self._layer_compress_ratio(lid)
            )
            ori_idx = mapping.ori_cache_idx
            gid = (
                self.caches_info[lid][ori_idx].group_id
                if ori_idx >= 0
                and lid < len(self.caches_info)
                and ori_idx < len(self.caches_info[lid])
                else -1 - lid
            )
            if gid in seen:
                row_ids.append(seen[gid])
                continue
            tensor = self._window_src_for_group(
                dsa_metadata, lid, ori_idx, token, valid, n_seqs
            )
            seen[gid] = len(merged)
            row_ids.append(len(merged))
            merged.append(tensor)
        out_rows = merged
        self._win_src_row_map = row_ids
        merged_buf = get_execution_buffer(
            ("win_src_all",),
            lambda: torch.stack(out_rows).to(self.device)
            if out_rows
            else torch.zeros(1, n_seqs, win, dtype=torch.int64, device=self.device),
        )
        unique = (
            torch.stack(out_rows)
            if out_rows
            else torch.zeros(1, n_seqs, win, dtype=torch.int64)
        )
        if merged_buf.shape != unique.shape:
            state = get_forward_context().execution_state
            merged_buf = unique.to(self.device)
            if state is not None:
                state.persistent_buffers[("win_src_all",)] = merged_buf
        else:
            merged_buf.copy_(unique.to(self.device), non_blocking=True)

    def _graph_window_compact(
        self,
        ori_kv: torch.Tensor,
        seq_kv: torch.Tensor,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather the newest SWA window into a 0-based buffer (in-graph).

        The source slots come from the per-layer ``("win_src", layer_id)``
        execution buffer that the refresh rebuilds every step (physical
        block-table addressing). Everything here is device-side and
        bucket-static: an in-graph gather, a validity mask from the clamped
        window length, and a zero-based per-sequence block table. The
        head-aligned layout keeps the eager semantics (compact position j
        holds token ``used-k+j`` with ``k = min(used, window)``).
        """
        block_size = ori_kv.size(1)
        head_dim = ori_kv.size(-1)
        win = self.window_size
        n_seqs = seq_kv.size(0)
        device = ori_kv.device

        src_all = get_execution_buffer(
            ("win_src_all",),
            lambda: torch.zeros(
                (1, n_seqs, win), dtype=torch.int64, device=device
            ),
        )
        row_map = getattr(self, "_win_src_row_map", None)
        row = row_map[layer_id] if row_map and layer_id < len(row_map) else 0
        src_idx = src_all[row]  # (B, W) view into the merged buffer
        offs = torch.arange(win, device=device).unsqueeze(0)  # (1, W)
        k = seq_kv.clamp(max=win).to(torch.int64)  # (B,)
        valid = offs < k.unsqueeze(1)  # (B, W)

        kv_flat = ori_kv.reshape(-1, head_dim)
        # Safety clamp: out-of-bounds gather indices produce device error
        # 507011 (no .item() here -- this runs inside the captured graph).
        src_idx = src_idx.clamp(0, kv_flat.size(0) - 1)
        gathered = kv_flat[src_idx]  # (B, W, D)
        compact = torch.where(
            valid.unsqueeze(-1),
            gathered,
            torch.zeros((), dtype=ori_kv.dtype, device=device),
        )
        # One window-sized block per sequence: (B, bs, 1, D) with bs == win.
        compact_kv = compact.reshape(n_seqs, block_size, 1, head_dim) \
            if win == block_size else compact.reshape(
                n_seqs * ((win + block_size - 1) // block_size),
                block_size, 1, head_dim,
            )
        n_blocks = compact_kv.size(0)
        bt = torch.arange(
            n_blocks, device=device, dtype=torch.int32
        ).reshape(n_seqs, -1)
        seqused = k.to(torch.int32)
        return compact_kv, bt, seqused

    def _persist_tensor(self, key: tuple[object, ...], tensor: torch.Tensor) -> torch.Tensor:
        """Land ``tensor`` in an execution-state persistent buffer (copy_).

        Only called on the per-replay refresh path (outside the graph). Shapes
        are bucket-stable once captured; growth re-allocates, which is only
        safe before capture (the runner freezes bucket shapes first).
        """
        buf = get_execution_buffer(key, lambda: tensor.clone())
        if buf.shape != tensor.shape:
            state = get_forward_context().execution_state
            buf = tensor.clone()
            if state is not None:
                state.persistent_buffers[key] = buf
        else:
            buf.copy_(tensor)
        return buf

    def _pack_metadata_to_device(self, dsa: DsaMetadata) -> None:
        """Pack CPU metadata into one pinned H2D + view-bind (C++ deepseek_v4.h).

        Replaces the per-field ``tensor.to(device)`` (one H2D + one persist
        copy_ per field) with a single byte-packed copy, mirroring
        ``deepseek_v4_pack_dsa_metadata_to_device``: collect CPU tensors
        (content-dedup), 64B-align them into a pinned host byte buffer, one
        async H2D into a persistent device byte buffer, then bind typed views
        into that buffer at each tensor's offset. ``view(dtype)`` reinterpret
        on the NPU device buffer shares storage, so the captured graph reads
        stable addresses refreshed by a single copy_ per step.
        """
        device = self.device
        scalar_names = (
            "seq_lens",
            "seq_lens_q",
            "actual_seq_lengths_query",
            "actual_seq_lengths_kv",
            "kv_cu_seq_lens",
            "max_seqlen_q",
            "max_seqlen_kv",
            "input_positions",
            "c4_pad_positions",
            "c128_pad_positions",
            "start_pos",
            "hadamard",
        )
        specs: list[dict] = []
        seen: dict[tuple, int] = {}

        def add(tensor: torch.Tensor, setter) -> None:
            if tensor is None:
                return
            if tensor.device.type != "cpu":
                # Already on device: leave the address stable.
                return
            if tensor.numel() == 0:
                # Empty tensors (unused manager slots) move to device like the
                # original path but carry no bytes to pack.
                setter(tensor.to(device))
                return
            t = tensor.contiguous()
            nbytes = t.numel() * t.element_size()
            key = (t.data_ptr(), nbytes, t.dtype, tuple(t.shape))
            idx = seen.get(key)
            if idx is not None:
                specs[idx]["targets"].append(setter)
            else:
                seen[key] = len(specs)
                specs.append({
                    "cpu": t, "nbytes": nbytes, "dtype": t.dtype,
                    "sizes": list(t.shape), "targets": [setter],
                })

        for name in scalar_names:
            add(getattr(dsa, name, None), lambda v, n=name: setattr(dsa, n, v))
        # block_tables / slot_mappings: the builder shares one CPU tensor per
        # manager across every layer referencing it; data_ptr dedup collapses
        # them so all layers rebind the same packed view.
        for layer_tensors in dsa.block_tables:
            for index, tensor in enumerate(layer_tensors):
                add(tensor, lambda v, lt=layer_tensors, i=index: lt.__setitem__(i, v))
        for layer_tensors in dsa.slot_mappings:
            for index, tensor in enumerate(layer_tensors):
                add(tensor, lambda v, lt=layer_tensors, i=index: lt.__setitem__(i, v))

        if not specs:
            return

        # 64-byte-aligned layout (view(dtype) requires offset alignment).
        alignment = 64
        total_bytes = 0
        for spec in specs:
            total_bytes = (total_bytes + alignment - 1) // alignment * alignment
            spec["offset"] = total_bytes
            total_bytes += spec["nbytes"]

        # Pinned host staging (allocated once, reused across steps).
        host = getattr(self, "_packed_host", None)
        if host is None or host.numel() < total_bytes:
            host = torch.empty(total_bytes, dtype=torch.uint8).pin_memory()
            self._packed_host = host
        host_view = host.narrow(0, 0, total_bytes)
        for spec in specs:
            host_view.narrow(0, spec["offset"], spec["nbytes"]).copy_(
                spec["cpu"].reshape(-1).view(torch.uint8)
            )

        # Single async H2D into the persistent device byte buffer.
        state = get_forward_context().execution_state
        dev_buf = get_execution_buffer(
            ("dsa_packed",),
            lambda: torch.empty(total_bytes, dtype=torch.uint8, device=device),
        )
        if dev_buf.numel() < total_bytes:
            dev_buf = torch.empty(total_bytes, dtype=torch.uint8, device=device)
            if state is not None:
                state.persistent_buffers[("dsa_packed",)] = dev_buf
        dev_buf.narrow(0, 0, total_bytes).copy_(host_view, non_blocking=True)

        # Bind typed views into the device buffer.
        for spec in specs:
            view = (
                dev_buf.narrow(0, spec["offset"], spec["nbytes"])
                .view(spec["dtype"])
                .reshape(spec["sizes"])
            )
            for setter in spec["targets"]:
                setter(view)

    def _move_metadata_to_device(
        self, dsa: DsaMetadata, persistent: bool = False
    ) -> None:
        """Mirror ``deepseek_v4_move_dsa_metadata_to_device`` for eager mode."""
        if persistent:
            # Packed single-copy path: scalar fields + manager block tables +
            # slot mappings all ride one pinned H2D and rebind as views into
            # the persistent device byte buffer (see _pack_metadata_to_device).
            self._pack_metadata_to_device(dsa)
        else:
            tensor_fields = (
                "seq_lens",
                "seq_lens_q",
                "actual_seq_lengths_query",
                "actual_seq_lengths_kv",
                "kv_cu_seq_lens",
                "max_seqlen_q",
                "max_seqlen_kv",
                "input_positions",
                "c4_pad_positions",
                "c128_pad_positions",
                "start_pos",
                "hadamard",
            )
            for name in tensor_fields:
                tensor = getattr(dsa, name, None)
                if tensor is None:
                    continue
                setattr(dsa, name, tensor.to(self.device))
            # Keep CPU originals of the manager block tables so the eager
            # decode window gather reads them without per-layer D2H syncs
            # (the builder dedups per manager; ~6 unique tensors).
            _seen: set[int] = set()
            host_cache: dict[int, torch.Tensor] = {}
            for lid, layer_tensors in enumerate(dsa.block_tables):
                for index, tensor in enumerate(layer_tensors):
                    if tensor is not None:
                        if tensor.device.type == "cpu" and tensor.numel() > 0:
                            t_id = id(tensor)
                            if t_id not in _seen:
                                _seen.add(t_id)
                                dev_t = tensor.to(self.device)
                                layer_tensors[index] = dev_t
                                # Key by the device tensor's data_ptr so the
                                # eager window gather can look it up from the
                                # device tensor it receives in execute().
                                host_cache[dev_t.data_ptr()] = tensor
                                continue
                        layer_tensors[index] = tensor.to(self.device)
            self._eager_bt_host = host_cache
            for layer_tensors in dsa.slot_mappings:
                for index, tensor in enumerate(layer_tensors):
                    if tensor is not None:
                        layer_tensors[index] = tensor.to(self.device)
        if persistent:
            # Rope gathers re-allocate every step; persist them so the
            # captured forward reads address-stable tables. The main
            # cos/sin tables are chunk views of a persistent model buffer
            # (address-stable by construction) and their consumers use
            # index_select (no contiguity requirement), so skip persisting
            # them to avoid the 2x64MB per-layer copy on every refresh.
            for name in (
                "c4_cos", "c4_sin", "c128_cos", "c128_sin",
            ):
                tensor = getattr(dsa, name, None)
                if tensor is not None and tensor.numel() > 0:
                    setattr(
                        dsa, name,
                        self._persist_tensor(("dsa_rope", name), tensor),
                    )


# ---------------------------------------------------------------------------
# Helpers (faithful ports of C++ free functions).
# ---------------------------------------------------------------------------


def _tensor_max_or_zero(tensor: torch.Tensor | None) -> int:
    if tensor is None or tensor.numel() == 0:
        return 0
    return int(tensor.max().item())


def _build_dsa_forward_meta(
    dsa: DsaMetadata, metadata: AttentionMetadata
) -> _DsaForwardMeta:
    """Mirror the C++ max-seqlen inputs used by build_precomputed_metadata.

    C++ computes sparse metadata max sizes from ModelInputParams::meta plus the
    host q/kv length vectors:
      max(params.meta.q_max_seq_len, max(host.q_seq_lens))
      max(params.meta.kv_max_seq_len, max(host.kv_seq_lens))
    """

    q_max = int(getattr(metadata, "max_query_len", dsa.max_query_len))
    kv_max = int(getattr(metadata, "max_seq_len", dsa.max_seq_len))
    q_max = max(q_max, _tensor_max_or_zero(getattr(metadata, "q_seq_lens_host", None)))
    kv_max = max(kv_max, _tensor_max_or_zero(getattr(metadata, "kv_seq_lens_host", None)))
    q_max = max(q_max, int(dsa.max_query_len))
    kv_max = max(kv_max, int(dsa.max_seq_len))
    return _DsaForwardMeta(q_max_seq_len=q_max, kv_max_seq_len=kv_max)


def _build_prefill_pa_nd_kv(
    kv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    block_table_hint: torch.Tensor | None,
    block_size: int,
    cu_seqlens_dst: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Python port of C++ build_prefill_pa_nd_kv (deepseek_sparse_attention.cpp:272-368).

    Builds a temporary PA_ND format KV cache from the current forward's kv
    tensor, for full prefill attention (no paged cache needed).
    """
    if kv is None or cu_seqlens is None or cu_seqlens.numel() <= 1 or block_size <= 0:
        return torch.empty(0), torch.empty(0)

    batch_size = cu_seqlens.numel() - 1
    cu_cpu = cu_seqlens.to(torch.device("cpu")).to(torch.int64)
    cu = cu_cpu.tolist()

    dst_cu = None
    if cu_seqlens_dst is not None and cu_seqlens_dst.numel() == batch_size + 1:
        dst_cu = cu_seqlens_dst.to(torch.device("cpu")).to(torch.int64).tolist()

    # Compute per-request lengths and block counts.
    dst_lens = []
    total_blocks = 0
    max_blocks_per_req = 0
    for i in range(batch_size):
        q_len = (
            dst_cu[i + 1] - dst_cu[i]
            if dst_cu is not None
            else cu[i + 1] - cu[i]
        )
        dst_lens.append(q_len)
        blocks = (q_len + block_size - 1) // block_size
        total_blocks += blocks
        max_blocks_per_req = max(max_blocks_per_req, blocks)

    if total_blocks <= 0:
        return torch.empty(0), torch.empty(0)

    table_cols = max(
        block_table_hint.size(1) if block_table_hint is not None and block_table_hint.dim() > 1 else 0,
        max_blocks_per_req,
    )

    # 0-based blocks with no padding block: CANN sparse_flash_mla addresses
    # the PA cache strictly through the block table and produces zero output
    # when block ids skip a zero-filled leading block (the xllm_ops kernel
    # tolerated the 1-based layout; SFM does not).
    packed_kv = torch.zeros(
        total_blocks, block_size, kv.size(1), kv.size(2),
        dtype=kv.dtype, device=kv.device,
    )

    table_data = [0] * (batch_size * table_cols)
    next_block = 0
    for req in range(batch_size):
        q_start = cu[req]
        src_len = cu[req + 1] - q_start
        q_len = dst_lens[req]
        blocks = (q_len + block_size - 1) // block_size
        if q_len <= 0 or blocks <= 0:
            continue
        for j in range(blocks):
            table_data[req * table_cols + j] = next_block + j
        copy_len = min(q_len, src_len)
        if copy_len > 0:
            target = packed_kv[next_block:next_block + blocks].view(
                blocks * block_size, kv.size(1), kv.size(2)
            )
            target[q_len - copy_len:q_len].copy_(kv[q_start:q_start + copy_len])
        next_block += blocks

    table = torch.tensor(table_data, dtype=torch.int32, device=kv.device).view(
        batch_size, table_cols
    )
    return packed_kv, table


def _get_layer_cache_tensor(
    layer_tensors: list[list[torch.Tensor]],
    layer_id: int,
    cache_idx: int,
) -> torch.Tensor | None:
    """Python port of ``get_layer_cache_tensor`` (deepseek_sparse_attention.cpp:80)."""
    if (
        layer_id < 0
        or layer_id >= len(layer_tensors)
        or cache_idx < 0
        or cache_idx >= len(layer_tensors[layer_id])
    ):
        return None
    return layer_tensors[layer_id][cache_idx]


def _scatter_by_slot(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Python port of ``scatter_by_slot`` (deepseek_sparse_attention.cpp:200).

    Writes ``value`` rows into the paged ``cache`` at the physical slots given by
    ``slot_mapping`` (= block_id * block_size + offset).
    """
    if (
        cache is None
        or cache.numel() == 0
        or slot_mapping is None
        or slot_mapping.numel() == 0
        or value is None
        or value.numel() == 0
    ):
        return
    value_2d = value.reshape(-1, value.size(-1))
    cache_2d = cache.view(-1, value_2d.size(1))
    slots = slot_mapping.reshape(-1).to(torch.long).to(cache.device)
    update_rows = min(slots.size(0), value_2d.size(0))
    if update_rows <= 0:
        return

    slots_slice = slots[:update_rows]
    value_slice = value_2d[:update_rows]

    if cache.device.type != "cpu":
        # Faithful port of C++ scatter_by_slot (deepseek_sparse_attention.cpp:230-235):
        # clamp -1 slots to 0, mask them so their old values are written back
        # unchanged, then scatter_nd_update.
        from xllm.python import kernels

        safe_slots = slots_slice.clamp_min(0)
        valid_mask = (slots_slice >= 0).unsqueeze(1)
        old_values = cache_2d.index_select(0, safe_slots)
        safe_values = torch.where(valid_mask, value_slice, old_values)
        kernels.scatter_nd_update(
            cache_2d, safe_slots.reshape(-1, 1), safe_values
        )
        return

    valid_mask = slots_slice >= 0
    valid_slots = slots_slice[valid_mask]
    if valid_slots.numel() == 0:
        return
    valid_values = value_slice[valid_mask]
    cache_2d.index_copy_(0, valid_slots, valid_values)


def _sparse_attn_sharedkv(**kwargs):
    """Two-stage sparse attention via CANN built-in ``sparse_flash_mla``.

    Thin indirection that maps the ``sparse_attn_sharedkv`` calling convention
    onto the CANN ``sparse_flash_mla`` interface (patched for Ascend950), and
    snapshots the exact input/output tensors for the later golden comparison.
    """
    from xllm.python import kernels

    layer = kwargs.pop("_dump_layer", None)
    # --- sparse_attn_sharedkv -> sparse_flash_mla 参数映射 ---
    seqused_kv = kwargs.pop("seqused_kv", None)          # -> seqused_ori_kv
    cmp_ratio = kwargs.get("cmp_ratio", 1)
    cmp_kv = kwargs.get("cmp_kv", None)
    has_cmp = cmp_kv is not None and cmp_ratio > 1
    # layout: PA_ND -> PA_BBND（物理 layout 相同，仅字符串不同）
    kwargs["layout_kv"] = "PA_BBND"
    # PA_BBND 下不允许传 cu_seqlens_ori_kv / cu_seqlens_cmp_kv
    kwargs["cu_seqlens_ori_kv"] = None
    kwargs["cu_seqlens_cmp_kv"] = None
    # seqused_kv 拆成 seqused_ori_kv + seqused_cmp_kv；补算 cmp_residual_kv。
    # Callers may pass the three values directly (graph mode: the ori side
    # runs a clamped window while cmp keeps the true length).
    if kwargs.get("seqused_ori_kv") is None:
        kwargs["seqused_ori_kv"] = seqused_kv
    if has_cmp:
        if kwargs.get("seqused_cmp_kv") is None:
            if seqused_kv is not None:
                kwargs["seqused_cmp_kv"] = seqused_kv // cmp_ratio
                kwargs["cmp_residual_kv"] = seqused_kv % cmp_ratio
            else:
                kwargs["seqused_cmp_kv"] = None
                kwargs["cmp_residual_kv"] = None
    else:
        kwargs["seqused_cmp_kv"] = None
        kwargs["cmp_residual_kv"] = None
    # SWA（无 cmp_kv）时 cmp_mask_mode 必须 0
    if not has_cmp:
        kwargs["cmp_mask_mode"] = 0
    # 预留/新增属性。ori_topk_length 仅在 SWA 稀疏 ori_kv 场景由调用方
    # 传入（每行有效索引条目数，左对齐），其余场景保持 None。
    kwargs.setdefault("ori_topk_length", None)
    kwargs.setdefault("cmp_topk_length", None)
    kwargs["topk_value_mode"] = 1

    if layer is not None:
        dsa_dump.snap(
            "sparse_flash_mla",
            dict(kwargs),
            layer=layer.layer_id,
            kind="moe" if layer.layer_id else "dense",
            extra={
                "source_layout_kv": "PA_ND",
                "kernel_layout_kv": kwargs["layout_kv"],
                "has_cmp": has_cmp,
                "cmp_ratio": cmp_ratio,
                "cu_seqlens_forced_none": True,
            },
        )
    out, lse = kernels.sparse_flash_mla(**kwargs)
    if layer is not None:
        dsa_dump.snap(
            "sparse_flash_mla_out",
            {"out": out, "lse": lse},
            layer=layer.layer_id,
            kind="moe" if layer.layer_id else "dense",
        )
    return out, lse

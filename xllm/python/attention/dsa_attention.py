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
        )
        self._populate_dsa_rope(dsa_metadata, metadata)
        self._move_metadata_to_device(dsa_metadata, persistent=True)
        self._build_precomputed_metadata(dsa_metadata, metadata, persistent=True)
        self._graph_dsa = dsa_metadata
        # Rebuild the per-layer window source indices (physical block-table
        # addressing) into the execution buffers the graph gathers through.
        self._build_window_src_indices(dsa_metadata, kv_seq_lens)
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
        chunks = cos_sin_cache.chunk(2, dim=-1)
        dsa.layer_id = layer_id
        dsa.cos_table = chunks[0].contiguous()
        dsa.sin_table = chunks[1].contiguous()

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
            dsa.cos_table, dsa.sin_table = (
                tensor.contiguous() for tensor in css.chunk(2, dim=-1)
            )
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
        # Decode fix for the CANN sparse_flash_mla operator: the kernel
        # produces zero output when block_table contains non-zero block IDs
        # at decode shapes (T=1). Copy the referenced cache blocks into a
        # fresh 0-based compact buffer and set all block_table entries to 0.
        # Graph mode passes the real block table directly instead: the copy
        # depends on per-step host reads (.item()) that cannot be captured,
        # and the standalone capture experiment (smoke/RESULTS.md X1) shows
        # correct decode output with non-zero block IDs under replay.
        if not use_prefill_attn and not self._graph_mode:
            def _compact_cache(kv, bt, total_tokens=None, newest_slot=None,
                               linear_base=None, copy_slots=0):
                """Copy referenced blocks into a 0-based buffer, return
                (compact_kv, compact_bt) with all bt entries = 0.

                The framework sliding-window block table can lag one block
                behind the newest partially-written block (and reference
                blocks already reclaimed by the SWA ring), which zeroes the
                newest window tokens in the compact buffer and decays the
                attention output.

                With ``slot_base`` (the ring slot of token 0, derived from the
                current step slot mapping) the compact buffer is laid out at
                ABSOLUTE token positions: token t lives at pool slot
                ``(slot_base + t) % ring_slots``. Only the newest
                ``copy_slots`` tokens are gathered; older absolute slots stay
                zero and are masked by the kernel Band mask.

                With ``total_tokens`` but no ``slot_base`` (compressed caches)
                the pool is sequential: block b holds entries
                ``[b*block_size, ...)``.

                Without ``total_tokens`` the caller-provided block table is
                followed verbatim (allocator-agnostic fallback)."""
                if kv is None or bt is None:
                    return kv, bt
                n_seqs = bt.size(0)
                n_cols = bt.size(1)
                block_size = kv.size(1)
                if total_tokens is not None and newest_slot is not None:
                    used = max(int(total_tokens), 1)
                    # Round up to whole blocks: the kernel addresses the
                    # compact buffer block-wise via the all-zero block table.
                    total_slots = (
                        (used - 1) // block_size + 1
                    ) * block_size
                    compact = torch.zeros(
                        1, total_slots, kv.size(2), kv.size(3),
                        dtype=kv.dtype, device=kv.device,
                    )
                    compact_flat = compact.reshape(-1, kv.size(3))
                    kv_flat = kv.reshape(-1, kv.size(3))
                    # SWA ring: the sliding-window manager recycles slots
                    # with a 2*window_size cycle inside
                    # [block_size, block_size + 2*window_size); token t lives
                    # at slot newest_slot - (newest_index - t) wrapped into
                    # that range. Only the newest copy_slots tokens are
                    # recoverable; older absolute slots stay zero and are
                    # masked by the kernel Band mask.
                    cycle = 2 * self.window_size
                    slot_lo = block_size
                    start_t = max(0, used - copy_slots)
                    offs = torch.arange(start_t, used, device=kv.device)
                    s = newest_slot - (used - 1 - offs)
                    s = (s - slot_lo) % cycle + slot_lo
                    compact_flat[start_t:used] = kv_flat[s]
                elif total_tokens is not None and linear_base is not None:
                    # Sequential pool (compressed caches): entry e lives at
                    # slot linear_base + e.
                    used = max(int(total_tokens), 1)
                    total_slots = (
                        (used - 1) // block_size + 1
                    ) * block_size
                    compact = torch.zeros(
                        1, total_slots, kv.size(2), kv.size(3),
                        dtype=kv.dtype, device=kv.device,
                    )
                    compact_flat = compact.reshape(-1, kv.size(3))
                    kv_flat = kv.reshape(-1, kv.size(3))
                    n_slots = min(
                        used, kv.size(0) * block_size - linear_base
                    )
                    idx = linear_base + torch.arange(
                        n_slots, device=kv.device
                    )
                    compact_flat[:n_slots] = kv_flat[idx]
                else:
                    bt_cpu = bt.cpu()
                    max_id = 0
                    for b in range(n_seqs):
                        for j in range(n_cols):
                            v = int(bt_cpu[b, j])
                            if v > max_id:
                                max_id = v
                    total_slots = (max_id + 1) * block_size
                    compact = torch.zeros(
                        1, total_slots, kv.size(2), kv.size(3),
                        dtype=kv.dtype, device=kv.device,
                    )
                    compact_flat = compact.reshape(-1, kv.size(3))
                    kv_flat = kv.reshape(-1, kv.size(3))
                    for b in range(n_seqs):
                        for j in range(n_cols):
                            blk = int(bt_cpu[b, j])
                            if blk < 0:
                                continue
                            src_start = blk * block_size
                            dst_start = j * block_size
                            n = min(block_size, total_slots - dst_start)
                            if n > 0:
                                compact_flat[dst_start:dst_start + n] = \
                                    kv_flat[src_start:src_start + n]
                new_bt = torch.zeros(
                    n_seqs, n_cols, dtype=torch.int32, device=kv.device
                )
                return compact, new_bt

            ori_total = None
            if seq_kv is not None and seq_kv.numel() > 0:
                ori_total = int(seq_kv.max().item())
            # Slot of the newest token (index ori_total-1), read from the
            # current step slot mapping.
            ori_newest_slot = None
            if (
                ori_total is not None
                and ori_slot is not None
                and ori_slot.numel() > 0
                and ori_kv_for_attn is not None
            ):
                ori_newest_slot = int(ori_slot.flatten()[0].item())
            if os.environ.get("DSA_BYPASS_ORI", "1") == "1":
                ori_kv_for_attn, ori_block_table_for_attn = _compact_cache(
                    ori_kv_for_attn, ori_block_table_for_attn,
                    total_tokens=ori_total,
                    newest_slot=ori_newest_slot,
                    copy_slots=(
                        2 * self.window_size if ori_newest_slot is not None else 0
                    ),
                )
            if cmp_kv is not None:
                cmp_total = (
                    ori_total // compress_ratio
                    if ori_total is not None and compress_ratio > 1
                    else None
                )
                # The compressed cache shares the reserved-block ring layout
                # (entry e lives at slot base+e); derive the base from the
                # newest committed entry slot, memoizing per layer for the
                # non-commit steps where cmp_slot is absent.
                cmp_base = None
                if (
                    cmp_total
                    and cmp_slot is not None
                    and cmp_slot.numel() > 0
                ):
                    cmp_base = (
                        int(cmp_slot.flatten()[0].item()) - (cmp_total - 1)
                    )
                    if not hasattr(self, "_cmp_linear_bases"):
                        self._cmp_linear_bases = {}
                    self._cmp_linear_bases[layer_id] = cmp_base
                if cmp_base is None:
                    cmp_base = getattr(self, "_cmp_linear_bases", {}).get(
                        layer_id
                    )
                if os.environ.get("DSA_BYPASS_CMP", "1") == "1":
                    cmp_kv, cmp_block_table_for_kernel = _compact_cache(
                        cmp_kv, cmp_block_table_for_kernel,
                        total_tokens=cmp_total,
                        linear_base=cmp_base,
                    )
        if self._graph_mode and not use_prefill_attn:
            # Static window compact for the ori (SWA-ring) side. CANN
            # sparse_flash_mla mis-decodes the real ring block table at
            # decode shapes (bisection: ori-direct garbles, cmp-direct is
            # fine -- smoke/RESULTS.md X3), so the graph gathers the newest
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
        out, _lse = _sparse_attn_sharedkv(
            _dump_layer=layer,
            q=q,
            ori_kv=ori_kv_for_attn,
            cmp_kv=cmp_kv if compress_ratio > 1 else None,
            ori_sparse_indices=None,
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
            ori_mask_mode=_MASK_MODE_COMPRESS,
            cmp_mask_mode=_MASK_MODE_RIGHT_DOWN_CAUSAL,
            ori_win_left=self.window_size - 1,
            ori_win_right=0,
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
                    # Graph decode runs the ori side over the clamped window
                    # compact; the tiling metadata must describe the same
                    # length the kernel receives, or the AICPU tiling reads
                    # past the 2*window compact capacity.
                    seq_kv.clamp(max=2 * self.window_size)
                    if self._graph_mode
                    else seq_kv
                ),
                seqused_cmp_kv=seqused_cmp_kv,
                cmp_residual_kv=cmp_residual_kv,
                ori_topk_length=None,
                cmp_topk_length=None,
                batch_size=batch_size,
                max_seqlen_q=max_q,
                max_seqlen_ori_kv=max_kv,
                max_seqlen_cmp_kv=max_seqlen_cmp_kv,
                ori_topk=0,
                cmp_topk=cmp_topk,
                cmp_ratio=ratio,
                ori_mask_mode=_MASK_MODE_COMPRESS,
                cmp_mask_mode=(0 if not has_cmp else _MASK_MODE_RIGHT_DOWN_CAUSAL),
                ori_win_left=max(self.window_size - 1, 0),
                ori_win_right=0,
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

    def _build_window_src_indices(
        self,
        dsa_metadata: DsaMetadata,
        kv_seq_lens: Sequence[int],
    ) -> None:
        """Refresh the per-layer window source-slot indices (host, per step).

        For every layer's SWA manager the newest ``window`` tokens are
        addressed through the REAL block table (physical addressing, no ring
        wrap assumptions -- the eager wrap formula breaks once the manager
        allocates blocks beyond the first, see smoke/RESULTS.md X3/ISS-7).
        The indices land in execution buffers keyed by layer; the captured
        graph gathers through them, so refreshing the content is enough.
        """
        bs = 128
        win = self.window_size
        n_layers = len(self.caches_info)
        n_seqs = max(len(kv_seq_lens), 1)
        for layer_id in range(n_layers):
            mapping = self._resolve_cache_mapping(
                layer_id, self._layer_compress_ratio(layer_id)
            )
            ori_idx = mapping.ori_cache_idx
            if ori_idx < 0:
                continue
            bt = (
                dsa_metadata.block_tables[layer_id][ori_idx]
                if layer_id < len(dsa_metadata.block_tables)
                and ori_idx < len(dsa_metadata.block_tables[layer_id])
                else None
            )
            slots = (
                dsa_metadata.slot_mappings[layer_id][ori_idx]
                if layer_id < len(dsa_metadata.slot_mappings)
                and ori_idx < len(dsa_metadata.slot_mappings[layer_id])
                else None
            )
            if bt is None or slots is None or bt.numel() == 0:
                continue
            bt_host = bt.detach().cpu().tolist()
            slot_host = (
                slots.detach().cpu().tolist()
                if slots.numel() >= n_seqs else [0] * n_seqs
            )
            rows = len(bt_host)
            idx_rows: list[list[int]] = []
            for seq in range(n_seqs):
                used = int(kv_seq_lens[seq]) if seq < len(kv_seq_lens) else 0
                k = min(used, win)
                row = bt_host[seq] if seq < rows else []
                # Effective (non -1) columns define the ring's block cycle.
                eff = [b for b in row if b >= 0]
                cols_eff = max(len(eff), 1)
                src: list[int] = []
                for j in range(win):
                    if j < k:
                        token = used - k + j
                        blk_pos = (token // bs) % cols_eff
                        blk = (
                            eff[blk_pos]
                            if eff
                            else (slot_host[seq] // bs if slot_host[seq] >= 0 else 0)
                        )
                        src.append(blk * bs + (token % bs))
                    else:
                        src.append(0)
                idx_rows.append(src)
            idx_tensor = torch.tensor(
                idx_rows, dtype=torch.int64, device=self.device
            )
            self._persist_tensor(
                ("win_src", layer_id), idx_tensor.to(self.device)
            )

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

        src_idx = get_execution_buffer(
            ("win_src", layer_id),
            lambda: torch.zeros(
                (n_seqs, win), dtype=torch.int64, device=device
            ),
        )
        offs = torch.arange(win, device=device).unsqueeze(0)  # (1, W)
        k = seq_kv.clamp(max=win).to(torch.int64)  # (B,)
        valid = offs < k.unsqueeze(1)  # (B, W)

        kv_flat = ori_kv.reshape(-1, head_dim)
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

    def _move_metadata_to_device(
        self, dsa: DsaMetadata, persistent: bool = False
    ) -> None:
        """Mirror ``deepseek_v4_move_dsa_metadata_to_device`` for eager mode."""
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
            if persistent:
                setattr(
                    dsa, name,
                    self._persist_tensor(("dsa_dev", name), tensor.to(self.device)),
                )
            else:
                setattr(dsa, name, tensor.to(self.device))
        for layer_idx, layer_tensors in enumerate(dsa.block_tables):
            for index, tensor in enumerate(layer_tensors):
                if tensor is None:
                    continue
                if persistent:
                    layer_tensors[index] = self._persist_tensor(
                        ("dsa_bt", layer_idx, index), tensor.to(self.device)
                    )
                else:
                    layer_tensors[index] = tensor.to(self.device)
        for layer_idx, layer_tensors in enumerate(dsa.slot_mappings):
            for index, tensor in enumerate(layer_tensors):
                if tensor is None:
                    continue
                if persistent:
                    layer_tensors[index] = self._persist_tensor(
                        ("dsa_slot", layer_idx, index), tensor.to(self.device)
                    )
                else:
                    layer_tensors[index] = tensor.to(self.device)
        if persistent:
            # Rope gathers re-allocate every step; persist them so the
            # captured forward reads address-stable tables.
            for name in (
                "cos_table", "sin_table", "c4_cos", "c4_sin",
                "c128_cos", "c128_sin",
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
    # 预留/新增属性
    kwargs["ori_topk_length"] = None
    kwargs["cmp_topk_length"] = None
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

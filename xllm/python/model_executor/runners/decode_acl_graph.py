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

"""NPU (Ascend) ACL graph runner for the Python model executor.

Captures and replays decode-step graphs using ``torch.npu.NPUGraph``.
Mirrors the structure of ``decode_cuda_graph.py`` but adds NPU-specific
logic:

* ``torch.npu.graph_task_group_begin/end`` around FIA ``.out`` calls during
  capture.
* ``torch.npu.graph_task_update_begin/end`` to refresh FIA host params before
  replay.
* Static ``block_table`` and ``slot_mapping`` tensors so the graph records
  fixed addresses whose *contents* are updated via ``_fill_entry`` each step.
* C++ ACLNN ops (RMSNorm, SiLU, reshape_paged_cache) are used in both eager
  and capture modes — no PyTorch fallbacks needed.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from scripts.logger import logger
from xllm.python import kernels
from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
)
from xllm.python.attention.dsa_metadata import DSA_CACHE_TOKEN
from xllm.python.attention.expanded_decode_metadata import (
    ExpandedDecodeMetadata,
    resolve_expanded_decode_metadata,
)
from xllm.python.model_executor.forward_context import (
    AclGraphCaptureContext,
    AclGraphExecutionState,
    AclGraphTask,
    EplbRuntimeState,
    ForwardContext,
    LayerSynchronizer,
    forward_context,
)
from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.model_executor.runners.base import BaseRunner, ModelExecutionOutput
from xllm.python.model_executor.runners.decode_cuda_graph import (
    _CAPTURE_WARMUP_STEPS,
    _decode_bucket,
)


def _require_positive_execution_counts(execution_counts: Sequence[int]) -> None:
    if any(count <= 0 for count in execution_counts):
        raise RuntimeError(f"DP execution token counts must be positive, got {execution_counts}")


def _dp_active_execution_counts(
    execution_counts: Sequence[int],
    raw_counts: Sequence[int] | None,
) -> list[int]:
    """Drop empty-rank placeholders. A raw 0 is not a verify sequence."""
    counts = [int(count) for count in execution_counts]
    if raw_counts is not None and len(raw_counts) == len(counts):
        active = [count for count, raw in zip(counts, raw_counts) if int(raw) > 0]
        if active:
            return active
    return counts


def _dp_spec_sequence_batch(
    execution_counts: Sequence[int],
    raw_counts: Sequence[int] | None,
    local_width: int,
) -> int | None:
    """Fold expanded-verify tokens into sequences.

    A raw 0 is an empty-rank placeholder. Width 1 is plain decode and is never
    folded. ``None`` means an active count is not a whole verify group, so every
    rank must stay eager. Both ranks must pass that same width.
    """
    counts = [int(count) for count in execution_counts]
    if local_width <= 1:
        return max(counts)
    active = _dp_active_execution_counts(counts, raw_counts)
    if any(count % local_width != 0 for count in active):
        return None
    return max(active) // local_width


def _complete_kpool_groups(padded_num_tokens: int, verify_width: int) -> tuple[int, ...]:
    """Groups of ``verify_width`` that fit in a bucket. A short tail is omitted."""
    if verify_width <= 1 or padded_num_tokens <= 0:
        return ()
    full_groups = padded_num_tokens // verify_width
    if full_groups <= 0:
        return ()
    return (verify_width,) * full_groups


class _CaptureAttentionMetadata:
    """Keep the batch layout while routing linear state to reserved slot 0."""

    def __init__(self, metadata: AttentionMetadata) -> None:
        self._metadata = metadata
        for name in ("linear_state_indices", "linear_state_read_indices", "linear_state_write_indices"):
            indices = getattr(metadata, name, None)
            setattr(self, name, torch.zeros_like(indices) if indices is not None else None)
        indices = self.linear_state_indices
        self.has_initial_state = torch.zeros_like(indices, dtype=torch.int32) if indices is not None else None
        accepted = getattr(metadata, "num_accepted_tokens", None)
        self.num_accepted_tokens = torch.ones_like(accepted) if accepted is not None else None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._metadata, name)


def _padded_rank_slices(
    token_counts: Sequence[int],
    dp_size: int,
    per_rank_stride: int,
) -> list[tuple[int, int]]:
    """Validate per-rank counts and return offsets into a padded DP buffer."""
    counts = tuple(int(count) for count in token_counts)
    if len(counts) != dp_size:
        raise RuntimeError(f"DP decode step requires {dp_size} token counts, got {len(counts)}")
    if any(count < 0 or count > per_rank_stride for count in counts):
        raise RuntimeError(
            f"DP token counts must fit the ACL graph batch bucket: counts={counts}, bucket={per_rank_stride}"
        )
    return [(rank * per_rank_stride, count) for rank, count in enumerate(counts)]


@dataclass(slots=True)
class _StaticAttentionMetadata:
    slot_mapping: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    qo_indptr: torch.Tensor | None = None
    q_cu_seq_lens: torch.Tensor | None = None
    kv_cu_seq_lens: torch.Tensor | None = None
    kv_seq_lens_host: torch.Tensor | None = None
    kv_seq_lens_host_values: list[int] | None = None
    new_cache_slots_host_values: list[int] | None = None
    paged_kv_indptr_host: torch.Tensor | None = None
    paged_kv_last_page_len_host: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    kv_seq_lens: torch.Tensor | None = None
    linear_state_indices: torch.Tensor | None = None
    linear_state_checkpoint_indices: torch.Tensor | None = None
    num_accepted_tokens: torch.Tensor | None = None
    has_initial_state: torch.Tensor | None = None
    dp_execution_token_counts: tuple[int, ...] = ()
    dp_is_decode: tuple[int, ...] = ()
    q_seq_lens: torch.Tensor | None = None
    kpool_query_lens: tuple[int, ...] = ()
    kpool_query_lens_device: torch.Tensor | None = None
    # Host-side copy of the (per-entry constant) q_cu for prepare()'s
    # sequence-lens read: a .cpu() on the device buffer would block the host
    # until the whole prior step's device queue drains, serializing the
    # scheduler behind the replay.
    q_cu_host_values: list[int] | None = None
    expanded_decode_metadata: ExpandedDecodeMetadata | None = None
    is_prefill: bool = False
    is_chunked_prefill: bool = False
    mega_moe_token_mask: torch.Tensor | None = None
    is_mixed: bool = False
    is_spec_verify: bool = False
    is_dflash_proposal: bool = False
    is_dummy: bool = False
    # Verify group width baked into this captured graph. 0 means the KDA
    # path derives equal groups from the padded row count.
    spec_group_width: int = 0
    local_slot_mapping: torch.Tensor | None = None
    kv_split_size: int = 1
    kv_split_rank: int = 0
    has_kv_shard: bool = False
    multi_block_tables: Sequence[torch.Tensor | None] = ()
    dsa_metadata: object | None = None
    dsa_positions: torch.Tensor | None = None
    dsa_cos_sin: torch.Tensor | None = None
    dsa_c4_cos_sin: torch.Tensor | None = None
    dsa_c128_cos_sin: torch.Tensor | None = None
    dsa_graph_mode: bool = False
    dsa_graph_block_table_cols: int = 0
    # Token-capacity bound of the CSA2 committed-row buckets (0 = backend does
    # not bucket); consumed by the backend's capacity gathers.
    dsa_graph_token_capacity: int = 0


class _DecodeGraphEntry:
    __slots__ = (
        "batch_size",
        "graph",
        "static_output",
        "static_input_ids",
        "static_positions",
        "static_input_embedding",
        "static_metadata",
        "kv_seq_lens_delta",
        "graph_tasks",
        "execution_state",
        "eplb",
        "execution_contexts",
        "is_padding",
    )


class _DsaDerivedMetadataView:
    """Read-through metadata view deriving the flat paging fields for DSA backends.

    The C++ scheduler's multi-block export path (DeepSeek-V4.1 DSA) fills only
    ``multi_block_tables`` and leaves the flat ``block_table`` undefined while
    ``slot_mapping`` arrives as an empty [0] tensor. CSA2 never reads those
    flat fields (DSA paging is rebuilt from the multi-block tables), but the
    decode-graph admission/fill contract expects a well-formed flat table.
    The view derives ``block_table`` from the SWA group table (manager 0) and
    a zero ``slot_mapping`` so the generic graph machinery stays
    shape-consistent; both are zero-reader fields for CSA2.
    """

    __slots__ = ("_base", "block_table", "slot_mapping", "multi_block_tables")

    def __init__(
        self,
        base: AttentionMetadata,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        multi_block_tables: tuple[torch.Tensor, ...],
    ) -> None:
        self._base = base
        self.block_table = block_table
        self.slot_mapping = slot_mapping
        self.multi_block_tables = multi_block_tables

    def __getattr__(self, name: str) -> object:
        return getattr(self._base, name)


_GraphKey = tuple[
    int,
    bool,
    int,
    torch.dtype | None,
    torch.device | None,
    tuple[int, ...] | None,
    tuple[int, ...],
    bool,
    int,
]


class DecodeAclGraphRunner(BaseRunner):
    """Decode graph runner for NPU (Ascend) using ACL graph capture/replay."""

    def __init__(
        self,
        model: nn.Module,
        attention_backend: AttentionBackend,
        device: torch.device,
        max_batch: int,
        max_model_len: int,
        dp_size: int = 1,
        dp_rank: int = 0,
        decode_batch_size_limit: int | None = None,
        num_decoding_tokens: int = 1,
        enable_mega_moe_token_mask: bool = False,
        *,
        is_spec_draft: bool = False,
        draft_query_width: int = 1,
        spec_verify_rows: int = 1,
        eplv2_graph_token_limit: int | None = None,
    ) -> None:
        super().__init__(model, attention_backend, device)
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        # Each rank may receive every sequence in the batch. The scheduler's
        # remaining_seq_budget is the full max_seqs_per_batch, not an even
        # split, so the graph sequence capacity matches that budget.
        self.sequence_batch_capacity = max(1, int(max_batch))
        # max_batch is the largest token-row bucket this runner can capture.
        # Speculative target verification expands each sequence to several
        # independent decode rows; only the executor knows that the model is
        # running that mode, so it supplies the row width explicitly. Direct
        # runner callers keep the previous sequence-only behavior.
        spec_verify_rows = max(1, int(spec_verify_rows))
        self.max_batch = self.sequence_batch_capacity
        if spec_verify_rows > 1:
            self.max_batch = _decode_bucket(self.sequence_batch_capacity * spec_verify_rows)
        # Token rows per logical sequence: 1 for plain decode, num_spec+1 for
        # MTP spec-verify. Static paging buffers are sized in token rows so the
        # expanded-verify graph gets matching capacity.
        self.num_decoding_tokens = max(1, int(num_decoding_tokens))
        # Optional memory guardrail: buckets above the limit fall back to
        # eager instead of capturing ever-larger graphs (0 disables).
        self.decode_batch_size_limit = int(decode_batch_size_limit or 0)
        self.max_model_len = max_model_len
        self._enable_mega_moe_token_mask = enable_mega_moe_token_mask
        self._is_spec_draft = is_spec_draft
        # Draft graphs stay width 1. This width only folds token rows into
        # sequences for the max_seqs_per_batch check.
        self._draft_query_width = max(1, int(draft_query_width))
        self._eplv2_graph_token_limit = eplv2_graph_token_limit
        self._graphs: dict[_GraphKey, _DecodeGraphEntry] = {}
        self._paged_kv_indices_buffer: torch.Tensor | None = None
        self._max_blocks_per_sequence: int = 0
        self._pinned_block_table_staging: dict[tuple[str, torch.dtype, torch.Size], torch.Tensor] = {}
        self._stream: torch.npu.Stream | None = None
        self._update_stream: torch.npu.Stream | None = None
        self._replay_done_event: torch.npu.Event | None = None
        # One-shot latch for the "host-offload staging staged nothing" warning
        # (see ``_warn_engram_stage_incomplete``): a replay runs once per step,
        # so an unlatched warning would flood the log for a persistent failure.
        self._engram_stage_skip_warned = False
        # Pre-size the host-offloaded engram graph-visible staging block for the
        # largest decode token bucket this runner can capture. Capture is lazy
        # and buckets warm up in ascending order, so without this floor the
        # block would be fixed at the first (smallest) bucket and a later,
        # larger bucket would overflow it (see EngramEmbedding._ensure_graph_staging).
        configure_engram = getattr(self.model, "configure_engram_graph_capacity", None)
        if configure_engram is not None:
            configure_engram(self._engram_graph_token_capacity())

    def _engram_graph_token_capacity(self) -> int:
        """Floor, in token rows, for the graph-visible engram staging block.

        The captured graph packs ``num_decoding_tokens`` token rows per
        sequence at the verify slots (1 for plain decode) -- the same
        per-sequence width ``_shared_graph_plan`` folds into ``verify_width``.
        A spec-draft runner
        additionally emits up to ``draft_query_width`` query rows per sequence;
        that is a different quantity from the verify rows and may be
        larger, so fold the wider of the two, keeping the block from being sized
        below what a wider draft replay touches. ``decode_batch_size_limit``,
        when set, keeps larger buckets on the eager path and off this floor.

        This value pre-sizes the graph-visible engram staging block, whose
        address is baked into the captured ACL graph and never grows afterwards
        (``EngramEmbedding._ensure_graph_staging`` raises rather than moving it),
        so it must be a lower bound (floor) covering the worst-case replay:
        sizing it too small overflows the block on a wider replay, while sizing
        it large only wastes HBM. ``_decode_bucket`` then aligns this floor to
        the graph key's padded token-row bucket.
        """
        seq_capacity = self.sequence_batch_capacity
        if self.decode_batch_size_limit > 0:
            seq_capacity = min(seq_capacity, self.decode_batch_size_limit)
        rows_per_seq = self.num_decoding_tokens
        if self._is_spec_draft:
            rows_per_seq = max(rows_per_seq, self._draft_query_width)
        return _decode_bucket(seq_capacity * rows_per_seq)

    def _eplv2_graph_admits(self, token_rows: int) -> bool:
        """Keep MC2 capture inside the HCCL window; All-to-AllV stays eager."""
        limit = self._eplv2_graph_token_limit
        return limit is None or _decode_bucket(token_rows) <= limit

    def _shared_graph_plan(
        self,
        metadata: AttentionMetadata,
        batch_size: int,
        local_expanded: bool | None = None,
    ) -> tuple[bool, int, int | None]:
        """Return the graph layout every DP rank must capture.

        ``(is_expanded, verify_width, admission_batch)``. ``admission_batch`` is
        ``None`` when active token counts are not whole verify groups. Under DP
        the width comes only from ``is_spec_verify`` and ``num_decoding_tokens``,
        so an empty rank and a busy rank share one graph key. A local expanded
        layout that is not that group-wide verify raises instead of capturing
        a private graph. Plain decode stays width 1.
        """
        if local_expanded is None:
            local_expanded = resolve_expanded_decode_metadata(metadata) is not None
        marked_spec_verify = bool(getattr(metadata, "is_spec_verify", False))
        spec_verify = marked_spec_verify or local_expanded
        local_width = 1
        if local_expanded and batch_size > 0:
            linear_idx = getattr(metadata, "linear_state_indices", None)
            seq_count = linear_idx.numel() if linear_idx is not None and linear_idx.numel() > 0 else batch_size
            if batch_size % seq_count == 0:
                local_width = batch_size // seq_count
        if self.dp_size > 1:
            # Local expanded rows must not choose a width the empty rank cannot see.
            if marked_spec_verify and self.num_decoding_tokens > 1:
                if local_width > 1 and local_width != self.num_decoding_tokens:
                    raise RuntimeError(
                        "DP spec-verify layout does not match num_decoding_tokens: "
                        f"measured={local_width}, num_decoding_tokens={self.num_decoding_tokens}"
                    )
                verify_width = self.num_decoding_tokens
            else:
                if local_width > 1:
                    raise RuntimeError(
                        "DP expanded decode is not a group-wide spec verify: "
                        f"measured={local_width}, num_decoding_tokens={self.num_decoding_tokens}"
                    )
                verify_width = 1
        elif local_width > 1:
            verify_width = local_width
        elif spec_verify and self.num_decoding_tokens > 1:
            verify_width = self.num_decoding_tokens
        else:
            verify_width = 1
        is_expanded = verify_width > 1
        if self.dp_size > 1:
            admission_batch = _dp_spec_sequence_batch(
                metadata.dp_execution_token_counts,
                getattr(metadata, "raw_dp_execution_token_counts", None),
                verify_width,
            )
        elif verify_width > 1 and batch_size % verify_width == 0:
            admission_batch = batch_size // verify_width
        else:
            admission_batch = batch_size
        if self._is_spec_draft and verify_width <= 1 and self._draft_query_width > 1:
            # Capacity is in sequences. Draft rows are query_width per sequence
            # but the captured graph stays width 1, so fold only this gate.
            if self.dp_size > 1:
                active_counts = _dp_active_execution_counts(
                    metadata.dp_execution_token_counts,
                    getattr(metadata, "raw_dp_execution_token_counts", None),
                )
                if active_counts and all(count % self._draft_query_width == 0 for count in active_counts):
                    admission_batch = max(active_counts) // self._draft_query_width
            elif batch_size % self._draft_query_width == 0:
                admission_batch = batch_size // self._draft_query_width
        return is_expanded, verify_width, admission_batch

    def can_execute(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
    ) -> bool:
        metadata = self._normalize_dsa_metadata(metadata)
        ok = self._can_execute_inner(input_ids, metadata, input_embedding)
        return ok

    def _normalize_dsa_metadata(
        self,
        metadata: AttentionMetadata,
    ) -> AttentionMetadata:
        """Derive or synthesize the flat/multi paging contract for DSA backends.

        Two normalizations, both no-ops for non-DSA backends:

        1. Busy DSA batches receive ``multi_block_tables`` only (the C++
           scheduler's multi-block export leaves the flat ``block_table``
           undefined); derive the flat table from the SWA group table.
        2. Empty DP shards receive a single dummy token and NO multi-block
           tables (the worker's empty-shard fake input is never routed
           through the composite block exporter). Left on the eager runner,
           the shard's fake-token full-model forward dominates every DP step
           (all ranks wait for the slowest group), so synthesize one-row
           all-zero tables and let the shard replay the bucket graph the
           busy ranks also replay. Block 0 mirrors the eager table-less
           fallback (ring block 0); the dummy row's output is discarded.
           Real decode batches always carry tables (verified: busy-shard
           decode exports n=3), so "table-less DSA batch" uniquely
           identifies the empty-shard dummy.

        Every rank derives the same contract from the same exported shape, so
        the whole DP group takes one runner (graph or eager) for a step; a
        mixed step would desynchronize the HCCL collectives.
        """
        group_infos = getattr(self.attention_backend, "group_infos", None)
        if group_infos is None:
            return metadata
        tables = list(getattr(metadata, "multi_block_tables", ()) or ())
        if not tables:
            if not bool(getattr(metadata, "is_dummy", False)):
                # A busy DSA batch must carry manager tables; a table-less
                # non-dummy batch is an exporter regression, not an empty shard.
                return metadata
            tables = [torch.zeros((1, 1), dtype=torch.int32) for _ in group_infos]
        elif len(tables) != len(group_infos) or tables[0].dim() != 2:
            return metadata
        block_table = metadata.block_table
        if block_table is None:
            # The C++ exporter stages the multi-block tables on the CPU. Keep
            # the derived flat table where it is: this view is rebuilt by all
            # three runner entry points (can_execute/warmup/execute), and a
            # device staging here would add two extra pageable H2D copies per
            # decode step, each draining the device queue. _decode_metadata
            # already performs the single device conversion the fill needs.
            block_table = tables[0]
        slot_mapping = metadata.slot_mapping
        if slot_mapping is None or slot_mapping.numel() != block_table.shape[0]:
            slot_dtype = slot_mapping.dtype if slot_mapping is not None else torch.int64
            slot_mapping = torch.zeros(block_table.shape[0], dtype=slot_dtype, device=block_table.device)
        return _DsaDerivedMetadataView(
            metadata,
            block_table,
            slot_mapping,
            tuple(tables),
        )

    def _can_execute_inner(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
    ) -> bool:
        # Static MoE graph-capability gate (51361671 A): the fp8/none checkpoint
        # modes serve ``DeepseekV41MoE``, whose forward syncs to the host and
        # loops over data-dependent expert segments, so no ACL graph can capture
        # it. The marker is derived once from the shared model config at backend
        # construction, so it does not depend on this rank's per-step metadata:
        # every rank of a DP group returns False together and none drops to
        # eager alone. Backends without the marker (non-V4.1) keep graph access.
        if not getattr(self.attention_backend, "moe_graph_capturable", True):
            return False
        if self.dp_size == 1 and input_ids.dim() != 1:
            return False
        # Debug switch. Under DP every rank must see it, not only the expanded one.
        disable_verify_graph = os.environ.get("XLLM_NO_VERIFY_GRAPH") == "1"
        if disable_verify_graph and self.dp_size > 1 and self.num_decoding_tokens > 1:
            return False
        batch_size = input_ids.numel()
        expanded_view = resolve_expanded_decode_metadata(metadata)
        is_expanded_spec_verify = expanded_view is not None
        # The DSpark draft backend owns expanded draft rows, so its dedicated
        # block-size configuration identifies the explicit-index graph path.
        is_dspark_draft_expanded = (
            self._is_spec_draft
            and bool(getattr(self.attention_backend, "dspark_use_explicit_indices", False))
            and int(getattr(self.attention_backend, "dspark_block_size", 0)) > 0
            and not metadata.is_prefill
        )
        if disable_verify_graph and is_expanded_spec_verify:
            return False
        # max_batch is a token-row bucket, while expanded-verify metadata
        # reports one row per logical sequence. Gate those rows by sequence
        # count so an expanded group cannot bypass the scheduler sequence
        # budget; plain decode and unexpanded speculative target verification
        # stay in token-row units.
        if self.dp_size == 1 and is_expanded_spec_verify:
            lsi = getattr(metadata, "linear_state_indices", None)
            has_kda_layers = lsi is not None and lsi.numel() > 0
            seq_count = lsi.numel() if has_kda_layers else batch_size
            size_check_bs = seq_count
        elif self.dp_size == 1:
            size_check_bs = batch_size
        if self.dp_size == 1 and self._is_spec_draft and self._draft_query_width > 1:
            if size_check_bs % self._draft_query_width != 0:
                return False
            size_check_bs //= self._draft_query_width
        kpool_query_lens = tuple(int(length) for length in getattr(metadata, "kpool_query_lens", ()))
        if kpool_query_lens:
            spans_cover_rows = all(length > 0 for length in kpool_query_lens) and sum(kpool_query_lens) == batch_size
            if not spans_cover_rows:
                # Only this rank sees the span list. Returning eager here would
                # leave a peer that has no spans on the shared graph.
                if self.dp_size > 1:
                    raise RuntimeError(
                        "DP KPool graph spans must cover this rank's token rows with positive lengths: "
                        f"spans={kpool_query_lens}, tokens={batch_size}"
                    )
                return False
            query_width = kpool_query_lens[0]
            if any(length != query_width for length in kpool_query_lens[1:]):
                # Only this rank sees the span list. Returning eager here would
                # leave a peer that has no spans on the shared graph.
                if self.dp_size > 1:
                    raise RuntimeError(
                        f"DP KPool graph spans must be uniform across requests: spans={kpool_query_lens}"
                    )
                return False
        if self.dp_size > 1:
            # A verify rank is chunked-prefill locally while a peer can still
            # be marked decode. dp_is_decode is the replicated step type, so
            # only a real non-decode step leaves the graph. A local chunked
            # flag must not make one rank eager and the other replay.
            dp_is_decode = getattr(metadata, "dp_is_decode", None)
            if dp_is_decode is None or len(dp_is_decode) != self.dp_size:
                raise RuntimeError(
                    "DP decode step requires dp_is_decode on every rank: "
                    f"got {dp_is_decode!r}, expected length {self.dp_size}"
                )
            all_decode = all(dp_is_decode)
            if input_ids.dim() != 1:
                if all_decode:
                    raise RuntimeError(
                        "DP decode input_ids must be one-dimensional: "
                        f"rank={self.dp_rank}, shape={tuple(input_ids.shape)}"
                    )
                return False
            local_non_decode = (metadata.is_prefill or metadata.is_chunked_prefill) and not (
                is_expanded_spec_verify or is_dspark_draft_expanded
            )
            if local_non_decode and not all_decode:
                return False
            # DP ranks share one graph shape, so a missing or malformed
            # Execution counts cannot silently fall back to eager: divergent
            # execution paths across ranks would deadlock HCCL collectives.
            execution_counts = getattr(
                metadata,
                "dp_execution_token_counts",
                None,
            )
            if execution_counts is None or len(execution_counts) != self.dp_size:
                raise RuntimeError(
                    "DP decode step requires valid dp_execution_token_counts "
                    f"(got {execution_counts!r}, "
                    f"expected length {self.dp_size}). All DP ranks must use the same graph shape."
                )
            _require_positive_execution_counts(execution_counts)
            if not self._eplv2_graph_admits(max(execution_counts)):
                return False
            if not all_decode:
                return False
            # MegaMoe shares one fixed graph shape across DP ranks, so the
            # local rows must equal this rank's advertised execution count;
            # a mismatch would dispatch the wrong shape and desync the EP
            # collective rather than silently fall back to eager.
            if execution_counts[self.dp_rank] != input_ids.shape[0]:
                raise RuntimeError(
                    "DP execution token count does not match the local input: "
                    f"rank={self.dp_rank}, rows={input_ids.shape[0]}, "
                    f"counts={execution_counts}"
                )
            _, verify_width, global_batch = self._shared_graph_plan(
                metadata,
                batch_size,
                local_expanded=is_expanded_spec_verify,
            )
            if global_batch is None:
                return False
            max_global_tokens = max(int(c) for c in execution_counts)
            if self._is_spec_draft and verify_width <= 1 and self._draft_query_width > 1:
                raw_counts = getattr(metadata, "raw_dp_execution_token_counts", None)
                active_counts = _dp_active_execution_counts(execution_counts, raw_counts)
                if any(count % self._draft_query_width != 0 for count in active_counts):
                    return False
            if verify_width > 1 and kpool_query_lens:
                self._require_shared_kpool_layout(
                    metadata,
                    batch_size,
                    _decode_bucket(max_global_tokens),
                    verify_width,
                )
            if input_embedding is not None and input_embedding.shape[0] != batch_size:
                raise RuntimeError(
                    "DP decode input_embedding does not match token rows: "
                    f"rank={self.dp_rank}, tokens={batch_size}, "
                    f"embedding_rows={int(input_embedding.shape[0])}"
                )
            ok = (
                self._has_compatible_decode_metadata(
                    input_ids,
                    metadata,
                    dp_all_decode=True,
                    local_expanded=is_expanded_spec_verify,
                    expanded_view=expanded_view,
                )
                and _decode_bucket(global_batch) <= self.max_batch
                and (self.decode_batch_size_limit <= 0 or _decode_bucket(global_batch) <= self.decode_batch_size_limit)
            )
            return ok
        bucket_size = _decode_bucket(size_check_bs)
        if not self._eplv2_graph_admits(batch_size):
            return False
        ok = (
            (
                (not metadata.is_prefill and not metadata.is_chunked_prefill)
                or is_expanded_spec_verify
                or is_dspark_draft_expanded
            )
            and self._has_compatible_decode_metadata(
                input_ids,
                metadata,
                local_expanded=is_expanded_spec_verify,
                expanded_view=expanded_view,
            )
            and (input_embedding is None or input_embedding.shape[0] == batch_size)
            and bucket_size <= self.max_batch
            and (self.decode_batch_size_limit <= 0 or bucket_size <= self.decode_batch_size_limit)
        )
        return ok

    def _decode_metadata(
        self, metadata: AttentionMetadata
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[int] | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return per-row KV and paging metadata for decode graph replay."""
        expanded = resolve_expanded_decode_metadata(metadata, block_size=self.attention_backend.page_size)
        block_table = expanded.block_table if expanded is not None else self._effective_block_table(metadata)
        kv_seq_lens = expanded.kv_seq_lens if expanded is not None else metadata.kv_seq_lens
        kv_seq_lens_host_values = (
            expanded.kv_seq_lens_host_values
            if expanded is not None
            else getattr(metadata, "kv_seq_lens_host_values", None)
        )
        paging_kv_seq_lens_host_values = kv_seq_lens_host_values
        if block_table is None or kv_seq_lens is None:
            raise RuntimeError("decode graph requires block and KV metadata")
        kv_seq_lens = kv_seq_lens.to(torch.int32)
        block_table = block_table.to(
            device=kv_seq_lens.device,
            dtype=torch.int32,
        ).contiguous()
        is_mla = getattr(self.attention_backend, "is_mla", False)
        requires_host_kv_lengths = getattr(
            self.attention_backend,
            "requires_host_kv_lengths",
            not is_mla,
        )
        if not requires_host_kv_lengths:
            # The C++ engine sizes kv_seq_lens_host_values by the global DP
            # batch, while block_table holds only this rank's local rows. When
            # the backend does not consume host KV lengths (e.g. sparse MLA),
            # drop them so the mismatched global length never reaches shape
            # validation (which would otherwise reject the bucket and silently
            # fall a DP>1 MLA decode back to eager).
            kv_seq_lens_host_values = None
        elif kv_seq_lens_host_values is None:
            raise RuntimeError("decode graph requires scheduler-provided host KV lengths")

        paged_kv_indptr = expanded.paged_kv_indptr if expanded is not None else metadata.paged_kv_indptr
        paged_kv_indices = expanded.paged_kv_indices if expanded is not None else metadata.paged_kv_indices
        paged_kv_last_page_len = (
            expanded.paged_kv_last_page_len if expanded is not None else metadata.paged_kv_last_page_len
        )
        # Speculative row builders provide row-aligned paging through the
        # regular attention metadata. Keep the fallback for older producers,
        # but steady-state replay reuses the packed tensors without Python
        # loops or extra H2D copies.
        _paged_missing = paged_kv_indptr is None or paged_kv_indices is None or paged_kv_last_page_len is None
        _paged_mismatch = (not _paged_missing) and (
            paged_kv_last_page_len.numel() != block_table.shape[0]
            or paged_kv_indptr.numel() != block_table.shape[0] + 1
        )
        if _paged_missing and expanded is None:
            raise RuntimeError("decode graph requires paged KV metadata")
        if _paged_missing or _paged_mismatch:
            (
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
            ) = self._build_row_aligned_paged_kv_metadata(
                block_table,
                paging_kv_seq_lens_host_values,
            )
        self._validate_decode_metadata_shapes(
            block_table,
            kv_seq_lens,
            kv_seq_lens_host_values,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        )
        return (
            block_table,
            kv_seq_lens,
            kv_seq_lens_host_values,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        )

    def _build_row_aligned_paged_kv_metadata(
        self,
        block_table: torch.Tensor,
        kv_seq_lens_host_values: Sequence[int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build row paging metadata without dynamic-shape device operators."""
        page_size = int(self.attention_backend.page_size)
        if page_size <= 0:
            raise RuntimeError("decode graph page size must be positive")
        if kv_seq_lens_host_values is None or len(kv_seq_lens_host_values) != block_table.shape[0]:
            raise RuntimeError("row-aligned graph paging requires one host KV length per token row")

        flat_page_indices: list[int] = []
        paged_kv_indptr_values = [0]
        paged_kv_last_page_len_values: list[int] = []
        table_width = int(block_table.shape[1])
        for row, kv_seq_len in enumerate(kv_seq_lens_host_values):
            effective_kv_seq_len = max(int(kv_seq_len), 1)
            page_count = (effective_kv_seq_len + page_size - 1) // page_size
            if page_count > table_width:
                raise RuntimeError(
                    "row-aligned graph paging exceeds the block table width: "
                    f"row={row}, pages={page_count}, capacity={table_width}"
                )
            paged_kv_indptr_values.append(paged_kv_indptr_values[-1] + page_count)
            paged_kv_last_page_len_values.append((effective_kv_seq_len - 1) % page_size + 1)
            row_offset = row * table_width
            flat_page_indices.extend(row_offset + page for page in range(page_count))

        flat_page_indices_tensor = torch.tensor(
            flat_page_indices,
            dtype=torch.int64,
            device=block_table.device,
        )
        paged_kv_indices = block_table.reshape(-1).index_select(0, flat_page_indices_tensor).contiguous()
        paged_kv_indptr = torch.tensor(
            paged_kv_indptr_values,
            dtype=torch.int32,
            device=block_table.device,
        )
        paged_kv_last_page_len = torch.tensor(
            paged_kv_last_page_len_values,
            dtype=torch.int32,
            device=block_table.device,
        )
        return (
            paged_kv_indptr.contiguous(),
            paged_kv_indices,
            paged_kv_last_page_len.contiguous(),
        )

    @staticmethod
    def _validate_decode_metadata_shapes(
        block_table: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        kv_seq_lens_host_values: list[int] | None,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
    ) -> None:
        if block_table.dim() != 2:
            raise RuntimeError("decode block_table must be two-dimensional")
        sequence_count = block_table.shape[0]
        per_sequence_tensors = (
            ("kv_seq_lens", kv_seq_lens),
            ("paged_kv_last_page_len", paged_kv_last_page_len),
        )
        for name, tensor in per_sequence_tensors:
            if tensor.dim() != 1 or tensor.numel() != sequence_count:
                raise RuntimeError(f"decode {name} must contain one value per sequence")
        if kv_seq_lens_host_values is not None and len(kv_seq_lens_host_values) != sequence_count:
            raise RuntimeError("decode kv_seq_lens_host_values must contain one value per sequence")
        if paged_kv_indptr.dim() != 1 or paged_kv_indptr.numel() != sequence_count + 1:
            raise RuntimeError("decode paged_kv_indptr must contain one offset per sequence plus the terminal offset")
        if paged_kv_indices.dim() != 1 or paged_kv_indices.numel() == 0:
            raise RuntimeError("decode paged_kv_indices must be a non-empty flat page list")

    @staticmethod
    def _validate_decode_token_layout(
        input_ids: torch.Tensor,
        positions: torch.Tensor | None,
        slot_mapping: torch.Tensor,
        sequence_count: int,
    ) -> None:
        if input_ids.dim() != 1 or input_ids.numel() != sequence_count:
            raise RuntimeError("ACL graph decode input_ids must contain one token per sequence")
        if slot_mapping.dim() != 1 or slot_mapping.numel() != sequence_count:
            raise RuntimeError("ACL graph decode slot_mapping must contain one slot per token")
        if positions is not None and (positions.dim() != 1 or positions.numel() != sequence_count):
            raise RuntimeError("ACL graph decode positions must contain one value per token")

    def _has_compatible_decode_metadata(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
        *,
        dp_all_decode: bool = False,
        local_expanded: bool | None = None,
        expanded_view: object | None = None,
    ) -> bool:
        """Check the one-token-per-sequence contract of ACL decode graphs.

        Pure shape/contract admission check — no on-device paging build and no
        exception-based capability detection (§7: expected incompatible shapes
        return False explicitly; genuinely unexpected/malformed states are left
        to raise from _decode_metadata/_validate during fill, propagating rather
        than being swallowed into a silent eager fallback).
        """
        if not self._is_shape_compatible(
            input_ids,
            metadata,
            expanded_view=expanded_view,
            expanded_resolved=local_expanded is not None,
        ):
            self._require_dp_all_decode_row_layout(input_ids, metadata, dp_all_decode)
            self._require_dp_all_decode_block_capacity(input_ids, metadata, dp_all_decode)
            return False
        batch_size = input_ids.numel()
        is_expanded = (
            local_expanded if local_expanded is not None else resolve_expanded_decode_metadata(metadata) is not None
        )
        is_dspark_draft_expanded = (
            self._is_spec_draft
            and bool(getattr(self.attention_backend, "dspark_use_explicit_indices", False))
            and int(getattr(self.attention_backend, "dspark_block_size", 0)) > 0
            and not metadata.is_prefill
        )
        if not is_expanded and metadata.kv_cu_seq_lens is not None:
            if metadata.kv_cu_seq_lens.numel() not in (
                batch_size,
                batch_size + 1,
            ):
                self._reject_dp_all_decode(
                    dp_all_decode,
                    "DP decode kv_cu_seq_lens does not match token rows: "
                    f"rank={self.dp_rank}, tokens={batch_size}, kv_cu={metadata.kv_cu_seq_lens.numel()}",
                )
                return False
        if metadata.q_cu_seq_lens is not None and not is_expanded:
            if metadata.q_cu_seq_lens.numel() not in (
                batch_size,
                batch_size + 1,
            ):
                self._reject_dp_all_decode(
                    dp_all_decode,
                    "DP decode q_cu_seq_lens does not match token rows: "
                    f"rank={self.dp_rank}, tokens={batch_size}, q_cu={metadata.q_cu_seq_lens.numel()}",
                )
                return False
        has_initial_state = getattr(metadata, "has_initial_state", None)
        if has_initial_state is not None:
            state_count = has_initial_state.numel()
            if state_count != batch_size and not (
                (is_expanded and state_count > 0 and batch_size % state_count == 0) or is_dspark_draft_expanded
            ):
                self._reject_dp_all_decode(
                    dp_all_decode,
                    "DP decode has_initial_state does not match token rows: "
                    f"rank={self.dp_rank}, tokens={batch_size}, states={state_count}",
                )
                return False
        linear_idx = getattr(metadata, "linear_state_indices", None)
        if (
            linear_idx is not None
            and linear_idx.numel() != batch_size
            and not (
                (is_expanded and linear_idx.numel() > 0 and batch_size % linear_idx.numel() == 0)
                or is_dspark_draft_expanded
            )
        ):
            self._reject_dp_all_decode(
                dp_all_decode,
                "DP decode linear_state_indices does not match token rows: "
                f"rank={self.dp_rank}, tokens={batch_size}, indices={linear_idx.numel()}",
            )
            return False
        # Linear-attention (KDA) layers read per-sequence conv/ssm state via
        # linear_state_indices; without it the captured graph would index
        # state slots with a None buffer. A spec-verify batch may carry one
        # slot id per logical sequence — the fill expands it
        # per token row.
        needs_linear_state = any(getattr(cache, "conv", None) is not None for cache in self.layer_caches)
        if needs_linear_state and linear_idx is None:
            raw_counts = getattr(metadata, "raw_dp_execution_token_counts", None)
            # A spec-verify placeholder has no local slots. It still replays
            # the busy rank's expanded graph, whose padding slot is filled later.
            placeholder_verify = (
                raw_counts is not None
                and 0 <= self.dp_rank < len(raw_counts)
                and int(raw_counts[self.dp_rank]) <= 0
                and bool(getattr(metadata, "is_spec_verify", False))
                and self.num_decoding_tokens > 1
            )
            if not placeholder_verify:
                self._reject_dp_all_decode(
                    dp_all_decode,
                    f"DP decode is missing linear_state_indices: rank={self.dp_rank}, tokens={batch_size}",
                )
                return False
        # V4.1 CSA2 multi-manager paging: graph replay refreshes every DSA
        # manager's block table (_fill_dsa_block_tables raises without them),
        # so a batch whose manager table count does not match the backend's
        # group count must stay on the eager runner, whose CSA2 path has
        # explicit table-less fallbacks (ring block 0 / plain window). Table-less
        # batches were already normalized above (empty DP shard dummy), so this
        # only rejects genuinely malformed multi-manager exports.
        group_infos = getattr(self.attention_backend, "group_infos", None)
        if group_infos is not None and not is_dspark_draft_expanded:
            source_tables = list(getattr(metadata, "multi_block_tables", ()) or ())
            if not source_tables or len(source_tables) != len(group_infos):
                return False
        # V4.1 CSA2 committed-row capacity buckets: the captured graph's
        # gathers are sized by max_model_len, so a context beyond it (or a
        # bucket the DP group does not share) must take the eager runner.
        granularity = getattr(self.attention_backend, "graph_token_capacity_granularity", 0)
        if granularity:
            bucket = self._token_capacity_bucket(metadata)
            if bucket <= 0:
                # 0 == not graph-admissible: either this rank cannot read the
                # host-planned global lengths under DP, or the max context is
                # unknown. Keep the eager runner rather than capture a bucket
                # the DP group may not share.
                return False
            if bucket > (self.max_model_len + int(granularity) - 1) // int(granularity):
                # A bucket above ceil(max_model_len / granularity) sits beyond
                # every capacity the graph may gather: fall back to the eager
                # runner (admission == False) instead of capturing an
                # out-of-range gather. This is the fail-closed upper bound, not
                # a crash path.
                return False
        return True

    @staticmethod
    def _effective_block_table(metadata: AttentionMetadata) -> torch.Tensor | None:
        """Return the primary per-row table used by DSV4 decode."""
        expanded = resolve_expanded_decode_metadata(metadata)
        if expanded is not None:
            return expanded.block_table
        if metadata.block_table is not None:
            return metadata.block_table
        multi_block_tables = tuple(getattr(metadata, "multi_block_tables", ()) or ())
        return multi_block_tables[0] if multi_block_tables else None

    @staticmethod
    def _has_effective_slot_mapping(
        metadata: AttentionMetadata,
        token_count: int,
    ) -> bool:
        slot_mapping = getattr(metadata, "slot_mapping", None)
        if slot_mapping is not None and slot_mapping.dim() == 1 and slot_mapping.numel() == token_count:
            return True
        host_slots = getattr(metadata, "new_cache_slots_host_values", None)
        if host_slots is not None and len(host_slots) == token_count:
            return True
        # V4.1 DSA multi-manager batches: a sequence with composite blocks is
        # exported through ``multi_block_tables`` and the C++ builder appends
        # nothing to ``new_token_slot_ids``, so the flat scheduler slot list is
        # empty by construction. The DSA builder then derives every committed
        # row's physical slot from the persistent manager tables, which makes
        # the flat contract a zero-reader field for CSA2 -- require the manager
        # table set instead so both busy and empty DP shards admit the graph.
        return bool(getattr(metadata, "multi_block_tables", ()))

    @staticmethod
    def _effective_slot_mapping(
        metadata: AttentionMetadata,
        token_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Resolve scheduler cache slots when DSV4 omits the top-level tensor."""
        slot_mapping = getattr(metadata, "slot_mapping", None)
        if slot_mapping is not None and slot_mapping.dim() == 1 and slot_mapping.numel() == token_count:
            return slot_mapping.to(device=device, dtype=torch.int32).contiguous()
        host_slots = getattr(metadata, "new_cache_slots_host_values", None)
        if host_slots is not None and len(host_slots) == token_count:
            return torch.tensor(host_slots, dtype=torch.int32, device=device)
        if getattr(metadata, "multi_block_tables", ()):
            # DSA flat slots are unread by CSA2 (see _has_effective_slot_mapping);
            # zeros keep the captured static buffer shape-consistent.
            return torch.zeros(token_count, dtype=torch.int32, device=device)
        raise RuntimeError("ACL graph decode requires one scheduler cache slot per token")

    def _reject_dp_all_decode(self, dp_all_decode: bool, message: str) -> None:
        if dp_all_decode:
            raise RuntimeError(message)

    def _require_dp_all_decode_row_layout(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
        dp_all_decode: bool,
    ) -> None:
        """Raise when one all-decode rank would otherwise drop to eager alone.

        Block-table rows, KV lengths, and cache slots are local. A peer with a
        matching layout would still enter the graph.
        """
        if input_ids.dim() != 1 or not dp_all_decode:
            return
        token_rows = int(input_ids.numel())
        block_table = self._effective_block_table(metadata)
        kv_seq_lens = metadata.kv_seq_lens
        expanded = resolve_expanded_decode_metadata(metadata)
        if expanded is not None:
            block_table = expanded.block_table
            kv_seq_lens = expanded.kv_seq_lens
            host_values = expanded.kv_seq_lens_host_values
        else:
            host_values = getattr(metadata, "kv_seq_lens_host_values", None)
        block_rows = int(block_table.shape[0]) if block_table is not None and block_table.dim() == 2 else None
        kv_rows = int(kv_seq_lens.numel()) if kv_seq_lens is not None and kv_seq_lens.dim() == 1 else None
        requires_host = not getattr(self.attention_backend, "is_mla", False) or getattr(
            self.attention_backend,
            "requires_host_kv_lengths",
            False,
        )
        host_rows = len(host_values) if host_values is not None else None
        slot_rows_match = self._has_effective_slot_mapping(metadata, token_rows)
        rows_match = (
            block_rows == token_rows
            and kv_rows == token_rows
            and slot_rows_match
            and (not requires_host or host_rows == token_rows)
        )
        if rows_match:
            return
        raise RuntimeError(
            "DP decode row layout does not match token rows: "
            f"rank={self.dp_rank}, tokens={token_rows}, block_rows={block_rows}, "
            f"kv_rows={kv_rows}, host_kv_rows={host_rows}, slots_match={slot_rows_match}"
        )

    def _require_dp_all_decode_block_capacity(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
        dp_all_decode: bool,
    ) -> None:
        """Raise when the block table cannot hold this rank's KV lengths.

        A peer whose table is wide enough would still enter the graph. Rows that
        do not match are reported separately. A complete paged layout is left to
        the graph.
        """
        if input_ids.dim() != 1 or not dp_all_decode:
            return
        block_table = self._effective_block_table(metadata)
        expanded = resolve_expanded_decode_metadata(metadata)
        if expanded is not None:
            block_table = expanded.block_table
            host_values = expanded.kv_seq_lens_host_values
            paged_kv_indptr = expanded.paged_kv_indptr
            paged_kv_indices = expanded.paged_kv_indices
            paged_kv_last_page_len = expanded.paged_kv_last_page_len
        else:
            host_values = getattr(metadata, "kv_seq_lens_host_values", None)
            paged_kv_indptr = metadata.paged_kv_indptr
            paged_kv_indices = metadata.paged_kv_indices
            paged_kv_last_page_len = metadata.paged_kv_last_page_len
        if block_table is None or block_table.dim() != 2 or host_values is None:
            return
        sequence_count = int(block_table.shape[0])
        if len(host_values) != sequence_count:
            return
        paged_missing = paged_kv_indptr is None or paged_kv_indices is None or paged_kv_last_page_len is None
        paged_mismatch = (not paged_missing) and (
            paged_kv_last_page_len.numel() != sequence_count or paged_kv_indptr.numel() != sequence_count + 1
        )
        if not paged_missing and not paged_mismatch:
            return
        page_size = int(self.attention_backend.page_size)
        table_width = int(block_table.shape[1])
        if (
            page_size > 0
            and table_width > 0
            and all((max(int(kv_seq_len), 1) + page_size - 1) // page_size <= table_width for kv_seq_len in host_values)
        ):
            return
        raise RuntimeError(
            "DP decode block table cannot hold the KV length: "
            f"rank={self.dp_rank}, page_size={page_size}, columns={table_width}, "
            f"kv_lengths={list(host_values)}"
        )

    def _is_shape_compatible(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
        expanded_view: object | None = None,
        *,
        expanded_resolved: bool = False,
    ) -> bool:
        """Pure, side-effect-free shape/contract pre-check.

        Mirrors the precondition branches of _decode_metadata and
        _validate_decode_metadata_shapes/_validate_decode_token_layout so that
        the EXPECTED incompatible shapes return False here (admit→eager) while
        any state this does not cover raises later (let-it-crash, §7) instead
        of being silently swallowed. No on-device paging construction.
        """
        expanded = expanded_view if expanded_resolved else resolve_expanded_decode_metadata(metadata)
        if expanded is not None:
            block_table = expanded.block_table
            kv_seq_lens = expanded.kv_seq_lens
        else:
            block_table = self._effective_block_table(metadata)
            kv_seq_lens = metadata.kv_seq_lens
        if block_table is None or kv_seq_lens is None:
            return False
        if block_table.dim() != 2 or kv_seq_lens.dim() != 1:
            return False
        sequence_count = block_table.shape[0]
        if kv_seq_lens.numel() != sequence_count:
            return False
        host_values = (
            expanded.kv_seq_lens_host_values
            if expanded is not None
            else getattr(metadata, "kv_seq_lens_host_values", None)
        )
        # Backends with host length inputs require a valid list; a missing
        # or mis-sized host list is an expected "not compatible" (→ eager).
        is_mla = getattr(self.attention_backend, "is_mla", False)
        requires_host = getattr(
            self.attention_backend,
            "requires_host_kv_lengths",
            not is_mla,
        )
        if requires_host:
            if host_values is None or len(host_values) != sequence_count:
                return False
        paged_kv_indptr = expanded.paged_kv_indptr if expanded is not None else metadata.paged_kv_indptr
        paged_kv_indices = expanded.paged_kv_indices if expanded is not None else metadata.paged_kv_indices
        paged_kv_last_page_len = (
            expanded.paged_kv_last_page_len if expanded is not None else metadata.paged_kv_last_page_len
        )
        paged_missing = paged_kv_indptr is None or paged_kv_indices is None or paged_kv_last_page_len is None
        paged_mismatch = (not paged_missing) and (
            paged_kv_last_page_len.numel() != sequence_count or paged_kv_indptr.numel() != sequence_count + 1
        )
        if paged_missing or paged_mismatch:
            if host_values is None or len(host_values) != sequence_count:
                return False
            page_size = int(self.attention_backend.page_size)
            table_width = int(block_table.shape[1])
            if page_size <= 0 or table_width <= 0:
                return False
            if any((max(int(kv_seq_len), 1) + page_size - 1) // page_size > table_width for kv_seq_len in host_values):
                return False
        # One-token-per-sequence (per-row) contract.
        if input_ids.dim() != 1 or input_ids.numel() != sequence_count:
            return False
        if not self._has_effective_slot_mapping(metadata, sequence_count):
            return False
        return True

    @staticmethod
    def _cumulative_lengths(
        sequence_lengths: torch.Tensor,
        cumulative_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Normalize NPU sequence ends to a cumulative tensor with a zero."""
        batch_size = sequence_lengths.numel()
        if cumulative_lengths is None:
            return torch.cat(
                (
                    torch.zeros(
                        1,
                        dtype=torch.int32,
                        device=sequence_lengths.device,
                    ),
                    torch.cumsum(sequence_lengths, dim=0),
                )
            )
        cumulative_lengths = cumulative_lengths.to(torch.int32)
        if cumulative_lengths.numel() == batch_size + 1:
            return cumulative_lengths
        if cumulative_lengths.numel() == batch_size:
            return torch.cat(
                (
                    torch.zeros(
                        1,
                        dtype=torch.int32,
                        device=cumulative_lengths.device,
                    ),
                    cumulative_lengths,
                )
            )
        raise RuntimeError(
            "cumulative sequence lengths must contain either one value per "
            "sequence or a leading zero plus one value per sequence"
        )

    def warmup(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
    ) -> None:
        metadata = self._normalize_dsa_metadata(metadata)
        batch_size = input_ids.shape[0]
        # Admission is in token-row buckets; _shared_graph_plan folds expanded
        # verify rows back to logical sequences.
        _, _, admission_batch = self._shared_graph_plan(
            metadata,
            batch_size,
            local_expanded=resolve_expanded_decode_metadata(metadata) is not None,
        )
        if admission_batch is None or _decode_bucket(admission_batch) > self.max_batch:
            raise ValueError("decode batch exceeds ACL graph capacity")

        # Graph capture is performed lazily by ``execute`` on the first decode
        # of each (bucket, expanded, embedding) key -- including the synthetic
        # batches driven by the C++ graph-warmup phase -- so this entry point
        # only validates capacity.  Pre-capturing here would duplicate
        # ``execute``'s first-capture path; deferring keeps a single capture
        # code path and matches the lazy-capture V3 wiring.

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        layer_synchronizer: LayerSynchronizer | None = None,
        eplb: EplbRuntimeState | None = None,
        input_batch: InputBatch | None = None,
    ) -> ModelExecutionOutput:
        metadata = self._normalize_dsa_metadata(metadata)
        batch_size = input_ids.shape[0]
        # Admission is in token-row buckets. The graph is keyed/captured at the
        # full padded token bucket.
        if self.dp_size > 1:
            # DP ranks share one captured graph shape: the bucket must be the
            # GLOBAL max token count, not this rank's local batch_size, so
            # every rank captures the same shape (the baked
            # dp_execution_token_counts agree across ranks and the fixed-shape
            # all_gather matches). The bucket is in TOKEN rows — same unit as
            # the non-DP _decode_bucket(batch_size) — because the static
            # buffers / q_cu / q_seq_lens are all per token row. The shared
            # verify layout also covers an empty DP rank.
            # Assert this rank's count equals its batch_size.
            execution_counts = getattr(metadata, "dp_execution_token_counts", None)
            if execution_counts is None or len(execution_counts) != self.dp_size:
                raise RuntimeError(
                    "DP decode step requires valid dp_execution_token_counts "
                    f"(got {execution_counts!r}, expected length {self.dp_size})."
                )
            this_rank_count = int(execution_counts[self.dp_rank])
            if this_rank_count != batch_size:
                raise RuntimeError(
                    f"dp_execution_token_counts[{self.dp_rank}]={this_rank_count} "
                    f"!= local batch_size {batch_size}; the DP graph shape is fixed "
                    "by the shared counts so they must agree per rank."
                )
            max_global_tokens = max(int(c) for c in execution_counts)
            padded_batch_size = _decode_bucket(max_global_tokens)
        else:
            padded_batch_size = _decode_bucket(batch_size)
        expanded_view = resolve_expanded_decode_metadata(metadata)
        is_expanded, verify_width, admission_batch = self._shared_graph_plan(
            metadata,
            batch_size,
            local_expanded=expanded_view is not None,
        )
        if admission_batch is None or _decode_bucket(admission_batch) > self.max_batch:
            raise ValueError("decode batch exceeds ACL graph capacity")

        kpool_query_lens = self._graph_kpool_query_lens(
            metadata,
            batch_size,
            padded_batch_size,
            verify_width,
        )
        token_capacity_bucket = self._token_capacity_bucket(metadata)
        graph_key = self._graph_key(
            padded_batch_size,
            is_expanded,
            input_embedding,
            kpool_query_lens,
            verify_width=verify_width,
            is_dflash_proposal=bool(getattr(metadata, "is_dflash_proposal", False)),
            token_capacity_bucket=token_capacity_bucket,
        )
        entry = self._graphs.get(graph_key)
        first_capture = entry is None
        if first_capture:
            entry = self._allocate_entry(
                padded_batch_size,
                input_ids,
                positions,
                metadata,
                input_batch,
                verify_width=verify_width,
            )
            entry.static_metadata.dsa_graph_token_capacity = token_capacity_bucket * int(
                getattr(self.attention_backend, "graph_token_capacity_granularity", 0) or 0
            )
            entry.eplb = self._allocate_graph_eplb_state(eplb, padded_batch_size)
        entry_eplb = getattr(entry, "eplb", None)
        if not first_capture and (entry_eplb is None) != (eplb is None):
            raise RuntimeError("EPLB state changed after ACL graph capture")
        if not first_capture and eplb is not None and entry_eplb is not None:
            if entry_eplb.expert_load_data.data_ptr() != eplb.expert_load_data.data_ptr():
                raise RuntimeError("EPLB expert-load tensor changed after ACL graph capture")
            if (entry_eplb.decode_token_mask is None) != (eplb.decode_token_mask is None):
                raise RuntimeError("EPLB decode-mask availability changed after ACL graph capture")
            entry_eplb.is_graph_warmup = eplb.is_graph_warmup
        self._fill_graph_eplb_decode_mask(entry, eplb, metadata)

        if self._stream is None:
            self._stream = torch.npu.Stream(device=input_ids.device)
            self._update_stream = torch.npu.Stream(device=input_ids.device, priority=-1)
            self._replay_done_event = torch.npu.Event()

        capture_metadata = None
        if first_capture and any(
            getattr(cache, "conv", None) is not None and getattr(cache, "ssm", None) is not None
            for cache in self.layer_caches
        ):
            # Both KDA checkpoint tables and model-specific metadata builders
            # must see padding slots before any warmup forward can run.
            capture_metadata = _CaptureAttentionMetadata(metadata)
        self._fill_entry(
            entry,
            input_ids,
            positions,
            capture_metadata if capture_metadata is not None else metadata,
            batch_size,
            input_embedding,
            input_batch,
            verify_width=verify_width,
            kpool_query_lens=kpool_query_lens,
            local_expanded=expanded_view is not None,
        )

        prepare_context = ForwardContext(
            self.attention_backend,
            self.device,
            entry.static_metadata,
            self.layer_caches,
            execution_state=entry.execution_state,
            eplb=entry_eplb,
            execution_contexts=entry.execution_contexts,
        )
        with forward_context(prepare_context):
            self.attention_backend.prepare(entry.static_metadata, graph_mode=True)
            if not first_capture:
                refresh_dsa = getattr(
                    self.attention_backend,
                    "refresh_dsa_metadata_for_graph_replay",
                    None,
                )
                if refresh_dsa is not None:
                    refresh_dsa(entry.static_metadata)
            # Host-offloaded engram tables: the graph only reads the staged
            # rows, so the gather must run here -- once per step, outside the
            # captured region (replay does not re-run Python). No-op models
            # without the hook and non-offload configurations skip it: the hook
            # existence is checked here so entry.static_input_ids / positions
            # are not dereferenced at all for models without the hook (the
            # helper's own early return happens after its arguments are
            # evaluated, so it cannot protect this call site). The hook's return
            # value is checked on an offload-active model: a False on a capture
            # is refused by _engram_first_capture_gate before the graph is
            # captured or cached (fail-fast, symmetric with the replay raise
            # below), while a False on a replay is fail-fast -- the helper
            # raises instead of letting the replay re-read the previous step's
            # stale rows. Refusing (rather than caching an Engram-less graph)
            # leaves the bucket re-capturable once staging recovers.
            staged = False
            if getattr(self.model, "stage_engram_rows_outside_graph", None) is not None:
                staged = self._stage_engram_rows_outside_graph(
                    entry.static_metadata,
                    entry.static_input_ids,
                    entry.static_positions,
                    capturing=first_capture,
                )

        if first_capture:
            self._engram_first_capture_gate(staged=staged)
            self._capture(entry)
            if capture_metadata is not None:
                # Capture may still read the persistent indices. Order real
                # metadata writes after it, then rebuild every derived table
                # in the same storage before the first request replay.
                torch.npu.current_stream().wait_stream(self._stream)
                self._fill_entry(
                    entry,
                    input_ids,
                    positions,
                    metadata,
                    batch_size,
                    input_embedding,
                    input_batch,
                    verify_width=verify_width,
                    kpool_query_lens=kpool_query_lens,
                    local_expanded=expanded_view is not None,
                )
                with forward_context(prepare_context):
                    self.attention_backend.prepare(entry.static_metadata, graph_mode=True)
            # Register the entry only once the capture has actually succeeded.
            # Registering before ``_capture`` would leave a half-built entry
            # (``entry.graph is None``) cached under ``graph_key`` whenever the
            # staging gate or ``_capture`` itself raises, and the next step for
            # this bucket would take the replay branch and dereference
            # ``entry.graph`` -> ``AttributeError``. Nothing between the
            # ``_graphs.get`` lookup above and here reads the dict, so deferring
            # the write is safe.
            self._graphs[graph_key] = entry

        # Besides ordering input updates, this wait protects the output view
        # returned by the previous replay.  The graph cannot overwrite its
        # static output until consumers queued on the current stream finish.
        self._stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self._stream):
            entry.graph.replay()
            output = self._slice_output(entry.static_output, batch_size)

        with torch.npu.stream(self._update_stream):
            self._update_stream.wait_event(self._replay_done_event)
            self._update_graph_tasks(self._update_stream, entry.graph_tasks)

        self._replay_done_event.record(self._stream)

        torch.npu.current_stream().wait_stream(self._stream)
        return output

    def _stage_engram_rows_outside_graph(
        self,
        static_metadata: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        capturing: bool,
    ) -> bool:
        """Run the model's pre-graph host-offload staging pass, once per step.

        Returns the hook's result. ``False`` means nothing was staged this step
        (e.g. the SWA cache is unbound, or the model is not offloading at all).
        When the model *is* offloading, a ``False`` result is a correctness
        hazard: the captured graph reads the persistent staging block, so the
        replay re-reads whatever rows the previous step left behind. On a
        replay (``capturing=False``) that is a hard error -- the graph is
        already captured with engram, so continuing would serve the previous
        step's stale rows as a silent wrong output (the eager path skips engram
        in the same condition), hence the fail-fast below. On a capture
        (``capturing=True``) this helper still returns ``False`` (it only runs
        the staging plus the replay-side check); the caller refuses the capture
        via ``_engram_first_capture_gate`` before the graph is captured or
        cached, so no Engram-less graph is ever stored. Models without the hook,
        and non-offload configurations, are unaffected.
        """
        stage_engram = getattr(self.model, "stage_engram_rows_outside_graph", None)
        if stage_engram is None:
            return False
        staged = bool(stage_engram(self.attention_backend, static_metadata, input_ids, positions))
        if not staged:
            if not capturing and bool(getattr(self.model, "engram_offload_active", False)):
                # Replay of an engram-baked graph with nothing staged this step:
                # the captured graph would re-read the previous step's rows.
                raise RuntimeError(
                    "engram host-offload staging returned False on a replay/reused graph; "
                    "the captured graph would re-read the previous step's rows"
                )
            self._warn_engram_stage_incomplete(capturing=capturing)
        return staged

    def _warn_engram_stage_incomplete(self, *, capturing: bool) -> None:
        """Latch one warning when the staging pass staged nothing while offloading.

        Gated on ``engram_offload_active`` so non-offload and engram-off models
        (whose hook is a legitimate no-op returning ``False``) stay silent. The
        warning is latched per runner instance: a replay runs once per step and
        would otherwise flood the log, and the underlying condition (the model
        keeps offloading but cannot stage) is not per-step diagnosable. It is a
        companion diagnostic to the capture fail-fast path: this warning is
        reachable only on a capture, where ``_engram_first_capture_gate`` then
        raises before the graph is captured or cached -- on a replay
        ``_stage_engram_rows_outside_graph`` raises *before* this call, so it is
        never reached there. It names capture-vs-replay, the engram layer count
        and the graph-capacity hints so the offending model is identifiable from
        the log alone.
        """
        offload_active = bool(getattr(self.model, "engram_offload_active", False))
        if not offload_active or self._engram_stage_skip_warned:
            return
        self._engram_stage_skip_warned = True
        # Duck-typed, not imported: the runner must not depend on
        # ``xllm.python.models`` (runners -> models is a layering inversion). The
        # model exposes the accessor as a method; a model without it (e.g. an
        # eagerly-run test stub) has no engram layers to name, so fall back to the
        # empty summary rather than importing the model-side helper.
        warning_fields = getattr(self.model, "engram_offload_warning_fields", None)
        if warning_fields is not None:
            engram_layer_count, capacity_hints = warning_fields()
        else:
            engram_layer_count, capacity_hints = 0, []
        logger.warning(
            "engram host-offload staging staged nothing "
            "(stage_engram_rows_outside_graph returned False) on an offload-active "
            "model: the first capture was refused (engram staging returned False); "
            "see the accompanying error -- the bucket will re-capture once staging "
            "recovers. offload_active=%s capturing=%s engram_layers=%d "
            "graph_capacity_hints=%s",
            offload_active,
            capturing,
            engram_layer_count,
            capacity_hints,
        )

    def _engram_first_capture_gate(self, *, staged: bool) -> None:
        """Refuse a first capture that cannot stage while engram is offloading.

        ``execute`` registers the entry in ``self._graphs`` only after a
        successful ``_capture``, so this gate runs before anything is cached:
        refusing simply means never registering. If the first capture cannot
        stage while the model actively offloads engram rows, the graph being
        captured would read the persistent staging block that a replay never
        refreshes -- an Engram-less graph latched forever, silently skipping
        engram on every later step. This gate raises before ``_capture`` runs,
        so the next step re-runs the first capture for this bucket (with a fresh
        staging attempt) and a recovered staging pass captures an engram-baked
        graph.

        Chosen over silently falling back to eager decode: this runner has no
        eager decode path (the ACL graph *is* the decode kernel), and a
        fail-fast refusal is symmetric with the replay-side raise in
        ``_stage_engram_rows_outside_graph`` -- a staging outage surfaces the
        same way whether it hits the first capture or a later replay. Refusing
        before the capture (rather than capturing and caching) is what keeps the
        bucket re-capturable: nothing is written to ``self._graphs`` on this
        path.
        """
        if staged or not bool(getattr(self.model, "engram_offload_active", False)):
            return
        raise RuntimeError(
            "engram host-offload staging returned False on the first ACL graph "
            "capture: refusing to cache an Engram-less graph (it would replay "
            "without engram forever). Likely the SWA group metadata was "
            "unavailable, the engram ensure_cache() failed, or the host staging "
            "table was not ready. The graph was not captured or cached, so this "
            "bucket re-captures on a later step once staging recovers; to disable "
            "engram host-offload and use the HBM path instead, set "
            "engram_config.cpu_offload=false. This is a startup/restart "
            "configuration (it is not switchable online): restart the server with "
            '--engram-config \'{"cpu_offload":false,"dp_shared_memory":false}\'.'
        )

    def _allocate_graph_eplb_state(
        self,
        eplb: EplbRuntimeState | None,
        padded_batch_size: int,
    ) -> EplbRuntimeState | None:
        if eplb is None:
            return None
        decode_token_mask = None
        if eplb.decode_token_mask is not None:
            decode_token_mask = torch.zeros(
                padded_batch_size * self.dp_size,
                dtype=eplb.decode_token_mask.dtype,
                device=eplb.decode_token_mask.device,
            )
        return EplbRuntimeState(
            expert_load_data=eplb.expert_load_data,
            decode_token_mask=decode_token_mask,
            is_graph_warmup=eplb.is_graph_warmup,
        )

    def _fill_graph_eplb_decode_mask(
        self,
        entry: _DecodeGraphEntry,
        eplb: EplbRuntimeState | None,
        metadata: AttentionMetadata,
    ) -> None:
        entry_eplb = getattr(entry, "eplb", None)
        if entry_eplb is None or entry_eplb.decode_token_mask is None:
            return
        if eplb is None or eplb.decode_token_mask is None:
            raise RuntimeError("EPLB decode-mask state is missing for a captured graph")

        destination = entry_eplb.decode_token_mask
        source = eplb.decode_token_mask.reshape(-1)
        destination.zero_()
        if self.dp_size == 1:
            if source.numel() > entry.batch_size:
                raise RuntimeError("EPLB decode mask exceeds graph bucket capacity")
            destination[: source.numel()].copy_(source)
            return

        raw_counts = tuple(metadata.raw_dp_execution_token_counts)
        destination_slices = _padded_rank_slices(raw_counts, self.dp_size, entry.batch_size)
        if source.numel() != sum(raw_counts):
            raise RuntimeError("EPLB decode mask does not match DP token counts")
        source_begin = 0
        for destination_begin, count in destination_slices:
            destination[destination_begin : destination_begin + count].copy_(
                source[source_begin : source_begin + count]
            )
            source_begin += count

    @staticmethod
    def _graph_key(
        padded_batch_size: int,
        is_expanded: bool,
        input_embedding: torch.Tensor | None,
        kpool_query_lens: tuple[int, ...] = (),
        verify_width: int = 1,
        is_dflash_proposal: bool = False,
        token_capacity_bucket: int = 0,
    ) -> _GraphKey:
        """Fix graph-captured attention mode as well as execution shape.

        ``token_capacity_bucket`` partitions graphs by the max context length
        for backends whose forward gathers grow with it (V4.1 CSA2
        committed-row capacity); it stays 0 for backends that do not opt in,
        keeping the legacy key space unchanged.
        """
        if input_embedding is None:
            return (
                padded_batch_size,
                is_expanded,
                verify_width,
                None,
                None,
                None,
                kpool_query_lens,
                is_dflash_proposal,
                token_capacity_bucket,
            )
        return (
            padded_batch_size,
            is_expanded,
            verify_width,
            input_embedding.dtype,
            input_embedding.device,
            tuple(input_embedding.shape[1:]),
            kpool_query_lens,
            is_dflash_proposal,
            token_capacity_bucket,
        )

    def _token_capacity_bucket(self, metadata: AttentionMetadata) -> int:
        """Bucket index over the batch's max context length (0 when disabled).

        Backends that gather per-context rows (the V4.1 CSA2 indexer /
        attention) need a shape-stable capacity inside the captured graph, so
        each bucket fixes the capacity at its upper bound; crossing into a new
        bucket captures a fresh graph lazily.

        Every DP rank in a communication group must derive the same bucket or
        their capture/replay schedules desynchronize the HCCL collectives, so
        DP uses the scheduler's host-planned global KV maximum
        (``dp_global_kv_max_seq_lens``, the same source the C++ MLA capture
        bucket uses) instead of this rank's local lengths.
        """
        granularity = int(getattr(self.attention_backend, "graph_token_capacity_granularity", 0) or 0)
        if not granularity:
            return 0
        if self.dp_size > 1:
            global_lengths = getattr(metadata, "dp_global_kv_max_seq_lens", None)
            if global_lengths is None or not global_lengths:
                # Conservative DP contract: without the scheduler's host-planned
                # global lengths, do not derive a per-rank bucket from this
                # rank's local lengths (0 == not graph-admissible), otherwise
                # ranks could pick different buckets and desync HCCL.
                return 0
            max_ctx = max((int(value) for value in global_lengths), default=0)
            return (max_ctx + granularity - 1) // granularity
        expanded = resolve_expanded_decode_metadata(metadata)
        host_values = (
            expanded.kv_seq_lens_host_values
            if expanded is not None
            else getattr(metadata, "kv_seq_lens_host_values", None)
        )
        if not host_values:
            kv_seq_lens = expanded.kv_seq_lens if expanded is not None else metadata.kv_seq_lens
            host_values = kv_seq_lens.tolist() if kv_seq_lens is not None else []
        max_ctx = max((int(value) for value in host_values), default=0)
        return (max_ctx + granularity - 1) // granularity

    def _padded_kpool_query_lens(
        self,
        metadata: AttentionMetadata,
        num_tokens: int,
        padded_num_tokens: int,
        verify_width: int = 1,
    ) -> tuple[int, ...]:
        """Return uniform KPool request spans for the complete groups in a bucket.

        A tail shorter than the request width stays in the tensor as padding and
        is not another request. Expanded verify with no metadata spans uses
        ``verify_width``, so an empty rank and a busy rank share one graph key.
        """
        query_lens = tuple(int(length) for length in getattr(metadata, "kpool_query_lens", ()))
        if not query_lens:
            if not self._uses_compressed_kpool_tail():
                return ()
            if verify_width > 1:
                return _complete_kpool_groups(padded_num_tokens, verify_width)
            query_lens = (1,) * num_tokens
        if any(length <= 0 for length in query_lens):
            raise RuntimeError(f"KPool graph query spans must be positive: {query_lens}")
        covered_tokens = sum(query_lens)
        if covered_tokens != num_tokens:
            raise RuntimeError(
                f"KPool graph query spans must cover every input token: covered={covered_tokens}, tokens={num_tokens}"
            )
        query_width = query_lens[0]
        if any(length != query_width for length in query_lens[1:]):
            raise RuntimeError(f"KPool ACL graph requires uniform request spans: {query_lens}")
        pad_rows = padded_num_tokens - num_tokens
        full_groups = pad_rows // query_width
        return query_lens + (query_width,) * full_groups

    def _uses_compressed_kpool_tail(self) -> bool:
        return any(
            cache.kpool_tail is not None and cache.kpool_tail.dim() == 4 and cache.kpool_tail.dtype == torch.bfloat16
            for cache in self.layer_caches
        )

    def _canonical_kpool_query_lens(self, padded_num_tokens: int, verify_width: int) -> tuple[int, ...]:
        """Complete-group spans every DP rank uses for one verify graph bucket.

        ``padded // width`` groups of ``width`` cover the rows that fill a
        request. A shorter tail is bucket padding and is omitted here.
        """
        if not self._uses_compressed_kpool_tail():
            return ()
        return _complete_kpool_groups(padded_num_tokens, verify_width)

    def _graph_kpool_query_lens(
        self,
        metadata: AttentionMetadata,
        num_tokens: int,
        padded_num_tokens: int,
        verify_width: int,
    ) -> tuple[int, ...]:
        if self.dp_size > 1 and verify_width > 1 and self._uses_compressed_kpool_tail():
            return _complete_kpool_groups(padded_num_tokens, verify_width)
        return self._padded_kpool_query_lens(
            metadata,
            num_tokens,
            padded_num_tokens,
            verify_width=verify_width,
        )

    def _require_shared_kpool_layout(
        self,
        metadata: AttentionMetadata,
        num_tokens: int,
        padded_num_tokens: int,
        verify_width: int,
    ) -> None:
        if self.dp_size <= 1 or verify_width <= 1 or not self._uses_compressed_kpool_tail():
            return
        local = self._padded_kpool_query_lens(
            metadata,
            num_tokens,
            padded_num_tokens,
            verify_width=verify_width,
        )
        shared = self._canonical_kpool_query_lens(padded_num_tokens, verify_width)
        if local != shared:
            raise RuntimeError(
                f"DP KPool graph spans do not match the shared bucket layout: local={local}, shared={shared}"
            )

    def _allocate_entry(
        self,
        padded_batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_batch: InputBatch | None = None,
        verify_width: int = 1,
    ) -> _DecodeGraphEntry:
        device = input_ids.device
        (
            _,
            _,
            _,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) = self._decode_metadata(metadata)
        if self._paged_kv_indices_buffer is None:
            page_size = self.attention_backend.page_size
            max_blocks_per_sequence = (self.max_model_len + page_size - 1) // page_size
            # Capacity in token rows: speculative verify captures N*width
            # independent paging rows, so size the flat paged_kv indices from
            # the sequence budget and verify width.
            self._paged_kv_indices_buffer = torch.zeros(
                self.sequence_batch_capacity * self.num_decoding_tokens * max_blocks_per_sequence,
                dtype=paged_kv_indices.dtype,
                device=device,
            )
            self._max_blocks_per_sequence = max_blocks_per_sequence

        static_block_table = torch.zeros(
            padded_batch_size,
            self._max_blocks_per_sequence,
            dtype=torch.int32,
            device=device,
        )

        entry = _DecodeGraphEntry()
        entry.batch_size = padded_batch_size
        entry.graph = None
        entry.static_output = None
        entry.graph_tasks = []
        entry.execution_state = AclGraphExecutionState({})
        entry.static_input_ids = torch.zeros(padded_batch_size, dtype=input_ids.dtype, device=device)
        entry.static_positions = torch.zeros(padded_batch_size, dtype=torch.int32, device=device)
        entry.is_padding = torch.ones(
            padded_batch_size,
            dtype=torch.bool,
            device=device,
        )
        entry.execution_contexts = {}
        if self.execution_metadata_builders:
            graph_input_batch = self._build_padded_input_batch(entry, input_batch)
            for builder in self.execution_metadata_builders:
                metadata_type = builder.metadata_type
                if metadata_type in entry.execution_contexts:
                    raise RuntimeError(f"duplicate execution metadata builder for {metadata_type.__name__}")
                entry.execution_contexts[metadata_type] = builder.allocate_persistent(
                    graph_input_batch,
                    metadata,
                )
        entry.static_input_embedding = None
        kpool_query_lens = self._graph_kpool_query_lens(
            metadata,
            input_ids.numel(),
            padded_batch_size,
            verify_width,
        )
        needs_accepted_tokens = getattr(metadata, "num_accepted_tokens", None) is not None or (
            self.num_decoding_tokens > 1
            and any(
                getattr(cache, "conv", None) is not None and getattr(cache, "ssm", None) is not None
                for cache in self.layer_caches
            )
        )
        entry.static_metadata = _StaticAttentionMetadata(
            slot_mapping=torch.zeros(
                padded_batch_size,
                dtype=torch.int32,
                device=device,
            ),
            paged_kv_indptr=torch.zeros(
                padded_batch_size + 1,
                dtype=paged_kv_indptr.dtype,
                device=device,
            ),
            paged_kv_indices=self._paged_kv_indices_buffer,
            paged_kv_last_page_len=torch.zeros(
                padded_batch_size,
                dtype=paged_kv_last_page_len.dtype,
                device=device,
            ),
            kv_cu_seq_lens=torch.zeros(
                padded_batch_size + 1,
                dtype=torch.int32,
                device=device,
            ),
            # paged_kv_indptr_host / paged_kv_last_page_len_host are consumed
            # only by the CUDA flashinfer backend (see attention/flashinfer.py);
            # the ACL decode path never reads them, so leaving them None keeps
            # capture allocation-only overhead off the CPU.
            kv_seq_lens_host_values=[1] * padded_batch_size,
            # DSA consumes live scheduler slots on the host before capture.
            new_cache_slots_host_values=[],
            block_table=static_block_table,
            # KDA (linear-attention) decode reads per-sequence conv/ssm state
            # slots from these buffers.  Static so the captured graph indexes a
            # fixed address; contents are refreshed by _fill_entry each step.
            linear_state_indices=torch.zeros(padded_batch_size, dtype=torch.int64, device=device),
            num_accepted_tokens=(
                torch.ones(
                    padded_batch_size,
                    dtype=torch.int32,
                    device=device,
                )
                if needs_accepted_tokens
                else None
            ),
            has_initial_state=torch.zeros(padded_batch_size, dtype=torch.int32, device=device),
            # DP layers read uniform padded counts off the captured metadata;
            # variable per-rank counts cannot be baked into a graph.
            # Sequence-parallel model layers also consume the execution row
            # count when DP=1. Keep the single-DP graph metadata explicit so
            # those layers use the same authority as DP MoE collectives.
            dp_execution_token_counts=(padded_batch_size,) * self.dp_size,
            dp_is_decode=tuple([1] * self.dp_size) if self.dp_size > 1 else (),
            is_dummy=bool(getattr(metadata, "is_dummy", False)),
            kpool_query_lens=kpool_query_lens,
            kpool_query_lens_device=(
                torch.tensor(kpool_query_lens, dtype=torch.int32, device=device) if kpool_query_lens else None
            ),
            # One mask slot per padded row across all DP ranks. aclnnMegaMoe
            # dispatches the fixed graph shape over EP, so padded lanes must be
            # marked inactive or they are routed as real tokens.
            mega_moe_token_mask=(
                torch.zeros(
                    padded_batch_size * self.dp_size,
                    dtype=torch.int8,
                    device=device,
                )
                if self._enable_mega_moe_token_mask
                else None
            ),
        )
        entry.static_metadata.q_cu_host_values = [0] * (padded_batch_size + 1)
        expanded_view = resolve_expanded_decode_metadata(metadata, block_size=self.attention_backend.page_size)
        local_expanded = expanded_view is not None
        group_expanded = verify_width > 1 or local_expanded
        entry.static_metadata.is_spec_verify = group_expanded
        entry.static_metadata.is_dflash_proposal = bool(getattr(metadata, "is_dflash_proposal", False))
        entry.kv_seq_lens_delta = torch.empty(padded_batch_size, dtype=torch.int32, device=device)
        # The graph metadata update writes per-sequence KV lengths into this
        # buffer.  MLA/SFA consumes the same stable buffer as its key lengths.
        entry.static_metadata.kv_seq_lens = entry.kv_seq_lens_delta
        if group_expanded:
            # Spec-verify rows per logical sequence: the static q_cu holds
            # GROUP boundaries [0, w, 2w, ...] so the in-graph KDA verify
            # grouping derives the sequence count without host syncs.
            # Derive the token-row count from the EXPANDED view (N*width),
            # not the top-level metadata.kv_seq_lens which stays at the logical
            # sequence count (N) for typed expanded verify — otherwise
            # spec_width stays 1 and q_seq_lens is built as N*w single-token
            # groups, breaking the KDA recurrent-state chain.
            # An empty rank has no expanded view. It still uses the shared
            # verify width so q_cu matches the busy rank's graph.
            w = verify_width if verify_width > 1 else 1
            if local_expanded and w == 1:
                expanded_kv = expanded_view.kv_seq_lens
                src_rows = expanded_kv.shape[0] if expanded_kv is not None else 0
                src_lsi = getattr(metadata, "linear_state_indices", None)
                src_seqs = src_lsi.shape[0] if src_lsi is not None else 0
                if src_seqs > 0 and src_rows > src_seqs and src_rows % src_seqs == 0:
                    w = src_rows // src_seqs
            # q_cu stays PER-ROW over the whole bucket, including padding.
            # q_seq_lens lists only complete verify groups. A tail that does
            # not fill another group stays padding and is not a fake sequence.
            n_groups = padded_batch_size // w
            if n_groups > 0:
                entry.static_metadata.q_cu_seq_lens = torch.arange(
                    0, padded_batch_size + 1, 1, dtype=torch.int32, device=device
                )
                entry.static_metadata.q_cu_host_values = list(range(padded_batch_size + 1))
                entry.static_metadata.q_seq_lens = torch.full((n_groups,), w, dtype=torch.int32, device=device)
                entry.static_metadata.spec_group_width = w
            entry.static_metadata.expanded_decode_metadata = ExpandedDecodeMetadata(
                kv_seq_lens=entry.kv_seq_lens_delta,
                block_table=entry.static_metadata.block_table,
                paged_kv_indptr=entry.static_metadata.paged_kv_indptr,
                paged_kv_indices=entry.static_metadata.paged_kv_indices,
                paged_kv_last_page_len=(entry.static_metadata.paged_kv_last_page_len),
                paged_attention_tiling_data=None,
                kv_seq_lens_host=None,
                kv_seq_lens_host_values=(
                    entry.static_metadata.kv_seq_lens_host_values
                    if getattr(
                        self.attention_backend,
                        "requires_host_kv_lengths",
                        not getattr(self.attention_backend, "is_mla", False),
                    )
                    else None
                ),
            )
        entry.static_metadata.multi_block_tables = self._build_static_multi_block_tables(
            padded_batch_size,
            device,
        )
        entry.static_metadata.dsa_graph_block_table_cols = self._max_blocks_per_sequence
        entry.static_metadata.dsa_graph_mode = True
        return entry

    def _build_static_multi_block_tables(
        self,
        padded_batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        """Build stable SWA/C4/C128 manager block tables for graph capture.

        Keep every manager's dimensions (and therefore its packed DSA metadata
        views) fixed across decodes of the same bucket. The first manager uses
        the runner-wide maximum block column count; compressed managers use the
        ``max_model_len``-derived upper bound. Unused entries point at reserved
        block 0.
        """
        max_swa_cols = self._max_blocks_per_sequence
        group_infos = getattr(self.attention_backend, "group_infos", None)
        if group_infos is None:
            # Test stubs and non-DSA backends only build a sliding-window row.
            manager_infos = [None]
        else:
            manager_infos = list(group_infos)
        tables: list[torch.Tensor] = []
        for group_info in manager_infos:
            if group_info is not None and getattr(group_info, "cache_type", None) == DSA_CACHE_TOKEN:
                ratio = max(int(getattr(group_info, "ratio", 1)), 1)
                block_size = max(int(getattr(group_info, "block_size", 1)), 1)
                compressed_block_size = ratio * block_size
                max_cols = max(
                    1,
                    (self.max_model_len + compressed_block_size - 1) // compressed_block_size,
                )
            else:
                max_cols = max_swa_cols
            tables.append(
                torch.zeros(
                    (padded_batch_size, max_cols),
                    dtype=torch.int32,
                    device=device,
                )
            )
        return tuple(tables)

    def _pinned_stage(self, source: torch.Tensor, kind: str) -> torch.Tensor:
        """Return a reusable pinned CPU view for asynchronous graph refreshes."""
        if source.device.type != "cpu" or source.is_pinned():
            return source
        key = (kind, source.dtype, source.shape)
        staged = self._pinned_block_table_staging.get(key)
        if staged is None:
            try:
                staged = torch.empty(source.shape, dtype=source.dtype, pin_memory=True)
                self._pinned_block_table_staging[key] = staged
            except RuntimeError:
                return source
        staged.copy_(source)
        return staged

    def _copy_block_table(self, target: torch.Tensor, source: torch.Tensor) -> None:
        if target.device.type == source.device.type:
            target.copy_(source)
            return
        staged = self._pinned_stage(source, "block_table")
        non_blocking = staged.is_pinned() and target.device.type in ("cuda", "npu")
        target.copy_(staged, non_blocking=non_blocking)

    def _fill_entry(
        self,
        entry: _DecodeGraphEntry,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        batch_size: int,
        input_embedding: torch.Tensor | None,
        input_batch: InputBatch | None = None,
        verify_width: int = 1,
        kpool_query_lens: tuple[int, ...] | None = None,
        local_expanded: bool | None = None,
    ) -> None:
        padded_batch_size = entry.batch_size
        static_metadata = entry.static_metadata
        static_metadata.is_dummy = bool(getattr(metadata, "is_dummy", False))
        if kpool_query_lens is None:
            kpool_query_lens = self._graph_kpool_query_lens(
                metadata,
                batch_size,
                padded_batch_size,
                verify_width,
            )
        if kpool_query_lens != static_metadata.kpool_query_lens:
            raise RuntimeError(
                "KPool query-span layout changed for an existing ACL graph: "
                f"captured={static_metadata.kpool_query_lens}, current={kpool_query_lens}"
            )
        (
            block_table,
            kv_seq_lens,
            kv_seq_lens_host_values,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) = self._decode_metadata(metadata)
        self._fill_graph_dsa_positions(entry, positions)
        if batch_size != block_table.shape[0]:
            raise RuntimeError("ACL graph decode batch size must match metadata sequences")
        slot_mapping = self._effective_slot_mapping(
            metadata,
            block_table.shape[0],
            input_ids.device,
        )
        self._validate_decode_token_layout(
            input_ids,
            positions,
            slot_mapping,
            block_table.shape[0],
        )
        is_expanded = (
            local_expanded if local_expanded is not None else resolve_expanded_decode_metadata(metadata) is not None
        )
        cumulative_kv_seq_lens = self._cumulative_lengths(
            kv_seq_lens,
            None if is_expanded else metadata.kv_cu_seq_lens,
        )
        graph_positions = positions.to(torch.int32).contiguous()
        kernels.update_decode_graph_metadata(
            input_ids,
            graph_positions,
            slot_mapping,
            cumulative_kv_seq_lens,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            entry.static_input_ids,
            entry.static_positions,
            static_metadata.slot_mapping,
            static_metadata.kv_cu_seq_lens,
            entry.kv_seq_lens_delta,
            static_metadata.paged_kv_indptr,
            static_metadata.paged_kv_indices,
            static_metadata.paged_kv_last_page_len,
            padded_batch_size,
        )
        # linear_state_indices is filled once below (the robust
        # repeat_interleave slice path), together with has_initial_state, so
        # there is no first redundant fill here.
        self._fill_host_metadata(entry, kv_seq_lens_host_values, batch_size)
        self._fill_new_cache_slots_host_metadata(entry, metadata, batch_size)

        if input_embedding is not None:
            if input_embedding.shape[0] != batch_size:
                raise ValueError("ACL graph input_embedding token count must match input_ids")
            if entry.static_input_embedding is None:
                entry.static_input_embedding = torch.zeros(
                    entry.batch_size,
                    input_embedding.shape[-1],
                    dtype=input_embedding.dtype,
                    device=input_embedding.device,
                )
            elif entry.static_input_embedding.shape[1:] != input_embedding.shape[1:]:
                raise ValueError("ACL graph input_embedding shape changed for a graph bucket")
            entry.static_input_embedding[:batch_size].copy_(input_embedding)
            if entry.batch_size > batch_size:
                entry.static_input_embedding[batch_size:].zero_()
        elif entry.static_input_embedding is not None:
            entry.static_input_embedding.zero_()

        if static_metadata.block_table is not None:
            src_bt = block_table
            copy_cols = min(
                src_bt.shape[1],
                static_metadata.block_table.shape[1],
            )
            self._copy_block_table(
                static_metadata.block_table[:batch_size, :copy_cols],
                src_bt[:batch_size, :copy_cols],
            )
            if padded_batch_size > batch_size:
                static_metadata.block_table[batch_size:].zero_()

        # Padded lanes must remain valid inputs for sparse MLA tiling.  Their
        # token and slot mapping are dummy values, so one KV token is safe.
        if padded_batch_size > batch_size:
            static_metadata.slot_mapping[batch_size:].fill_(-1)
            entry.kv_seq_lens_delta[batch_size:].fill_(1)

        # KDA (linear-attention) static state slots.  Padded lanes point at
        # slot 0, which the linear-state block manager reserves as its padding
        # slot (block_manager_impl padding_block_), so their conv/ssm writes
        # never touch a live sequence's state.  A spec-verify
        # batch carries one slot id per logical sequence: expand per row.
        if static_metadata.linear_state_indices is not None:
            src_idx = getattr(metadata, "linear_state_indices", None)
            if src_idx is not None:
                if src_idx.numel() >= batch_size:
                    static_metadata.linear_state_indices[:batch_size].copy_(src_idx[:batch_size])
                else:
                    width = batch_size // src_idx.numel()
                    static_metadata.linear_state_indices[:batch_size].copy_(
                        src_idx.repeat_interleave(width)[:batch_size]
                    )
            if padded_batch_size > batch_size:
                static_metadata.linear_state_indices[batch_size:].zero_()
        src_accepted = getattr(metadata, "num_accepted_tokens", None)
        static_accepted = static_metadata.num_accepted_tokens
        if static_accepted is not None:
            static_accepted.fill_(1)
            if not static_metadata.is_dummy and src_accepted is None:
                raise RuntimeError("accepted-token metadata is missing during ACL graph replay")
            if not static_metadata.is_dummy:
                if src_accepted.ndim != 1 or not 0 < src_accepted.numel() <= batch_size:
                    raise ValueError("accepted-token metadata must contain one count per logical sequence")
                static_accepted[: src_accepted.numel()].copy_(src_accepted.to(torch.int32))
        if static_metadata.has_initial_state is not None:
            src_his = getattr(metadata, "has_initial_state", None)
            if src_his is None:
                # The eager runtime omits the validity mask at decode
                # (has_initial_state is None), which execute_linear reads as
                # "leave the gathered slot state untouched". Reproduce that
                # through the static buffer by marking every lane warm —
                # torch.where(warm, state, zeros) then passes the state
                # through unchanged.
                static_metadata.has_initial_state.fill_(1)
            else:
                if not isinstance(src_his, torch.Tensor):
                    src_his = torch.tensor(
                        src_his,
                        dtype=static_metadata.has_initial_state.dtype,
                        device=static_metadata.has_initial_state.device,
                    )
                # has_initial_state is sequence-scoped (N entries) while the
                # static buffer is per token ROW (N*width under MTP expanded
                # verify); expand each sequence's mask across its spec rows
                # before copying, otherwise copy_ hits a shape mismatch.
                if src_his.numel() < batch_size and src_his.numel() > 0 and batch_size % src_his.numel() == 0:
                    width = batch_size // src_his.numel()
                    src_his = src_his.repeat_interleave(width)
                static_metadata.has_initial_state[:batch_size].copy_(src_his[:batch_size])
                if padded_batch_size > batch_size:
                    static_metadata.has_initial_state[batch_size:].zero_()
        self._fill_dsa_block_tables(
            static_metadata,
            metadata,
            block_table,
            batch_size,
        )
        self._fill_mega_moe_token_mask(entry, metadata, batch_size)
        if self.execution_metadata_builders:
            graph_input_batch = self._build_padded_input_batch(entry, input_batch)
            for builder in self.execution_metadata_builders:
                persistent_metadata = entry.execution_contexts.get(builder.metadata_type)
                if persistent_metadata is None:
                    raise RuntimeError(f"missing persistent execution metadata for {builder.metadata_type.__name__}")
                builder.update_persistent(
                    persistent_metadata,
                    graph_input_batch,
                    metadata,
                )

    def _build_padded_input_batch(
        self,
        entry: _DecodeGraphEntry,
        input_batch: InputBatch | None,
    ) -> InputBatch:
        if input_batch is None:
            raise RuntimeError("execution metadata builders require upstream InputBatch metadata")
        entry.is_padding.fill_(True)
        entry.is_padding[: input_batch.num_tokens].fill_(False)
        return input_batch.bind_graph_inputs(
            entry.static_input_ids,
            entry.static_positions,
            entry.is_padding,
        )

    def _fill_dsa_block_tables(
        self,
        static_metadata: _StaticAttentionMetadata,
        metadata: AttentionMetadata,
        block_table: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Refresh every stable DSA manager table for the current decode."""
        static_tables = list(static_metadata.multi_block_tables)
        if not static_tables:
            return
        source_tables = list(getattr(metadata, "multi_block_tables", ()) or ())
        if not source_tables:
            if getattr(metadata, "is_dummy", False):
                for target in static_tables:
                    target.zero_()
                return
            if getattr(self.attention_backend, "group_infos", None) is not None:
                raise RuntimeError("DeepSeek-V4 ACL graph requires all DSA manager block tables")
            source_tables = [block_table]
        if len(source_tables) < len(static_tables):
            raise RuntimeError(f"ACL graph DSA manager count changed: {len(source_tables)} != {len(static_tables)}")
        if len(source_tables) > len(static_tables):
            # MTP draft inputs inherit the target's SWA/C4/C128 tables, while
            # the draft graph registers only its own SWA prefix.
            source_tables = source_tables[: len(static_tables)]

        for manager_id, (target, source) in enumerate(zip(static_tables, source_tables, strict=True)):
            # Every graph padding row points at permanently reserved block 0.
            target.zero_()
            if source is None:
                continue
            if source.dim() != 2:
                raise RuntimeError(f"ACL graph DSA manager {manager_id} block table must be two-dimensional")
            source_rows = int(source.shape[0])
            if source_rows != batch_size:
                if source_rows <= 0 or batch_size % source_rows != 0:
                    raise RuntimeError(
                        f"ACL graph DSA manager {manager_id} row count "
                        f"{source_rows} does not match batch size {batch_size}"
                    )
                source = source.repeat_interleave(batch_size // source_rows, dim=0)
            if source.shape[1] > target.shape[1]:
                raise RuntimeError(
                    f"ACL graph DSA manager {manager_id} requires {source.shape[1]} "
                    f"columns, capacity is {target.shape[1]}"
                )
            self._copy_block_table(target[:batch_size, : source.shape[1]], source[:batch_size])

    def _fill_graph_dsa_positions(
        self,
        entry: _DecodeGraphEntry,
        positions: torch.Tensor,
    ) -> None:
        """Keep a stable DSA position tensor synced with the static bucket."""
        static_metadata = entry.static_metadata
        padded_batch_size = entry.batch_size
        if static_metadata.dsa_positions is None:
            static_metadata.dsa_positions = torch.zeros(
                padded_batch_size,
                dtype=torch.int64,
                device=entry.static_positions.device,
            )
        graph_positions = positions.to(device=entry.static_positions.device, dtype=torch.int64)
        copy_count = min(int(graph_positions.numel()), padded_batch_size)
        static_metadata.dsa_positions[:copy_count].copy_(graph_positions[:copy_count])
        if padded_batch_size > copy_count:
            static_metadata.dsa_positions[copy_count:].zero_()

    def _fill_mega_moe_token_mask(
        self,
        entry: _DecodeGraphEntry,
        metadata: AttentionMetadata,
        batch_size: int,
    ) -> None:
        mask = entry.static_metadata.mega_moe_token_mask
        if mask is None:
            return
        mask.zero_()
        if self.dp_size == 1:
            mask[:batch_size].fill_(1)
            return

        padded_batch_size = entry.batch_size
        rank_slices = _padded_rank_slices(
            metadata.dp_execution_token_counts,
            self.dp_size,
            padded_batch_size,
        )
        for start, count in rank_slices:
            mask[start : start + count].fill_(1)

    def _fill_host_metadata(
        self,
        entry: _DecodeGraphEntry,
        kv_seq_lens: list[int] | None,
        batch_size: int,
    ) -> None:
        if not getattr(
            self.attention_backend,
            "requires_host_kv_lengths",
            not getattr(self.attention_backend, "is_mla", False),
        ):
            return
        if kv_seq_lens is None:
            raise RuntimeError("decode ACL graph requires scheduler-provided host KV lengths")
        if len(kv_seq_lens) != batch_size:
            raise RuntimeError("decode ACL graph requires per-sequence host KV lengths")

        padded_batch_size = entry.batch_size
        static_metadata = entry.static_metadata
        static_kv_seq_lens = static_metadata.kv_seq_lens_host_values
        if static_kv_seq_lens is None:
            raise RuntimeError("decode ACL graph host KV buffer is missing")
        static_kv_seq_lens[:batch_size] = kv_seq_lens
        if padded_batch_size > batch_size:
            static_kv_seq_lens[batch_size:] = [1] * (padded_batch_size - batch_size)

    @staticmethod
    def _fill_new_cache_slots_host_metadata(
        entry: _DecodeGraphEntry,
        metadata: AttentionMetadata,
        batch_size: int,
    ) -> None:
        """Copy scheduler-resolved physical DSA slots into graph metadata."""
        source_slots = getattr(metadata, "new_cache_slots_host_values", None)
        target_slots = entry.static_metadata.new_cache_slots_host_values
        if target_slots is None:
            return
        if source_slots is None:
            entry.static_metadata.new_cache_slots_host_values = []
            return
        if not source_slots and getattr(metadata, "is_dummy", False):
            # Empty-DP dummy rows have no scheduler-owned cache slot. Their
            # DSA tables point entirely at reserved block 0, so let the DSA
            # builder derive a safe slot from that table.
            entry.static_metadata.new_cache_slots_host_values = []
            return
        if len(source_slots) < batch_size:
            if not getattr(metadata, "multi_block_tables", ()):
                raise RuntimeError("decode ACL graph requires one host cache slot per token")
            # DSA multi-manager exports publish no flat scheduler slots (the
            # C++ builder skips ``new_token_slot_ids`` for composite-block
            # sequences). Both the C++ and Python DSA builders fall back to
            # deriving each committed row's slot from the persistent manager
            # tables when the list length does not match, so degrade to the
            # empty list instead of failing the capture.
            entry.static_metadata.new_cache_slots_host_values = []
            return
        real_slots = [int(slot) for slot in source_slots[:batch_size]]
        padding_slots = [0] * (entry.batch_size - batch_size)
        entry.static_metadata.new_cache_slots_host_values = real_slots + padding_slots

    def _capture(self, entry: _DecodeGraphEntry) -> None:
        # Warmup forwards execute for real. execute() routes linear state to
        # reserved slot 0 so these passes cannot advance live request state.
        context = ForwardContext(
            self.attention_backend,
            self.device,
            entry.static_metadata,
            self.layer_caches,
            execution_state=entry.execution_state,
            eplb=getattr(entry, "eplb", None),
            execution_contexts=entry.execution_contexts,
        )
        # Metadata was filled on the caller stream; warmup must consume the
        # completed padding indices and checkpoint tables.
        self._stream.wait_stream(torch.npu.current_stream())
        with forward_context(context), torch.npu.stream(self._stream):
            for _ in range(_CAPTURE_WARMUP_STEPS):
                self._forward_static(entry)
        torch.npu.synchronize()
        entry.graph = torch.npu.NPUGraph()
        capture_context = AclGraphCaptureContext(self._stream, [])
        context = ForwardContext(
            self.attention_backend,
            self.device,
            entry.static_metadata,
            self.layer_caches,
            acl_graph=capture_context,
            execution_state=entry.execution_state,
            eplb=getattr(entry, "eplb", None),
            execution_contexts=entry.execution_contexts,
        )
        with forward_context(context), torch.npu.graph(entry.graph, stream=self._stream):
            entry.static_output = self._forward_static(entry)
        entry.graph_tasks = capture_context.tasks

    def _forward_static(self, entry: _DecodeGraphEntry) -> ModelExecutionOutput:
        if entry.static_input_embedding is None:
            return self.model(entry.static_input_ids, entry.static_positions)
        return self.model(
            entry.static_input_ids,
            entry.static_positions,
            entry.static_input_embedding,
        )

    @staticmethod
    def _update_graph_tasks(
        stream: torch.npu.Stream,
        graph_tasks: list[AclGraphTask],
    ) -> None:
        for task in graph_tasks:
            torch.npu.graph_task_update_begin(stream, task.handle)
            task.update()
            torch.npu.graph_task_update_end(stream)
            task.event.record(stream)

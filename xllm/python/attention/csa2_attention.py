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

"""DeepSeek-V4.1 Flash CSA2 attention backend.

The pure-torch reference path implements the official forward semantics on
top of the runtime :class:`DsaMetadata` contract built by :mod:`dsa_metadata`:

  * SWA ring writes/reads through the SWA-group slot mappings;
  * the ratio-2 softmax-gated compressor with per-token state parking in the
    SWA ring slots, ratio-1 plain projection;
  * the CSA2 indexer (INT8 query/key, relu-weighted scoring, hierarchical
    candidate pool, top-k with the window-row offset baked in);
  * the sparse attention gather over ``[window rows ; top-k compressed rows]``
    with the attention-sink normalizer term.

DeepSeek-V4.1 is NPU-only (no CPU/CUDA fallback). The decode ACL-graph path
routes through the official fused ops, selected by their static guards and
registration state (no env gating): ``quant_lightning_indexer_v2_va``
(QLI v3) replaces the indexer chain, and the interleaved RoPE has a fused
partial-rotary path. On A3, BF16 sparse attention uses the migrated DSV4.1
SMLA kernel; the reference implementation remains available for other shapes
and numerical validation. The model owns all weights; the backend only
orchestrates caches and math.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Sequence

import torch

from scripts.logger import logger
from xllm.python.attention.csa_attention import (
    DsaAttentionBackend,
    _CompressedAttentionCacheMapping,
    _get_layer_cache_tensor,
    _scatter_by_slot,
)
from xllm.python.attention.dsa_metadata import (
    DSA_CACHE_SLIDING_WINDOW,
    DSA_CACHE_TOKEN,
    READ_TABLE_LAYOUT_RING,
    DsaMetadata,
    DsaMetadataBuilder,
    build_cache_specs_v41,
    last_ring_writes,
)
from xllm.python.model_executor.dsa_cp import DsaCpContext, get_current_dsa_cp_runtime
from xllm.python.model_executor.forward_context import get_forward_context, get_forward_context_or_none
from xllm.python.models.deepseek_v41 import (
    V41_QUANT_ASCEND,
    V41_QUANT_NONE,
)

# Softmax-scale exponent base of the sparse kernel's running-max floor.
_NEG_INF_FLOOR = -1e30
# Maximum query rows per attention tile. CSA2 materializes selected KV per
# query, unlike V4's shared-KV tail: bound the two simultaneous FP32 copies
# (cast gather and masked result) as well as retaining the score-row cap.
_ATTENTION_QUERY_TILE = 256
_ATTENTION_KV_TILE_BYTES = 32 * 1024 * 1024
# Query rows per indexer scoring tile. The fp32 score intermediate is
# [tile, index_heads, capacity]; a whole-prompt chunk against a long-context
# committed capacity reaches multiple GiB per layer (30k ctx x 2k chunk x 32
# heads x 4B ~= 4 GiB), which overflows the activation headroom of the
# W8A8+engram full model. Top-k selection is per query row, so tiling over
# the query axis is exact; decode/graph batches (q_len <= tile) take the
# single-tile path unchanged.
_INDEXER_QUERY_TILE = 256


@dataclass
class _SwaGroupMetadata:
    """SWA-group tensors shared by every layer (one ring layout per request)."""

    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    block_size: int
    cache: torch.Tensor


@lru_cache(maxsize=1)
def _is_smla_a3() -> bool:
    import xllm_runtime

    return xllm_runtime.is_ascend_a3()


# The A3 metadata validator accepts sparse compressed widths of 512 or 1024.
_SMLA_CMP_TOPK_BLOCK = 512
_SMLA_CMP_TOPK_MAX = 1024
_SMLA_ORI_MASK_MODE = 4  # Window/compress masking over original KV rows.
_SMLA_CMP_MASK_MODE = 3  # Right-down causal masking over compressed rows.
_SMLA_NO_MASK_MODE = 0


def _supports_smla_va() -> bool:
    """Check schemas and SoC; the linked OPP must also provide the ACLNN APIs.

    Probe registration on each call so a late-loaded extension can be used.
    ACLNN symbol or execution failures remain deployment errors.
    """
    return (
        getattr(torch.ops.xllm_ops, "smla_va", None) is not None
        and getattr(torch.ops.xllm_ops, "smla_va_metadata", None) is not None
        and _is_smla_a3()
    )


@dataclass
class _SmlaForwardState:
    """Layer-shared inputs owned by one ForwardContext, never a request cache."""

    inputs: dict[tuple[object, ...], dict[str, Any]]


@dataclass
class Csa2LayerContext:
    """Per-layer activations the model hands to the backend.

    ``x`` is the layer's attention input (the post-``attn_norm`` collapsed
    stream that also feeds its Q / SWA-KV / compressor projections), ``qr``
    the shared q-lora latent, and ``cos_sin_cache`` the layer's own RoPE table
    (the compress-theta table for every ratio > 0 layer).

    ``qr_scale`` carries the per-token INT8 scale when ``qr`` holds
    pre-quantized codes; the current model always leaves it ``None`` (the
    eager norm keeps ``qr`` a bf16 latent).
    """

    x: torch.Tensor
    qr: torch.Tensor
    positions: torch.Tensor
    cos_sin_cache: torch.Tensor
    qr_scale: torch.Tensor | None = None


@dataclass(frozen=True)
class Csa2QueryLayout:
    """Immutable query geometry supplied by the model for one forward."""

    query_width: int = 1
    token_rows: bool = False
    shared_window: bool = False


@dataclass
class CSA2SharedRuntime:
    """Cross-layer hand-off state for one model forward.

    Mirrors the reference ``SharedAttentionRuntime`` (model.py:1166-1180):
    kv-source layers publish their caches, index-source layers publish the
    top-k, and the candidate-source layer publishes the candidate pool. The
    candidate pool is per sequence (the reference keeps a ``[b, s, t]`` mask;
    each sequence's pool only covers its own committed rows). Layers run in
    order, so one rolling slot per field is enough.
    """

    query_layout: Csa2QueryLayout | None = None
    key_cache: torch.Tensor | None = None
    key_block_table: torch.Tensor | None = None
    index_cache: torch.Tensor | None = None
    index_block_table: torch.Tensor | None = None
    # FP16 per-token scale plane of the INT8 ``index_cache`` (the reader
    # dequantizes the gathered index rows with it). Shared by the kv-source
    # layer that owns the group and every Reindex/Reuse layer reading it.
    index_scale: torch.Tensor | None = None
    compress_ratio: int = 0
    topk_idxs: torch.Tensor | None = None
    topk_layer: int = -1
    # Eager bool-mask pool (``[q_len, capacity]`` per sequence), published by
    # the candidate-source layer on the pure-torch path; the prefill path
    # still uses it.
    candidates: list[torch.Tensor | None] | None = None
    candidates_layer: int = -1
    # Config-enabled fused pool: the block-index candidate tensor
    # (``[rows, 1, candidate_topk_blocks]`` INT32, the op's third output) that
    # the QLI v3 source layer (``candidate_mode=1``) publishes for this
    # forward's consumers (``candidate_mode=2``). It is a pure in-forward data
    # dependency (source layer 20 runs before the consumers), so it needs no
    # persistent buffer; the layer id is kept for diagnostics.
    candidate_blocks: torch.Tensor | None = None
    candidate_blocks_layer: int = -1


# --- Fused interleaved RoPE path (always on when the guards hold) ------------
#
# Replaces the eager ``apply_interleaved_rope`` chain with the registered NPU
# op ``npu_inplace_partial_rotary_mul``: in-place rotation of the trailing
# ``rope_head_dim`` slice of the last dim, with the per-token cos/sin table
# pre-expanded to the kernel's per-element duplicated interleave layout.
# CANN trap: the vendor kernel takes cos/sin in the SAME dtype as x only (no
# fp32-table exception), so the expanded rows are cast to x's dtype.
# ACL-graph safety: a cache build attempted under capture is refused
# (``torch.npu.is_current_stream_capturing``) and the caller stays per-call,
# so no host/device allocation is recorded in the graph.
#
# R2: the INVERSE (output de-rotation) call site runs the same fused op,
# gated by the same guards as the forward call site. The pre-expanded table
# cache is the default form; the per-call expansion remains as the
# tagging-failure / OOM / capture-in-progress fallback.
# Attribute name under which the per-dtype expanded (cos, sin) pairs are
# memoized on the persistent cos/sin table tensor.
_ROPE_EXPANDED_ATTR = "_xllm_v41_expanded_rope"

_partial_rotary_op: Any = None


def _get_partial_rotary_op() -> Any:
    """Resolve the semantic partial-rotary wrapper once its op is registered."""
    global _partial_rotary_op
    if _partial_rotary_op is None:
        if getattr(torch.ops.xllm_ops, "npu_inplace_partial_rotary_mul", None) is None:
            return None
        try:
            from xllm.python.kernels_npu.rotary_embedding import (
                npu_inplace_partial_rotary_mul,
            )
        except Exception:  # noqa: BLE001 - torch_npu absent on CPU-only hosts.
            return None
        _partial_rotary_op = npu_inplace_partial_rotary_mul
    return _partial_rotary_op


def _rope_fused_device_supported(x: torch.Tensor) -> bool:
    """Whether ``x`` lives on an NPU (the fused op's only device contract)."""
    return x.device.type == "npu"


def _rope_fused_supported(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    inverse: bool,
) -> bool:
    """Guards for the fused partial rotary op.

    ``inverse`` selects the conjugated de-rotation form of the same op; it
    shares every guard with the forward call site.

    The fused path rounds ``cos``/``sin`` to ``x``'s bf16 (the vendor kernel's
    dtype contract) while the eager path keeps the fp32 table, so the two
    differ by up to ~1 bf16 ulp; the acceptance bound is
    ``diff <= 2**-6 + 2**-6 * |eager|``. The bound holds while the decode
    ``attn_out`` component magnitude stays roughly <= 8: a real run must
    sample ``attn_out`` absmax once and re-evaluate the bound if it is
    materially above that.
    """
    half = cos_sin_cache.size(-1) // 2
    dim = x.size(-1)
    return (
        _rope_fused_device_supported(x)
        and x.dim() in (2, 3)
        and x.dtype == torch.bfloat16
        # [kernel hard constraint] the vendor aclnnInplacePartialRotaryMul is
        # only correct for a rope slice of exactly 64 (rope_head_dim < 64
        # raises a vector-core error); V4.1's qk_rope_head_dim is 64.
        and 2 * half == 64
        # P2-a: ``size(-1) // 2 == 32`` alone accepts an ODD table width (e.g.
        # 65), which would silently truncate one element; require exact halves.
        and cos_sin_cache.size(-1) == 2 * half
        and dim <= 1024
        and dim % 2 == 0
        and 2 * half <= dim
        and cos_sin_cache.dim() == 2
        and cos_sin_cache.dtype == torch.float32
        and x.is_contiguous()
        and _get_partial_rotary_op() is not None
    )


_fused_rope_active_logged = False
_fused_rope_fallback_logged = False
_fused_rope_inverse_active_logged = False


def _log_fused_rope_active() -> None:
    """One-shot info when the fused RoPE path first engages."""
    global _fused_rope_active_logged
    if not _fused_rope_active_logged:
        _fused_rope_active_logged = True
        logger.info("interleaved RoPE fused path active")


def _log_fused_rope_inverse_active() -> None:
    """One-shot info when the fused INVERSE RoPE path first engages (R2)."""
    global _fused_rope_inverse_active_logged
    if not _fused_rope_inverse_active_logged:
        _fused_rope_inverse_active_logged = True
        logger.info("interleaved RoPE fused INVERSE path active")


def _log_fused_rope_fallback(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    inverse: bool,
) -> None:
    """One-shot warning when a guard keeps the fused path eager."""
    global _fused_rope_fallback_logged
    if _fused_rope_fallback_logged:
        return
    _fused_rope_fallback_logged = True
    logger.warning(
        f"the fused partial rotary op is unavailable "
        f"(device={x.device.type} dtype={x.dtype} dim={x.dim()} inverse={inverse} "
        f"cache_dtype={cos_sin_cache.dtype} op_registered={_get_partial_rotary_op() is not None}); "
        "staying on the eager path"
    )


def _expand_rope_rows(
    cos_sin_cache: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand full-table cos/sin halves to the kernel's interleave layout.

    ``cos_sin_cache`` is ``[max_pos, 2 * half]`` (``[cos | sin]`` halves);
    the returned pair is the per-element duplicated interleave expansion
    (``out[..., 2j] == out[..., 2j + 1] == cache[..., j]``) cast to
    ``dtype`` -- the same op chain the per-call path applies to gathered
    rows, applied here to the whole table once.
    """
    half = cos_sin_cache.size(-1) // 2
    cos = cos_sin_cache[..., :half].repeat_interleave(2, dim=-1).to(dtype).contiguous()
    sin = cos_sin_cache[..., half:].repeat_interleave(2, dim=-1).to(dtype).contiguous()
    return cos, sin


_rope_preexpand_fallback_logged = False


def _log_rope_preexpand_fallback(reason: str) -> None:
    """One-shot warning when the pre-expanded cache is unavailable."""
    global _rope_preexpand_fallback_logged
    if _rope_preexpand_fallback_logged:
        return
    _rope_preexpand_fallback_logged = True
    logger.warning(
        f"interleaved RoPE pre-expanded table cache unavailable ({reason}); staying on the per-call expansion path"
    )


def _expanded_rope_tables(
    cos_sin_cache: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Memoized whole-table expansion for the fused rotary op.

    Returns the ``(cos, sin)`` pair in the kernel's per-element duplicated
    interleave layout and ``dtype``, or ``None`` when the cache cannot be
    built; the caller then selects the per-call expansion path before
    running the kernel.

    Bitwise equivalence with the per-call path: slice, repeat_interleave(2)
    and the dtype cast are row-local and element-local, so
    ``expand(table)[positions] == expand(table[positions])`` for every
    position -- gathering rows from the pre-expanded table reproduces the
    per-call expansion bit for bit.

    The memo lives as an attribute on the table tensor itself, so its
    lifetime is bound to the persistent rotary buffer (the V4.1 tables are
    shared across layers and across same-descriptor models). A cache HIT is
    served even under graph capture (the memoized tensors are static and
    exactly what the graph should record), but BUILDING under capture would
    bake the expansion ops into the graph and allocate from the graph's
    memory pool, so a capture-in-progress stream refuses the build and the
    caller stays per-call for that capture (the graph runner's eager warmup
    populates the cache first).
    """
    per_dtype = getattr(cos_sin_cache, _ROPE_EXPANDED_ATTR, None)
    if per_dtype is not None and dtype in per_dtype:
        return per_dtype[dtype]
    if _is_stream_capturing():
        return None
    if per_dtype is None:
        per_dtype = {}
        try:
            setattr(cos_sin_cache, _ROPE_EXPANDED_ATTR, per_dtype)
        except Exception as exc:  # noqa: BLE001
            _log_rope_preexpand_fallback(f"cannot tag table tensor: {exc}")
            return None
    try:
        pair = _expand_rope_rows(cos_sin_cache, dtype)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        # Path selection before the kernel runs: the proven per-call
        # expansion keeps serving on an allocation-constrained device.
        _log_rope_preexpand_fallback(f"expansion OOM: {exc}")
        return None
    per_dtype[dtype] = pair
    return pair


def _apply_interleaved_rope_fused(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """In-place partial interleaved RoPE through the semantic NPU wrapper.

    ``x`` is modified in place (callers assign the return value, and every
    call site consumes the tensor only after the rotation); the return value
    keeps ``x``'s original shape. The compact ``[cos | sin]`` half-width
    table row is expanded to the per-element duplicated interleave layout the
    kernel expects and cast to ``x``'s dtype.

    ``inverse`` is forwarded to the wrapper, which conjugates the rotation by
    negating ``sin`` (R2 output de-rotation); the wrapper owns that sign, so
    the caller keeps passing the unmodified table.

    The expansion is served from the whole-table cache
    (``_expanded_rope_tables``): per call only two index_select ops gather
    the already-expanded bf16 rows (bitwise identical to expanding the
    gathered rows). A cache miss (tagging failure / build OOM / build under
    capture) falls back to the original per-call expansion.
    """
    half = cos_sin_cache.size(-1) // 2
    pos = positions.reshape(-1).to(torch.int64)
    expanded = _expanded_rope_tables(cos_sin_cache, x.dtype)
    if expanded is not None:
        cos = expanded[0].index_select(0, pos)
        sin = expanded[1].index_select(0, pos)
    else:
        rows = cos_sin_cache.index_select(0, pos)
        cos = rows[..., :half].repeat_interleave(2, dim=-1).to(x.dtype).contiguous()
        sin = rows[..., half:].repeat_interleave(2, dim=-1).to(x.dtype).contiguous()
    dim = x.size(-1)
    # The kernel contract is 3D [M, n_head, head_dim]; a 2D latent gets a
    # singleton head axis (a view, so the in-place write still lands in x).
    x3d = x if x.dim() == 3 else x.unsqueeze(1)
    _get_partial_rotary_op()(x3d, cos, sin, dim - 2 * half, 2 * half, inverse=inverse)
    return x


def apply_interleaved_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Interleaved-pair complex RoPE on the last ``2 * half`` dims (model.py:392-406).

    Adjacent element pairs ``(x[2i], x[2i+1])`` form one complex number that
    is multiplied by ``exp(i * theta_p)``; ``inverse`` conjugates the rotation,
    which de-rotates the attention output by the query's own rotation. The
    per-position table broadcasts over any leading head/batch axes of ``x``
    (the position axis is always ``x``'s first axis). The math runs in fp32
    and the result is cast back to ``x``'s dtype.
    """
    if _rope_fused_supported(x, positions, cos_sin_cache, inverse):
        if inverse:
            _log_fused_rope_inverse_active()
        else:
            _log_fused_rope_active()
        return _apply_interleaved_rope_fused(x, positions, cos_sin_cache, inverse)
    _log_fused_rope_fallback(x, positions, cos_sin_cache, inverse)
    half = cos_sin_cache.size(-1) // 2
    rows = cos_sin_cache.index_select(0, positions.reshape(-1).to(torch.int64))
    cos = rows[..., :half].to(torch.float32)
    sin = rows[..., half:].to(torch.float32)
    if inverse:
        sin = -sin
    for _ in range(x.dim() - 2):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    dim = x.size(-1)
    tail = x[..., dim - 2 * half :].to(torch.float32)
    even = tail[..., 0::2]
    odd = tail[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    rotated = rotated.flatten(-2).to(x.dtype)
    if 2 * half == dim:
        return rotated
    return torch.cat((x[..., : dim - 2 * half], rotated), dim=-1)


def _is_stream_capturing() -> bool:
    """ACL-graph capture probe (always ``False`` where ``torch_npu`` is absent)."""
    npu = getattr(torch, "npu", None)
    return bool(npu is not None and npu.is_available() and npu.is_current_stream_capturing())


# --- V4.1 compressor Triton pooling (always on when guards hold) -------------
#
# Replaces the eager softmax-gated pooling of the ratio-2 compressor with the
# Triton kernel ported from vllm-ascend dsv41: the ring state is aligned with
# vllm-ascend (fixed 32 rows per block, ``kDsv41CompressStateRows``) but the
# kernel is adapted to xLLM's two split fp32 planes (compress_kv_state /
# compress_score_state), not a packed ring. The pure-torch path (same ring-32
# addressing) remains the fallback for any guard or op failure.
_COMPRESSOR_TRITON_RING = 32

_v41_compressor_fallback_logged = False


def _log_v41_compressor_fallback(reason: str) -> None:
    """One-shot warning when the fused compressor declines (guard failure)."""
    global _v41_compressor_fallback_logged
    if _v41_compressor_fallback_logged:
        return
    _v41_compressor_fallback_logged = True
    logger.warning(f"fused V4.1 compressor is inactive ({reason}); staying on the pure-torch pooling path")


_v41_compressor_active_logged = False


def _log_v41_compressor_active() -> None:
    """One-shot info when the fused V4.1 Triton compressor first returns a result."""
    global _v41_compressor_active_logged
    if not _v41_compressor_active_logged:
        _v41_compressor_active_logged = True
        logger.info("fused V4.1 compressor Triton path active (op=xllm.python.kernels_npu.triton.compressor)")


_V41_COMPRESSOR_MODULE: Any = None


def _get_v41_compressor_module() -> Any:
    """Import the ported compressor module once (None where triton is absent)."""
    global _V41_COMPRESSOR_MODULE
    if _V41_COMPRESSOR_MODULE is None:
        try:
            from xllm.python.kernels_npu.triton import compressor as _module
        except Exception:  # noqa: BLE001 - torch_npu/triton absent on CPU-only hosts.
            return None
        _V41_COMPRESSOR_MODULE = _module
    return _V41_COMPRESSOR_MODULE


def _v41_compressor_triton_reason(
    ratio: int,
    kv_state: torch.Tensor | None,
    score_state: torch.Tensor | None,
) -> str | None:
    """Why the fused ratio-2 compressor must fall back to torch (None = usable).

    Every clause is static (ratio, dtype, shape, device, registered module)
    so the branch decision is graph-capture safe. The op itself runs both eagerly
    and inside ACL-graph capture; the precondition is that the pre-capture warmup
    forward the graph runner executes first has completed the Triton JIT
    (decode_acl_graph.py, the pre-capture warmup forward).
    """
    if ratio != 2:
        return f"unsupported ratio {ratio}"
    if kv_state is None or score_state is None:
        return "no compressor state ring"
    if kv_state.dtype != torch.float32 or score_state.dtype != torch.float32:
        return "ring state not fp32"
    if kv_state.dim() != 3 or score_state.dim() != 3:
        return "ring state not [blocks, rows, dim]"
    if kv_state.size(1) != _COMPRESSOR_TRITON_RING or score_state.size(1) != _COMPRESSOR_TRITON_RING:
        return "ring rows != 32"
    if kv_state.device.type not in ("npu", "privateuseone"):
        return "not an NPU device"
    if _get_v41_compressor_module() is None:
        return "triton compressor unavailable"
    return None


def _v41_compressor_swa_gate(
    dsa: Any,
    layer_id: int,
    ori_cache_idx: int,
    ring_blocks: int | None = None,
) -> tuple[torch.Tensor | None, str | None]:
    """Resolve the SWA read table and the fused compressor's ring-block gate.

    Returns ``(block_table, reason)`` where a ``None`` reason admits the fused
    op. The gate is the *config-derived* ring-block count the caller passes in
    (``(window - 1) // block + 1``, production == 1), never the physical table
    width. The caller resolves the state block from the current absolute
    position; released SWA blocks can leave invalid earlier table columns.
    A missing/placeholder table and a count != 1 (an absent count is 0) fall
    back conservatively. Pure static logic with no device reads, so CPU-testable.
    """
    block_table = _get_layer_cache_tensor(dsa.block_tables, layer_id, ori_cache_idx)
    if block_table is None or block_table.numel() == 0:
        return block_table, "no swa block table"
    count = int(ring_blocks) if ring_blocks is not None else 0
    if count != 1:
        return block_table, f"swa ring blocks per sequence {count} != 1"
    return block_table, None


def _v41_compressor_num_cores() -> int:
    module = _get_v41_compressor_module()
    if module is None:
        return 0
    try:
        return int(module._cube_core_num())
    except Exception as exc:  # noqa: BLE001 - missing device properties.
        _log_v41_compressor_fallback(f"core count unavailable: {exc}")
        return 0


def _v41_compressor_pool_triton(
    kv_seq: torch.Tensor,
    score_seq: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    start_pos: torch.Tensor,
    used_len: int,
    block_id: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor | None:
    """Pool one segment's ratio-2 groups with the Triton kernel.

    Returns token-aligned pooled rows in the activation dtype (the caller
    gathers the committed group rows and applies RMSNorm), or None when the op
    declines so the caller keeps the pure-torch path. ``start_pos`` / ``block_id``
    are device scalars, so the control metadata is built with device arithmetic
    and stays ACL-graph friendly.
    """
    module = _get_v41_compressor_module()
    if module is None:
        return None
    q_len = int(kv_seq.size(0))
    if q_len <= 0:
        return None
    num_cores = _v41_compressor_num_cores()
    if num_cores <= 0:
        return None
    dev = kv_seq.device
    head_dim = int(kv_seq.size(1))
    meta = torch.stack(
        (
            start_pos.to(torch.int32).reshape(()),
            torch.full((), int(used_len), dtype=torch.int32, device=dev),
            torch.zeros((), dtype=torch.int32, device=dev),
            torch.zeros((), dtype=torch.int32, device=dev),
            block_id.to(torch.int32).reshape(()),
        )
    ).reshape(5, 1)
    out = torch.empty(q_len, head_dim, dtype=output_dtype, device=dev)
    try:
        module.compressor_from_projected(
            kv_seq.contiguous(),
            score_seq.contiguous(),
            kv_state,
            score_state,
            meta.contiguous(),
            out,
            max_query_len=q_len,
            num_cores=num_cores,
        )
    except Exception as exc:  # noqa: BLE001 - a declining op must not break serving.
        _log_v41_compressor_fallback(f"op raised: {exc}")
        return None
    return out


def _ensure_seq_lens_tensor(
    ctx_lens: torch.Tensor | Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Coerce sequence lengths to a tensor (graph safety: never a host read).

    The eager path may still hand over plain lists; the graph path always
    receives the persistent ``dsa.seq_lens`` tensor whose values are refreshed
    per replay, so every consumer below must treat it as device data.
    """
    if torch.is_tensor(ctx_lens):
        return ctx_lens
    return torch.tensor(list(ctx_lens), dtype=torch.int64, device=device)


def _indexer_score(
    q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Indexer scores ``score[s, t] = sum_h w[s, h] * relu(q[s, h] . k[t])``.

    Mirrors model.py:556-557: the per-head dot products are rectified BEFORE
    the learned per-head weights are applied and summed.
    """
    score = torch.einsum("shd,td->sht", q.float(), index_k.float())
    score = score.relu_() * weights.float().unsqueeze(-1)
    return score.sum(dim=1)


def _select_candidate_blocks(
    score: torch.Tensor,
    compress_lens: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k (model.py:583-610).

    ``score`` is ``[s, t]`` with unreachable positions already ``-inf``.
    ``compress_lens`` holds the per-query committed counts. Returns a bool
    mask shaped like ``score``.
    """
    width = score.size(-1)
    padded = torch.nn.functional.pad(score, (0, -width % block_size), value=float("-inf"))
    blocks = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = blocks.size(-1)
    # Pin the block holding the query's newest committed position: it is only
    # partly filled and could otherwise be outscored by an older, full block.
    last = torch.div(compress_lens - 1, block_size, rounding_mode="floor")
    blocks = blocks.masked_fill(
        torch.arange(num_blocks, device=score.device) == last.unsqueeze(-1),
        float("inf"),
    )
    top = blocks.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(blocks, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
    # repeat_interleave is pathological for 1-byte dtypes on Ascend, so the
    # unsqueeze->expand->flatten form replaces it.
    expanded = keep.unsqueeze(-1).expand(*keep.shape, block_size)
    return expanded.flatten(-2)[..., :width]


def _indexer_topk(
    score: torch.Tensor,
    compress_lens: torch.Tensor,
    index_topk: int,
    offset: int,
) -> torch.Tensor:
    """Top-k selection re-sorted by position; unreachable picks become -1 (model.py:577-580).

    ``compress_lens`` is the per-query visibility count ``(p + 1) // ratio``;
    valid picks are shifted by ``offset`` (the window-row count of the
    concatenated gather tensor). A pick is recoverable only when its score is
    finite *and* its position is visible: the Reindex candidate pool masks
    out-of-pool rows with ``-inf`` while those rows still fall inside the
    visible range, so a pool narrower than ``index_topk`` would otherwise leak
    them into the picks -- the finite check drops them onto the -1 sentinel.
    Ties are broken toward lower positions via a
    stable descending sort: ``torch.topk`` leaves tie order
    implementation-defined, which would make a whole-prompt prefill and its
    token-by-token replay disagree on the zero-score ties the relu-weighted
    scoring produces.
    """
    topk = min(index_topk, score.size(-1))
    order = torch.argsort(score, dim=-1, descending=True, stable=True)
    idxs = order[..., :topk]
    idxs = idxs.sort(dim=-1).values
    finite = torch.isfinite(score.gather(-1, idxs))
    visible = compress_lens.unsqueeze(-1)
    return torch.where((idxs < visible) & finite, idxs + offset, -1).to(torch.int32)


# The rebase ascending sort runs its FP32 kernel instead of INT32 for the
# eager prefill hand-off: the sentinel (``_QLI_V3_REBASE_FP32_SENTINEL`` =
# 1 << 24) is exactly representable in FP32 and strictly above every valid
# compressed row index, so the FP32 sort matches the INT32 one bit for bit.
# The decode call site keeps the INT32 kernel (captured path unchanged).
_QLI_V3_REBASE_FP32_SENTINEL = float(1 << 24)


def _rebase_qli_picks(
    picks: torch.Tensor,
    visible: torch.Tensor,
    offset: int,
    *,
    fp32_sort: bool = False,
) -> torch.Tensor:
    """Filter, sort and rebase QLI op picks onto the eager window-offset form.

    ``picks`` is the op's ``[rows, topk]`` INT32 selection of 0-based compressed
    rows (negative = padding); ``visible`` is ``[rows, 1]``, the per-sequence
    count of committed compressed rows the op was given as PA_BBND
    ``seqused_k`` (``floor(seq_lens / ratio)``, which is exactly the eager
    indexer's visibility bound ``(p + 1) // ratio`` for ``p = seq_lens - 1``).

    A pick at or past ``visible`` is not a cached row: the eager indexer never
    produces one (``idxs < visible`` in :func:`_indexer_topk`), and rebasing it
    by ``offset`` would silently address an unrelated window row. Such picks
    become ``-1``. Valid picks are sorted ascending (invalid slots last), which
    is what the eager indexer's position sort produces; fused consumers of the
    picks walk only the left-aligned valid prefix of each index row, so the
    ascending order is part of the picks contract. Everything is device
    arithmetic -- no host read, no data-dependent shape -- so a captured graph
    replays it unchanged.

    ``fp32_sort`` swaps only the *kernel* the ascending sort runs on (INT32 ->
    FP32); the output is bit-identical for the compressed-row index range (see
    ``_QLI_V3_REBASE_FP32_SENTINEL``). The eager prefill call site passes the
    FP32 kernel, while the decode call site keeps the INT32 kernel of its
    captured ACL-graph path.
    """
    valid = (picks >= 0) & (picks < visible)
    if fp32_sort:
        sentinel_f = _QLI_V3_REBASE_FP32_SENTINEL
        key = torch.where(
            valid,
            picks.to(torch.float32),
            torch.full(picks.shape, sentinel_f, dtype=torch.float32, device=picks.device),
        )
        key = key.sort(dim=-1).values
        picks = torch.where(
            key < sentinel_f,
            key.to(torch.int32),
            torch.full_like(picks, -1),
        )
    else:
        sentinel = torch.iinfo(torch.int32).max
        sortable = torch.where(valid, picks, torch.full_like(picks, sentinel))
        sortable = sortable.sort(dim=-1).values
        picks = torch.where(sortable >= sentinel, torch.full_like(sortable, -1), sortable)
    return torch.where(picks >= 0, picks + offset, torch.full_like(picks, -1)).to(torch.int32)


def _window_visible_idxs(
    positions_seq: torch.Tensor,
    previous_context: torch.Tensor,
    win: int,
) -> torch.Tensor:
    """Fixed chronological SWA picks into ``[win ring rows; current chunk]``.

    The caller gathers ring rows at increasing absolute positions
    ``previous_context - win .. previous_context - 1`` before appending the
    current chunk. Physical block IDs may wrap or be unordered; gather output
    still follows the supplied positions, not physical block order.
    Each query has exactly ``win`` columns independent of the chunk length.
    Negative positions are padding; every real position maps to its ring or
    current-chunk row. Keeping this reduction support fixed avoids summing
    over additional masked columns as chunk/batch sizes change.
    """
    window_positions = positions_seq.unsqueeze(-1) - win + 1 + torch.arange(win, device=positions_seq.device)
    return torch.where(window_positions >= 0, window_positions - previous_context + win, -1)


def _block_query_lens(token_rows: int, query_width: int) -> list[int]:
    """Regroup token-row metadata into complete query blocks."""
    if query_width <= 0 or token_rows % query_width:
        raise RuntimeError(f"Token rows must contain complete blocks: rows={token_rows}, width={query_width}")
    return [query_width] * (token_rows // query_width)


def _block_visible_idxs(
    prev_ctx: torch.Tensor,
    q_len: int,
    col_pos: torch.Tensor,
    cols: torch.Tensor,
    win: int,
) -> torch.Tensor:
    """A non-causal block shares one context window across its queries.

    A draft block is sampled in parallel, so all ``q_len`` queries condition
    on the same rows: the trailing ``win`` context rows before the block plus
    the whole block itself (``[prev_ctx - win, prev_ctx + q_len - 1]``),
    non-causal within the block. The target's per-query causal SWA window
    (:func:`_window_visible_idxs`) must not be used for the draft block.
    Masked columns use the same ``-1`` padding sentinel as causal window picks.
    """
    window_lo = torch.clamp(prev_ctx - win, min=0)
    window_hi = prev_ctx + q_len - 1
    valid = (col_pos >= window_lo) & (col_pos <= window_hi)
    return torch.where(valid, cols, -1).unsqueeze(0).expand(q_len, col_pos.numel())


def _token_cache_slots(
    block_table: torch.Tensor | None,
    cache: torch.Tensor | None,
    rows: torch.Tensor,
    seqs: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Physical slots of compressed rows in a paged TOKEN-group cache.

    The compact ``slot_mappings`` layout (one entry per newly committed row)
    packs sequences in commit order and therefore cannot be consumed inside a
    captured graph; the slots are recomputed from the persistent block table
    instead, with ``mask`` folding the commit decision into -1 (scatter skips)
    so no host branch remains. ``rows``/``seqs`` are flat per-row tensors; the
    -1 slots of uncommitted or padded rows are never written.
    """
    if block_table is None or cache is None:
        return torch.full_like(rows, -1)
    if block_table.numel() == 0:
        return torch.full_like(rows, -1)
    block_size = cache.size(1)
    cols = torch.div(rows, block_size, rounding_mode="floor") % block_table.size(1)
    blocks = block_table[seqs, cols].to(torch.int64).clamp_min(0)
    slots = blocks * block_size + rows % block_size
    if mask is not None:
        slots = torch.where(mask, slots, -1)
    return slots


# Serving-context cap applied before deriving the default graph token-capacity
# granularity. The model's ``max_position_embeddings`` (1048576 for V4.1) is the
# theoretical RoPE horizon, not any served context, and deriving from it yields
# an impractical 65536-token bucket whose indexer gather touches 65536 rows per
# layer; the dual-node launcher historically pinned 2048 to keep that gather
# bounded. Capping the derivation source at 32768 reproduces exactly that 2048
# floor (32768 -> ceil(32768 / 16) = 2048) while still scaling with shorter
# served contexts.
_GRAPH_TOKEN_CAPACITY_SERVING_CAP = 32768


def _derive_graph_token_capacity_granularity(max_model_len: int) -> int:
    """Default ACL-graph token-capacity granularity from the served context.

    The granularity is the token width of one captured capacity bucket
    (``_committed_capacity`` gathers ``granularity // ratio`` compressed rows
    per bucket). The derivation source is the served context length capped at
    :data:`_GRAPH_TOKEN_CAPACITY_SERVING_CAP` so a large
    ``max_position_embeddings`` cannot blow up the per-layer gather. Aligning
    ``ceil(serving_capped / 16)`` up to a 1024 multiple keeps the captured
    bucket count at roughly 16 across context lengths, while the 1024 floor
    avoids over-fragmenting short contexts. Returns 0 when the context length
    is unknown, keeping the fail-closed capture guard.
    """
    if max_model_len <= 0:
        return 0
    serving_capped = min(max_model_len, _GRAPH_TOKEN_CAPACITY_SERVING_CAP)
    scaled = (serving_capped + 15) // 16
    return max(1024, ((scaled + 1023) // 1024) * 1024)


# --- QLI v3 fused indexer (always on when guards hold) -----------------------
#
# Replaces the pure-torch indexer chain of ``_run_indexer`` with the native
# ``xllm_ops`` ``quant_lightning_indexer_v2_va`` single op: INT8 query/key
# with FP16 scales (``quant_mode=2``), ``layout_k`` fixed to ``PA_BBND``
# (the A3 tiling rejects every other layout), reading the policy's INT8
# index plane plus its FP16 per-token scale directly -- the same planes the
# compressor writes, no additive shadow plane.
# The two-level candidate pool is config-driven: when the model config sets
# ``candidate_source_layer_id >= 0``, the source layer runs
# ``candidate_mode=1`` (its third output is the block-index pool), the
# consuming layers run ``candidate_mode=2`` (pool-restricted top-k).
# ACL-graph safety: the batch descriptors and tiling metadata live in a
# per-bucket :class:`_QliGraphState` built in the runner's eager warmup and
# refreshed in place before every replay; nothing is allocated and no
# metadata op first runs inside a capture.

# The native ``xllm_ops`` handles of the fused QLI v3 path: the indexer is
# the migrated "_va" form of the V2 kernel (dispatched to
# ``aclnnQuantLightningIndexerV2Va``; one call collapsing the historical
# indexer/metadata/candidate trio, so no separate "v3" handle exists), and
# the tiling-metadata producer is the V2 metadata "_va" binding next to it.
_QLI_V3_OP_NAME = "quant_lightning_indexer_v2_va"
_QLI_V3_METADATA_OP_NAME = "quant_lightning_indexer_v2_metadata_va"

# Quantized int8 indexer modes supported by the operator (INT8 only).
_QLI_V3_QUANT_MODE = 2
# Index head width of the operator contract.
_QLI_V3_HEAD_DIM = 128
# Fixed size of the ``quant_lightning_indexer_v2_metadata_va`` schedule buffer.
_QLI_V3_METADATA_SIZE = 1024
# Compression ratios whose index rows the operator can score.
_QLI_V3_SUPPORTED_RATIOS = (1, 2)
# Symmetric INT8 rounding matching vllm-ascend's ``quantize_indexer_query``:
# ``scale = fp16(absmax / 127)`` floored at the smallest normal, then
# ``q = round_half_to_even(x / scale)`` clamped to ``[-127, 127]``.
_QLI_V3_INT8_MAX = 127.0
_QLI_V3_MIN_SCALE = 2.0**-24
# PA_BBND paged index-key geometry the A3 tiling expects
# (``quant_lightning_indexer_v2_va/op_host/quant_lightning_indexer_v2_va_tiling.cpp`` ``CheckShapeDim``/``CheckScaleShape``):
# the INT8 paged index key is ``[blocks, tokens_per_block, num_k_heads,
# head_dim]`` with ``num_k_heads == 1`` (``GetAndCheckN2Size``), and the FP16
# dequant scale drops the head dim -> ``[blocks, tokens_per_block,
# num_k_heads]``. ASSUMPTION (not yet on-device): the scale-plane rank matches
# the key minus the head dim exactly; this is the ONE constant to adjust if the
# device rejects it.
_QLI_V3_KEY_NUM_HEADS = 1
# ``mask_mode`` / AscendC ``sparse_mode``: the compressed rows relate to a query
# right-down-causally (row ``j`` is visible to query ``p`` iff
# ``j < (p + 1) // ratio``), i.e. ``SPARSE_MODE_LOWER = 3``
# (``quant_lightning_indexer_v2_va/op_host/quant_lightning_indexer_v2_va_tiling.h``; the tiling accepts only 0 or 3,
# ``...tiling.cpp:393-398``). The AscendC op def (``..._def.cpp:90``) defaults
# ``mask_mode`` to ``0`` while the torch schema (the candidate entry ported from
# vllm-ascend) defaults it to ``3`` -- the two ends are NOT aligned. Every call
# site pins this constant explicitly so neither default is relied on;
# ``mask_mode=3`` is also required for ratio>1, whose ``cmp_residual_k`` must be
# non-null only when ``cmpRatio != 1 && sparseMode != 0``
# (``...tiling.cpp:691-697``).
_QLI_V3_MASK_MODE = 3

# Two-level candidate pool of the op, driven by the model config: when
# ``candidate_source_layer_id >= 0`` the candidate-source layer runs
# ``candidate_mode=1`` (publishes its block-index pool on the op's third
# output) and each layer that consumes that pool runs ``candidate_mode=2``
# (restrict its own top-k to the pool); every other layer stays
# ``candidate_mode=3``.
# ``candidate_mode`` values, matching the migrated op def
# (``quant_lightning_indexer_v2_va_def.cpp``, op name QuantLightningIndexerV2Va).
_QLI_V3_CAND_MODE_SOURCE = 1
_QLI_V3_CAND_MODE_CONSUMER = 2
_QLI_V3_CAND_MODE_OFF = 3
# The A3 tiling only implements ``candidate_block_size == 8`` with a
# ``candidate_topk_blocks`` in ``(0, 2048]`` and a multiple of 64
# (see ``quant_lightning_indexer_v2_va_tiling.cpp``). With the pool
# configured they become load-bearing operands, so a violation is a
# misconfiguration the dispatcher must report instead of silently degrading.
_QLI_V3_CAND_BLOCK_SIZE = 8
_QLI_V3_CAND_TOPK_BLOCKS_MAX = 2048
_QLI_V3_CAND_TOPK_BLOCKS_ALIGN = 64

# Prefill extension of the fused indexer: an *eager* prefill step (any
# ``q_len > 1``) runs the fused op too, with the dispatch's two decode-shape
# gates relaxed for it and the TND batch descriptors built per call
# (``cu_seqlens_q`` over the chunk, PA_BBND ``seqused_k`` = the
# per-sequence committed compressed rows) instead of the decode graph's
# persistent bucket state.

_qli_v3_ops: tuple[Any, Any] | None = None


def _get_qli_v3_ops() -> tuple[Any, Any] | None:
    """Resolve ``(quant_lightning_indexer_v2_va, quant_lightning_indexer_v2_metadata_va)``.

    Both handles are registered by the ``xllm_ops`` extension itself: the
    candidate indexer is the migrated "_va" binding of the V2 kernel (dispatched
    to ``aclnnQuantLightningIndexerV2Va``) and the metadata producer is the V2
    metadata "_va" binding built next to it. The probe is best-effort (the
    extension and ``torch_npu`` are absent on CPU-only hosts) and cheap enough
    to run on every forward; only a successful resolution is cached, so a later
    registration is picked up.

    Resolution is NOT readiness: ``getattr`` only proves both Torch schemas are
    present in ``torch.ops.xllm_ops``. The authoritative readiness check is the
    ABI-level one -- the ACLNN entry points are bound into the loaded
    ``libcust_opapi.so`` -- and the on-device acceptance run.
    """
    global _qli_v3_ops
    if _qli_v3_ops is not None:
        return _qli_v3_ops
    namespace = getattr(torch.ops, "xllm_ops", None)
    if namespace is None:
        return None
    indexer = getattr(namespace, _QLI_V3_OP_NAME, None)
    metadata = getattr(namespace, _QLI_V3_METADATA_OP_NAME, None)
    if indexer is None or metadata is None:
        return None
    _qli_v3_ops = (indexer, metadata)
    return _qli_v3_ops


_qli_v3_active_logged = False
_qli_v3_fallback_logged = False
_qli_v3_candidate_logged = False

# Fallback reasons that are the path's DESIGNED behaviour (a prefill/eager
# step, or a layer that takes part in the shared candidate pool) rather than
# a misconfiguration: they stay silent and do not consume the one-shot, so a
# later genuine guard failure can still report itself. A prefill that cannot
# fuse is refused by a deliberately non-silent guard, so no new fallback can
# hide.
_QLI_V3_SILENT_FALLBACK_REASONS = frozenset({"not an ACL-graph decode step", "query length != 1"})


def _log_qli_v3_active() -> None:
    """One-shot info when the QLI v3 indexer path first engages."""
    global _qli_v3_active_logged
    if not _qli_v3_active_logged:
        _qli_v3_active_logged = True
        logger.info("DeepSeek-V4.1 QLI v3 fused indexer active")


def _log_qli_v3_candidate_active(layer_id: int) -> None:
    """One-shot info when a two-level candidate layer first fuses.

    A positive rank-log leg distinct from ``_log_qli_v3_active`` (which fires on
    any fused indexer layer): this one proves the candidate operands actually
    reached the op, so a config that is set but whose candidate layers all fall
    back cannot masquerade as active. Fires for the first fused candidate layer
    of the rank -- source (mode 1) or consumer (mode 2); in a real forward the
    source layer always runs before its consumers.
    """
    global _qli_v3_candidate_logged
    if not _qli_v3_candidate_logged:
        _qli_v3_candidate_logged = True
        logger.info(f"DeepSeek-V4.1 QLI v3 candidate indexer active (layer {layer_id})")


def _log_qli_v3_fallback(reason: str) -> None:
    """One-shot warning when a guard keeps the QLI v3 indexer eager.

    Designed-behaviour reasons (see ``_QLI_V3_SILENT_FALLBACK_REASONS``) do not
    warn and do not consume the one-shot.
    """
    global _qli_v3_fallback_logged
    if _qli_v3_fallback_logged or reason in _QLI_V3_SILENT_FALLBACK_REASONS:
        return
    _qli_v3_fallback_logged = True
    logger.warning(f"the QLI v3 indexer path is inactive ({reason}); staying on the pure-torch indexer")


_indexer_quant_fused_logged = False


def _quantize_indexer_rows_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric INT8 quantization of the last axis.

    Matches vllm-ascend's ``quantize_indexer_query`` rounding: the FP16 scale
    is ``absmax / 127`` (floored at the smallest positive value so a zero row
    stays defined) and the codes are the half-to-even rounded normalised values
    clamped to ``[-127, 127]``. Every step is row-local/pointwise, so the helper
    is ACL-graph safe (static shapes, no host sync).

    Parity note (recorded at the definition so it governs every call site):
    this helper runs bare, exactly like vllm-ascend's upstream
    ``quantize_indexer_query`` -- no inf/NaN input defense is performed here;
    evaluating such defenses is deferred to a later round.
    """
    # The fused kernel is validated for contiguous BF16 rows of width 128.
    if x.device.type == "npu" and x.dtype == torch.bfloat16 and x.ndim >= 2 and x.size(-1) == 128 and x.is_contiguous():
        from xllm.python.kernels_npu.triton.indexer_quant import quantize_indexer_rows_int8

        global _indexer_quant_fused_logged
        if not _indexer_quant_fused_logged:
            logger.info("DeepSeek-V4.1 fused INT8 indexer quantization active")
            _indexer_quant_fused_logged = True
        return quantize_indexer_rows_int8(x)
    absmax = x.float().abs().amax(dim=-1)
    scale = (absmax / _QLI_V3_INT8_MAX).to(torch.float16).to(torch.float32).clamp_min(_QLI_V3_MIN_SCALE)
    normalized = x.float() / scale.unsqueeze(-1)
    codes = torch.round(normalized).clamp(-_QLI_V3_INT8_MAX, _QLI_V3_INT8_MAX)
    return codes.to(torch.int8), scale.to(torch.float16)


@dataclass
class _QliGraphState:
    """ACL-graph-stable operands of the QLI v3 decode path for one bucket.

    Built once during the runner's eager warmup (which also JIT-compiles the
    metadata op) and refreshed in place on every replay; the captured graph
    records these persistent addresses, so no buffer may be allocated and no
    metadata op may first run inside capture. ``capacity`` is the token-level
    committed-row bound (``graph_token_capacity``); the per-ratio compressed
    capacity is ``max(capacity // ratio, 1)``, exactly like ``_run_indexer``'s.
    The paged key is addressed by the runtime ``block_table`` plus the per-row
    ``seqused_k`` (PA_BBND), so no ``cu_seqlens_k`` operand exists.
    """

    rows: int
    capacity: int
    cu_seqlens_q: torch.Tensor
    seqused_k: dict[int, torch.Tensor]
    cmp_residual: dict[int, torch.Tensor]
    metadata: dict[int, torch.Tensor]


class Csa2AttentionBackend(DsaAttentionBackend):
    """Pure-torch CSA2 backend for DeepSeek-V4.1 Flash.

    Consumes the V4.1 cache layout of contract section 2 through the standard
    ``LayerCache`` slots (key=0, index=2, swa=5, compress_kv_state=6,
    compress_score_state=7) and the ``DsaMetadataBuilder`` block tables / slot
    mappings. The AICPU precomputed-metadata step of the V4 backend is not
    needed (no NPU kernels are dispatched here).

    ACL graph integration: every data-dependent branch in the forward is
    tensor arithmetic (see :meth:`execute_csa2_layer`), so one captured graph
    stays correct across parity and length changes. The one remaining dynamic
    shape -- the committed-row gather of the indexer / sparse attention -- is
    pinned to a constant row capacity instead of a per-replay host read:
    ``graph_token_capacity_granularity`` is the largest decode context length
    the serving run can reach, and the captured graph gathers at
    ``granularity // ratio`` rows for every replay.

    The granularity is resolved in :meth:`__init__`: it is derived from
    ``max_model_len`` capped at :data:`_GRAPH_TOKEN_CAPACITY_SERVING_CAP`
    (0 keeps the eager exact-max behaviour).

    MoE graph capability: only the Ascend W8A8 export routes the experts
    through the grouped kernel chain (MC2) an ACL graph can replay. The
    ``fp8``/``none`` checkpoint modes serve ``DeepseekV41MoE``, whose forward
    syncs to the host and loops over data-dependent expert segments, so no
    captured graph can replay it. :attr:`moe_graph_capturable` records that
    verdict (derived from the shared model config) for the decode-graph
    admission; :attr:`acl_graph_enabled` gates the startup capacity check.
    """

    # The captured CSA2 forward computes its window logic in-graph from the
    # refreshed metadata; it must not build (or inherit the state for) the
    # V4 backend's out-of-graph SWA window-source indices. This backend does
    # not run the base constructor, so the base's A5 graph branch would
    # otherwise crash on the missing V4-only window-source state.
    graph_uses_swa_window_sources = False

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
        kv_source_layer_ids: list[int],
        index_source_layer_ids: list[int],
        candidate_source_layer_id: int,
        candidate_topk_blocks: int,
        candidate_block_size: int,
        max_model_len: int = 0,
        quant_mode: str = V41_QUANT_NONE,
        acl_graph_enabled: bool = False,
    ) -> None:
        self.caches_info, self.group_infos = build_cache_specs_v41(
            compress_ratios,
            kv_source_layer_ids,
            index_source_layer_ids,
            window_size,
            n_layers,
        )
        self._builder = DsaMetadataBuilder(self.caches_info, self.group_infos, read_table_layout=READ_TABLE_LAYOUT_RING)
        self.window_size = window_size
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.num_heads = num_heads
        self.head_dim = attn_head_dim
        self.device = device
        self.dtype = dtype
        self.scale = attn_head_dim**-0.5

        self.index_source_layer_ids = sorted({int(source) for source in index_source_layer_ids})
        self.candidate_source_layer_id = int(candidate_source_layer_id)
        self.candidate_topk_blocks = int(candidate_topk_blocks)
        self.candidate_block_size = int(candidate_block_size)
        if self.candidate_source_layer_id >= 0 and self.candidate_source_layer_id not in self.index_source_layer_ids:
            raise ValueError(
                "DeepSeek-V4.1 candidate source layer must also be an index source layer: "
                f"{self.candidate_source_layer_id}"
            )

        # MoE graph capability marker (see the class docstring): the fp8/none
        # torch MoE cannot be captured. Derived from the shared model config, so
        # every rank of a DP group computes the same value and the decode-graph
        # admission stays group-consistent.
        self.quant_mode = quant_mode
        self.moe_graph_capturable = quant_mode == V41_QUANT_ASCEND

        self.graph_token_capacity_granularity = _derive_graph_token_capacity_granularity(max_model_len)
        granularity_source = f"derived from max_model_len={max_model_len}"
        logger.info(
            f"DeepSeek-V4.1 CSA2 graph_token_capacity_granularity="
            f"{self.graph_token_capacity_granularity} ({granularity_source})"
        )

        # Startup checks (51361671 C). The graph-vs-eager decision and the
        # committed-row capacity are both static for the run, so fail (or warn)
        # here at construction instead of at the first captured forward, where
        # the root cause is far from the symptom.
        if acl_graph_enabled and not self.moe_graph_capturable:
            logger.warning(
                f"DeepSeek-V4.1 quant_mode={quant_mode} serves the torch MoE, which no ACL graph "
                "can capture; decode-graph admission is disabled and decode stays eager"
            )
        if acl_graph_enabled and self.moe_graph_capturable and self.graph_token_capacity_granularity <= 0:
            raise RuntimeError(
                "DeepSeek-V4.1 ACL-graph serving needs a static committed-row capacity, but "
                "max_model_len was not provided to derive one. Provide "
                "max_position_embeddings, or disable graph execution."
            )

        self._kv_caches: list[Any] = []
        # Config-derived SWA ring blocks per sequence; refreshed on bind so it is
        # a plain Python int computed before capture (never a captured tensor).
        self._swa_ring_blocks = self._compute_swa_ring_blocks()
        self._metadata: Any | None = None
        self._v41_shared: CSA2SharedRuntime | None = None
        # Persistent QLI v3 indexer operands, one entry per (batch_rows,
        # token capacity) graph bucket; built in warmup, refreshed under capture.
        self._qli_state: dict[tuple[int, int], _QliGraphState] = {}

    # -- V4 backend overrides ----------------------------------------------

    def _compute_swa_ring_blocks(self) -> int:
        """SWA ring blocks per sequence, derived from the config.

        ``get_swa_blocks_per_seq(window, block) = (window - 1) // block + 1``
        (block_utils.cpp), the value the scheduler allocates; production
        ``window == block == 128`` yields 1. The SWA page width is the cache's
        own block size; before the caches are bound fall back to
        ``window_size``, which yields the same production 1. A plain Python int
        computed before capture.
        """
        swa_block = self.window_size
        if self._kv_caches:
            swa = getattr(self._kv_caches[0], "swa", None)
            if swa is not None and swa.dim() >= 2 and int(swa.size(1)) > 0:
                swa_block = int(swa.size(1))
        if swa_block <= 0:
            return 1
        return (self.window_size - 1) // swa_block + 1

    def bind_kv_caches(self, kv_caches: list[Any]) -> None:
        super().bind_kv_caches(kv_caches)
        if self._kv_caches and self._kv_caches[0].swa is not None:
            block_size = int(self._kv_caches[0].swa.size(1))
            # SWA pages use the bound physical width, not the attention window.
            # Replace descriptors in the lists already held by the builder;
            # TOKEN descriptors and matching-width bindings stay unchanged.
            for index, group in enumerate(self.group_infos):
                if group.cache_type == DSA_CACHE_SLIDING_WINDOW and group.block_size != block_size:
                    self.group_infos[index] = replace(group, block_size=block_size)
            for layer_caches in self.caches_info:
                for index, cache in enumerate(layer_caches):
                    if cache.cache_type == DSA_CACHE_SLIDING_WINDOW and cache.block_size != block_size:
                        layer_caches[index] = replace(cache, block_size=block_size)
        self._swa_ring_blocks = self._compute_swa_ring_blocks()

    def reset_forward(self, metadata: Any | None = None) -> None:
        super().reset_forward(metadata)
        self._v41_shared = None

    def refresh_dsa_metadata_for_graph_replay(self, metadata: Any | None = None) -> None:
        """Refresh DSA tensors and the QLI operand buffers outside capture.

        The inherited refresh rebuilds the DSA metadata and copies the replay
        values into the graph-captured tensors; the QLI branch rewrites its
        per-ratio sequence lengths / residuals plus the indexer's tiling
        metadata. This runs before every graph replay, so it must never
        allocate a new state, build one under capture, or read a device value
        back to the host.
        """
        super().refresh_dsa_metadata_for_graph_replay(metadata)
        resolved = self._metadata if metadata is None else metadata
        dsa = getattr(resolved, "dsa_metadata", None)
        if dsa is None:
            return
        for state in self._qli_replay_states(dsa):
            self._refresh_qli_state(dsa, state)

    def _qli_replay_states(self, dsa: DsaMetadata) -> list[_QliGraphState]:
        """The QLI states matching one replay's (rows, capacity) bucket, if any."""
        key = (
            int(dsa.seq_lens.numel()),
            int(getattr(dsa, "graph_token_capacity", 0) or 0),
        )
        state = self._qli_state.get(key)
        return [state] if state is not None else []

    def _build_dsa_metadata_for_forward(self, metadata: Any, *, group_limit: int | None = None) -> DsaMetadata:
        dsa = super()._build_dsa_metadata_for_forward(metadata, group_limit=group_limit)
        if bool(getattr(metadata, "is_dummy", False)):
            # The shared V4 builder gives empty-DP rows a synthetic 512-token
            # context for native operator tiling. CSA2 runs tensor attention:
            # that context would expose uninitialized reserved-block history.
            # Retain one query row for collective symmetry, but no committed
            # history or writes. These persistent values refresh on every
            # graph replay, including empty -> active -> empty transitions.
            dsa.seq_lens.zero_()
            dsa.actual_seq_lengths_kv.zero_()
            dsa.kv_cu_seq_lens.zero_()
            dsa.max_seqlen_kv.zero_()
            dsa.max_seq_len = 0
            # Preserve start_pos = KV length - query length, including the
            # synthetic query. CSA2 RoPE uses ctx.positions instead; negative
            # history is clamped for gathers and masked before cache writes.
            dsa.start_pos.copy_(-dsa.seq_lens_q)
            for layer_slots in dsa.slot_mappings:
                for slots in layer_slots:
                    slots.fill_(-1)
        return dsa

    def _build_precomputed_metadata(
        self,
        compressed_metadata: DsaMetadata,
        metadata: Any,
        cu_seqlens_ori_kv_override: torch.Tensor | None = None,
        max_query_len_override: int | None = None,
        max_seq_len_override: int | None = None,
    ) -> None:
        """The pure-torch path dispatches no AICPU kernels; nothing to build."""
        del (
            compressed_metadata,
            metadata,
            cu_seqlens_ori_kv_override,
            max_query_len_override,
            max_seq_len_override,
        )

    def _resolve_cache_mapping(self, layer_id: int, compress_ratio: int) -> _CompressedAttentionCacheMapping:
        """V4.1 mapping: TOKEN entries exist for ratio-1 sources too."""
        mapping = _CompressedAttentionCacheMapping()
        if layer_id < 0 or layer_id >= len(self.caches_info):
            return mapping
        token_indices: list[int] = []
        swa_indices: list[int] = []
        for cache_idx, ci in enumerate(self.caches_info[layer_id]):
            if ci.cache_type == DSA_CACHE_TOKEN:
                token_indices.append(cache_idx)
            elif ci.cache_type == DSA_CACHE_SLIDING_WINDOW:
                swa_indices.append(cache_idx)
        if token_indices and compress_ratio > 0:
            mapping.cmp_cache_idx = token_indices[0]
        if len(token_indices) > 1:
            mapping.index_cache_idx = token_indices[1]
        if swa_indices:
            mapping.ori_cache_idx = swa_indices[0]
        if len(swa_indices) > 1:
            mapping.kv_state_cache_idx = swa_indices[1]
        if len(swa_indices) > 2:
            mapping.score_state_cache_idx = swa_indices[2]
        return mapping

    def _layer_compress_ratio(self, layer_id: int) -> int:
        caches = self.caches_info[layer_id] if 0 <= layer_id < len(self.caches_info) else []
        for ci in caches:
            if ci.cache_type == DSA_CACHE_TOKEN:
                return ci.ratio
        return 0

    def swa_history_cache(self) -> torch.Tensor:
        """Return the cache whose slot layout also addresses Engram history."""
        if not self._kv_caches:
            raise RuntimeError("Engram history restore requires a bound SWA cache")
        mapping = self._resolve_cache_mapping(0, 0)
        cache = self._kv_caches[0].swa
        if cache is None or mapping.ori_cache_idx < 0:
            raise RuntimeError(
                "Engram history restore requires a bound SWA cache with a valid sliding-window slot mapping"
            )
        return cache

    def swa_group_metadata(self, dsa: DsaMetadata | None) -> _SwaGroupMetadata | None:
        """The SWA-group slot mapping / block table / cache shared by all layers.

        Every layer's first cache entry addresses the SWA group and the group
        shares one block-table/slot layout (contract section 2), so layer 0's
        view serves any consumer that needs position-stable ring slots (the
        engram hash cache).
        """
        if dsa is None or not self._kv_caches:
            return None
        mapping = self._resolve_cache_mapping(0, 0)
        layer_cache = self._kv_caches[0]
        if layer_cache.swa is None or mapping.ori_cache_idx < 0:
            return None
        slot_mapping = _get_layer_cache_tensor(dsa.slot_mappings, 0, mapping.ori_cache_idx)
        block_table = _get_layer_cache_tensor(dsa.block_tables, 0, mapping.ori_cache_idx)
        if slot_mapping is None or block_table is None:
            return None
        return _SwaGroupMetadata(
            slot_mapping=slot_mapping,
            block_table=block_table,
            block_size=layer_cache.swa.size(1),
            cache=layer_cache.swa,
        )

    # -- model-facing entry point -------------------------------------------

    def execute_csa2_layer(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        layer: Any,
    ) -> torch.Tensor:
        """Run one V4.1 layer (SWA-only or CSA2) end to end.

        ``q`` is ``[T, H, D]`` (RoPE applied at token positions) and ``kv`` is
        ``[T, D]`` (RoPE applied, bf16) from the model's projections; the
        layer carries its CSA2 mode and the :class:`Csa2LayerContext`.

        ACL-graph safety: ``seq_lens_q`` is read as a Python list because its
        value is constant per captured graph (one row per sequence on the
        decode path), while ``seq_lens`` changes every step and is therefore
        consumed strictly as a device tensor (the persistent buffer refreshed
        by ``refresh_dsa_metadata_for_graph_replay``). Every data-dependent
        branch below is expressed as tensor arithmetic so one captured graph
        stays correct across parity / length changes.
        """
        metadata = self._current_forward_metadata()
        dsa = getattr(metadata, "dsa_metadata", None)
        if dsa is None:
            raise RuntimeError("DeepSeek-V4.1 requires prepared DSA metadata before attention")
        mode = layer.mode
        layer_id = layer.layer_id
        layer_cache = self._kv_caches[layer_id]
        mapping = self._resolve_cache_mapping(layer_id, mode.compress_ratio)
        ctx = layer._csa2_ctx
        if ctx is None:
            raise RuntimeError(f"DeepSeek-V4.1 layer {layer_id} ran without its CSA2 context")
        cp_runtime = get_current_dsa_cp_runtime()
        cp_context = cp_runtime.plan if cp_runtime is not None else None
        context = get_forward_context_or_none()
        query_layout = context.execution_contexts.get(Csa2QueryLayout) if context is not None else None
        shared_window = query_layout is not None and query_layout.shared_window
        token_rows = query_layout is not None and (dsa.is_acl_graph or query_layout.token_rows)
        if shared_window and cp_context is not None:
            raise RuntimeError("Block attention does not support context parallelism")
        if cp_context is not None:
            q_lens = list(cp_context.global_q_lens)
        elif dsa.is_acl_graph or token_rows:
            # Row counts and the model's query layout are capture-stable host
            # metadata. Never copy device sequence lengths back during capture.
            width = query_layout.query_width if query_layout is not None else 1
            q_lens = _block_query_lens(int(dsa.seq_lens_q.numel()), width)
        else:
            q_lens = [int(value) for value in dsa.seq_lens_q.tolist()]
        ctx_lens = _ensure_seq_lens_tensor(dsa.seq_lens, ctx.x.device)
        if token_rows:
            ctx_lens = ctx_lens[:: query_layout.query_width]

        # A multi-token query reads history that later tokens would overwrite
        # in the SWA ring. The torch tail already reads this chunk directly
        # from kv, so commit its ring writes only after all queries finish.
        # Single-token decode retains write-before-read.
        defer_swa_write = any(length > 1 for length in q_lens)
        swa_slot = _get_layer_cache_tensor(dsa.slot_mappings, layer_id, mapping.ori_cache_idx)
        if not defer_swa_write and layer_cache.swa is not None and swa_slot is not None:
            _scatter_by_slot(layer_cache.swa, swa_slot, kv)

        shared = self._shared_runtime()
        shared.query_layout = query_layout
        # 2) Compressor: only kv-source (Full) layers produce the shared main KV.
        if mode.is_full:
            self._run_compressor(layer, ctx, dsa, layer_cache, mapping, q_lens, ctx_lens, shared)
        # Cache producers own the complete token stream. Query/indexer work
        # owns a contiguous query token interval, including decode graph rows.
        if cp_runtime is not None:
            ctx_lens = cp_runtime.local_context_lengths
            q_lens = list(cp_context.local_q_lens)
            ctx = replace(
                ctx,
                x=cp_context.shard(ctx.x),
                qr=cp_context.shard(ctx.qr),
                positions=cp_context.shard(ctx.positions),
            )
            # DSA metadata remains global for cache layout. Query consumers
            # receive local q_lens/ctx_lens explicitly.
        # 3) Indexer: Full/Reindex layers compute a fresh top-k; Reuse layers
        #    consume the tensor object their group source published.
        if mode.is_full or mode.is_reindex:
            topk = self._run_indexer(layer, ctx, dsa, q_lens, ctx_lens, shared)
        elif mode.is_reuse:
            topk = shared.topk_idxs
            if topk is None and dsa.block_tables:
                # Real batches always publish the group source's top-k;
                # profiling batches (empty block tables) degrade to the
                # plain window path.
                raise RuntimeError(
                    f"DeepSeek-V4.1 reuse layer {layer_id} has no shared top-k; its group source must run first"
                )
        else:
            topk = None
        layer._csa2_used_topk = topk
        # 4) Sparse attention over [window rows ; top-k compressed rows].
        output = self._csa2_attention(q, kv, layer, dsa, q_lens, ctx_lens, topk, shared, cp_context)
        if defer_swa_write and layer_cache.swa is not None and swa_slot is not None:
            # Captured blocks keep fixed row counts; eager chunks may wrap
            # the ring and must select the latest write for each slot.
            slots, values = (swa_slot, kv) if dsa.is_acl_graph else last_ring_writes(swa_slot, kv)
            _scatter_by_slot(layer_cache.swa, slots, values)
        return output

    # -- compressor ---------------------------------------------------------

    def _run_compressor(
        self,
        layer: Any,
        ctx: Csa2LayerContext,
        dsa: DsaMetadata,
        layer_cache: Any,
        mapping: _CompressedAttentionCacheMapping,
        q_lens: list[int],
        ctx_lens: torch.Tensor,
        shared: CSA2SharedRuntime,
    ) -> None:
        """Compress the layer's own attention input into the shared main KV.

        Ratio 1 commits one row per token (plain ``RMSNorm(wkv(x))``); ratio 2
        pools each non-overlapping group of 2 with an fp32 softmax gate,
        parking partial-group tokens in the state ring (reference model.py:458-485,
        chunked-prefill generalization per the semantics doc section 4).

        ACL-graph safety: at most ``ratio - 1`` tail rows per sequence are
        parked with a tensor mask; commits use a fixed ``q_len // ratio + 1`` row
        capacity per sequence with a tensor ``commit_mask``; member values are
        selected between the current chunk and the parked ring with
        ``torch.where`` instead of host branches, and every cache slot is
        computed from the persistent block tables as device arithmetic.
        """
        ratio = layer.mode.compress_ratio
        layer_id = layer.layer_id
        ctx_lens = _ensure_seq_lens_tensor(ctx_lens, ctx.x.device)
        kv_state = layer_cache.compress_kv_state
        score_state = layer_cache.compress_score_state
        kv_state_flat = kv_state.view(-1, layer_cache.swa.size(-1)) if kv_state is not None else None
        score_state_flat = score_state.view(-1, layer_cache.swa.size(-1)) if score_state is not None else None
        # Ring length of the compressor state (V4.1: fixed 32, see
        # deepseek_v4_cache_policy.h kDsv41CompressStateRows); the state planes
        # are [swa_count, state_rows, head_dim].
        state_rows = kv_state.size(1) if kv_state is not None else 0

        latent_rows: list[torch.Tensor] = []
        group_starts: list[torch.Tensor] = []
        slot_rows: list[torch.Tensor] = []
        slot_seqs: list[torch.Tensor] = []
        slot_masks: list[torch.Tensor] = []
        token_offset = 0
        for seq, q_len in enumerate(q_lens):
            if q_len == 0:
                # No chunk members exist to pool or park for this request.
                # Decode graphs retain one synthetic row instead.
                continue
            ctx_len = ctx_lens[seq]
            prev_ctx = ctx_len - q_len
            # Padded graph rows (ctx <= 0) stay fully inert: no parking, no
            # commit, so a garbage block-table entry can never be written.
            seq_valid = ctx_len > 0
            x_seq = ctx.x[token_offset : token_offset + q_len]
            rows = torch.arange(q_len, device=ctx.x.device)
            # Fixed commit capacity: a chunk of q_len tokens completes at most
            # q_len // ratio + 1 groups (the group straddling each boundary).
            capacity = q_len // ratio + 1
            g = torch.arange(capacity, device=ctx.x.device)
            group_idx = torch.div(prev_ctx, ratio, rounding_mode="floor") + g
            commit_mask = g < (
                torch.div(ctx_len, ratio, rounding_mode="floor") - torch.div(prev_ctx, ratio, rounding_mode="floor")
            )
            commit_mask = commit_mask & seq_valid
            if ratio == 1:
                # One token per group: nothing to pool, no gate, no state. The
                # commit count is exactly q_len (ctx - prev == q_len), so the
                # slot segment stays q_len rows and aligns 1:1 with the latent
                # rows in the concatenated scatter.
                kv_seq, _ = layer.compress_kv_score(x_seq)
                latent_rows.append(layer.compress_rmsnorm(kv_seq))
                group_starts.append(ctx.positions[token_offset : token_offset + q_len].to(torch.int64))
                rows_segment = group_idx[:q_len]
                mask_segment = commit_mask[:q_len]
            else:
                kv_seq, score_seq = layer.compress_kv_score(x_seq)
                # Fused Triton pooling: pool the committed groups at their
                # completion-token rows and
                # update the state ring in place. The op consumes host control
                # metadata and the pre-capture warmup forward JIT-compiles it, so
                # it runs in both eager and ACL-graph capture; the reference torch
                # path below (same ring-32 addressing) is the fallback.
                reason = _v41_compressor_triton_reason(ratio, kv_state, score_state)
                triton_pooled = None
                if reason is None:
                    if any(int(q) != 1 for q in q_lens):
                        # The op is validated on the single-token decode rows
                        # only; a q_len > 1 row (prefill / speculative chunk) is
                        # unverified, so it keeps the torch path.
                        reason = "non-decode row"
                    else:
                        # Resolve state from the current absolute position,
                        # just like the torch path. SWA releases old blocks;
                        # column zero can be invalid while later blocks live.
                        block_table, reason = _v41_compressor_swa_gate(
                            dsa, layer_id, mapping.ori_cache_idx, self._swa_ring_blocks
                        )
                        if block_table is not None and reason is None:
                            # Empty-DP tables contain only zero/padding blocks:
                            # even start_pos=-1 resolves block 0, protected by
                            # the Triton pool/write kernels before state access.
                            triton_pooled = _v41_compressor_pool_triton(
                                kv_seq,
                                score_seq,
                                kv_state,
                                score_state,
                                start_pos=prev_ctx,
                                used_len=q_len,
                                block_id=torch.div(
                                    self._v41_state_slots(
                                        dsa, layer_id, mapping.ori_cache_idx, seq, prev_ctx.reshape(1), state_rows
                                    ),
                                    state_rows,
                                    rounding_mode="floor",
                                ),
                                output_dtype=torch.float32 if ctx.x.dtype == torch.float32 else torch.bfloat16,
                            )
                            if triton_pooled is None:
                                reason = "op declined"
                if reason is not None:
                    _log_v41_compressor_fallback(reason)
                if triton_pooled is not None:
                    _log_v41_compressor_active()
                    # Garbage rows (beyond this sequence's commit count) read
                    # back zeroed rows and stay out of the slot mask.
                    offsets = group_idx * ratio + (ratio - 1) - prev_ctx
                    pooled_rows = triton_pooled[offsets.clamp(0, q_len - 1)]
                    latent_rows.append(layer.compress_rmsnorm(pooled_rows.to(ctx.x.dtype)))
                    group_starts.append((group_idx * ratio).clamp(0, ctx.cos_sin_cache.size(0) - 1))
                    rows_segment = group_idx
                    mask_segment = commit_mask
                else:
                    # Read-before-write: the pooling below must read the ring
                    # residual (the group member just before this chunk) before
                    # this chunk re-parks into the ring. The Triton kernel keeps
                    # the same K2-read -> K3-write order; with the ring shrunk to
                    # 32 rows, a q_len >= 32 chunk whose prev_ctx is odd would
                    # otherwise overwrite the residual slot (pos prev_ctx-1) with
                    # its own parked row (pos prev_ctx+31, same slot mod 32).
                    pos = prev_ctx + rows
                    park_mask = ((pos + 1) % ratio != 0) & seq_valid
                    # Only the trailing min(q_len, state_rows) tokens can be the
                    # next chunk's pre-segment residual, so keep one writer per
                    # ring slot exactly like the Triton kernel's K3. Parking
                    # the whole chunk would write the same slot twice whenever
                    # q_len > state_rows, and the NPU scatter's
                    # duplicate-index write order is unspecified.
                    tail_start = max(0, q_len - state_rows)
                    park_mask = park_mask & (rows >= tail_start)
                    # Pool the groups committed by this forward; a member inside
                    # the chunk reads the fresh projection, one before the chunk
                    # start reads its parked copy from the state ring.
                    member_pos = group_idx.unsqueeze(-1) * ratio + torch.arange(ratio, device=ctx.x.device)
                    # Trailing singleton so the [cap, ratio] mask right-aligns
                    # with the [cap, ratio, dim] member tensors in torch.where.
                    in_chunk = (member_pos >= prev_ctx).unsqueeze(-1)
                    member_row = (member_pos - prev_ctx).clamp(0, q_len - 1)
                    if kv_state_flat is not None and score_state_flat is not None:
                        parked_slot = self._v41_state_slots(
                            dsa, layer_id, mapping.ori_cache_idx, seq, member_pos.clamp_min(0), state_rows
                        )
                        member_kv = torch.where(in_chunk, kv_seq[member_row], kv_state_flat[parked_slot])
                        member_score = torch.where(in_chunk, score_seq[member_row], score_state_flat[parked_slot])
                    else:
                        member_kv = kv_seq[member_row]
                        member_score = score_seq[member_row]
                    # Softmax-gated pooling over the group members, one gate per
                    # channel (wgate projects to the full latent width;
                    # model.py:480-485).
                    gate = member_score.softmax(dim=1)
                    pooled = (member_kv * gate).sum(dim=1)
                    # Residuals read: now park every token whose group does not
                    # complete this forward. The parity branch is folded into the
                    # slot mask (-1 skips).
                    if kv_state_flat is not None and score_state_flat is not None:
                        # State ring slot of each chunk token (V4.1 fixed-length
                        # ring), resolved from absolute positions via the SWA
                        # block table. The parked rows must be addressed with
                        # the state ring stride, not the SWA slot stride.
                        park_slot = torch.where(
                            park_mask,
                            self._v41_state_slots(dsa, layer_id, mapping.ori_cache_idx, seq, pos, state_rows),
                            -1,
                        )
                        _scatter_by_slot(kv_state_flat, park_slot, kv_seq)
                        _scatter_by_slot(score_state_flat, park_slot, score_seq)
                    # RMSNorm AFTER pooling, BEFORE RoPE (model.py:485).
                    latent_rows.append(layer.compress_rmsnorm(pooled.to(ctx.x.dtype)))
                    # Garbage rows (beyond this sequence's commit count) keep a
                    # valid non-negative rope position; the slot mask keeps them
                    # unwritten. The lower clamp also catches fully padded rows
                    # (ctx_len <= 0 -> group_idx = -1), which stay inert via the
                    # mask but still flow through apply_interleaved_rope.
                    group_starts.append((group_idx * ratio).clamp(0, ctx.cos_sin_cache.size(0) - 1))
                    rows_segment = group_idx
                    mask_segment = commit_mask
            slot_rows.append(rows_segment)
            slot_seqs.append(torch.full_like(rows_segment, seq))
            slot_masks.append(mask_segment)
            token_offset += q_len

        latent = torch.cat(latent_rows, dim=0) if latent_rows else None
        all_starts = torch.cat(group_starts, dim=0) if group_starts else None
        all_slot_rows = torch.cat(slot_rows, dim=0) if slot_rows else None
        all_slot_seqs = torch.cat(slot_seqs, dim=0) if slot_seqs else None
        all_slot_mask = torch.cat(slot_masks, dim=0) if slot_masks else None
        index_block_table = _get_layer_cache_tensor(dsa.block_tables, layer_id, mapping.index_cache_idx)
        key_block_table = _get_layer_cache_tensor(dsa.block_tables, layer_id, mapping.cmp_cache_idx)
        # The indexer consumes the PRE-RoPE latent before the cache write.
        if latent is not None and layer.indexer is not None and layer.indexer.owns_k:
            index_k = layer.indexer_key(latent)
            index_k = apply_interleaved_rope(index_k, all_starts, ctx.cos_sin_cache)
            if layer_cache.index is not None:
                index_slots = _token_cache_slots(
                    index_block_table,
                    layer_cache.index,
                    all_slot_rows,
                    all_slot_seqs,
                    all_slot_mask,
                )
                # The index plane is always INT8 (NPU platform policy): the
                # cache keeps per-row symmetric int8 codes plus an fp16
                # per-token scale, matching vllm-ascend's index K cache.
                index_codes, index_scales = _quantize_indexer_rows_int8(index_k)
                _scatter_by_slot(layer_cache.index, index_slots, index_codes)
                if layer_cache.indexer_scale is not None:
                    _scatter_by_slot(
                        layer_cache.indexer_scale,
                        index_slots,
                        index_scales.unsqueeze(-1),
                    )
        if latent is not None:
            rotated = apply_interleaved_rope(latent, all_starts, ctx.cos_sin_cache)
            if layer_cache.key is not None:
                key_slots = _token_cache_slots(
                    key_block_table,
                    layer_cache.key,
                    all_slot_rows,
                    all_slot_seqs,
                    all_slot_mask,
                )
                _scatter_by_slot(layer_cache.key, key_slots, rotated)
        # Publish the shared caches for this group's Reindex/Reuse layers. The
        # index scale plane travels with the INT8 index plane.
        shared.key_cache = layer_cache.key
        shared.key_block_table = key_block_table
        shared.index_cache = layer_cache.index
        shared.index_block_table = index_block_table
        shared.index_scale = layer_cache.indexer_scale
        shared.compress_ratio = ratio

    # -- indexer ------------------------------------------------------------

    def _run_indexer(
        self,
        layer: Any,
        ctx: Csa2LayerContext,
        dsa: DsaMetadata,
        q_lens: list[int],
        ctx_lens: torch.Tensor,
        shared: CSA2SharedRuntime,
    ) -> torch.Tensor:
        """Score the shared index K and select the top-k compressed rows.

        Full layers own their K (written by :meth:`_run_compressor`); Reindex
        layers re-score the group's shared K restricted to the candidate pool
        (reference model.py:527-580).

        ACL-graph safety: the score matrix is gathered at the batch-level
        committed-row capacity (the graph's static capacity bound under
        capture, the exact batch maximum eagerly) so every tensor shape is
        graph-stable; rows beyond a sequence's committed count are masked to
        ``-inf`` and can never enter a top-k pick.
        """
        layer_id = layer.layer_id
        ratio = shared.compress_ratio
        ctx_lens = _ensure_seq_lens_tensor(ctx_lens, ctx.x.device)
        if ratio <= 0 or shared.index_cache is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 indexer for layer {layer_id} has no shared index cache; "
                "its group kv source must run first"
            )
        if shared.index_block_table is None:
            # Profiling / warmup batches carry no block tables (see the
            # zeros ring rows in _csa2_attention); skip the scoring pass so
            # the sparse attention runs the plain window path. Reuse layers
            # degrade the same way via the empty dsa.block_tables check.
            return None
        if ctx.qr.size(0) == 0:
            topk = torch.empty((len(q_lens), 0, 0), dtype=torch.int32, device=ctx.x.device)
            shared.topk_idxs = topk
            shared.topk_layer = layer_id
            if layer_id == self.candidate_source_layer_id:
                shared.candidates = [None] * len(q_lens)
                shared.candidates_layer = layer_id
            return topk
        index_cache = shared.index_cache
        block_table = shared.index_block_table
        block_size = index_cache.size(1)
        capacity = self._committed_capacity(dsa, ratio, block_table, block_size, ctx_lens)

        # With the fused q norm+quant on, the shared latent is already
        # INT8-quantized (ctx.qr_scale present) and the indexer reuses it; when
        # off (the default) qr is the bf16 latent and qr_scale is None. The
        # single-arg call is kept for backends/fakes that predate the switch.
        q_idx = layer.indexer_query(ctx.qr, ctx.qr_scale) if ctx.qr_scale is not None else layer.indexer_query(ctx.qr)
        q_idx = apply_interleaved_rope(q_idx, ctx.positions, ctx.cos_sin_cache)
        qli_reason = self._qli_v3_fallback_reason(layer, dsa, q_lens, shared, index_cache)
        if qli_reason is None:
            _log_qli_v3_active()
            return self._run_indexer_qli_v3(
                layer,
                q_idx,
                ctx,
                dsa,
                q_lens,
                shared,
                block_table,
                capacity,
            )
        _log_qli_v3_fallback(qli_reason)
        # The eager indexer scores in the INT8 query domain: the query is
        # per-row symmetric INT8 quantized and dequantized back to fp32, the
        # same representation the fused path uses for its query side. This
        # matches the INT8 index cache write exactly.
        q_codes, q_scales = _quantize_indexer_rows_int8(q_idx)
        q_idx = q_codes.float() * q_scales.unsqueeze(-1).float()
        weights = layer.indexer_weights(ctx.x)

        candidates_per_seq: list[torch.Tensor | None] | None = None
        if layer_id == self.candidate_source_layer_id:
            candidates_per_seq = [None] * len(q_lens)
        rows_per_seq: list[torch.Tensor] = []
        token_offset = 0
        for seq, q_len in enumerate(q_lens):
            if q_len == 0:
                rows_per_seq.append(torch.empty((0, 0), dtype=torch.int32, device=q_idx.device))
                continue
            q_seq = q_idx[token_offset : token_offset + q_len]
            w_seq = weights[token_offset : token_offset + q_len]
            positions_seq = ctx.positions[token_offset : token_offset + q_len].to(torch.int64)
            ctx_len = ctx_lens[seq]
            committed = torch.div(ctx_len, ratio, rounding_mode="floor")
            if capacity <= 0:
                rows_per_seq.append(torch.full((q_len, 0), -1, dtype=torch.int32, device=q_idx.device))
                token_offset += q_len
                continue
            index_k = self._gather_cache_rows(index_cache, block_table, seq, capacity, block_size)
            # Capacity gathers may read uninitialized tail rows. Mask their
            # values before math, not only the later attention/indexer scores.
            committed_rows = torch.arange(capacity, device=q_idx.device) < committed
            index_k = torch.where(committed_rows.unsqueeze(-1), index_k, 0)
            # The index plane is always INT8 (NPU platform policy): the cache
            # holds per-row codes; dequantize with the per-token scale plane
            # (same physical slots, gathered the same way).
            index_scales = self._gather_cache_rows(shared.index_scale, block_table, seq, capacity, block_size)
            index_scales = torch.where(committed_rows.unsqueeze(-1), index_scales, 0)
            index_k = index_k.float() * index_scales.float()
            # A compressed group is visible to query p once p has passed its
            # last token: visible(p) = (p + 1) // ratio (model.py:562-567);
            # rows past this sequence's committed count (or past the graph's
            # capacity bound) are padding and can never be picked.
            visible = torch.div(positions_seq + 1, ratio, rounding_mode="floor")
            t_range = torch.arange(capacity, device=q_idx.device)
            # The compressed picks address the concatenated
            # [window_kv ; compress_kv] tensor. The window is a fixed-capacity
            # [ring candidates ; chunk] block, so the compressed region starts
            # at a constant column and no ctx-length term remains.
            offset = self.window_size + q_len
            # Score in query tiles: the [tile, heads, capacity] fp32
            # intermediate is GiB-scale for whole-prompt chunks against long
            # contexts (see _INDEXER_QUERY_TILE). Top-k and the candidate
            # pool are per query row, so tiling is exact.
            row_tiles: list[torch.Tensor] = []
            pool_tiles: list[torch.Tensor] = []
            for tile_start in range(0, q_len, _INDEXER_QUERY_TILE):
                tile_end = min(tile_start + _INDEXER_QUERY_TILE, q_len)
                score = _indexer_score(q_seq[tile_start:tile_end], index_k, w_seq[tile_start:tile_end])
                tile_visible = visible[tile_start:tile_end]
                score.masked_fill_(t_range >= tile_visible.unsqueeze(-1), float("-inf"))
                score.masked_fill_(t_range >= committed.unsqueeze(-1), float("-inf"))
                if layer_id == self.candidate_source_layer_id:
                    assert candidates_per_seq is not None
                    pool_tiles.append(
                        _select_candidate_blocks(
                            score,
                            tile_visible,
                            self.candidate_topk_blocks,
                            self.candidate_block_size,
                        )
                    )
                elif self._uses_candidates(layer_id):
                    if shared.candidates is None or seq >= len(shared.candidates):
                        raise RuntimeError(
                            f"DeepSeek-V4.1 reindex layer {layer_id} requires the candidate pool "
                            f"from layer {self.candidate_source_layer_id}"
                        )
                    pool = shared.candidates[seq]
                    if pool is None:
                        raise RuntimeError(
                            f"DeepSeek-V4.1 reindex layer {layer_id} has no candidate pool for sequence {seq}"
                        )
                    score.masked_fill_(~pool[tile_start:tile_end], float("-inf"))
                row_tiles.append(_indexer_topk(score, tile_visible, self.index_topk, offset))
            if pool_tiles and candidates_per_seq is not None:
                candidates_per_seq[seq] = torch.cat(pool_tiles, dim=0)
            if row_tiles:
                rows_per_seq.append(row_tiles[0] if len(row_tiles) == 1 else torch.cat(row_tiles, dim=0))
            else:
                rows_per_seq.append(torch.full((q_len, 0), -1, dtype=torch.int32, device=q_idx.device))
            token_offset += q_len

        # One rectangular tensor (padded with -1) keeps the shared hand-off a
        # single object; padding columns are invalid and contribute nothing.
        seq_count = len(q_lens)
        s_max = max(q_lens) if q_lens else 0
        k_width = min(self.index_topk, capacity) if capacity > 0 else 0
        topk = torch.full((seq_count, s_max, k_width), -1, dtype=torch.int32, device=q_idx.device)
        for seq, rows in enumerate(rows_per_seq):
            if rows.numel() > 0:
                topk[seq, : rows.size(0), : rows.size(1)] = rows
        shared.topk_idxs = topk
        shared.topk_layer = layer_id
        if candidates_per_seq is not None:
            shared.candidates = candidates_per_seq
            shared.candidates_layer = layer_id
        return topk

    # -- QLI v3 fused indexer --------------------------------------------------

    def _qli_v3_fallback_reason(
        self,
        layer: Any,
        dsa: DsaMetadata,
        q_lens: list[int],
        shared: CSA2SharedRuntime,
        index_cache: torch.Tensor | None,
    ) -> str | None:
        """``None`` when the fused QLI v3 op may run, otherwise why it stays eager.

        Every clause is a host constant, a tensor *shape*/dtype or a
        registered-op probe (never a device value), so the dispatch decision is
        identical on capture and replay. The reason string feeds the one-shot
        fallback log. Candidate-pool layers (config-driven via
        ``candidate_source_layer_id``) fall through to the plane checks; only an
        out-of-contract candidate config (which the op's tiling would reject)
        is reported, and that reason is deliberately NOT silent.

        The prefill extension opens the two decode-shape gates (ACL-graph
        step, ``q_len == 1``) for an *eager* prefill step only. Every other
        step -- decode, an ACL-graph step -- keeps the historical clause
        order and reason strings exactly (same returns, logs and ops).
        """
        acl_graph = bool(getattr(dsa, "is_acl_graph", False))
        prefill_step = any(int(length) != 1 for length in q_lens)
        # Eager-only: the prefill extension opens the two decode-shape gates for
        # an *eager* prefill step. ACL-graph steps are single-token by
        # construction (``execute_csa2_layer`` hardcodes ``q_lens = [1] * ...``
        # whenever ``dsa.is_acl_graph``), so the ``not acl_graph`` clause is a
        # defensive invariant rather than a live branch -- but it keeps the
        # "eager only" contract explicit for any future caller that feeds a
        # graph step a chunk-shaped query.
        if not (prefill_step and not acl_graph):
            # Not an eager prefill: the exact historical two checks (order
            # included) so the decision and its reason never move.
            if not acl_graph:
                return "not an ACL-graph decode step"
            if any(int(length) != 1 for length in q_lens):
                return "query length != 1"
        if self.index_head_dim != _QLI_V3_HEAD_DIM:
            return f"unsupported index head dim {self.index_head_dim} (op requires {_QLI_V3_HEAD_DIM})"
        if int(shared.compress_ratio) not in _QLI_V3_SUPPORTED_RATIOS:
            return f"unsupported compression ratio {shared.compress_ratio}"
        # The static per-bucket capacity is a decode-graph contract; an eager
        # prefill derives its committed capacity from the live context lengths
        # (``_committed_capacity``), so the clause does not apply there.
        bucketed_capacity = int(getattr(dsa, "graph_token_capacity", 0) or 0)
        if not prefill_step and bucketed_capacity <= 0:
            return "missing static graph token capacity"
        is_candidate_layer = layer.layer_id == self.candidate_source_layer_id or self._uses_candidates(layer.layer_id)
        if is_candidate_layer:
            config_reason = self._qli_v3_candidate_config_reason()
            if config_reason is not None:
                return config_reason
        index_reason = self._qli_index_plane_ready(shared, index_cache)
        if index_reason is not None:
            return index_reason
        if _get_qli_v3_ops() is None:
            return "quant_lightning_indexer_v2_va / quant_lightning_indexer_v2_metadata_va are not registered"
        return None

    def _qli_index_plane_ready(
        self,
        shared: CSA2SharedRuntime,
        index_cache: torch.Tensor | None,
    ) -> str | None:
        """Static readiness of the INT8 index plane operands the op consumes.

        The fused op reads the *main* INT8 index cache (``shared.index_cache``)
        plus its FP16 per-token scale (``shared.index_scale``), both allocated
        by the platform policy and addressed by the TOKEN-group block table.
        Every clause is a tensor *shape*/dtype or a host None-check, never a
        device value. Returns ``None`` when the plane is ready, otherwise the
        reason it is not (the caller forwards it to the one-shot fallback log).

        Known interaction: the index plane is always INT8 (the model-dtype
        route was removed), so the None/dtype clauses can no longer fire on a
        healthy policy; they are kept as a conservative guard against a policy
        or platform regression. The QLI side degrades gracefully when the plane
        is missing (this clause reports it and the caller falls back to the
        eager indexer), but note the eager path itself requires the INT8 plane
        -- including its scale -- to be ready: it gathers ``shared.index_scale``
        unconditionally (``_gather_cache_rows`` reads ``cache.device`` on its
        first line), so a missing scale faults there instead of degrading. On
        NPU the plane and its scale are allocated in pairs, so this branch is
        unreachable there.
        """
        key = index_cache if index_cache is not None else shared.index_cache
        scale = shared.index_scale
        if key is None or scale is None:
            missing = "key" if key is None else "scale"
            return f"index INT8 plane not ready ({missing} is None)"
        if key.dtype != torch.int8 or scale.dtype != torch.float16:
            return f"index plane dtype mismatch (key {key.dtype}, scale {scale.dtype})"
        if key.dim() != 4 or int(key.size(2)) != _QLI_V3_KEY_NUM_HEADS:
            return f"index key must be [blocks, tokens, {_QLI_V3_KEY_NUM_HEADS}, head] (got {tuple(key.shape)})"
        if scale.dim() != 3 or int(scale.size(2)) != _QLI_V3_KEY_NUM_HEADS:
            return f"index scale must be [blocks, tokens, {_QLI_V3_KEY_NUM_HEADS}] (got {tuple(scale.shape)})"
        if int(key.size(3)) != self.index_head_dim:
            return f"index key head dim {int(key.size(3))} does not match {self.index_head_dim}"
        if int(scale.size(0)) != int(key.size(0)) or int(scale.size(1)) != int(key.size(1)):
            return f"index scale geometry {tuple(scale.shape)} does not match the index key {tuple(key.shape)}"
        if shared.index_block_table is None:
            return "index INT8 PA_BBND needs the TOKEN-group block table"
        return None

    def _qli_v3_candidate_config_reason(self) -> str | None:
        """``None`` when the two-level candidate config matches the op tiling.

        The A3 tiling only implements ``candidate_block_size == 8`` with a
        positive ``candidate_topk_blocks`` that is a multiple of 64 and at most
        2048 (``quant_lightning_indexer_v2_va/op_host/quant_lightning_indexer_v2_va_tiling.cpp``). With the
        candidate pool configured these values are handed to the op, so a
        violation is a misconfiguration: fall back eagerly and report it (the
        reason is not in ``_QLI_V3_SILENT_FALLBACK_REASONS``), never silently
        feed the op an out-of-contract configuration.
        """
        if self.candidate_block_size != _QLI_V3_CAND_BLOCK_SIZE:
            return f"candidate block size {self.candidate_block_size} (op requires {_QLI_V3_CAND_BLOCK_SIZE})"
        if (
            self.candidate_topk_blocks <= 0
            or self.candidate_topk_blocks > _QLI_V3_CAND_TOPK_BLOCKS_MAX
            or self.candidate_topk_blocks % _QLI_V3_CAND_TOPK_BLOCKS_ALIGN != 0
        ):
            return (
                f"candidate topk blocks {self.candidate_topk_blocks} (op requires a "
                f"positive multiple of {_QLI_V3_CAND_TOPK_BLOCKS_ALIGN}, "
                f"<= {_QLI_V3_CAND_TOPK_BLOCKS_MAX})"
            )
        return None

    def _qli_state_for(self, dsa: DsaMetadata) -> _QliGraphState:
        """Return (or, outside capture, build) the state of this graph bucket."""
        token_capacity = int(getattr(dsa, "graph_token_capacity", 0) or 0)
        if token_capacity <= 0:
            # Non-graph (eager) path: there is no static per-bucket capacity, so
            # derive the token-level bound from the actual sequence lengths. The
            # host sync is free in eager (this branch is never taken under
            # capture, where graph_token_capacity > 0).
            seq_lens = dsa.seq_lens
            token_capacity = int(seq_lens.max().item()) if seq_lens is not None and seq_lens.numel() > 0 else 0
        key = (
            int(dsa.seq_lens.numel()),
            token_capacity,
        )
        state = self._qli_state.get(key)
        if state is not None:
            return state
        if _is_stream_capturing():
            raise RuntimeError(
                "DeepSeek-V4.1 QLI v3 state must be built in the eager warmup, before "
                f"ACL graph capture; it is missing for bucket (rows={key[0]}, capacity={key[1]})"
            )
        state = self._build_qli_state(dsa, key)
        self._qli_state[key] = state
        return state

    def _build_qli_state(self, dsa: DsaMetadata, key: tuple[int, int]) -> _QliGraphState:
        """Allocate one bucket's persistent QLI operands (never under capture).

        The initial refresh both fills the buffers and runs the metadata op for
        the first time, so its JIT compilation lands in the eager warmup the
        graph runner performs before capture.
        """
        rows, capacity = key
        device = dsa.seq_lens.device
        state = _QliGraphState(
            rows=rows,
            capacity=capacity,
            cu_seqlens_q=torch.arange(rows + 1, dtype=torch.int32, device=device),
            seqused_k={},
            cmp_residual={},
            metadata={},
        )
        for ratio in _QLI_V3_SUPPORTED_RATIOS:
            # PA_BBND addresses the paged shadow key through the runtime block
            # table plus the per-row ``seqused_k`` (compressed committed rows),
            # so the only constant descriptor is the query side.
            state.seqused_k[ratio] = torch.zeros(rows, dtype=torch.int32, device=device)
            state.cmp_residual[ratio] = torch.zeros(rows, dtype=torch.int32, device=device)
            state.metadata[ratio] = torch.zeros(_QLI_V3_METADATA_SIZE, dtype=torch.int32, device=device)
        self._refresh_qli_state(dsa, state)
        return state

    def _refresh_qli_state(self, dsa: DsaMetadata, state: _QliGraphState) -> None:
        """Write replay values into one state's buffers (in place, no sync)."""
        seq_lens = dsa.seq_lens
        for ratio in state.metadata:
            state.seqused_k[ratio].copy_(torch.div(seq_lens, ratio, rounding_mode="floor"))
            state.cmp_residual[ratio].copy_(torch.remainder(seq_lens, ratio))
            self._fill_qli_metadata(dsa, state, ratio)

    def _fill_qli_metadata(self, dsa: DsaMetadata, state: _QliGraphState, ratio: int) -> None:
        """Rebuild one ratio class's tiling metadata into its persistent buffer."""
        ops = _get_qli_v3_ops()
        if ops is None:
            return
        _, metadata_op = ops
        cap = max(state.capacity // ratio, 1)
        value = metadata_op(
            self.index_n_heads,
            1,
            self.index_head_dim,
            min(self.index_topk, cap),
            _QLI_V3_QUANT_MODE,
            cu_seqlens_q=state.cu_seqlens_q,
            # PA_BBND resolves K positions through the runtime block table and
            # ``seqused_k``; the tiling requires cu_seqlens_k to be absent
            # (``...tiling.cpp:619-631``). ``max_seqlen_k`` is the paged-cache
            # length bound: vllm-ascend feeds ``max_cache_seq_len`` (the largest
            # committed compressed length of the batch, ``dsa_v41.py:925-954``);
            # xllm's graph-stable upper bound is the per-bucket per-sequence
            # capacity ``cap``. ASSUMPTION (not yet on-device): the metadata
            # accepts the bound instead of the exact max.
            cu_seqlens_k=None,
            seqused_k=state.seqused_k[ratio],
            # ``cmp_ratio==1`` has no residual semantics: pass ``None`` exactly
            # like the indexer op does. The vendor host check no longer
            # dereferences ``cmp_residual_k`` before its IsTensorExist guard
            # (fixed in xllm_ops ``...metadata_va_check.h``), so ``None`` is now
            # the canonical value for ratio==1.
            cmp_residual_k=state.cmp_residual[ratio] if ratio > 1 else None,
            batch_size=state.rows,
            max_seqlen_q=1,
            max_seqlen_k=cap,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=_QLI_V3_MASK_MODE,
            cmp_ratio=ratio,
            device=str(state.cu_seqlens_q.device),
        )
        state.metadata[ratio].copy_(value)

    def _run_indexer_qli_v3(
        self,
        layer: Any,
        q_idx: torch.Tensor,
        ctx: Csa2LayerContext,
        dsa: DsaMetadata,
        q_lens: list[int],
        shared: CSA2SharedRuntime,
        block_table: torch.Tensor,
        capacity: int,
    ) -> torch.Tensor:
        """One fused ``quant_lightning_indexer_v2_va`` call for a decode-graph step.

        ``q_idx`` is the RoPE-applied index query (pre-INT8-quant). The read side
        quantizes only the query per (token, head): the key is the persistent
        INT8 index cache the compressor wrote (with its FP16 per-token scale),
        addressed by the TOKEN-group ``block_table`` + ``seqused_k`` -- matching
        vllm-ascend's call shape. The returned per-sequence 0-based compressed
        rows are re-based to the window-offset indices ``_csa2_attention``
        consumes (``_indexer_topk`` semantics with ``q_len == 1``), and ``-1``
        padding survives.

        With a configured candidate pool (``candidate_source_layer_id >= 0``)
        the candidate source layer runs ``candidate_mode=1`` and publishes the
        op's third output (the ``[rows, 1, candidate_topk_blocks]`` INT32
        block-index pool) on ``shared.candidate_blocks``; the layers that
        consume the pool run ``candidate_mode=2`` and feed that tensor back as
        ``candidate_topk_index``. Every other layer runs
        ``candidate_mode=3`` with no candidate operand, byte for byte the
        historical call.

        An *eager* prefill step (any ``q_len > 1``) routes to
        :meth:`_run_indexer_qli_v3_prefill`; every other step stays on the
        decode call below.

        Eager-only coupling (explicit anchor): the dispatch's prefill relaxation
        carries a ``not is_acl_graph`` clause, and ``execute_csa2_layer``
        hardcodes ``q_lens = [1] * ...`` whenever ``dsa.is_acl_graph``, so this
        branch is only ever reached for an eager step. The ``q_len != 1`` test is
        therefore load-bearing (it is what separates a real prefill from an eager
        single-token step) and must not be dropped.
        """
        if any(int(length) != 1 for length in q_lens):
            return self._run_indexer_qli_v3_prefill(layer, q_idx, ctx, dsa, q_lens, shared, block_table, capacity)
        ops = _get_qli_v3_ops()
        if ops is None:
            raise RuntimeError("DeepSeek-V4.1 QLI v3 op became unavailable after dispatch")
        indexer, _ = ops
        ratio = int(shared.compress_ratio)
        state = self._qli_state_for(dsa)
        # Width matches the eager contract (``k_width = min(index_topk, capacity)``).
        topk_width = min(self.index_topk, capacity)
        key = shared.index_cache
        key_scale = shared.index_scale
        if key is None or key_scale is None:
            raise RuntimeError("DeepSeek-V4.1 QLI v3 INT8 index plane disappeared after dispatch")
        query_int8, query_scale = _quantize_indexer_rows_int8(q_idx)
        weights = layer.indexer_weights(ctx.x).to(torch.float16)
        # Two-level candidate pool: the config-driven mode adds the candidate
        # operands; a non-candidate call stays identical.
        candidate_mode = _QLI_V3_CAND_MODE_OFF
        candidate_kwargs: dict[str, Any] = {}
        if self.candidate_source_layer_id >= 0:
            if layer.layer_id == self.candidate_source_layer_id:
                candidate_mode = _QLI_V3_CAND_MODE_SOURCE
            elif self._uses_candidates(layer.layer_id):
                candidate_mode = _QLI_V3_CAND_MODE_CONSUMER
                pool = shared.candidate_blocks
                # Defensive invariant: every guard above is step-uniform, so a
                # consumer in this forward must have been preceded by the source
                # layer, which republishes ``candidate_blocks`` and stamps its id.
                # If a future path ever hands a consumer a pool this forward did
                # not (re)produce -- e.g. an ACL-graph replay that skipped the
                # source -- the marker mismatches and the layer fails loud instead
                # of reading the previous forward's stale pool.
                if (
                    pool is None
                    or int(pool.numel()) == 0
                    or shared.candidate_blocks_layer != self.candidate_source_layer_id
                ):
                    raise RuntimeError(
                        f"DeepSeek-V4.1 reindex layer {layer.layer_id} requires the candidate "
                        f"pool from layer {self.candidate_source_layer_id} (this forward "
                        f"published candidate_blocks_layer={shared.candidate_blocks_layer})"
                    )
                candidate_kwargs["candidate_topk_index"] = pool
            if candidate_mode != _QLI_V3_CAND_MODE_OFF:
                candidate_kwargs["candidate_topk_blocks"] = self.candidate_topk_blocks
                candidate_kwargs["candidate_block_size"] = self.candidate_block_size
        sparse_indices, _, candidate_out = indexer(
            query_int8.contiguous(),
            key,
            weights.contiguous(),
            query_scale.contiguous(),
            key_scale,
            topk_width,
            _QLI_V3_QUANT_MODE,
            # PA_BBND: the paged index plane is addressed by the block table and
            # the per-row seq-used lengths; cu_seqlens_k must NOT be provided
            # (``...tiling.cpp:619-631``), exactly like vllm-ascend's call.
            block_table=block_table,
            cu_seqlens_q=state.cu_seqlens_q,
            cu_seqlens_k=None,
            seqused_k=state.seqused_k[ratio],
            cmp_residual_k=state.cmp_residual[ratio] if ratio > 1 else None,
            metadata=state.metadata[ratio],
            max_seqlen_q=1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=_QLI_V3_MASK_MODE,
            cmp_ratio=ratio,
            candidate_mode=candidate_mode,
            **candidate_kwargs,
        )
        if candidate_mode != _QLI_V3_CAND_MODE_OFF:
            # The candidate operands reached the op (mode 1/2), so the fused
            # two-level pool is genuinely in effect; the non-candidate path
            # never calls this (candidate_mode stays OFF), keeping its run
            # byte-identical.
            _log_qli_v3_candidate_active(layer.layer_id)
        if candidate_mode == _QLI_V3_CAND_MODE_SOURCE:
            # Publish the block-index pool for this forward's consumers: the op
            # returns ``[rows, 1, candidate_topk_blocks]`` INT32 for mode 1 (an
            # empty tensor otherwise). The tensor is a graph-internal op output
            # (the same allocation class as ``sparse_indices``), so the Python
            # reference on the shared runtime adds no capture-time allocation.
            shared.candidate_blocks = candidate_out
            shared.candidate_blocks_layer = layer.layer_id
        # Per-sequence 0-based compressed rows -> the window-row offset of the
        # eager indexer (``offset = window_size + q_len`` with q_len == 1).
        # The op is the only place that can return a row at or past the
        # sequence's committed count (a padded / out-of-range lane), so filter
        # by PA_BBND's own ``seqused_k`` (the exact count the op addressed)
        # and sort ascending before rebasing (``_rebase_qli_picks``).
        offset = self.window_size + 1
        picks = sparse_indices[:, 0, :]
        visible = state.seqused_k[ratio].unsqueeze(-1)
        rows_out = _rebase_qli_picks(picks, visible, offset)
        topk = rows_out.unsqueeze(1)
        shared.topk_idxs = topk
        shared.topk_layer = layer.layer_id
        return topk

    def _run_indexer_qli_v3_prefill(
        self,
        layer: Any,
        q_idx: torch.Tensor,
        ctx: Csa2LayerContext,
        dsa: DsaMetadata,
        q_lens: list[int],
        shared: CSA2SharedRuntime,
        block_table: torch.Tensor,
        capacity: int,
    ) -> torch.Tensor:
        """One fused ``quant_lightning_indexer_v2_va`` call for an eager prefill step.

        Reachable on a step whose query has at least one sequence with
        ``q_len > 1`` (the dispatch opens the two decode-shape gates for this
        eager-prefill shape). Unlike the decode path
        there is no ACL-graph bucket state: the TND batch descriptors are built
        per call from the chunk geometry, which the eager path owns as host
        constants (``q_lens`` is a Python list here, never a device read).

        ``q_idx`` packs every sequence's RoPE-applied index query in order
        (``sum(q_lens)`` rows); the op returns the packed per-token selections,
        which are rebased per sequence (``offset = window_size + q_len``) and
        scattered into the same rectangular ``[seq, s_max, k_width]`` tensor the
        pure-torch indexer publishes (``-1`` padding in the unused rows/columns).

        On-device points that this CPU wiring cannot settle (see the C1 plan
        assumptions A1/A3): whether the A3 tiling accepts the eager TND prefill
        shape, and whether the op's internal ``mask_mode=3`` causal bound for a
        chunked continuation (``prev_ctx > 0``) matches the eager per-query
        visibility ``(p + 1) // ratio``. The mechanism-door trace and the
        per-layer picks diff are the on-device checks for both.
        """
        ops = _get_qli_v3_ops()
        if ops is None:
            raise RuntimeError("DeepSeek-V4.1 QLI v3 op became unavailable after dispatch")
        indexer, metadata_op = ops
        ratio = int(shared.compress_ratio)
        device = q_idx.device
        seq_count = len(q_lens)
        total_q = sum(int(length) for length in q_lens)
        s_max = max(int(length) for length in q_lens) if q_lens else 0
        k_width = min(self.index_topk, capacity) if capacity > 0 else 0
        # The eager hand-off is the rectangular top-k of ``_run_indexer``; build
        # it first so the degenerate cases below publish the identical object.
        topk = torch.full((seq_count, s_max, k_width), -1, dtype=torch.int32, device=device)
        if capacity <= 0 or total_q == 0 or k_width == 0:
            shared.topk_idxs = topk
            shared.topk_layer = layer.layer_id
            return topk

        key = shared.index_cache
        key_scale = shared.index_scale
        if key is None or key_scale is None:
            raise RuntimeError("DeepSeek-V4.1 QLI v3 INT8 index plane disappeared after dispatch")

        ctx_lens = _ensure_seq_lens_tensor(dsa.seq_lens, device)
        # TND packed-query offsets: host arithmetic over the chunk geometry, so
        # no capture-time allocation constraint applies (eager path).
        offsets = [0]
        for length in q_lens:
            offsets.append(offsets[-1] + int(length))
        cu_seqlens_q = torch.tensor(offsets, dtype=torch.int32, device=device)
        # PA_BBND per-sequence addressed length: the committed compressed rows
        # floor(ctx_len / ratio), i.e. the eager indexer's visibility bound at
        # the sequence's last query (``(p + 1) // ratio`` for ``p = ctx_len - 1``).
        committed = torch.div(ctx_lens, ratio, rounding_mode="floor").to(torch.int32)
        cmp_residual = torch.remainder(ctx_lens, ratio).to(torch.int32) if ratio > 1 else None
        # ratio==1 has no residual semantics: pass ``None`` exactly like the
        # indexer call below. The vendor host check no longer dereferences
        # ``cmp_residual_k`` before its IsTensorExist guard (fixed in xllm_ops
        # ``...metadata_va_check.h``), so ``None`` is now canonical.
        metadata_residual = cmp_residual
        max_q = s_max
        metadata = metadata_op(
            self.index_n_heads,
            1,
            self.index_head_dim,
            k_width,
            _QLI_V3_QUANT_MODE,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=None,
            seqused_k=committed,
            cmp_residual_k=metadata_residual,
            batch_size=seq_count,
            max_seqlen_q=max_q,
            max_seqlen_k=capacity,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=_QLI_V3_MASK_MODE,
            cmp_ratio=ratio,
            device=str(device),
        )
        query_int8, query_scale = _quantize_indexer_rows_int8(q_idx)
        weights = layer.indexer_weights(ctx.x).to(torch.float16)
        # Two-level candidate pool: kept in lockstep with the decode branch of
        # :meth:`_run_indexer_qli_v3` (same modes, same fail-loud invariant).
        candidate_mode = _QLI_V3_CAND_MODE_OFF
        candidate_kwargs: dict[str, Any] = {}
        if self.candidate_source_layer_id >= 0:
            if layer.layer_id == self.candidate_source_layer_id:
                candidate_mode = _QLI_V3_CAND_MODE_SOURCE
            elif self._uses_candidates(layer.layer_id):
                candidate_mode = _QLI_V3_CAND_MODE_CONSUMER
                pool = shared.candidate_blocks
                if (
                    pool is None
                    or int(pool.numel()) == 0
                    or shared.candidate_blocks_layer != self.candidate_source_layer_id
                ):
                    raise RuntimeError(
                        f"DeepSeek-V4.1 reindex layer {layer.layer_id} requires the candidate "
                        f"pool from layer {self.candidate_source_layer_id} (this forward "
                        f"published candidate_blocks_layer={shared.candidate_blocks_layer})"
                    )
                candidate_kwargs["candidate_topk_index"] = pool
            if candidate_mode != _QLI_V3_CAND_MODE_OFF:
                candidate_kwargs["candidate_topk_blocks"] = self.candidate_topk_blocks
                candidate_kwargs["candidate_block_size"] = self.candidate_block_size
        sparse_indices, _, candidate_out = indexer(
            query_int8.contiguous(),
            key,
            weights.contiguous(),
            query_scale.contiguous(),
            key_scale,
            k_width,
            _QLI_V3_QUANT_MODE,
            block_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=None,
            seqused_k=committed,
            cmp_residual_k=cmp_residual,
            metadata=metadata,
            max_seqlen_q=max_q,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=_QLI_V3_MASK_MODE,
            cmp_ratio=ratio,
            candidate_mode=candidate_mode,
            **candidate_kwargs,
        )
        if candidate_mode != _QLI_V3_CAND_MODE_OFF:
            _log_qli_v3_candidate_active(layer.layer_id)
        if candidate_mode == _QLI_V3_CAND_MODE_SOURCE:
            shared.candidate_blocks = candidate_out
            shared.candidate_blocks_layer = layer.layer_id
        # Packed per-token picks -> the eager rectangular hand-off. Each
        # sequence's slice is filtered/sorted/rebased with that sequence's own
        # window offset (``window_size + q_len``), exactly the eager
        # ``_indexer_topk(..., offset)`` contract; ``visible`` is the per-query
        # visibility ``(p + 1) // ratio`` the eager indexer masks with, so an
        # out-of-range op lane is dropped instead of re-based onto a window row.
        picks = sparse_indices[:, 0, :]
        token_offset = 0
        for seq, q_len in enumerate(q_lens):
            q_len = int(q_len)
            if q_len <= 0:
                continue
            segment = picks[token_offset : token_offset + q_len]
            positions_seq = ctx.positions[token_offset : token_offset + q_len].to(torch.int64)
            visible = torch.div(positions_seq + 1, ratio, rounding_mode="floor").unsqueeze(-1)
            rows_out = _rebase_qli_picks(
                segment,
                visible,
                self.window_size + q_len,
                # FP32 kernel; bit-equal to the INT32 one (see the sentinel
                # note above).
                fp32_sort=True,
            )
            width = min(int(rows_out.size(1)), k_width)
            if width > 0:
                topk[seq, :q_len, :width] = rows_out[:, :width]
            token_offset += q_len
        shared.topk_idxs = topk
        shared.topk_layer = layer.layer_id
        return topk

    def _uses_candidates(self, layer_id: int) -> bool:
        return 0 <= self.candidate_source_layer_id < layer_id

    # -- sparse attention ----------------------------------------------------

    def _csa2_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        layer: Any,
        dsa: DsaMetadata,
        q_lens: list[int],
        ctx_lens: torch.Tensor,
        topk: torch.Tensor | None,
        shared: CSA2SharedRuntime,
        cp_context: DsaCpContext | None = None,
    ) -> torch.Tensor:
        """Gather ``[window rows ; top-k compressed rows]`` and run the sink softmax.

        Per (batch, query) the window covers positions ``max(0, p - win + 1)..p``
        at a fixed ``win + q_len`` row capacity (ring candidates masked per
        query to ``pos >= max(p - win + 1, 0)``, the current chunk appended
        after), the compressed rows
        come from the group's shared key cache at the committed capacity, and
        the attention sink contributes only to the softmax normalizer
        (kernel.py:310-389). All index math is tensor arithmetic, so the whole
        method replays correctly inside a captured ACL graph.
        """
        mode = layer.mode
        layer_id = layer.layer_id
        ratio = mode.compress_ratio
        win = self.window_size
        head_dim = q.size(-1)
        ctx_lens = _ensure_seq_lens_tensor(ctx_lens, q.device)
        layer_cache = self._kv_caches[layer_id]
        mapping = self._resolve_cache_mapping(layer_id, ratio)
        swa_bt = _get_layer_cache_tensor(dsa.block_tables, layer_id, mapping.ori_cache_idx)
        query_layout = shared.query_layout
        shared_window = query_layout is not None and query_layout.shared_window
        token_rows = query_layout is not None and (dsa.is_acl_graph or query_layout.token_rows)
        if token_rows and swa_bt is not None:
            swa_bt = swa_bt[:: query_layout.query_width]
        swa_bs = layer_cache.swa.size(1) if layer_cache.swa is not None else win
        global_q_lens = list(cp_context.global_q_lens) if cp_context is not None else q_lens
        global_ctx_lens = _ensure_seq_lens_tensor(dsa.seq_lens, q.device) if cp_context is not None else ctx_lens

        sinks = layer.attn_sink.to(torch.float32)
        swa_flat = layer_cache.swa.view(-1, head_dim) if layer_cache.swa is not None else None
        out = torch.empty_like(q)
        token_offset = 0
        global_token_offset = 0
        for seq, q_len in enumerate(q_lens):
            global_q_len = global_q_lens[seq]
            global_prev_ctx = global_ctx_lens[seq] - global_q_len
            kv_seq = kv[global_token_offset : global_token_offset + global_q_len]
            global_token_offset += global_q_len
            if q_len == 0:
                continue
            ctx_len = ctx_lens[seq]
            prev_ctx = ctx_len - q_len
            q_seq = q[token_offset : token_offset + q_len]
            # Window rows at fixed capacity: [win ring candidates ; chunk].
            # Ring candidate c sits at absolute position prev - win + c; which
            # candidates a given query may read is decided per query below.
            # Native SMLA is selected only for the validated V4.1 Flash dimensions.
            # SMLA uses a causal window; shared-window blocks also need future
            # rows within the block and retain the non-causal reference path.
            use_smla = (
                not shared_window
                and q.device.type == "npu"
                and head_dim == 512
                and q.dtype == torch.bfloat16
                and win == 128
                and (topk is None or topk.size(-1) <= _SMLA_CMP_TOPK_MAX)
                and _supports_smla_va()
            )
            if use_smla and global_q_len == 1 and swa_flat is not None and swa_bt is not None:
                from xllm.python.kernels_npu.triton.smla_window_pack import (
                    can_pack_smla_window,
                    pack_smla_window,
                )

                if can_pack_smla_window(swa_flat, swa_bt, kv_seq, global_ctx_lens, seq, swa_bs, win):
                    window_kv = pack_smla_window(swa_flat, swa_bt, kv_seq, global_ctx_lens, seq, swa_bs, win)
                    out[token_offset : token_offset + q_len] = self._smla_attention(
                        q_seq,
                        window_kv,
                        layer,
                        ctx_lens,
                        global_ctx_lens,
                        global_q_len,
                        q_len,
                        seq,
                        topk,
                        shared,
                        sinks,
                    )
                    token_offset += q_len
                    continue
            ring_pos = (global_prev_ctx - win) + torch.arange(win, device=q.device)
            if swa_flat is not None and swa_bt is not None:
                ring_rows = self._gather_swa_positions(swa_flat, swa_bt, seq, ring_pos.clamp_min(0), swa_bs)
            else:
                # Profiling / warmup batches carry no block tables: keep the
                # traced shapes intact with zeros; the mask below drops them.
                ring_rows = q.new_zeros((win, kv_seq.size(-1)))
            # The production ring is allocated with empty(), not zeros().
            # Negative history candidates (and padded graph requests) are not
            # real KV. Masking only scores is insufficient: 0 * NaN in the
            # value reduction still poisons every output, even on first prefill.
            ring_rows = torch.where((ring_pos >= 0).unsqueeze(-1), ring_rows, 0)
            kv_seq = torch.where(global_ctx_lens[seq] > 0, kv_seq, 0)
            window_kv = torch.cat((ring_rows, kv_seq), dim=0)
            if use_smla:
                out[token_offset : token_offset + q_len] = self._smla_attention(
                    q_seq,
                    window_kv,
                    layer,
                    ctx_lens,
                    global_ctx_lens,
                    global_q_len,
                    q_len,
                    seq,
                    topk,
                    shared,
                    sinks,
                )
                token_offset += q_len
                continue
            # Window index matrix: query p (position prev+i) sees exactly the
            # window rows whose absolute position lies in
            # [max(p - win + 1, 0), p]; the lower bound tightens with the query
            # (only the chunk's first token shares the chunk-level bound).
            positions_seq = prev_ctx + torch.arange(q_len, device=q.device)
            if shared_window:
                col_pos = torch.cat((ring_pos, positions_seq), dim=0)
                cols = torch.arange(window_kv.size(0), device=q.device)
                window_idxs = _block_visible_idxs(prev_ctx, q_len, col_pos, cols, win)
            else:
                # Fixed chronological picks keep target reductions independent
                # of chunk length; the draft retains its complete query block.
                window_idxs = _window_visible_idxs(positions_seq, global_prev_ctx, win)

            # Compressed rows for the top-k picks (shared group cache),
            # gathered at the committed capacity; padding rows are unreachable
            # because every top-k pick past `committed` is -1.
            capacity = 0
            if ratio > 0 and topk is not None and shared.key_cache is not None:
                capacity = self._committed_capacity(
                    dsa,
                    ratio,
                    shared.key_block_table,
                    shared.key_cache.size(1),
                    ctx_lens,
                )
            compress_kv = None
            if capacity > 0:
                shared_key_cache = shared.key_cache
                shared_key_table = shared.key_block_table
                assert shared_key_cache is not None and shared_key_table is not None
                compress_kv = self._gather_cache_rows(
                    shared_key_cache,
                    shared_key_table,
                    seq,
                    capacity,
                    shared_key_cache.size(1),
                )
                committed = torch.div(ctx_len, ratio, rounding_mode="floor")
                valid_rows = torch.arange(capacity, device=q.device) < committed
                compress_kv = torch.where(valid_rows.unsqueeze(-1), compress_kv, 0)
            if compress_kv is not None:
                assert topk is not None
                kv_cat = torch.cat((window_kv, compress_kv), dim=0)
                compressed_idxs = topk[seq][:q_len]
                if cp_context is not None:
                    compressed_idxs = torch.where(
                        compressed_idxs >= 0, compressed_idxs + global_q_len - q_len, compressed_idxs
                    )
                idxs = torch.cat((window_idxs, compressed_idxs), dim=-1)
            else:
                kv_cat = window_kv
                idxs = window_idxs
            # Keep top-k padding width independent of committed capacity.
            # Padding and unreachable cache rows are zeroed before *any* math:
            # score masking alone cannot stop 0 * NaN in the value reduction.
            pick_width = window_idxs.size(-1) + (self.index_topk if ratio > 0 else 0)
            idxs = torch.nn.functional.pad(idxs, (0, pick_width - idxs.size(-1)), value=-1)
            # Shape-only budget: no device reads or data-dependent graph branch.
            # Keep every query's picks/reduction unchanged; split query rows only.
            kv_bytes_per_query = 2 * pick_width * head_dim * 4
            query_tile = max(1, min(_ATTENTION_QUERY_TILE, _ATTENTION_KV_TILE_BYTES // kv_bytes_per_query))
            for tile_start in range(0, q_len, query_tile):
                tile_end = min(tile_start + query_tile, q_len)
                picks = idxs[tile_start:tile_end].to(torch.int64)
                valid = (picks >= 0) & (picks < kv_cat.size(0))
                picked_kv = kv_cat[picks.clamp(0, kv_cat.size(0) - 1)].float()
                picked_kv = torch.where(valid.unsqueeze(-1), picked_kv, 0)
                scores = torch.einsum("shd,snd->shn", q_seq[tile_start:tile_end].float(), picked_kv)
                scores = scores * self.scale
                scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
                row_max = scores.amax(dim=-1).clamp_min(_NEG_INF_FLOOR)
                probs = torch.exp(scores - row_max.unsqueeze(-1))
                # The sink adds exp(sink - row_max) to the normalizer only.
                sum_exp = probs.sum(dim=-1) + torch.exp(sinks - row_max)
                tile_out = torch.einsum("shn,snd->shd", probs, picked_kv)
                tile_out = tile_out / sum_exp.unsqueeze(-1)
                out[token_offset + tile_start : token_offset + tile_end] = tile_out.to(q.dtype)
                # Do not retain this large tensor during the next tile's gather.
                del picked_kv
            token_offset += q_len
        return out

    def _smla_attention(
        self,
        q: torch.Tensor,
        window_kv: torch.Tensor,
        layer: Any,
        ctx_lens: torch.Tensor,
        global_ctx_lens: torch.Tensor,
        global_q_len: int,
        q_len: int,
        seq: int,
        topk: torch.Tensor | None,
        shared: CSA2SharedRuntime,
        sinks: torch.Tensor,
    ) -> torch.Tensor:
        """Run the vLLM DSV4.1 A3 SMLA kernel with the existing cache layout.

        The SWA ring and current chunk form one temporary paged KV view.
        Compressed KV stays in its shared paged cache. Independent compressed
        lengths/residuals preserve causal positions even for a CP query shard.
        Inference length tensors must remain immutable between layers of one
        ForwardContext; start a new context before updating their values.
        """
        ratio = layer.mode.compress_ratio
        # Ratio 1 uses the separate projected KV cache without pooling; it
        # still needs the compressed-cache input of the SMLA interface.
        has_compressed = ratio > 0 and topk is not None and topk.size(-1) > 0 and shared.key_cache is not None
        if has_compressed:
            cache = shared.key_cache
            valid_layout = cache.ndim == 3 or (cache.ndim == 4 and cache.size(2) == 1)
            if not valid_layout or cache.size(-1) != q.size(-1):
                raise ValueError(
                    "SMLA compressed KV cache must have shape [num_blocks, block_size, head_dim] "
                    "or [num_blocks, block_size, 1, head_dim]"
                )
        ratio = ratio if has_compressed else 0
        compressed_width = (
            (topk.size(-1) + _SMLA_CMP_TOPK_BLOCK - 1) // _SMLA_CMP_TOPK_BLOCK * _SMLA_CMP_TOPK_BLOCK
            if has_compressed
            else 0
        )
        context = get_forward_context()
        state = context.layer_shared_cache.get(_SmlaForwardState)
        if state is None:
            state = _SmlaForwardState({})
            context.layer_shared_cache[_SmlaForwardState] = state
        compressed_capacity = shared.key_block_table.size(1) * shared.key_cache.size(1) if has_compressed else 0
        # Keep tensor identities and available version counters so distinct
        # CP plans and eager in-place updates cannot reuse stale inputs.
        # Inference tensors have no version counter and follow the immutable
        # per-forward length contract documented above.
        # Graph replay recomputes the captured device arithmetic from live
        # lengths; reading values with .item() here would break capture.
        key = (
            seq,
            q_len,
            global_q_len,
            q.size(1),
            q.size(2),
            q.device,
            self.window_size,
            window_kv.size(0),
            ratio,
            compressed_width,
            compressed_capacity,
            ctx_lens,
            global_ctx_lens,
            None if torch.is_inference(ctx_lens) else ctx_lens._version,
            None if torch.is_inference(global_ctx_lens) else global_ctx_lens._version,
        )
        inputs = state.inputs.get(key)
        if inputs is None:
            # Device arithmetic creates graph-safe constants and live lengths.
            starts = torch.arange(2, device=q.device, dtype=torch.int32) * q_len
            previous = global_ctx_lens[seq : seq + 1] - global_q_len
            local_end = ctx_lens[seq : seq + 1].to(torch.int32)
            ori_lens = (self.window_size + local_end - previous).clamp(min=0, max=window_kv.size(0)).to(torch.int32)
            visible = (
                (local_end - q_len + torch.arange(1, q_len + 1, device=q.device, dtype=torch.int32))
                .clamp(min=0, max=self.window_size)
                .view(-1, 1)
            )
            dropped_rows = getattr(context.metadata, "swa_bounded_replay_dropped_rows", ())
            if seq < len(dropped_rows) and dropped_rows[seq] > 0:
                # The retained suffix has no preceding consumer-layer rows.
                # Mask that missing prefix even when absolute positions are high.
                retained = torch.arange(1, q_len + 1, device=q.device, dtype=torch.int32).view(-1, 1)
                visible = torch.minimum(visible, retained)
            cmp_lens = torch.div(local_end, ratio, rounding_mode="floor") if ratio else None
            residual = local_end % ratio if ratio else None
            inputs = dict(
                cu_seqlens_q=starts,
                seqused_ori_kv=ori_lens,
                seqused_cmp_kv=cmp_lens,
                cmp_residual_kv=residual,
                ori_topk_length=visible,
            )
            inputs["metadata"] = torch.ops.xllm_ops.smla_va_metadata(
                q.size(1),
                1,
                q.size(2),
                **inputs,
                batch_size=1,
                max_seqlen_q=q_len,
                max_seqlen_ori_kv=window_kv.size(0),
                max_seqlen_cmp_kv=compressed_capacity,
                cmp_topk=compressed_width,
                cmp_ratio=ratio,
                ori_mask_mode=_SMLA_ORI_MASK_MODE,
                cmp_mask_mode=_SMLA_CMP_MASK_MODE if ratio else _SMLA_NO_MASK_MODE,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_BBND",
                has_ori_kv=True,
                has_cmp_kv=has_compressed,
            )
            state.inputs[key] = inputs
        block_size = self.window_size
        padding = (-window_kv.size(0)) % block_size
        ori_kv = torch.nn.functional.pad(window_kv, (0, 0, 0, padding)).view(-1, block_size, 1, q.size(-1))
        ori_table = torch.arange(ori_kv.size(0), device=q.device, dtype=torch.int32).view(1, -1)
        cmp_kv = cmp_table = cmp_indices = None
        if has_compressed:
            assert topk is not None and shared.key_cache is not None and shared.key_block_table is not None
            cmp_kv = shared.key_cache.view(*shared.key_cache.shape[:2], 1, q.size(-1))
            cmp_table = shared.key_block_table[seq : seq + 1]
            indices = topk[seq, :q_len]
            # The indexer encodes compressed picks after window_size + local
            # q_len rows. Under CP, window_kv also contains the full global
            # chunk, so its row count is NOT the indexer's offset.
            # The reference adds global_q_len - q_len to address the global
            # chunk, then subtracting window_size + global_q_len to address
            # the separate cache cancels that shift exactly.
            # SMLA consumes logical compressed rows and -1 padding. Reject
            # window picks and out-of-range rows instead of clamping them to
            # an unrelated valid KV row; all checks stay on the device.
            indexer_offset = self.window_size + q_len
            logical_indices = indices - indexer_offset
            valid_indices = (
                (indices >= indexer_offset)
                & (logical_indices < inputs["seqused_cmp_kv"].view(1, 1))
                & (logical_indices < compressed_capacity)
            )
            cmp_indices = torch.where(valid_indices, logical_indices, -1)
            cmp_indices = torch.nn.functional.pad(cmp_indices, (0, compressed_width - indices.size(-1)), value=-1)
            cmp_indices = cmp_indices.to(torch.int32).unsqueeze(1).contiguous()
        output, _ = torch.ops.xllm_ops.smla_va(
            q,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            cmp_sparse_indices=cmp_indices,
            ori_block_table=ori_table,
            cmp_block_table=cmp_table,
            **inputs,
            sinks=sinks,
            softmax_scale=self.scale,
            cmp_ratio=ratio,
            ori_mask_mode=_SMLA_ORI_MASK_MODE,
            cmp_mask_mode=_SMLA_CMP_MASK_MODE if ratio else _SMLA_NO_MASK_MODE,
            ori_win_left=self.window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_BBND",
        )
        return output

    # -- cache gather helpers -------------------------------------------------

    def _committed_capacity(
        self,
        dsa: DsaMetadata | None,
        ratio: int,
        block_table: torch.Tensor | None,
        block_size: int,
        ctx_lens: torch.Tensor,
    ) -> int:
        """Compressed-row gather capacity (rows) for one forward.

        Under ACL graph capture this is a Python constant baked into the
        graph: the runner's per-bucket bound ``graph_token_capacity`` when the
        captured entry carries one (the decode-graph runner sets it to
        ``bucket * granularity``), otherwise ``graph_token_capacity_granularity
        // ratio`` -- so no replay ever sync-reads the device sequence lengths.
        Eagerly it is the exact batch maximum, read on the host where the sync
        is free.
        """
        if ratio <= 0:
            return 0
        if bool(getattr(dsa, "is_acl_graph", False)):
            bucketed_capacity = int(getattr(dsa, "graph_token_capacity", 0) or 0)
            if bucketed_capacity > 0:
                return max(bucketed_capacity // ratio, 1)
            if self.graph_token_capacity_granularity > 0:
                return max(self.graph_token_capacity_granularity // ratio, 1)
            raise RuntimeError(
                "DeepSeek-V4.1 ACL graph capture needs a static committed-row "
                "capacity, but the serving entry did not derive one (no "
                "max_model_len was provided). The engine should have derived "
                "a default at construction; a missing value means the "
                "deployment entry point passed no max_model_len."
            )
        if ctx_lens.numel() == 0:
            return 0
        return max(int(torch.div(ctx_lens.max(), ratio, rounding_mode="floor").item()), 0)

    def _gather_cache_rows(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        seq: int,
        count: int,
        block_size: int,
    ) -> torch.Tensor:
        """Gather logical rows ``0..count-1`` of one sequence from a paged cache.

        ``count`` may exceed the sequence's committed rows (capacity gather):
        the unassigned tail blocks clamp to block 0 and produce garbage rows
        the caller masks out, keeping the gather shape graph-stable.
        """
        rows = torch.arange(count, device=cache.device)
        cols = torch.div(rows, block_size, rounding_mode="floor") % block_table.size(1)
        blocks = block_table[seq, cols].to(torch.int64).clamp_min(0)
        slots = blocks * block_size + (rows % block_size)
        return cache.view(-1, cache.size(-1)).index_select(0, slots)

    def _gather_swa_positions(
        self,
        swa_flat: torch.Tensor,
        block_table: torch.Tensor,
        seq: int,
        positions: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Read ring rows in the supplied absolute-position order.

        Each position resolves its own physical slot; index_select preserves
        that order even when block IDs wrap or are non-monotonic.
        Unassigned blocks clamp to block 0: the resulting garbage rows are
        produced only for out-of-window / padded candidates whose mask keeps
        them out of every score, but the clamped gather never indexes below
        the cache (which would crash a captured graph).
        """
        cols = torch.div(positions, block_size, rounding_mode="floor") % block_table.size(1)
        blocks = block_table[seq, cols].to(torch.int64).clamp_min(0)
        slots = blocks * block_size + (positions % block_size)
        return swa_flat.index_select(0, slots)

    def _v41_state_slots(
        self,
        dsa: DsaMetadata,
        layer_id: int,
        swa_cache_idx: int,
        seq: int,
        positions: torch.Tensor,
        state_rows: int,
    ) -> torch.Tensor:
        """V4.1 compressor state ring slots of absolute positions.

        The ratio-2 compressor state (``compress_kv_state`` /
        ``compress_score_state``) is a per-sequence ring of ``state_rows`` rows
        (vllm-ascend's ``STATE_RING_ROWS`` = 32), addressed by the SWA ring block
        and the position modulo the ring. It shares the SWA group's block table,
        so the ring block id is resolved by indexing the block table at
        ``position // block_size`` (wrapping over its columns) but the in-block
        offset wraps at ``state_rows`` instead of the SWA block size. Graph-safe:
        only device arithmetic, unassigned blocks clamp to block 0 (garbage rows
        the caller's commit mask keeps unwritten).
        """
        block_table = _get_layer_cache_tensor(dsa.block_tables, layer_id, swa_cache_idx)
        if block_table is None:
            return positions % state_rows
        swa = self._kv_caches[layer_id].swa
        block_size = swa.size(1) if swa is not None else self.window_size
        cols = torch.div(positions, block_size, rounding_mode="floor") % block_table.size(1)
        blocks = block_table[seq, cols].to(torch.int64).clamp_min(0)
        return blocks * state_rows + positions % state_rows

    def _shared_runtime(self) -> CSA2SharedRuntime:
        if self._v41_shared is None:
            self._v41_shared = CSA2SharedRuntime()
        return self._v41_shared

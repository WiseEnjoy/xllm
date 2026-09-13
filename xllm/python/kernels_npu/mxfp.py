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

"""MXFP8/MXFP4 microscaling quantization kernels (Ascend 950).

Serves DeepSeek-V4 official FP8 checkpoints on Ascend 950PR:

* Dense projections store ``float8_e4m3fn`` weights with one ``ue8m0``
  scale per 128x128 weight block. The NPU matmul kernels instead consume a
  per-1x32-group scale laid out as ``[K/64, N, 2]`` bytes, so the loader
  expands each block scale by ``repeat_interleave`` -- lossless because a
  ``ue8m0`` byte is a power of two and the 128-row/128-col blocks tile the
  32-group grid exactly.
* Routed experts store packed ``float4_e2m1fn`` weights (uint8 storage,
  low nibble = even k) with one ``ue8m0`` scale per 1x32 group.

All ops are plain torch_npu kernels; no C++ extension is involved.
"""

from __future__ import annotations

import torch
import torch_npu

# torch_npu extensions dtypes may be missing on older CANN builds (non-A5
# devices keep serving W8A8 checkpoints with the same source tree); resolve
# them lazily so importing this module never breaks those deployments.
try:
    E8M0 = torch_npu.float8_e8m0fnu
    FP4_X2 = torch_npu.float4_e2m1fn_x2
    MXFP_SUPPORTED = True
except AttributeError:  # pragma: no cover - pre-MX CANN
    E8M0 = None  # type: ignore[assignment]
    FP4_X2 = None  # type: ignore[assignment]
    MXFP_SUPPORTED = False

_MX_GROUP_SIZE = 32
_FRACTAL_NZ_FORMAT = 29


def dynamic_mx_quant(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to MXFP8 (e4m3 + ue8m0 per-32-group scale).

    Args:
        value: Activation tensor whose last dim is a multiple of 32.

    Returns:
        The e4m3-quantized tensor and the ue8m0 scale as uint8 bytes in the
        ``[..., K/64, 2]`` layout the quant matmuls expect.
    """
    quantized, scale = torch_npu.npu_dynamic_mx_quant(
        value, dst_type=torch.float8_e4m3fn, scale_alg=0, round_mode="rint"
    )
    return quantized, scale.view(torch.uint8)


def rms_norm_dynamic_mx_quant(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RMSNorm + MXFP8 quantization (e4m3 + ue8m0 scale).

    Drop-in FP8 counterpart of ``kernels.rms_norm_dynamic_quant`` (int8).
    The fused CANN kernel produces bit-identical results to running
    ``rms_norm`` then :func:`dynamic_mx_quant`.

    Args:
        value: Input tensor ``[..., K]``.
        weight: RMSNorm gamma over the last dim.
        eps: RMSNorm epsilon.

    Returns:
        The e4m3-quantized normed tensor and its ue8m0 scale bytes
        (``[..., K/64, 2]`` uint8).
    """
    quantized, scale, _ = torch.ops.npu.npu_rms_norm_dynamic_mx_quant(
        value,
        weight,
        epsilon=eps,
        scale_alg=0,
        round_mode="rint",
        dst_type=torch.float8_e4m3fn,
    )
    return quantized, scale.view(torch.uint8)


def expand_dense_block_scale(
    scale_bytes: torch.Tensor,
    n_out: int,
    n_in: int,
    block_size: int = 128,
) -> torch.Tensor:
    """Expand a dense 128x128 block scale to the 1x32-group matmul layout.

    Args:
        scale_bytes: ``[n_out/block, n_in/block]`` uint8 (ue8m0) tensor.
        n_out: Output dim of the weight (rows).
        n_in: Input dim of the weight (cols).
        block_size: Checkpoint block size along both dims.

    Returns:
        ``[n_in/64, n_out, 2]`` uint8 bytes, contiguous.
    """
    expanded = (
        scale_bytes.repeat_interleave(block_size // _MX_GROUP_SIZE, dim=1)
        .repeat_interleave(block_size, dim=0)[:n_out, : n_in // _MX_GROUP_SIZE]
    )
    return expanded.reshape(n_out, -1, 2).transpose(0, 1).contiguous()


def quant_matmul_mx(
    x_quant: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    x_scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dense MXFP8 matmul with pre-quantized activation.

    Args:
        x_quant: e4m3 activation ``[..., K]`` (contiguous in the last dim).
        weight: e4m3 weight ``[K, N]``.
        scale: Weight scale bytes ``[K/64, N, 2]`` (from
            :func:`expand_dense_block_scale`).
        x_scale: Activation scale bytes ``[..., K/64, 2]`` from
            :func:`dynamic_mx_quant` / :func:`rms_norm_dynamic_mx_quant`.
        output_dtype: Result dtype.

    Returns:
        The product ``[..., N]`` in ``output_dtype``.
    """
    return torch_npu.npu_quant_matmul(
        x_quant,
        weight,
        scale,
        scale_dtype=E8M0,
        pertoken_scale=x_scale,
        pertoken_scale_dtype=E8M0,
        bias=None,
        output_dtype=output_dtype,
        group_sizes=[1, 1, _MX_GROUP_SIZE],
    )


def prepare_mxfp4_weight(weight_packed: torch.Tensor) -> torch.Tensor:
    """Cast packed FP4 expert weights to the grouped-matmul NZ layout.

    Args:
        weight_packed: uint8 bytes ``[experts, N, K/2]`` (low nibble = even k).

    Returns:
        The FRACTAL_NZ-cast tensor transposed to ``[experts, K/2, N]``.
    """
    return torch_npu.npu_format_cast(
        weight_packed,
        _FRACTAL_NZ_FORMAT,
        customize_dtype=torch.float8_e4m3fn,
        input_dtype=FP4_X2,
    ).transpose(1, 2)


def prepare_mxfp4_scale(scale_bytes: torch.Tensor) -> torch.Tensor:
    """Re-view expert scales as the grouped-matmul antiquant layout.

    Args:
        scale_bytes: ``[experts, N, K/32]`` uint8 (ue8m0) tensor.

    Returns:
        A strided ``[experts, K/64, N, 2]`` view. The value is deliberately
        NOT materialized: the grouped-matmul antiquant path identifies the
        scale layout from the transpose strides, and ``.contiguous()``
        reorders the N axis and silently corrupts non-uniform scales.
    """
    experts, n_out, groups = scale_bytes.shape
    return (
        scale_bytes.reshape(experts, n_out, groups // 2, 2)
        .view(torch.uint8)
        .transpose(-3, -2)
    )


def clipped_swiglu(
    gate_up: torch.Tensor, limit: float
) -> torch.Tensor:
    """SwiGLU with element clamping at ``limit`` (DSV4 swiglu_limit=10).

    ``gate_up`` holds the concatenated [gate | up] halves produced by the
    first grouped matmul.

    Args:
        gate_up: ``[..., 2 * inter]`` tensor.
        limit: Clamp bound applied to both halves.

    Returns:
        ``[..., inter]`` tensor of ``silu(gate) * up``.
    """
    return torch_npu.npu_clipped_swiglu(
        gate_up, dim=-1, alpha=1.0, limit=limit, bias=0.0, interleaved=False
    )


__all__ = [
    "E8M0",
    "FP4_X2",
    "clipped_swiglu",
    "dynamic_mx_quant",
    "expand_dense_block_scale",
    "prepare_mxfp4_scale",
    "prepare_mxfp4_weight",
    "quant_matmul_mx",
    "rms_norm_dynamic_mx_quant",
]

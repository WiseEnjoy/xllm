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

"""DeepSeek-V4 MXFP8/MXFP4 weight-format modules.

Counterparts of the W8A8 modules in ``deepseek_v32`` for DeepSeek-V4 official
FP8 checkpoints (e4m3 dense weights + ue8m0 128x128 block scales, packed-fp4
expert weights + ue8m0 1x32 group scales). The two formats are selected per
checkpoint at model build time via ``DeepseekV4Config.use_mxfp``; the W8A8
modules and their loading path stay untouched.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python import distributed

try:
    from xllm.python import kernels
except Exception:  # pragma: no cover - kernels need the compiled lib
    kernels = None  # type: ignore[assignment]


class MXFP8DynamicLinear(nn.Module):
    """Dynamic-activation MXFP8 linear (e4m3 weight + ue8m0 block scale).

    Checkpoint layout: ``weight`` [N, K] e4m3, ``weight_scale``
    [N/128, K/128] uint8 (ue8m0). After
    :meth:`process_weights_after_loading` the weight is transposed to [K, N]
    and the scale expanded to the per-1x32-group ``[K/64, N, 2]`` layout the
    quant matmul consumes (block boundaries tile the group grid exactly, and
    ue8m0 bytes are powers of two, so the expansion is lossless).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device,
        transpose_weight_after_loading: bool = True,
    ) -> None:
        super().__init__()
        del transpose_weight_after_loading  # see process_weights_after_loading
        self.in_features = in_features
        self.out_features = out_features
        self._weight_is_transposed = False
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, device=device),
            requires_grad=False,
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(
                out_features // 128,
                in_features // 128,
                dtype=torch.uint8,
                device=device,
            ),
        )

    def process_weights_after_loading(self) -> None:
        # npu_quant_matmul takes x2 in [K, N] layout with no transpose flag
        # (unlike the int8 quant_matmul's transpose2), so the [N, K]
        # checkpoint weight is always transposed once here regardless of the
        # constructor flag kept for interface parity with W8A8DynamicLinear.
        if not self._weight_is_transposed:
            self.weight.data = self.weight.data.transpose(0, 1).contiguous()
            self._weight_is_transposed = True
        self._kernel_scale = kernels.mxfp.expand_dense_block_scale(
            self.weight_scale.data, self.out_features, self.in_features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized, x_scale = kernels.mxfp.dynamic_mx_quant(x)
        return self.forward_quant(quantized, x_scale)

    def forward_quant(
        self, x_quant: torch.Tensor, x_scale: torch.Tensor
    ) -> torch.Tensor:
        """Run the MXFP8 matmul on an already-quantized activation.

        Args:
            x_quant: e4m3 activation from :func:`kernels.mxfp.dynamic_mx_quant` or
                :func:`kernels.mxfp.rms_norm_dynamic_mx_quant`.
            x_scale: Matching ue8m0 scale bytes.
        """
        return kernels.mxfp.quant_matmul_mx(
            x_quant, self.weight.data, self._kernel_scale, x_scale
        )


class MXFP8RowParallelLinear(nn.Module):
    """Row-parallel MXFP8 linear: K is sharded, output is all-reduced.

    Counterpart of the bf16 ``RowParallelLinear`` used for ``wo_b``: the
    checkpoint weight [N, K] is sharded along K (dim 1) with its block
    scale columns; each rank computes its [T, N] partial and the results are
    summed over the TP group.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int,
        tp_rank: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        assert in_features % tp_size == 0
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.in_local = in_features // tp_size
        self._processed = False
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, device=device),
            requires_grad=False,
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(
                out_features // 128,
                in_features // 128,
                dtype=torch.uint8,
                device=device,
            ),
        )

    def process_weights_after_loading(self) -> None:
        # Idempotency guard: the draft loader walks modules() and may reach
        # this node both directly and via the parent attention module; a
        # second K-shard narrow would apply to the already-sharded weight.
        if self._processed:
            return
        self._processed = True
        # Shard K (dim 1) with matching scale columns, then transpose to the
        # matmul layout [K_local, N]. The K shard is a whole number of
        # 128-blocks (K=8192, tp<=8), so scale columns stay aligned.
        weight = self.weight.data.narrow(
            1, self.tp_rank * self.in_local, self.in_local
        )
        scale_cols = self.weight_scale.data.size(1) // self.tp_size
        scale = self.weight_scale.data.narrow(
            1, self.tp_rank * scale_cols, scale_cols
        )
        self.weight.data = weight.transpose(0, 1).contiguous()
        self._kernel_scale = kernels.mxfp.expand_dense_block_scale(
            scale, self.out_features, self.in_local
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.in_local
        quantized, x_scale = kernels.mxfp.dynamic_mx_quant(x)
        out = kernels.mxfp.quant_matmul_mx(
            quantized, self.weight.data, self._kernel_scale, x_scale
        )
        if self.tp_size > 1:
            distributed.tp_all_reduce(out)
        return out


class MXFP8GroupedOProjection(nn.Module):
    """Grouped low-rank o-projection (``wo_a``) for MXFP8 checkpoints.

    ``wo_a`` maps each o-group's attention output slice to its low-rank
    space: ``x [T, n_local_groups, K] -> [T, n_local_groups * lora_rank]``.
    The fused transpose-quant batch-matmul is not usable on this CANN build
    (k/n hard-limited to 512/128), so each group runs one dense MXFP8
    ``quant_matmul`` -- verified equivalent to the reference einsum.
    """

    def __init__(
        self,
        group_hidden: int,
        o_lora_rank: int,
        n_local_groups: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.group_hidden = group_hidden
        self.o_lora_rank = o_lora_rank
        self.n_local_groups = n_local_groups
        n_out = n_local_groups * o_lora_rank
        self.weight = nn.Parameter(
            torch.empty(n_out, group_hidden, dtype=torch.float8_e4m3fn, device=device),
            requires_grad=False,
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(
                n_out // 128,
                group_hidden // 128,
                dtype=torch.uint8,
                device=device,
            ),
        )

    def process_weights_after_loading(self) -> None:
        self._group_weights = []
        self._group_scales = []
        scale_rows = self.weight_scale.data.size(0) // self.n_local_groups
        for g in range(self.n_local_groups):
            rows = slice(g * self.o_lora_rank, (g + 1) * self.o_lora_rank)
            self._group_weights.append(
                self.weight.data[rows].transpose(0, 1).contiguous()
            )
            self._group_scales.append(
                kernels.mxfp.expand_dense_block_scale(
                    self.weight_scale.data[
                        g * scale_rows:(g + 1) * scale_rows
                    ],
                    self.o_lora_rank,
                    self.group_hidden,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x [T, n_local_groups, K]`` to ``[T, n_local_groups*lora]``."""
        num_tokens = x.size(0)
        quantized, x_scale = kernels.mxfp.dynamic_mx_quant(x)
        outs = []
        for g in range(self.n_local_groups):
            outs.append(
                kernels.mxfp.quant_matmul_mx(
                    quantized[:, g, :].contiguous(),
                    self._group_weights[g],
                    self._group_scales[g],
                    x_scale[:, g, :].contiguous(),
                )
            )
        return torch.cat(outs, dim=-1).view(num_tokens, -1)


class MXFP8MLP(nn.Module):
    """Clamped-SwiGLU FFN with MXFP8 projections (shared experts / dense MLP).

    Counterpart of ``DeepseekV3MLP`` in its W8A8 configuration: ``gate_up``
    is sharded on the output dim, ``down`` on the input dim, and the final
    partial is reduced over the TP group.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int,
        tp_rank: int,
        device: torch.device,
        swiglu_limit: float = 0.0,
        skip_tp_reduce: bool = False,
    ) -> None:
        super().__init__()
        assert intermediate_size % tp_size == 0
        self.tp_size = tp_size
        self.skip_tp_reduce = skip_tp_reduce
        self.swiglu_limit = swiglu_limit
        inter_local = intermediate_size // tp_size
        self.gate_up_proj = MXFP8DynamicLinear(
            hidden_size, 2 * inter_local, device
        )
        # down_proj holds its K (intermediate) dim pre-sharded by the loader
        # (checkpoint full width -> narrow(dim=1)); N keeps the full hidden.
        self.down_proj = MXFP8DynamicLinear(
            inter_local, hidden_size, device
        )

    def process_weights_after_loading(self) -> None:
        self.gate_up_proj.process_weights_after_loading()
        self.down_proj.process_weights_after_loading()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        if 0.0 < self.swiglu_limit < 1_000_000.0:
            act = kernels.mxfp.clipped_swiglu(gate_up, self.swiglu_limit)
        else:
            assert kernels is not None
            act = kernels.silu_and_mul(gate_up)
        out = self.down_proj(act)
        if self.tp_size > 1 and not self.skip_tp_reduce:
            distributed.tp_all_reduce(out)
        return out


__all__ = [
    "MXFP8DynamicLinear",
    "MXFP8GroupedOProjection",
    "MXFP8MLP",
    "MXFP8RowParallelLinear",
]

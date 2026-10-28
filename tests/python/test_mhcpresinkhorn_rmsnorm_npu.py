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

"""NPU tests for the PyPTO-Pro mhcpresinkhorn_rmsnorm fused operator.

The host-side contract (argument validation, workspace cache, launch buffers)
is covered by ``test_mhcpresinkhorn_rmsnorm.py`` on CPU.  This module runs the
real kernel on an Ascend device and compares it against a pure-torch
re-implementation of the fused chain:

1. BF16 projection output: ``mix = (x.float() @ hc_fn.float().T).bfloat16()``.
2. Per-token inverse RMS over all ``4 * 4096`` input lanes.
3. Scale and bias the 4 pre, 4 post and 16 combination logits.
4. Sigmoid pre/post gates and 20 Sinkhorn iterations on the combination logits.
5. Collapse streams to BF16, then apply RMSNorm with the fused gamma.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

torch_npu = pytest.importorskip(
    "torch_npu",
    reason="mhcpresinkhorn_rmsnorm tests require torch_npu",
)
pytest.importorskip(
    "pypto_pro",
    reason="mhcpresinkhorn_rmsnorm tests require the pypto_pro package",
)

_MODULE_NAME = "xllm.python.kernels_npu.pypto_pro_mhcpresinkhorn_rmsnorm_npu"
_SOURCE = (Path(__file__).parents[2] / "xllm" / "python" / "kernels_npu"
           / "pypto-pro" / "mhcpresinkhorn_rmsnorm.py")


def _load_operator():
    """Load the operator module from its file path.

    ``pypto-pro`` is not a valid Python package name, so the module cannot be
    imported by name.
    """
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


op = _load_operator()

HC = op.HC
D = op.D
MIX_HC = op.MIX_HC
HC_D = op.HC_D
EPS = op.EPS
RMS_EPS = op.RMS_EPS
SINKHORN_ITERS = op.SINKHORN_ITERS


def _reference(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU mathematical reference; preserve the BF16 boundary before RMSNorm."""
    t = x.shape[0]
    x_fp32 = x.detach().cpu().float()
    fn_fp32 = hc_fn.detach().cpu().float()
    scale = hc_scale.detach().cpu().float()
    base = hc_base.detach().cpu().float()
    gamma_fp32 = gamma.detach().cpu().float()
    flattened = x_fp32.reshape(t, HC_D)
    # Match the production BF16 linear output after FP32 accumulation.
    mix = (flattened @ fn_fp32.T).to(torch.bfloat16).float()
    inv_rms = torch.rsqrt(flattened.square().mean(dim=-1, keepdim=True) + EPS)
    mix = mix * inv_rms
    pre = torch.sigmoid(mix[:, :HC] * scale[0] + base[:HC]) + EPS
    post = torch.sigmoid(mix[:, HC:2 * HC] * scale[1] + base[HC:2 * HC]) * 2.0
    logits = (mix[:, 2 * HC:] * scale[2] + base[2 * HC:]).reshape(t, HC, HC)

    # hc_pre emits BF16; the subsequent model RMSNorm consumes that value.
    y = (x_fp32 * pre.unsqueeze(-1)).sum(dim=1).to(torch.bfloat16).float()
    rstd = torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + RMS_EPS)
    y_norm = (y * rstd * gamma_fp32).to(torch.bfloat16)

    comb = torch.softmax(logits, dim=-1) + EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + EPS)
    for _ in range(SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + EPS)
    return y_norm, rstd, post, comb


def test_reference_constant_input() -> None:
    """Zero logits give uniform gates; constant streams collapse to exactly 2."""
    y, rstd, post, comb = _reference(
        torch.ones(1, HC, D, dtype=torch.bfloat16),
        torch.zeros(MIX_HC, HC_D, dtype=torch.bfloat16),
        torch.ones(3),
        torch.zeros(MIX_HC),
        torch.ones(D, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(y, torch.ones_like(y), rtol=0, atol=0)
    expected_rstd = torch.rsqrt(torch.full_like(rstd, 4.0 + RMS_EPS))
    torch.testing.assert_close(rstd, expected_rstd, rtol=0, atol=0)
    torch.testing.assert_close(post, torch.ones_like(post), rtol=0, atol=0)
    torch.testing.assert_close(comb, torch.full_like(comb, 0.25), rtol=0, atol=EPS)


def _inputs(t: int, seed: int = 2026) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return dict(
        x=torch.randn(t, HC, D, dtype=torch.bfloat16, device="npu"),
        hc_fn=torch.randn(MIX_HC, HC_D, dtype=torch.bfloat16, device="npu") * 0.02,
        hc_scale=torch.randn(3, dtype=torch.float32, device="npu"),
        hc_base=torch.randn(MIX_HC, dtype=torch.float32, device="npu"),
        gamma=torch.randn(D, dtype=torch.bfloat16, device="npu"),
    )


@pytest.mark.parametrize("t", [1, 15, 16, 17, 33])
@pytest.mark.parametrize("seed", [2026, 42])
def test_matches_reference(t: int, seed: int) -> None:
    torch_npu.npu.set_device(0)
    args = _inputs(t, seed)

    y, rstd, post, comb = op.mhcpresinkhorn_rmsnorm(**args)
    torch_npu.npu.synchronize()

    ref_y, ref_rstd, ref_post, ref_comb = _reference(**args)
    assert (tuple(y.shape), y.dtype) == ((t, D), torch.bfloat16)
    assert (tuple(rstd.shape), rstd.dtype) == ((t, 1), torch.float32)
    assert (tuple(post.shape), post.dtype) == ((t, HC), torch.float32)
    assert (tuple(comb.shape), comb.dtype) == ((t, HC, HC), torch.float32)

    torch.testing.assert_close(y.cpu(), ref_y.cpu(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(rstd.cpu(), ref_rstd.cpu(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(post.cpu(), ref_post.cpu(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(comb.cpu(), ref_comb.cpu(), rtol=2e-3, atol=2e-3)


def test_sinkhorn_output_is_doubly_stochastic_for_bounded_logits() -> None:
    torch_npu.npu.set_device(0)
    args = _inputs(8)
    # Fixed iterations need bounded logits to converge to both marginals.
    args["hc_scale"].zero_()
    args["hc_base"].clamp_(-1, 1)

    _, _, _, comb = op.mhcpresinkhorn_rmsnorm(**args)
    torch_npu.npu.synchronize()

    comb = comb.cpu()
    assert torch.isfinite(comb).all()
    assert (comb > 0).all()
    # Epsilon in each denominator leaves sums slightly below one.
    expected = 1.0
    torch.testing.assert_close(comb.sum(dim=-1), torch.full_like(comb[:, :, 0], expected),
                               rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(comb.sum(dim=-2), torch.full_like(comb[:, :, 0], expected),
                               rtol=1e-3, atol=1e-3)


def test_rmsnorm_output_matches_standalone_rmsnorm() -> None:
    torch_npu.npu.set_device(0)
    args = _inputs(4)

    y, rstd, _, _ = op.mhcpresinkhorn_rmsnorm(**args)
    torch_npu.npu.synchronize()

    # y is the normalized value, so mean(y^2) must be ~1 when gamma is all ones.
    ones = dict(args)
    ones["gamma"] = torch.ones(D, dtype=torch.bfloat16, device="npu")
    y_ones, rstd_ones, _, _ = op.mhcpresinkhorn_rmsnorm(**ones)
    torch_npu.npu.synchronize()

    mean_sq = y_ones.float().square().mean(dim=-1, keepdim=True)
    torch.testing.assert_close(mean_sq.cpu(), torch.ones_like(mean_sq).cpu(),
                               rtol=2e-2, atol=2e-2)
    # rstd must not depend on gamma.
    torch.testing.assert_close(rstd.cpu(), rstd_ones.cpu(), rtol=0, atol=0)
    assert torch.isfinite(y.float()).all()
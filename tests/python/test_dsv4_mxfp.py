# Copyright 2026 The xLLM Authors.
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

"""Unit tests for the DeepSeek-V4 MXFP8/MXFP4 weight-format path.

Numerical goldens compare the module output against an exact fp32 reference
built from the dequantized operands (e4m3/fp4 values and ue8m0 scales are
exact in fp32), so any mismatch above bf16 output rounding indicates a layout
or contract bug. CPU-only tests validate the scale-layout transforms; device
tests exercise the full kernels. NPU device tests require Ascend 950.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).parents[2]

torch.manual_seed(0)


def _load_module(relpath: str, name: str):
    path = _REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_mxfp():
    return _load_module(
        "xllm/python/kernels_npu/mxfp.py", "pr_mxfp"
    )


def _load_dsv4_mxfp():
    """File-level load with the kernels stub wired to the real mxfp module.

    conftest stubs ``xllm.python.kernels`` so tests run without compiled
    operators; the MXFP modules access ``kernels.mxfp`` at call time, so
    attaching the file-loaded mxfp here exercises the real math.
    """
    import sys
    import types

    mxfp = _load_mxfp()
    kernels_stub = sys.modules["xllm.python.kernels"]
    if not isinstance(kernels_stub, types.ModuleType) or kernels_stub.__name__ != "xllm.python.kernels":
        kernels_stub = types.ModuleType("xllm.python.kernels")
        sys.modules["xllm.python.kernels"] = kernels_stub
    kernels_stub.mxfp = mxfp
    # moe.py's MXFP4 path resolves mxfp the same way.
    moe = _load_module("xllm/python/kernels_npu/moe.py", "pr_npu_moe_mxfp")
    kernels_stub.mxfp4_moe_with_selected_experts = (
        moe.mxfp4_moe_with_selected_experts
    )
    return _load_module("xllm/python/models/dsv4_mxfp.py", "pr_dsv4_mxfp")


def _e8m0_to_float(scale_bytes: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, scale_bytes.to(torch.float32) - 127.0)


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Reference fp4 unpacking: uint8 [.., K/2] -> float [.., K], low nibble
    = even k."""
    lo = packed & 0xF
    hi = packed >> 4

    def nib_to_val(n: torch.Tensor) -> torch.Tensor:
        sign = (n >> 3).to(torch.float32) * -2.0 + 1.0
        e = ((n >> 1) & 0x3).to(torch.float32)
        m = (n & 0x1).to(torch.float32)
        mag = torch.where(e == 0, m * 0.5, torch.pow(2.0, e - 1) * (1.0 + 0.5 * m))
        return sign * mag

    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.float32)
    out[..., 0::2] = nib_to_val(lo)
    out[..., 1::2] = nib_to_val(hi)
    return out


# ---------------------------------------------------------------------------
# CPU-only layout tests
# ---------------------------------------------------------------------------


def test_expand_dense_block_scale_layout() -> None:
    mxfp = _load_mxfp()
    # 4 output blocks x 2 input blocks; distinct bytes track the mapping.
    scale = torch.arange(8, dtype=torch.uint8).reshape(4, 2)
    expanded = mxfp.expand_dense_block_scale(scale, n_out=512, n_in=256)
    assert expanded.shape == (256 // 64, 512, 2)
    # Row n, group k must see byte scale[n // 128, k // 4]; the [K/64, N, 2]
    # layout packs two adjacent k-groups per last-dim pair.
    for n in (0, 129, 300, 511):
        for k in (0, 1, 2, 3):
            expect = scale[n // 128, k // 4].item()
            got = expanded[k // 2, n, k % 2].item()
            assert got == expect, (n, k, got, expect)


def test_prepare_mxfp4_scale_keeps_strided_view() -> None:
    mxfp = _load_mxfp()
    scale = torch.arange(4 * 8 * 6, dtype=torch.uint8).reshape(4, 8, 6)
    viewed = mxfp.prepare_mxfp4_scale(scale)
    assert viewed.shape == (4, 3, 8, 2)
    # Materializing would reorder the N axis; the kernel needs the strided
    # view. Guard the contract this path depends on.
    assert not viewed.is_contiguous()
    # Element mapping: (e, k/2, n, j) == scale[e, n, k] for k = 2*(k/2)+j.
    for e in range(4):
        for kk in range(3):
            for n in range(8):
                for j in range(2):
                    assert viewed[e, kk, n, j].item() == scale[e, n, 2 * kk + j].item()


def test_e8m0_reference_semantics() -> None:
    # ue8m0 byte 127 -> 2^0, 130 -> 2^3 (validated against the CANN kernels).
    assert _e8m0_to_float(torch.tensor([127], dtype=torch.uint8)).item() == 1.0
    assert _e8m0_to_float(torch.tensor([130], dtype=torch.uint8)).item() == 8.0
    vals = _unpack_fp4(torch.tensor([[0x12, 0x86]], dtype=torch.uint8))
    # low nibble 2 -> 1.0 (k=0), high nibble 1 -> 0.5 (k=1);
    # low nibble 6 -> 4.0 (k=2), high nibble 8 -> -0.0 (k=3).
    assert vals[0, 0].item() == 1.0
    assert vals[0, 1].item() == 0.5
    assert vals[0, 2].item() == 4.0
    assert vals[0, 3].item() == 0.0


# ---------------------------------------------------------------------------
# Device numerical goldens (Ascend 950)
# ---------------------------------------------------------------------------

_npu = pytest.importorskip("torch_npu")  # noqa: F841

DEVICE = torch.device("npu:0") if torch.npu.is_available() else None

requires_npu = pytest.mark.skipif(
    DEVICE is None, reason="NPU device not available"
)


def _random_mxfp8_linear(in_features: int, out_features: int, module_cls):
    linear = module_cls(in_features, out_features, DEVICE)
    weight = (torch.randn(out_features, in_features, device=DEVICE) * 0.3)
    linear.weight.data.copy_(weight.to(torch.float8_e4m3fn))
    scale = torch.randint(120, 136, (out_features // 128, in_features // 128), dtype=torch.uint8)
    linear.weight_scale.copy_(scale)
    # Keep the checkpoint-layout copy for exact reference reconstruction
    # (process_weights_after_loading transposes the parameter in place).
    return linear, scale, weight.to(torch.float8_e4m3fn).cpu()


@requires_npu
def test_mxfp8_dynamic_linear_golden() -> None:
    dsv4_mxfp = _load_dsv4_mxfp()
    MXFP8DynamicLinear = dsv4_mxfp.MXFP8DynamicLinear
    mxfp = _load_mxfp()
    in_features, out_features, m_rows = 256, 128, 5
    linear, scale, weight_orig = _random_mxfp8_linear(
        in_features, out_features, MXFP8DynamicLinear
    )
    linear.process_weights_after_loading()
    x = torch.randn(m_rows, in_features, dtype=torch.bfloat16, device=DEVICE) * 2.7
    out = linear(x)
    # Exact reference over the kernel's own quantized activation.
    xq, xs = mxfp.dynamic_mx_quant(x)
    w_ref = weight_orig.float() * _e8m0_to_float(
        scale
    ).repeat_interleave(128, 0).repeat_interleave(128, 1)
    a_ref = xq.cpu().float() * _e8m0_to_float(
        xs.cpu().reshape(m_rows, -1)
    ).repeat_interleave(32, dim=1)
    ref = a_ref @ w_ref.t()
    err = ((out.cpu().float() - ref).abs() / ref.abs().max().clamp_min(1e-6)).max().item()
    assert err < 0.02, f"rel_err={err}"


@requires_npu
def test_rms_norm_dynamic_mx_quant_matches_manual_chain() -> None:
    mxfp = _load_mxfp()
    m_rows, k = 37, 4096
    x = torch.randn(m_rows, k, dtype=torch.bfloat16, device=DEVICE) * 2.3
    gamma = torch.rand(k, dtype=torch.bfloat16, device=DEVICE) * 2 + 0.5
    eps = 1e-6
    y, xs = mxfp.rms_norm_dynamic_mx_quant(x, gamma, eps)
    var = x.float().pow(2).mean(-1, keepdim=True)
    normed = (x.float() * torch.rsqrt(var + eps) * gamma.float()).to(torch.bfloat16)
    y2, xs2 = mxfp.dynamic_mx_quant(normed)
    # The fused kernel's RMS reduction may differ from the manual chain by an
    # ulp, which can flip a quantization code on ties; compare the
    # dequantized values instead of raw bytes.
    a_fused = y.cpu().float() * _e8m0_to_float(
        xs.cpu().reshape(m_rows, -1)
    ).repeat_interleave(32, dim=1)
    a_man = y2.cpu().float() * _e8m0_to_float(
        xs2.cpu().reshape(m_rows, -1)
    ).repeat_interleave(32, dim=1)
    rel = (
        (a_fused - a_man).abs() / a_man.abs().max().clamp_min(1e-6)
    ).max().item()
    bit_match = (
        (y.view(torch.uint8) == y2.view(torch.uint8)).float().mean().item()
    )
    # An ulp difference in the RMS reduction can flip a quantization code on
    # exact ties; require the outputs to agree nearly everywhere with only
    # isolated single-code flips.
    assert bit_match > 0.99, f"bit match rate={bit_match}"
    assert rel < 0.03, f"dequantized rel diff={rel}"


@requires_npu
def test_mxfp8_grouped_o_projection_golden() -> None:
    dsv4_mxfp = _load_dsv4_mxfp()
    MXFP8GroupedOProjection = dsv4_mxfp.MXFP8GroupedOProjection

    group_hidden, lora, n_local, m_rows = 256, 128, 2, 13
    proj = MXFP8GroupedOProjection(group_hidden, lora, n_local, DEVICE)
    weight = torch.randn(n_local * lora, group_hidden, device=DEVICE) * 0.4
    proj.weight.data.copy_(weight.to(torch.float8_e4m3fn))
    weight_orig = weight.to(torch.float8_e4m3fn).cpu()
    scale = torch.randint(120, 136, (n_local * lora // 128, group_hidden // 128), dtype=torch.uint8)
    # n_local*lora = 128 rows = 1 scale block per group boundary case.
    proj.weight_scale.copy_(scale)
    proj.process_weights_after_loading()
    x = torch.randn(m_rows, n_local, group_hidden, dtype=torch.bfloat16, device=DEVICE) * 1.7
    out = proj(x)
    assert out.shape == (m_rows, n_local * lora)

    mxfp = _load_mxfp()
    xq, xs = mxfp.dynamic_mx_quant(x)
    s_f = _e8m0_to_float(scale)
    w_ref = weight_orig.float() * s_f.repeat_interleave(128, 0).repeat_interleave(128, 1)
    xs2 = xs.cpu().reshape(m_rows, n_local, -1)
    a_ref = xq.cpu().float() * _e8m0_to_float(xs2).repeat_interleave(32, dim=2)
    ref = torch.cat(
        [a_ref[:, g, :] @ w_ref[g * lora:(g + 1) * lora].t()
         for g in range(n_local)], dim=-1)
    err = ((out.cpu().float() - ref).abs() / ref.abs().max().clamp_min(1e-6)).max().item()
    assert err < 0.02, f"rel_err={err}"


@requires_npu
def test_mxfp8_row_parallel_linear_shard_golden() -> None:
    dsv4_mxfp = _load_dsv4_mxfp()
    MXFP8RowParallelLinear = dsv4_mxfp.MXFP8RowParallelLinear

    in_features, out_features, m_rows = 512, 256, 9
    row = MXFP8RowParallelLinear(
        in_features, out_features, tp_size=1, tp_rank=0, device=DEVICE
    )
    weight = torch.randn(out_features, in_features, device=DEVICE) * 0.35
    row.weight.data.copy_(weight.to(torch.float8_e4m3fn))
    weight_orig = weight.to(torch.float8_e4m3fn).cpu()
    scale = torch.randint(120, 136, (out_features // 128, in_features // 128), dtype=torch.uint8)
    row.weight_scale.copy_(scale)
    row.process_weights_after_loading()
    assert row.weight.data.shape == (in_features, out_features)

    x = torch.randn(m_rows, in_features, dtype=torch.bfloat16, device=DEVICE) * 2.1
    out = row(x)
    mxfp = _load_mxfp()
    xq, xs = mxfp.dynamic_mx_quant(x)
    w_ref = weight_orig.float() * _e8m0_to_float(
        scale
    ).repeat_interleave(128, 0).repeat_interleave(128, 1)
    a_ref = xq.cpu().float() * _e8m0_to_float(
        xs.cpu().reshape(m_rows, -1)
    ).repeat_interleave(32, dim=1)
    ref = a_ref @ w_ref.t()
    err = ((out.cpu().float() - ref).abs() / ref.abs().max().clamp_min(1e-6)).max().item()
    assert err < 0.02, f"rel_err={err}"

    # TP=2 shard layout: the K shard and scale columns must land on the same
    # 128-block boundaries after process_weights_after_loading.
    row2 = MXFP8RowParallelLinear(
        in_features, out_features, tp_size=2, tp_rank=1, device=DEVICE
    )
    row2.weight.data.copy_(weight.to(torch.float8_e4m3fn))
    row2.weight_scale.copy_(scale)
    row2.process_weights_after_loading()
    assert row2.weight.data.shape == (in_features // 2, out_features)
    # rank 1 owns K columns [256, 512); scale cols [2, 4).
    w2_ref = weight[None, :, 256:].cpu().float() * _e8m0_to_float(
        scale[:, 2:]
    ).repeat_interleave(128, 0).repeat_interleave(128, 1)
    got = row2._kernel_scale  # noqa: SLF001 - layout assertion
    expect = (
        scale[:, 2:]
        .repeat_interleave(4, dim=1)
        .repeat_interleave(128, dim=0)
        .reshape(out_features, -1, 2)
        .transpose(0, 1)
    )
    assert torch.equal(got.cpu(), expect.contiguous())


@requires_npu
def test_mxfp8_mlp_golden() -> None:
    dsv4_mxfp = _load_dsv4_mxfp()
    MXFP8MLP = dsv4_mxfp.MXFP8MLP

    hidden, inter, m_rows, limit = 256, 256, 7, 10.0
    mlp = MXFP8MLP(hidden, inter, tp_size=1, tp_rank=0, device=DEVICE, swiglu_limit=limit)
    originals = {}
    for name, proj, out_dim, in_dim in (
        ("gate_up", mlp.gate_up_proj, 2 * inter, hidden),
        ("down", mlp.down_proj, hidden, inter),
    ):
        weight = torch.randn(out_dim, in_dim, device=DEVICE) * 0.3
        proj.weight.data.copy_(weight.to(torch.float8_e4m3fn))
        originals[name] = (
            weight.to(torch.float8_e4m3fn).cpu(),
            torch.randint(120, 136, (out_dim // 128, in_dim // 128), dtype=torch.uint8),
        )
        proj.weight_scale.copy_(originals[name][1])
    mlp.process_weights_after_loading()

    x = torch.randn(m_rows, hidden, dtype=torch.bfloat16, device=DEVICE) * 3.3
    out = mlp(x)

    mxfp = _load_mxfp()

    def deq_w(name):
        weight_orig, scale = originals[name]
        s_f = _e8m0_to_float(scale)
        return weight_orig.float() * s_f.repeat_interleave(128, 0).repeat_interleave(128, 1)

    w_gu = deq_w("gate_up")
    w_down = deq_w("down")
    # reference over the same quantized activations the kernel sees
    xq, xs = mxfp.dynamic_mx_quant(x)
    a = xq.cpu().float() * _e8m0_to_float(xs.cpu().reshape(m_rows, -1)).repeat_interleave(32, dim=1)
    gu = a @ w_gu.t()
    g, u = gu[:, :inter], gu[:, inter:]
    act = torch.nn.functional.silu(g.clamp(max=limit)) * u.clamp(-limit, limit)
    act_bf16 = act.to(torch.bfloat16).to(DEVICE)
    aq, asc = mxfp.dynamic_mx_quant(act_bf16)
    act_q = aq.cpu().float() * _e8m0_to_float(asc.cpu().reshape(m_rows, -1)).repeat_interleave(32, dim=1)
    ref = act_q @ w_down.t()
    err = ((out.cpu().float() - ref).abs() / ref.abs().max().clamp_min(1e-6)).max().item()
    assert err < 0.02, f"rel_err={err}"


@requires_npu
def test_mxfp4_moe_with_selected_experts_golden() -> None:
    mxfp = _load_mxfp()
    _load_dsv4_mxfp()  # wires the mxfp4 op onto the kernels stub
    import sys

    kernels = sys.modules["xllm.python.kernels"]
    num_experts, hidden, inter, topk, m_rows, limit = 4, 256, 128, 2, 7, 10.0
    # Packed fp4 weights + ue8m0 scales for all experts.
    w13_packed = torch.randint(
        0, 256, (num_experts, 2 * inter, hidden // 2), dtype=torch.uint8
    )
    w2_packed = torch.randint(
        0, 256, (num_experts, hidden, inter // 2), dtype=torch.uint8
    )
    s13 = torch.randint(
        120, 136, (num_experts, 2 * inter, hidden // 32), dtype=torch.uint8
    )
    s2 = torch.randint(
        120, 136, (num_experts, hidden, inter // 32), dtype=torch.uint8
    )
    # Expert 3 stays empty (no token selects it) to cover the empty-group path.
    topk_ids = torch.tensor(
        [[0, 1]] * 3 + [[1, 2]] * 4, dtype=torch.int32
    )
    topk_weights = torch.rand(m_rows, topk, dtype=torch.bfloat16)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)

    w13_k = mxfp.prepare_mxfp4_weight(w13_packed.to(DEVICE))
    w2_k = mxfp.prepare_mxfp4_weight(w2_packed.to(DEVICE))
    s13_k = mxfp.prepare_mxfp4_scale(s13.to(DEVICE))
    s2_k = mxfp.prepare_mxfp4_scale(s2.to(DEVICE))

    x = torch.randn(m_rows, hidden, dtype=torch.bfloat16, device=DEVICE) * 2.5
    out = kernels.mxfp4_moe_with_selected_experts(
        x, topk_weights.to(DEVICE), topk_ids.to(DEVICE),
        w13_k, w2_k, s13_k, s2_k,
        num_total_experts=num_experts, start_expert_id=0,
        num_experts_per_rank=num_experts, swiglu_limit=limit,
    )

    # Exact reference: per (token, slot) expert computation in fp32.
    w13_ref = _unpack_fp4(w13_packed) * _e8m0_to_float(s13).repeat_interleave(32, dim=2)
    w2_ref = _unpack_fp4(w2_packed) * _e8m0_to_float(s2).repeat_interleave(32, dim=2)
    xq, xs = mxfp.dynamic_mx_quant(x)
    a = xq.cpu().float() * _e8m0_to_float(xs.cpu().reshape(m_rows, -1)).repeat_interleave(32, dim=1)
    ref = torch.zeros(m_rows, hidden)
    for t in range(m_rows):
        for slot in range(topk):
            e = topk_ids[t, slot].item()
            gu = a[t] @ w13_ref[e].t()
            g, u = gu[:inter], gu[inter:]
            act = torch.nn.functional.silu(g.clamp(max=limit)) * u.clamp(-limit, limit)
            act_bf = act.to(torch.bfloat16).to(DEVICE)
            aq, asc = mxfp.dynamic_mx_quant(act_bf)
            act_q = aq.cpu().float() * _e8m0_to_float(
                asc.cpu().reshape(1, -1)
            ).repeat_interleave(32, dim=1)
            ref[t] += (
                act_q @ w2_ref[e].t()
            )[0] * topk_weights[t, slot].float()
    err = ((out.cpu().float() - ref).abs() / ref.abs().max().clamp_min(1e-6)).max().item()
    assert err < 0.02, f"rel_err={err}"


@requires_npu
def test_mxfp8_linear_graph_capture() -> None:
    dsv4_mxfp = _load_dsv4_mxfp()
    MXFP8DynamicLinear = dsv4_mxfp.MXFP8DynamicLinear

    in_features, out_features, m_rows = 256, 128, 16
    linear, _, _ = _random_mxfp8_linear(
        in_features, out_features, MXFP8DynamicLinear
    )
    linear.process_weights_after_loading()
    static_x = torch.randn(m_rows, in_features, dtype=torch.bfloat16, device=DEVICE)
    eager = linear(static_x).clone()

    g = torch.npu.NPUGraph()
    side = torch.npu.Stream()
    side.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(side):
        linear(static_x)
    torch.npu.current_stream().wait_stream(side)
    with torch.npu.stream(side):
        g.capture_begin()
        captured = linear(static_x)
        g.capture_end()
    torch.npu.current_stream().wait_stream(side)
    g.replay()
    torch.npu.synchronize()
    assert torch.equal(captured, eager)

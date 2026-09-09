/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// NPU acceptance test for the xllm_ops torch-op library.
//
// Mirrors the CUDA xllm_ops_test: verifies TORCH_LIBRARY registrations survive
// linking on NPU (PrivateUse1), ops are callable via the dispatcher, and the
// embedded Python interpreter sees torch.ops.xllm_ops.*.

#include <acl/acl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <filesystem>
#include <limits>
#include <optional>
#include <string>

#include "core/framework/parallel_state/mega_moe_comm_resource.h"
#include "core/kernels/npu/npu_ops_api.h"
#include "core/kernels/xllm_torch_ops.h"
#include "hccl/hccl.h"

namespace py = pybind11;

namespace xllm {
namespace {

torch::Tensor rms_norm_reference(const torch::Tensor& input,
                                 const torch::Tensor& weight,
                                 double eps) {
  auto x = input.to(torch::kFloat32);
  auto var = x.pow(2).mean(-1, /*keepdim=*/true);
  auto normed = x * torch::rsqrt(var + eps);
  return (normed * weight.to(torch::kFloat32)).to(input.scalar_type());
}

torch::Tensor silu_and_mul_reference(const torch::Tensor& input) {
  const int64_t d = input.size(-1) / 2;
  auto a = input.slice(-1, 0, d);
  auto b = input.slice(-1, d, 2 * d);
  return (a * torch::sigmoid(a)) * b;
}

void prepend_python_model_path() {
  std::filesystem::path repo_root(__FILE__);
  for (int i = 0; i < 5; ++i) {
    repo_root = repo_root.parent_path();
  }
  const std::string python_model_path = repo_root.string();
  py::list sys_path = py::module_::import("sys").attr("path");
  sys_path.attr("insert")(0, python_model_path);
}

bool is_npu_available() {
  return c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
             ->deviceCount() > 0;
}

bool is_ascend950_device() {
  const char* soc_name = aclrtGetSocName();
  return soc_name != nullptr &&
         std::string(soc_name).find("Ascend950") != std::string::npos;
}

torch::Tensor expand_kv_heads_reference(const torch::Tensor& tensor,
                                        int64_t num_heads) {
  const int64_t num_kv_heads = tensor.size(1);
  EXPECT_EQ(num_heads % num_kv_heads, 0);
  const int64_t expansion_factor = num_heads / num_kv_heads;
  return tensor.unsqueeze(2)
      .expand({tensor.size(0), num_kv_heads, expansion_factor, tensor.size(2)})
      .reshape({tensor.size(0), num_heads, tensor.size(2)});
}

torch::Tensor packed_causal_attention_reference(const torch::Tensor& query,
                                                const torch::Tensor& key,
                                                const torch::Tensor& value,
                                                double scale) {
  const auto query_float = query.to(torch::kFloat32).permute({1, 0, 2});
  const auto key_float =
      expand_kv_heads_reference(key.to(torch::kFloat32), query.size(1));
  const auto value_float =
      expand_kv_heads_reference(value.to(torch::kFloat32), query.size(1));
  auto scores =
      torch::matmul(query_float, key_float.permute({1, 2, 0})) * scale;
  const auto causal_mask =
      torch::ones({query.size(0), key.size(0)}, torch::kBool).triu(1);
  scores.masked_fill_(causal_mask, -std::numeric_limits<float>::infinity());
  return torch::matmul(torch::softmax(scores, -1),
                       value_float.permute({1, 0, 2}))
      .permute({1, 0, 2});
}

torch::Tensor decode_attention_reference(const torch::Tensor& query,
                                         const torch::Tensor& key,
                                         const torch::Tensor& value,
                                         double scale) {
  const auto query_float = query.to(torch::kFloat32).squeeze(0);
  const auto key_float =
      expand_kv_heads_reference(key.to(torch::kFloat32), query.size(1));
  const auto value_float =
      expand_kv_heads_reference(value.to(torch::kFloat32), query.size(1));
  const auto scores =
      torch::matmul(query_float.unsqueeze(1), key_float.permute({1, 2, 0})) *
      scale;
  return torch::matmul(torch::softmax(scores, -1),
                       value_float.permute({1, 0, 2}))
      .squeeze(1)
      .unsqueeze(0);
}

class NpuXllmOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    xllm::ensure_xllm_torch_ops_registered();
    if (!is_npu_available()) {
      GTEST_SKIP() << "NPU not available; skipping xllm_ops NPU test.";
    }
    if (!Py_IsInitialized()) {
      setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1);
      Py_InitializeEx(0);
    }
    py::gil_scoped_acquire gil;
    prepend_python_model_path();
    py::module_::import("xllm.python._npu_bootstrap");
    py::module_::import("xllm.python");
  }
};

TEST_F(NpuXllmOpsTest, DispatcherRmsNormMatchesReference) {
  py::gil_scoped_acquire gil;
  auto opts =
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kPrivateUse1);
  auto input = torch::randn({8, 128}, opts);
  auto weight = torch::randn({128}, opts);
  const double eps = 1e-6;

  auto op =
      c10::Dispatcher::singleton().findSchemaOrThrow("xllm_ops::rms_norm", "");
  auto out = op.typed<torch::Tensor(
      const torch::Tensor&, const torch::Tensor&, double)>()
                 .call(input, weight, eps);

  auto ref = rms_norm_reference(input, weight, eps);
  EXPECT_TRUE(
      torch::allclose(out.cpu(), ref.cpu(), /*rtol=*/1e-2, /*atol=*/1e-2))
      << "max abs diff = "
      << (out.cpu().to(torch::kFloat32) - ref.cpu().to(torch::kFloat32))
             .abs()
             .max()
             .item<float>();
}

// 复现问题: rms_norm 走 torch_npu custom_ops → atb RmsNormOperation，在 decode
// 阶段 （单/少 token + bf16 + 3D）下 aicore 超时（timeout or trap，device error
// 507015）。 prefill 的 [92,16,512] bf16 能跑，decode 的 [1,16,512] bf16
// 复现失败。
TEST_F(NpuXllmOpsTest, DispatcherRmsNormBf16DecodeSingleToken) {
  py::gil_scoped_acquire gil;
  auto opts = torch::TensorOptions()
                  .dtype(torch::kBFloat16)
                  .device(torch::kPrivateUse1);
  auto input = torch::randn({1, 16, 512}, opts);
  auto weight = torch::randn({512}, opts);
  const double eps = 1e-6;

  auto op =
      c10::Dispatcher::singleton().findSchemaOrThrow("xllm_ops::rms_norm", "");
  auto out = op.typed<torch::Tensor(
      const torch::Tensor&, const torch::Tensor&, double)>()
                 .call(input, weight, eps);

  auto ref = rms_norm_reference(input, weight, eps);
  EXPECT_TRUE(
      torch::allclose(out.cpu(), ref.cpu(), /*rtol=*/1e-2, /*atol=*/1e-2))
      << "max abs diff = "
      << (out.cpu().to(torch::kFloat32) - ref.cpu().to(torch::kFloat32))
             .abs()
             .max()
             .item<float>();
}

TEST_F(NpuXllmOpsTest, DispatcherRmsNormBf16PrefillShape) {
  py::gil_scoped_acquire gil;
  auto opts = torch::TensorOptions()
                  .dtype(torch::kBFloat16)
                  .device(torch::kPrivateUse1);
  auto input = torch::randn({92, 16, 512}, opts);
  auto weight = torch::randn({512}, opts);
  const double eps = 1e-6;

  auto op =
      c10::Dispatcher::singleton().findSchemaOrThrow("xllm_ops::rms_norm", "");
  auto out = op.typed<torch::Tensor(
      const torch::Tensor&, const torch::Tensor&, double)>()
                 .call(input, weight, eps);

  auto ref = rms_norm_reference(input, weight, eps);
  EXPECT_TRUE(
      torch::allclose(out.cpu(), ref.cpu(), /*rtol=*/1e-2, /*atol=*/1e-2))
      << "max abs diff = "
      << (out.cpu().to(torch::kFloat32) - ref.cpu().to(torch::kFloat32))
             .abs()
             .max()
             .item<float>();
}

TEST_F(NpuXllmOpsTest, DispatcherSiluAndMulMatchesReference) {
  py::gil_scoped_acquire gil;
  auto opts =
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kPrivateUse1);
  auto gate_up = torch::randn({8, 256}, opts);

  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "xllm_ops::silu_and_mul", "");
  auto out = op.typed<torch::Tensor(const torch::Tensor&)>().call(gate_up);

  auto ref = silu_and_mul_reference(gate_up);
  ASSERT_EQ(out.size(-1), 128);
  EXPECT_TRUE(
      torch::allclose(out.cpu(), ref.cpu(), /*rtol=*/1e-2, /*atol=*/1e-2))
      << "max abs diff = "
      << (out.cpu().to(torch::kFloat32) - ref.cpu().to(torch::kFloat32))
             .abs()
             .max()
             .item<float>();
}

TEST_F(NpuXllmOpsTest, EmbeddedInterpreterSeesOps) {
  py::gil_scoped_acquire gil;

  auto opts =
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kPrivateUse1);
  auto gate_up = torch::randn({8, 256}, opts);

  py::module_ torch_mod = py::module_::import("torch");
  py::object xllm_ops = torch_mod.attr("ops").attr("xllm_ops");
  py::object out_obj = xllm_ops.attr("silu_and_mul")(gate_up);
  auto out = out_obj.cast<torch::Tensor>();

  auto ref = silu_and_mul_reference(gate_up);
  ASSERT_EQ(out.size(-1), 128);
  EXPECT_TRUE(
      torch::allclose(out.cpu(), ref.cpu(), /*rtol=*/1e-2, /*atol=*/1e-2))
      << "max abs diff = "
      << (out.cpu().to(torch::kFloat32) - ref.cpu().to(torch::kFloat32))
             .abs()
             .max()
             .item<float>();
}

TEST_F(NpuXllmOpsTest, GroupGemmWrapperMatchesInt32Reference) {
  py::gil_scoped_acquire gil;

  py::exec(R"PY(
import torch

device = torch.device("privateuseone:0")
torch.manual_seed(20260814)
tokens, experts, input_dim, output_dim = 8, 2, 128, 256
x_cpu = torch.randint(-4, 5, (tokens, input_dim), dtype=torch.int8)
w_cpu = torch.randint(-4, 5, (experts, input_dim, output_dim), dtype=torch.int8)
group_list_cpu = torch.tensor([4, 4], dtype=torch.int64)

x = x_cpu.to(device)
w = w_cpu.to(device)
group_list = group_list_cpu.to(device)
out = torch.ops.xllm_ops.group_gemm(
    x, w, None, None, group_list, 2, 0, 1, torch.int32
)
torch.npu.synchronize()

expected = torch.cat((
    x_cpu[:4].to(torch.int32) @ w_cpu[0].to(torch.int32),
    x_cpu[4:].to(torch.int32) @ w_cpu[1].to(torch.int32),
), dim=0)
assert out.shape == (tokens, output_dim)
assert out.dtype == torch.int32
torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
)PY");
}

TEST_F(NpuXllmOpsTest, GroupGemmWrapperAcceptsScaleAndPerTokenScale) {
  py::gil_scoped_acquire gil;

  py::exec(R"PY(
import torch

device = torch.device("privateuseone:0")
tokens, experts, input_dim, output_dim = 8, 2, 128, 128
x = torch.randint(-4, 5, (tokens, input_dim), dtype=torch.int8, device=device)
w = torch.randint(-4, 5, (experts, input_dim, output_dim), dtype=torch.int8, device=device)
scale = torch.ones((experts, output_dim), dtype=torch.bfloat16, device=device)
per_token_scale = torch.ones((tokens,), dtype=torch.float32, device=device)
group_list = torch.tensor([4, 4], dtype=torch.int64, device=device)

out = torch.ops.xllm_ops.group_gemm(
    x, w, scale, per_token_scale, group_list, 2, 0, 1, torch.bfloat16
)
torch.npu.synchronize()
assert out.shape == (tokens, output_dim)
assert out.dtype == torch.bfloat16
)PY");
}

TEST_F(NpuXllmOpsTest, Dsv4PartialRotaryPythonWrapperRunsOnNpu) {
  py::gil_scoped_acquire gil;

  py::exec(R"PY(
import torch
from xllm.python.kernels_npu.rotary_embedding import (
    npu_inplace_partial_rotary_mul,
)

torch.manual_seed(2026)
x_cpu = torch.randn((8, 2, 128), dtype=torch.float32).to(torch.bfloat16)
cos_cpu = torch.randn((8, 64), dtype=torch.float32).to(torch.bfloat16)
sin_cpu = torch.randn((8, 64), dtype=torch.float32).to(torch.bfloat16)

expected = x_cpu.float().clone()
segment = x_cpu[..., 64:128].float()
swapped = torch.empty_like(segment)
swapped[..., 0::2] = segment[..., 1::2]
swapped[..., 1::2] = segment[..., 0::2]
sign = torch.ones_like(cos_cpu.float())
sign[..., 0::2] = -1
expected[..., 64:128] = (
    segment * cos_cpu.float().unsqueeze(1)
    + swapped * sin_cpu.float().unsqueeze(1) * sign.unsqueeze(1)
)
expected = expected.to(torch.bfloat16).float()

x = x_cpu.to("privateuseone:0")
cos = cos_cpu.to(x.device)
sin = sin_cpu.to(x.device)
result = npu_inplace_partial_rotary_mul(x, cos, sin, 64, 64)
torch.npu.synchronize()

assert result.data_ptr() == x.data_ptr()
torch.testing.assert_close(
    x.cpu().float(), expected, atol=2e-2, rtol=2e-2
)
)PY");
}

// 复现 DeepSeek-V4 真实 RoPE 参数：head_dim=512, nope_head_dim=448,
// rope_head_dim=64 → partial_slice=[448,512]（非零起点）。xllm_ops
// 自带的用例只测 start=0，未覆盖非零起点 + 大 head_dim，正是服务里 tiling MTE
// 越界 （JudgeSliceInfo sliceStart_=448 sliceEnd_=512, errcode 95）的触发条件。
TEST_F(NpuXllmOpsTest, Dsv4PartialRotaryRealShapeRepro) {
  py::gil_scoped_acquire gil;

  py::exec(R"PY(
import torch


def _golden(x, cos, sin, start):
    out = x.float().clone()
    seg = out[..., start:]
    c = cos.float().unsqueeze(1)  # [M, 1, 64] broadcast over S
    s = sin.float().unsqueeze(1)
    rot = torch.empty_like(seg)
    rot[..., 0::2] = seg[..., 1::2]
    rot[..., 1::2] = seg[..., 0::2]
    sign = torch.ones_like(c)
    sign[..., 0::2] = -1.0
    out[..., start:] = seg * c + rot * s * sign
    return out.to(x.dtype)


device = torch.device("privateuseone:0")

for label, shape in [("q-ish S=16", (92, 16, 512)), ("kv-ish S=1", (92, 1, 512))]:
    M, S, D = shape
    x_cpu = torch.randn(M, S, D, dtype=torch.float32).to(torch.bfloat16)
    cos_cpu = torch.randn(M, 64, dtype=torch.float32).to(torch.bfloat16)
    sin_cpu = torch.randn(M, 64, dtype=torch.float32).to(torch.bfloat16)
    golden = _golden(x_cpu, cos_cpu, sin_cpu, 448).float()  # [M, S, 512]

    x4d = x_cpu.to(device).unsqueeze(2)             # [M, S, 1, 512]
    cos4d = cos_cpu.to(device).view(M, 1, 1, 64)
    sin4d = sin_cpu.to(device).view(M, 1, 1, 64)

    torch.ops.xllm_ops.npu_inplace_partial_rotary_mul(
        x4d, cos4d, sin4d, "interleave", [448, 512]
    )
    torch.npu.synchronize()
    result = x4d.cpu().squeeze(2).float()           # [M, S, 512]
    torch.testing.assert_close(result, golden, atol=2e-2, rtol=2e-2)
    print("[repro]", label, "OK")
)PY");
}

TEST_F(NpuXllmOpsTest, Dsv4CompressorPythonWrapperRunsOnNpu) {
  py::gil_scoped_acquire gil;

  py::exec(R"PY(
import torch
from xllm.python.kernels_npu.dsa import compressor

device = torch.device("privateuseone:0")
torch.manual_seed(2025)
batch, tokens, hidden = 1, 128, 1024
ratio, head_dim, coff, rope_dim = 128, 512, 1, 64
compressed_tokens = tokens // ratio

x_cpu = (torch.randn(batch, tokens, hidden) * 0.1).to(torch.float16)
wkv_cpu = (torch.randn(coff * head_dim, hidden) * 0.05).to(torch.float16)
wgate_cpu = (torch.randn(coff * head_dim, hidden) * 0.05).to(torch.float16)
ape_cpu = (torch.randn(ratio, coff * head_dim) * 0.1).float()
norm_cpu = (torch.randn(head_dim) * 0.1 + 1).to(torch.float16)
rope_cos_cpu = (
    torch.randn(batch, compressed_tokens, rope_dim) * 0.1
).to(torch.float16)
rope_sin_cpu = (
    torch.randn(batch, compressed_tokens, rope_dim) * 0.1
).to(torch.float16)

projected_kv = x_cpu.float()[0] @ wkv_cpu.float().T
scores = x_cpu.float()[0] @ wgate_cpu.float().T + ape_cpu
pooled = (torch.softmax(scores, dim=0) * projected_kv).sum(0, keepdim=True)
variance = pooled.square().mean(-1, keepdim=True)
expected = pooled * torch.rsqrt(variance + 1e-6) * norm_cpu.float()
rope_segment = expected[:, -rope_dim:].clone()
half = rope_dim // 2
rotated = torch.cat((-rope_segment[:, half:], rope_segment[:, :half]), dim=-1)
expected[:, -rope_dim:] = (
    rope_segment * rope_cos_cpu.float()[0]
    + rotated * rope_sin_cpu.float()[0]
)
expected = expected.view(batch, compressed_tokens, head_dim).half().float()

x = x_cpu.to(device)
wkv = wkv_cpu.to(device)
wgate = wgate_cpu.to(device)
ape = ape_cpu.to(device)
norm_weight = norm_cpu.to(device)
rope_sin = rope_sin_cpu.to(device)
rope_cos = rope_cos_cpu.to(device)
kv_state = torch.zeros((1, 128, head_dim), dtype=torch.float32, device=device)
score_state = torch.zeros_like(kv_state)
kv_block_table = torch.tensor([[0]], dtype=torch.int32, device=device)
score_block_table = torch.tensor([[0]], dtype=torch.int32, device=device)

out, wkv_proj, softmax_res, norm_x, norm_rstd = compressor(
    x,
    wkv,
    wgate,
    kv_state,
    score_state,
    ape,
    norm_weight,
    rope_sin,
    rope_cos,
    kv_block_table,
    score_block_table,
    None,
    None,
    None,
    rope_dim,
    ratio,
    coff,
    1e-6,
    1,
    False,
)
torch.npu.synchronize()

assert out.shape == (batch, compressed_tokens, head_dim)
assert out.dtype == torch.float16
assert wkv_proj.numel() == 0
assert softmax_res.numel() == 0
assert norm_x.numel() == 0
assert norm_rstd.numel() == 0
torch.testing.assert_close(
    out.cpu().float(), expected, atol=2e-2, rtol=2e-2
)
)PY");
}

TEST_F(NpuXllmOpsTest, Dsv4QuantLightningIndexerPythonWrapperRunsOnNpu) {
  py::gil_scoped_acquire gil;

  py::exec(R"PY(
import torch
from xllm.python.kernels_npu.dsa import (
    quant_lightning_indexer,
    quant_lightning_indexer_metadata,
)

device = torch.device("privateuseone:0")
torch.manual_seed(2026)
tokens, heads, head_dim = 84, 64, 128
page_size, sparse_count = 128, 512
query_cpu = torch.randint(-8, 8, (tokens, heads, head_dim), dtype=torch.int8)
key_cpu = torch.randint(-8, 8, (1, page_size, 1, head_dim), dtype=torch.int8)
query = query_cpu.to(device)
key = key_cpu.to(device)
weights = torch.ones((tokens, heads), dtype=torch.float16, device=device)
query_scale = torch.ones((tokens, heads), dtype=torch.float16, device=device)
key_scale = torch.ones((1, page_size, 1), dtype=torch.float16, device=device)
query_lens = torch.tensor([tokens], dtype=torch.int32, device=device)
key_lens = torch.tensor([tokens], dtype=torch.int32, device=device)
block_table = torch.tensor([[0]], dtype=torch.int32, device=device)
metadata = quant_lightning_indexer_metadata(
    heads,
    1,
    head_dim,
    0,
    0,
    query_lens,
    key_lens,
    1,
    tokens,
    tokens,
    "TND",
    "PA_BSND",
    sparse_count,
    3,
    2**63 - 1,
    2**63 - 1,
    4,
    "npu",
)
metadata_again = quant_lightning_indexer_metadata(
    heads,
    1,
    head_dim,
    0,
    0,
    query_lens,
    key_lens,
    1,
    tokens,
    tokens,
    "TND",
    "PA_BSND",
    sparse_count,
    3,
    2**63 - 1,
    2**63 - 1,
    4,
    "npu",
)
indices, values = quant_lightning_indexer(
    query,
    key,
    weights,
    query_scale,
    key_scale,
    0,
    0,
    query_lens,
    key_lens,
    block_table,
    metadata,
    "TND",
    "PA_BSND",
    sparse_count,
    3,
    2**63 - 1,
    2**63 - 1,
    4,
    False,
)
torch.npu.synchronize()

assert indices.shape == (tokens, 1, sparse_count)
assert indices.dtype == torch.int32
assert values.numel() == 0
assert values.dtype == torch.float32
assert torch.equal(metadata.cpu(), metadata_again.cpu())

valid_key_count = tokens // 4
indices_cpu = indices.cpu().squeeze(1)
assert torch.all(
    (indices_cpu == -1)
    | ((indices_cpu >= 0) & (indices_cpu < valid_key_count))
)
keys = key_cpu[0, :valid_key_count, 0].float()
token_idx = tokens - 1
dots = query_cpu[token_idx].float() @ keys.T
expected_top8 = set(torch.topk(dots.clamp_min(0).sum(0), 8).indices.tolist())
actual_top8 = set(indices_cpu[token_idx, :8].tolist())
assert len(expected_top8 & actual_top8) >= 4, (
    sorted(expected_top8),
    sorted(actual_top8),
)
)PY");
}

TEST_F(NpuXllmOpsTest, Dsv4SparseAttentionPythonWrapperRunsOnNpu) {
  py::gil_scoped_acquire gil;

  py::exec(R"PY(
import torch
from xllm.python.kernels_npu.dsa import (
    sparse_attn_sharedkv,
    sparse_attn_sharedkv_metadata,
)

device = torch.device("privateuseone:0")
torch.manual_seed(1234)
batch, q_tokens, kv_tokens = 1, 4, 16
heads, head_dim, page_size = 64, 512, 16
query_cpu = (torch.randn(batch, q_tokens, heads, head_dim) * 0.1).half()
kv_cpu = (torch.randn(batch, kv_tokens, 1, head_dim) * 0.1).half()
sinks_cpu = (torch.randn(heads) * 0.1).float()
query = query_cpu.to(device)
ori_kv = kv_cpu.view(1, page_size, 1, head_dim).to(device)
block_table = torch.tensor([[0]], dtype=torch.int32, device=device)
cu_q = torch.tensor([0, q_tokens], dtype=torch.int32, device=device)
cu_kv = torch.tensor([0, kv_tokens], dtype=torch.int32, device=device)
seq_q = torch.tensor([q_tokens], dtype=torch.int32, device=device)
seq_kv = torch.tensor([kv_tokens], dtype=torch.int32, device=device)
sinks = sinks_cpu.to(device)
metadata = sparse_attn_sharedkv_metadata(
    heads,
    1,
    head_dim,
    cu_q,
    cu_kv,
    None,
    seq_q,
    seq_kv,
    batch,
    q_tokens,
    kv_tokens,
    0,
    0,
    1,
    4,
    3,
    127,
    0,
    "BSND",
    "PA_ND",
    True,
    False,
)
metadata_again = sparse_attn_sharedkv_metadata(
    heads,
    1,
    head_dim,
    cu_q,
    cu_kv,
    None,
    seq_q,
    seq_kv,
    batch,
    q_tokens,
    kv_tokens,
    0,
    0,
    1,
    4,
    3,
    127,
    0,
    "BSND",
    "PA_ND",
    True,
    False,
)
out, lse = sparse_attn_sharedkv(
    query,
    ori_kv,
    None,
    None,
    None,
    block_table,
    None,
    None,
    None,
    None,
    None,
    seq_kv,
    sinks,
    metadata,
    head_dim**-0.5,
    1,
    4,
    3,
    127,
    0,
    "BSND",
    "PA_ND",
    False,
)
torch.npu.synchronize()

assert out.shape == query.shape
assert out.dtype == query.dtype
assert lse.numel() == 0
assert torch.equal(metadata.cpu(), metadata_again.cpu())

expected = torch.zeros_like(query_cpu.float())
keys = kv_cpu[0, :, 0].float()
scale = head_dim**-0.5
for q_idx in range(q_tokens):
    diagonal = kv_tokens - q_tokens + q_idx
    left = max(diagonal - 127, 0)
    right = diagonal
    selected_keys = keys[left:right + 1]
    logits = query_cpu[0, q_idx].float() @ selected_keys.T * scale
    sink_logits = sinks_cpu[:, None]
    normalizer = torch.logsumexp(
        torch.cat((logits, sink_logits), dim=1), dim=1
    )
    probabilities = torch.exp(logits - normalizer[:, None])
    expected[0, q_idx] = probabilities @ selected_keys
expected = expected.half().float()
torch.testing.assert_close(
    out.cpu().float(), expected, atol=2e-2, rtol=2e-2
)
)PY");
}

TEST_F(NpuXllmOpsTest, Dsv4A5AttentionPartialRotary) {
  if (!is_ascend950_device()) {
    GTEST_SKIP() << "Ascend950 is required for the A5 attention path.";
  }
  constexpr int64_t kSequenceLength = 129;
  constexpr int64_t kQueryHeads = 6;
  constexpr int64_t kKvHeads = 1;
  constexpr int64_t kHeadDim = 256;
  constexpr double kScale = 1.0 / 16.0;
  torch::manual_seed(20260729);

  const auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
  const auto query_cpu =
      (0.25 * torch::randn({kSequenceLength, kQueryHeads, kHeadDim}, cpu_float))
          .to(torch::kBFloat16);
  const auto key_cpu =
      (0.25 * torch::randn({kSequenceLength, kKvHeads, kHeadDim}, cpu_float))
          .to(torch::kBFloat16);
  const auto value_cpu =
      torch::randn({kSequenceLength, kKvHeads, kHeadDim}, cpu_float)
          .to(torch::kBFloat16);
  const auto query = query_cpu.to(torch::kPrivateUse1);
  const auto key = key_cpu.to(torch::kPrivateUse1);
  const auto value = value_cpu.to(torch::kPrivateUse1);

  const auto [actual, softmax_lse] =
      xllm::kernel::npu::npu_fused_infer_attention(query,
                                                   key,
                                                   value,
                                                   std::nullopt,
                                                   std::nullopt,
                                                   {kSequenceLength},
                                                   {kSequenceLength},
                                                   kQueryHeads,
                                                   kKvHeads,
                                                   kScale,
                                                   /*block_size=*/128,
                                                   /*sparse_mode=*/0,
                                                   /*input_layout=*/"TND",
                                                   /*softmax_lse_flag=*/false);
  const auto expected =
      packed_causal_attention_reference(query_cpu, key_cpu, value_cpu, kScale);

  EXPECT_EQ(actual.sizes(), query.sizes());
  EXPECT_EQ(softmax_lse.numel(), 0);
  EXPECT_TRUE(torch::allclose(actual.cpu().to(torch::kFloat32),
                              expected,
                              /*rtol=*/5e-2,
                              /*atol=*/5e-2))
      << "max abs diff = "
      << (actual.cpu().to(torch::kFloat32) - expected)
             .abs()
             .max()
             .item<float>();
}

TEST_F(NpuXllmOpsTest, Qwen35_27B_TP4_KvCacheCrosses128TokenBoundary) {
  py::gil_scoped_acquire gil;
  if (!is_ascend950_device()) {
    GTEST_SKIP() << "Ascend950 is required for the A5 paged-cache path.";
  }

  constexpr int64_t kSequenceLength = 130;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kNumPhysicalBlocks = 3;
  constexpr int64_t kQueryHeads = 6;
  constexpr int64_t kKvHeads = 1;
  constexpr int64_t kHeadDim = 256;
  constexpr double kScale = 1.0 / 16.0;
  torch::manual_seed(20260730);

  const auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
  const auto key_cpu =
      torch::randn({kSequenceLength, kKvHeads, kHeadDim}, cpu_float)
          .to(torch::kBFloat16);
  const auto value_cpu =
      torch::randn({kSequenceLength, kKvHeads, kHeadDim}, cpu_float)
          .to(torch::kBFloat16);
  const auto query_cpu =
      (0.25 * torch::randn({1, kQueryHeads, kHeadDim}, cpu_float))
          .to(torch::kBFloat16);
  auto key = key_cpu.to(torch::kPrivateUse1);
  auto value_tensor = value_cpu.to(torch::kPrivateUse1);
  std::optional<torch::Tensor> value = value_tensor;

  const auto npu_bfloat = torch::TensorOptions()
                              .dtype(torch::kBFloat16)
                              .device(torch::kPrivateUse1);
  auto key_cache = torch::zeros(
      {kNumPhysicalBlocks, kBlockSize, kKvHeads, kHeadDim}, npu_bfloat);
  auto value_cache_tensor = torch::zeros_like(key_cache);
  std::optional<torch::Tensor> value_cache = value_cache_tensor;

  const auto first_block_slots =
      torch::arange(2 * kBlockSize,
                    3 * kBlockSize,
                    torch::TensorOptions().dtype(torch::kInt32));
  const auto second_block_slots =
      torch::arange(0, 2, torch::TensorOptions().dtype(torch::kInt32));
  const auto slot_mapping = torch::cat({first_block_slots, second_block_slots})
                                .to(torch::kPrivateUse1);
  xllm::kernel::npu::reshape_paged_cache(
      key, value, key_cache, value_cache, slot_mapping);

  auto expected_key_cache =
      torch::zeros({kNumPhysicalBlocks, kBlockSize, kKvHeads, kHeadDim},
                   torch::TensorOptions().dtype(torch::kBFloat16));
  auto expected_value_cache = torch::zeros_like(expected_key_cache);
  expected_key_cache[2].copy_(key_cpu.narrow(0, 0, kBlockSize));
  expected_value_cache[2].copy_(value_cpu.narrow(0, 0, kBlockSize));
  expected_key_cache[0].narrow(0, 0, 2).copy_(key_cpu.narrow(0, kBlockSize, 2));
  expected_value_cache[0].narrow(0, 0, 2).copy_(
      value_cpu.narrow(0, kBlockSize, 2));

  EXPECT_TRUE(torch::equal(key_cache.cpu(), expected_key_cache));
  EXPECT_TRUE(torch::equal(value_cache.value().cpu(), expected_value_cache));

  const auto query = query_cpu.to(torch::kPrivateUse1);
  const auto block_table =
      torch::tensor({{2, 0}}, torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::kPrivateUse1);
  const auto seq_lens =
      torch::tensor({kSequenceLength},
                    torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::kPrivateUse1);
  auto actual = torch::empty_like(query);
  xllm::kernel::npu::batch_decode(query,
                                  key_cache,
                                  value_cache.value(),
                                  kScale,
                                  block_table,
                                  seq_lens,
                                  actual);
  const auto expected =
      decode_attention_reference(query_cpu, key_cpu, value_cpu, kScale);

  EXPECT_TRUE(torch::allclose(actual.cpu().to(torch::kFloat32),
                              expected,
                              /*rtol=*/5e-2,
                              /*atol=*/5e-2))
      << "max abs diff = "
      << (actual.cpu().to(torch::kFloat32) - expected)
             .abs()
             .max()
             .item<float>();
}

TEST_F(NpuXllmOpsTest, ModelExecutorUsesExplicitRuntimeBatchLimit) {
  py::gil_scoped_acquire gil;
  prepend_python_model_path();

  py::exec(R"PY(
import torch
from unittest.mock import patch

from xllm.python.layers.attention import Attention
from xllm.python.model_executor import executor as executor_module


class FakeBackend:
    def __init__(self, **kwargs):
        pass

    def bind_kv_caches(self, kv_caches):
        pass

    def prepare(self, metadata, *, graph_mode=False):
        pass

    def execute(self, q, k, v, layer):
        return q

    @property
    def num_kv_blocks(self):
        return 0

    @property
    def page_size(self):
        return 1


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.zeros(1, device="privateuseone:0")
        )
        self.attention = Attention(1, 1, 8, 1.0, 0, 0)
        self.model = torch.nn.Identity()


with patch.object(
    executor_module, "_create_attention_backend", return_value=FakeBackend()
):
    model_executor = executor_module.ModelExecutor(
        FakeModel(),
        {"python_graph_backend": "off"},
        max_seqs_per_batch=3,
    )
    assert model_executor._num_attention_layers == 1
    assert model_executor.decode_graph_runner is None
    assert model_executor.inductor_runner is None
)PY");
}

TEST_F(NpuXllmOpsTest, LightningIndexerOutKeepsBuffersAcrossGraphReplay) {
  py::gil_scoped_acquire gil;

  py::exec(R"PY(
import torch

query = torch.randn(
    (1, 64, 128), dtype=torch.bfloat16, device="privateuseone:0"
)
key = torch.randn(
    (1, 16, 1, 128), dtype=torch.bfloat16, device="privateuseone:0"
)
weights = torch.randn(
    (1, 64), dtype=torch.bfloat16, device="privateuseone:0"
)
query_seq_lengths = torch.tensor(
    [1], dtype=torch.int32, device="privateuseone:0"
)
key_seq_lengths = torch.tensor(
    [16], dtype=torch.int32, device="privateuseone:0"
)
block_table = torch.tensor(
    [[0]], dtype=torch.int32, device="privateuseone:0"
)
sparse_indices = torch.empty(
    (1, 1, 4), dtype=torch.int32, device="privateuseone:0"
)
sparse_values = torch.empty(
    (1, 1, 4), dtype=torch.bfloat16, device="privateuseone:0"
)
indices_address = sparse_indices.data_ptr()
values_address = sparse_values.data_ptr()


def run_indexer():
    return torch.ops.xllm_ops.lightning_indexer_out(
        query,
        key,
        weights,
        query_seq_lengths,
        key_seq_lengths,
        block_table,
        "TND",
        "PA_BSND",
        4,
        3,
        2**63 - 1,
        2**63 - 1,
        False,
        sparse_indices,
        sparse_values,
    )


eager_result = run_indexer()
assert eager_result.shape == (1, 1, 4)
assert eager_result.dtype == torch.int32
assert eager_result.data_ptr() == indices_address
assert sparse_values.shape == (1, 1, 4)
assert sparse_values.dtype == torch.bfloat16
assert sparse_values.data_ptr() == values_address

stream = torch.npu.Stream()
graph = torch.npu.NPUGraph()
with torch.npu.stream(stream):
    run_indexer()
torch.npu.synchronize()
with torch.npu.stream(stream):
    with torch.npu.graph(graph, stream=stream):
        graph_result = run_indexer()
torch.npu.synchronize()

with torch.npu.stream(stream):
    query.add_(0.25)
    graph.replay()
    query.sub_(0.5)
    graph.replay()
torch.npu.synchronize()

assert graph_result.data_ptr() == indices_address
assert sparse_indices.data_ptr() == indices_address
assert sparse_values.data_ptr() == values_address
)PY");
}

TEST_F(NpuXllmOpsTest, SparseFlashAttentionOutKeepsBufferAcrossGraphReplay) {
  py::gil_scoped_acquire gil;

  py::exec(R"PY(
import torch

query = torch.randn(
    (1, 8, 512), dtype=torch.bfloat16, device="privateuseone:0"
)
key = torch.randn(
    (1, 16, 1, 512), dtype=torch.bfloat16, device="privateuseone:0"
)
value = torch.randn_like(key)
sparse_indices = torch.tensor(
    [[[0, 1, 2, 3]]], dtype=torch.int32, device="privateuseone:0"
)
block_table = torch.tensor(
    [[0]], dtype=torch.int32, device="privateuseone:0"
)
actual_seq_lengths_query = torch.tensor(
    [1], dtype=torch.int32, device="privateuseone:0"
)
actual_seq_lengths_kv = torch.tensor(
    [16], dtype=torch.int32, device="privateuseone:0"
)
query_rope = torch.randn(
    (1, 8, 64), dtype=torch.bfloat16, device="privateuseone:0"
)
key_rope = torch.randn(
    (1, 16, 1, 64), dtype=torch.bfloat16, device="privateuseone:0"
)
output = torch.empty_like(query)
output_address = output.data_ptr()


def run_attention():
    return torch.ops.xllm_ops.sparse_flash_attention_out(
        query,
        key,
        value,
        sparse_indices,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        1.0 / 16.0,
        1,
        "TND",
        "PA_BSND",
        3,
        output,
    )


eager_result = run_attention()
assert eager_result.shape == query.shape
assert eager_result.dtype == query.dtype
assert eager_result.data_ptr() == output_address

stream = torch.npu.Stream()
graph = torch.npu.NPUGraph()
with torch.npu.stream(stream):
    run_attention()
torch.npu.synchronize()
with torch.npu.stream(stream):
    with torch.npu.graph(graph, stream=stream):
        graph_result = run_attention()
torch.npu.synchronize()

with torch.npu.stream(stream):
    query.add_(0.25)
    graph.replay()
    query.sub_(0.5)
    graph.replay()
torch.npu.synchronize()

assert graph_result.data_ptr() == output_address
assert output.data_ptr() == output_address
)PY");
}

// ---------------------------------------------------------------------------
// MegaMoe kernel unit tests (single-rank, ep=1, W8A8 quantization path)
// ---------------------------------------------------------------------------

TEST_F(NpuXllmOpsTest, MegaMoeSingleRankW8A8MatchesReference) {
  if (!is_ascend950_device()) {
    GTEST_SKIP() << "Ascend950 is required for the MegaMoe kernel path.";
  }
  if (!xllm::kernel::npu::has_mega_moe()) {
    GTEST_SKIP() << "aclnnMegaMoe symbol not available in this build.";
  }

  // 1) Build a single-rank HCCL communicator + MegaMoe context.
  constexpr int32_t kRank = 0;
  constexpr int32_t kRankSize = 1;
  constexpr int32_t kDeviceIndex = 0;
  // HcclGetRootInfo requires an initialized ACL/HCCL runtime: bind the
  // device and init ACL first (the fixture may not have initialized it).
  ASSERT_EQ(aclrtSetDevice(kDeviceIndex), ACL_SUCCESS);
  HcclRootInfo root_info;
  HcclResult root_result = HcclGetRootInfo(&root_info);
  ASSERT_EQ(root_result, HCCL_SUCCESS) << "HcclGetRootInfo failed";
  HcclComm hccl_comm = nullptr;
  HcclResult comm_result = HcclCommInitRootInfo(
      /*nRanks=*/kRankSize, &root_info, /*rank=*/kRank, &hccl_comm);
  ASSERT_EQ(comm_result, HCCL_SUCCESS)
      << "HcclCommInitRootInfo single-rank failed";
  ASSERT_NE(hccl_comm, nullptr);

  xllm::MegaMoeCommSpec spec;
  spec.group_name = "mega_moe_test_single";
  spec.hccl_comm = hccl_comm;
  spec.ep_world_size = kRankSize;
  spec.device_index = kDeviceIndex;
  spec.max_num_tokens_per_rank = 8;
  auto resource = std::unique_ptr<xllm::MegaMoeCommResource>(
      xllm::MegaMoeCommResource::create(spec));
  ASSERT_NE(resource, nullptr);
  const auto& context = resource->context_tensor();
  ASSERT_TRUE(context.defined()) << "MegaMoe context tensor undefined";
  ASSERT_EQ(context.scalar_type(), torch::kInt) << "context must be int32";
  ASSERT_GT(context.numel(), 0);

  // 2) Prepare tiny W8A8 inputs: 3 tokens, 4 experts (all local at ep=1),
  //    hidden=32, intermediate=16, topk=2.
  constexpr int64_t kNumTokens = 3;
  constexpr int64_t kNumExperts = 4;
  constexpr int64_t kHidden = 32;
  constexpr int64_t kIntermediate = 16;
  constexpr int64_t kTopk = 2;
  const auto device = torch::Device(torch::kPrivateUse1, kDeviceIndex);
  torch::manual_seed(42);

  // Hidden states (bf16 kernel input).
  auto x = torch::randn(
               {kNumTokens, kHidden},
               torch::TensorOptions().dtype(torch::kBFloat16).device(device))
               .contiguous();
  // Global topk_ids (int32) and fp32 weights.
  auto topk_ids =
      torch::tensor({{0, 2}, {1, 3}, {2, 0}},
                    torch::TensorOptions().dtype(torch::kInt32).device(device))
          .contiguous();
  auto topk_weights =
      torch::tensor(
          {{0.6f, 0.4f}, {0.7f, 0.3f}, {0.5f, 0.5f}},
          torch::TensorOptions().dtype(torch::kFloat32).device(device))
          .contiguous();

  // W8A8 weights: per-expert [in, out] int8 + fp32 scale.
  std::vector<torch::Tensor> w1_list;
  std::vector<torch::Tensor> w2_list;
  std::vector<torch::Tensor> scale1_list;
  std::vector<torch::Tensor> scale2_list;
  torch::Tensor w1_fp32, w2_fp32;
  for (int64_t e = 0; e < kNumExperts; ++e) {
    // Gate+Up: [hidden, 2*intermediate]
    w1_fp32 = torch::randn({kHidden, 2 * kIntermediate}) * 0.05;
    // Down: [intermediate, hidden]
    w2_fp32 = torch::randn({kIntermediate, kHidden}) * 0.05;
    auto scale1 = w1_fp32.abs().max().clamp_min(1e-6).to(torch::kFloat32);
    auto scale2 = w2_fp32.abs().max().clamp_min(1e-6).to(torch::kFloat32);
    auto w1_i8 = (w1_fp32 / scale1)
                     .clamp(-127, 127)
                     .to(torch::kInt8)
                     .to(device)
                     .contiguous();
    auto w2_i8 = (w2_fp32 / scale2)
                     .clamp(-127, 127)
                     .to(torch::kInt8)
                     .to(device)
                     .contiguous();
    // Encode scale as int64 bit-cast (C++ convert_fp32_scale_to_int64).
    auto scale1_i64 =
        scale1.view(torch::kInt32).to(torch::kInt64).to(device).contiguous();
    auto scale2_i64 =
        scale2.view(torch::kInt32).to(torch::kInt64).to(device).contiguous();
    w1_list.push_back(w1_i8);
    w2_list.push_back(w2_i8);
    scale1_list.push_back(scale1_i64);
    scale2_list.push_back(scale2_i64);
  }

  // 3) Call the kernel via the torch dispatcher (dispatch_quant_mode=2 W8A8).
  auto result_tuple = xllm::kernel::npu::apply_npu_mega_moe(
      context,
      x,
      topk_ids,
      topk_weights,
      torch::TensorList(w1_list),
      torch::TensorList(w2_list),
      /*moe_expert_num=*/kNumExperts,
      /*ep_world_size=*/kRankSize,
      /*ccl_buffer_size=*/resource->ccl_buffer_size(),
      torch::TensorList(scale1_list),
      torch::TensorList(scale2_list),
      /*bias1=*/{},
      /*bias2=*/{},
      /*x_active_mask=*/{},
      /*max_recv_token_num=*/kNumTokens,
      /*dispatch_quant_mode=*/2,
      /*combine_quant_mode=*/0,
      /*comm_alg=*/"",
      /*num_max_tokens_per_rank=*/8,
      /*activation=*/"swiglu",
      /*activation_clamp=*/1.0e30,
      /*dispatch_quant_out_dtype=*/0,
      /*topo_type=*/0,
      /*rank_num_per_server=*/1);

  auto output = std::get<0>(result_tuple);
  ASSERT_TRUE(output.defined());
  ASSERT_EQ(output.sizes(), x.sizes());

  // 4) Reference: dequant -> gate_up -> swiglu -> down -> weighted sum.
  auto expected = torch::zeros({kNumTokens, kHidden}, torch::kFloat32);
  auto x_cpu = x.cpu().to(torch::kFloat32);
  auto ids_cpu = topk_ids.cpu();
  auto weights_cpu = topk_weights.cpu();
  for (int64_t t = 0; t < kNumTokens; ++t) {
    for (int64_t k = 0; k < kTopk; ++k) {
      auto eid = ids_cpu[t][k].item<int32_t>();
      auto weight = weights_cpu[t][k].item<float>();
      // Dequant w1/w2 back to fp32 (best case for reference).
      auto w1_ref =
          w1_list[eid].cpu().to(torch::kFloat32) *
          scale1_list[eid].cpu().to(torch::kInt32).view(torch::kFloat32);
      auto w2_ref =
          w2_list[eid].cpu().to(torch::kFloat32) *
          scale2_list[eid].cpu().to(torch::kInt32).view(torch::kFloat32);
      auto gate_up = x_cpu[t].matmul(w1_ref);
      auto half = kIntermediate;
      auto gate = gate_up.slice(0, 0, half);
      auto up = gate_up.slice(0, half, 2 * half);
      auto silu = gate * torch::sigmoid(gate);
      auto inter = silu * up;
      auto out = inter.matmul(w2_ref);
      expected[t] += weight * out;
    }
  }

  auto output_cpu = output.cpu().to(torch::kFloat32);
  EXPECT_TRUE(
      torch::allclose(output_cpu, expected, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "max abs diff = "
      << (output_cpu - expected).abs().max().to(torch::kFloat32).item<double>()
      << ", output[0] = " << output_cpu[0].slice(0, 0, 4)
      << ", expected[0] = " << expected[0].slice(0, 0, 4);

  // Cleanup comm.
  if (hccl_comm != nullptr) {
    HcclCommDestroy(hccl_comm);
  }
}

}  // namespace
}  // namespace xllm

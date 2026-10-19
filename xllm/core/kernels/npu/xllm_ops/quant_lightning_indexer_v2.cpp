/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <glog/logging.h>
#include <torch/library.h>

#include <string>
#include <tuple>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
#include <cstdint>
#include <optional>
#include <vector>

// QuantLightningIndexerV2Va, DeepSeek-V4.1 "_va" variant: the fused candidate
// entry of the two-level top-k index pool, dispatched to the ACLNN entry point
// aclnnQuantLightningIndexerV2Va (third_party/xllm_ops); the AscendC kernel is
// the same candidate-capable V2 kernel behind the non-"_va" entry. Naming
// contract: the access *file* name is source-agnostic (no "_va"); only the
// access *function* name carries the "_va" suffix, mirroring the operator's
// own OpType (QuantLightningIndexerV2Va) and its ACLNN interface name.
namespace xllm::kernel::npu {

namespace {

constexpr int64_t kQliSize = 8;
constexpr int64_t kQliDim0 = 0;
constexpr int64_t kQliDim1 = 1;
constexpr int64_t kQliDim2 = 2;

std::tuple<torch::Tensor, torch::Tensor>
construct_quant_lightning_indexer_v2_va_output(const torch::Tensor& query,
                                               const torch::Tensor& key,
                                               int64_t sparse_count,
                                               const std::string& query_layout,
                                               const std::string& key_layout,
                                               bool return_value) {
  at::SmallVector<int64_t, kQliSize> output_size;
  for (size_t i = 0; i < query.sizes().size(); i++) {
    CHECK(query.size(i) > 0)
        << "All values within query's shape should be greater "
           "than 0, but shape[" << i << "] is " << query.size(i);
  }
  for (size_t i = 0; i < key.sizes().size(); i++) {
    CHECK(key.size(i) > 0)
        << "All values within key's shape should be greater "
           "than 0, but shape[" << i << "] is " << key.size(i);
  }
  CHECK(sparse_count > 0)
      << "sparse count should be greater than 0, but now is " << sparse_count;
  const int64_t key_head_num =
      (key_layout == "TND") ? key.size(kQliDim1) : key.size(kQliDim2);
  if (query_layout == "BSND") {
    output_size = {
        query.size(kQliDim0), query.size(kQliDim1), key_head_num, sparse_count};
  } else {
    output_size = {query.size(kQliDim0), key_head_num, sparse_count};
  }
  torch::Tensor sparse_indices_out =
      torch::zeros(output_size, query.options().dtype(torch::kInt));
  torch::Tensor sparse_values_out;
  if (return_value) {
    sparse_values_out =
        torch::zeros(output_size, query.options().dtype(torch::kBFloat16));
  } else {
    sparse_values_out =
        torch::zeros({0}, query.options().dtype(torch::kBFloat16));
  }
  return std::tuple<torch::Tensor, torch::Tensor>(sparse_indices_out,
                                                  sparse_values_out);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
quant_lightning_indexer_v2_va(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& weights,
    const torch::Tensor& query_dequant_scale,
    const torch::Tensor& key_dequant_scale,
    int64_t topk,
    int64_t quant_mode,
    const std::optional<torch::Tensor>& candidate_topk_index,
    const std::optional<torch::Tensor>& cu_seqlens_q,
    const std::optional<torch::Tensor>& cu_seqlens_k,
    const std::optional<torch::Tensor>& seqused_q,
    const std::optional<torch::Tensor>& seqused_k,
    const std::optional<torch::Tensor>& cmp_residual_k,
    const std::optional<torch::Tensor>& block_table,
    const std::optional<torch::Tensor>& output_idx_offset,
    const std::optional<torch::Tensor>& metadata,
    int64_t max_seqlen_q,
    c10::string_view layout_q,
    c10::string_view layout_k,
    int64_t mask_mode,
    int64_t cmp_ratio,
    int64_t candidate_mode,
    int64_t candidate_topk_blocks,
    int64_t candidate_block_size) {
  CHECK(query.numel() > 0) << "Tensor query is empty.";
  CHECK(key.numel() > 0) << "Tensor key is empty.";
  // TODO(xllm_ops): the ACLNN interface requires quant_mode but the current
  // AscendC tiling supports the INT8 mode (2) only; revisit together with the
  // hardware/tiling version that adds other quant modes.
  CHECK(quant_mode == 2)
      << "Aurora QLI V2 currently supports INT8 quant_mode=2";
  CHECK(query.scalar_type() == torch::kInt8 &&
        key.scalar_type() == torch::kInt8)
      << "QLI V2 query/key must be INT8";
  CHECK(weights.scalar_type() == torch::kHalf &&
        query_dequant_scale.scalar_type() == torch::kHalf &&
        key_dequant_scale.scalar_type() == torch::kHalf)
      << "QLI V2 weights/scales must be FP16";
  CHECK(candidate_mode >= 1 && candidate_mode <= 3)
      << "Invalid candidate_mode";
  CHECK(candidate_mode != 2 || candidate_topk_index.has_value())
      << "Consumer requires candidate blocks";

  std::string query_layout_str = std::string(layout_q);
  std::string key_layout_str = std::string(layout_k);

  // Switch the current device to the input's device before allocating the
  // outputs (the op may be traced from a graph capture on another device key).
  const c10::OptionalDeviceGuard device_guard(c10::Device(query.device()));

  // The ACLNN interface requires the return_value argument: the op def
  // declares it as an Int attr (quant_lightning_indexer_v2_va_def.cpp
  // .Int(0)), so the call must receive an int64_t -- passing a C++ bool
  // here would make EXEC_NPU_CMD rebuild a mismatched (bool) call ABI with
  // no compile-time guard. The current AscendC tiling supports False only,
  // so the const local both fixes the value and makes an accidental true
  // call site a compile error (no runtime CHECK needed); callers must
  // ignore the empty {0} second output while the value path stays off.
  // TODO(xllm_ops): flip to true when a tiling/hardware version supports
  // the value output, and revisit the caller contract then.
  const bool return_value = false;
  // EXEC_NPU_CMD binds its arguments by non-const lvalue reference before
  // rebuilding the call ABI, so the int64_t value must be a named lvalue.
  const int64_t return_value_attr = static_cast<int64_t>(return_value);
  std::tuple<torch::Tensor, torch::Tensor> indexer_output =
      construct_quant_lightning_indexer_v2_va_output(
          query, key, topk, query_layout_str, key_layout_str, return_value);
  torch::Tensor sparse_indices_out = std::get<0>(indexer_output);
  torch::Tensor sparse_values_out = std::get<1>(indexer_output);

  // The third output only materialises for the candidate *source* layer
  // (candidate_mode == 1); every other mode returns an empty int32 tensor so
  // the caller's unpacking stays shape-stable across modes.
  torch::Tensor candidate_topk_index_out;
  if (candidate_mode == 1) {
    const int64_t key_head_num =
        (key_layout_str == "TND") ? key.size(kQliDim1) : key.size(kQliDim2);
    at::SmallVector<int64_t, kQliSize> candidate_size;
    if (query_layout_str == "BSND") {
      candidate_size = {query.size(kQliDim0),
                        query.size(kQliDim1),
                        key_head_num,
                        candidate_topk_blocks};
    } else {
      candidate_size = {
          query.size(kQliDim0), key_head_num, candidate_topk_blocks};
    }
    candidate_topk_index_out =
        torch::zeros(candidate_size, query.options().dtype(torch::kInt));
  } else {
    candidate_topk_index_out =
        torch::zeros({0}, query.options().dtype(torch::kInt));
  }

  if (query.device().is_meta()) {
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>(
        sparse_indices_out, sparse_values_out, candidate_topk_index_out);
  }

  // Interface contract: the aclnn entry point declares its layout arguments
  // as char* (C ABI), while the strings are const std::string locals owned
  // here for the call scope. The ACLNN side reads them without mutating, so
  // the const_cast is a signature adaptation, not a write escape hatch.
  char* query_layout_ptr = const_cast<char*>(query_layout_str.c_str());
  char* key_layout_ptr = const_cast<char*>(key_layout_str.c_str());
  // The A3 tiling cannot read tensor strides through the dynamic aclnn call, so
  // the key / key-scale first-dimension strides travel as explicit attributes
  // (equal to the compact values when the operands are contiguous).
  const int64_t key_stride0 = key.stride(0);
  const int64_t key_dequant_scale_stride0 = key_dequant_scale.stride(0);

  EXEC_NPU_CMD(aclnnQuantLightningIndexerV2Va,
               query,
               key,
               weights,
               query_dequant_scale,
               key_dequant_scale,
               cu_seqlens_q,
               cu_seqlens_k,
               seqused_q,
               seqused_k,
               cmp_residual_k,
               block_table,
               output_idx_offset,
               metadata,
               candidate_topk_index,
               topk,
               quant_mode,
               max_seqlen_q,
               query_layout_ptr,
               key_layout_ptr,
               mask_mode,
               cmp_ratio,
               return_value_attr,
               candidate_mode,
               candidate_topk_blocks,
               candidate_block_size,
               key_stride0,
               key_dequant_scale_stride0,
               sparse_indices_out,
               sparse_values_out,
               candidate_topk_index_out);

  return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>(
      sparse_indices_out, sparse_values_out, candidate_topk_index_out);
}

}  // namespace xllm::kernel::npu

namespace xllm::kernel::npu {

namespace {

torch::Tensor get_optional_tensor(const std::optional<torch::Tensor>& tensor) {
  return tensor.has_value() ? *tensor : torch::Tensor();
}

std::tuple<torch::Tensor, torch::Tensor>
construct_quant_lightning_indexer_v2_output_tensor(
    const torch::Tensor& query,
    const torch::Tensor& key,
    int64_t topk,
    const std::string& query_layout,
    const std::string& key_layout,
    bool return_value) {
  constexpr int64_t kDim0 = 0;
  constexpr int64_t kDim1 = 1;
  constexpr int64_t kDim2 = 2;

  for (int64_t size : query.sizes()) {
    CHECK(size > 0) << "All query dimensions must be positive, but got "
                    << size;
  }
  for (int64_t size : key.sizes()) {
    CHECK(size > 0) << "All key dimensions must be positive, but got " << size;
  }
  CHECK(topk > 0) << "topk must be positive, but got " << topk;

  const int64_t key_head_num =
      key_layout == "TND" ? key.size(kDim1) : key.size(kDim2);
  std::vector<int64_t> output_size;
  if (query_layout == "BSND") {
    output_size = {query.size(kDim0), query.size(kDim1), key_head_num, topk};
  } else {
    output_size = {query.size(kDim0), key_head_num, topk};
  }

  torch::Tensor sparse_indices_out =
      torch::zeros(output_size, query.options().dtype(torch::kInt));
  torch::Tensor sparse_values_out;
  if (return_value) {
    sparse_values_out =
        torch::zeros(output_size, query.options().dtype(torch::kBFloat16));
  } else {
    sparse_values_out =
        torch::zeros({0}, query.options().dtype(torch::kBFloat16));
  }

  return {sparse_indices_out, sparse_values_out};
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> quant_lightning_indexer_v2(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& weights,
    const torch::Tensor& query_descale,
    const torch::Tensor& key_descale,
    const std::optional<torch::Tensor>& cu_seqlens_q,
    const std::optional<torch::Tensor>& cu_seqlens_k,
    const std::optional<torch::Tensor>& seqused_q,
    const std::optional<torch::Tensor>& seqused_k,
    const std::optional<torch::Tensor>& cmp_residual_k,
    const std::optional<torch::Tensor>& block_table,
    const std::optional<torch::Tensor>& output_idx_offset,
    const std::optional<torch::Tensor>& metadata,
    int64_t topk,
    int64_t quant_mode,
    int64_t max_seqlen_q,
    c10::string_view layout_q,
    c10::string_view layout_k,
    int64_t mask_mode,
    int64_t cmp_ratio,
    int64_t return_value) {
  const std::string query_layout(layout_q);
  const std::string key_layout(layout_k);
  const char* query_layout_ptr = query_layout.c_str();
  const char* key_layout_ptr = key_layout.c_str();

  auto [sparse_indices_out, sparse_values_out] =
      construct_quant_lightning_indexer_v2_output_tensor(
          query,
          key,
          topk,
          query_layout,
          key_layout,
          static_cast<bool>(return_value));

  torch::Tensor cu_seqlens_q_tensor = get_optional_tensor(cu_seqlens_q);
  torch::Tensor cu_seqlens_k_tensor = get_optional_tensor(cu_seqlens_k);
  torch::Tensor seqused_q_tensor = get_optional_tensor(seqused_q);
  torch::Tensor seqused_k_tensor = get_optional_tensor(seqused_k);
  torch::Tensor cmp_residual_k_tensor = get_optional_tensor(cmp_residual_k);
  torch::Tensor block_table_tensor = get_optional_tensor(block_table);
  torch::Tensor output_idx_offset_tensor =
      get_optional_tensor(output_idx_offset);
  torch::Tensor metadata_tensor = get_optional_tensor(metadata);

  // v2 kernel supports int8 natively (quant_mode=2), no fp8 conversion needed.

  EXEC_NPU_CMD(aclnnQuantLightningIndexerV2,
               query,
               key,
               weights,
               query_descale,
               key_descale,
               cu_seqlens_q_tensor,
               cu_seqlens_k_tensor,
               seqused_q_tensor,
               seqused_k_tensor,
               cmp_residual_k_tensor,
               block_table_tensor,
               output_idx_offset_tensor,
               metadata_tensor,
               topk,
               quant_mode,
               max_seqlen_q,
               query_layout_ptr,
               key_layout_ptr,
               mask_mode,
               cmp_ratio,
               return_value,
               sparse_indices_out,
               sparse_values_out);

  return {sparse_indices_out, sparse_values_out};
}

}  // namespace xllm::kernel::npu

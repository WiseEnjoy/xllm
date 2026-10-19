/* Copyright 2026 The xLLM Authors.

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

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {

namespace {

torch::Tensor get_optional_tensor(const std::optional<torch::Tensor>& tensor) {
  // Absent optionals stay undefined on purpose: EXEC_NPU_CMD maps a
  // non-defined tensor to a null aclnn input pointer, which the op treats
  // as an absent optional argument. This sentinel was A/B-tested with this
  // op; do not swap in torch::empty({0}) here -- that is a present,
  // zero-size input and follows a different kernel path. The T4 metadata
  // builder (quant_lightning_indexer_v2_metadata.cpp) uses an empty int32
  // tensor for its distinct op on purpose; the two helpers are
  // intentionally not unified.
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

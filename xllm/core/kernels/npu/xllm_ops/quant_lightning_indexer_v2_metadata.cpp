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

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

namespace {

torch::Tensor get_optional_tensor(const std::optional<torch::Tensor>& tensor,
                                  const torch::Device& device) {
  if (tensor.has_value()) {
    return *tensor;
  }
  return torch::empty({0}, torch::dtype(torch::kInt32).device(device));
}

}  // namespace

torch::Tensor quant_lightning_indexer_v2_metadata(
    const std::optional<torch::Tensor>& cu_seqlens_q,
    const std::optional<torch::Tensor>& cu_seqlens_k,
    const std::optional<torch::Tensor>& seqused_q,
    const std::optional<torch::Tensor>& seqused_k,
    const std::optional<torch::Tensor>& cmp_residual_k,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    int64_t topk,
    int64_t quant_mode,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    c10::string_view layout_q,
    c10::string_view layout_k,
    int64_t mask_mode,
    int64_t cmp_ratio,
    const c10::string_view device) {
  torch::Device output_device = torch::Device(std::string(device));
  if (cu_seqlens_q.has_value()) {
    output_device = cu_seqlens_q->device();
  } else if (cu_seqlens_k.has_value()) {
    output_device = cu_seqlens_k->device();
  } else if (seqused_q.has_value()) {
    output_device = seqused_q->device();
  } else if (seqused_k.has_value()) {
    output_device = seqused_k->device();
  }

  torch::Tensor metadata =
      torch::zeros({kDsaMetadataBufferElements},
                   torch::dtype(torch::kInt32).device(output_device));
  torch::Tensor cu_seqlens_q_tensor =
      get_optional_tensor(cu_seqlens_q, output_device);
  torch::Tensor cu_seqlens_k_tensor =
      get_optional_tensor(cu_seqlens_k, output_device);
  torch::Tensor seqused_q_tensor =
      get_optional_tensor(seqused_q, output_device);
  torch::Tensor seqused_k_tensor =
      get_optional_tensor(seqused_k, output_device);
  torch::Tensor cmp_residual_k_tensor =
      get_optional_tensor(cmp_residual_k, output_device);

  const std::string query_layout(layout_q);
  const std::string key_layout(layout_k);
  const char* query_layout_ptr = query_layout.c_str();
  const char* key_layout_ptr = key_layout.c_str();

  EXEC_NPU_CMD(aclnnQuantLightningIndexerV2Metadata,
               cu_seqlens_q_tensor,
               cu_seqlens_k_tensor,
               seqused_q_tensor,
               seqused_k_tensor,
               cmp_residual_k_tensor,
               num_heads_q,
               num_heads_k,
               head_dim,
               topk,
               quant_mode,
               batch_size,
               max_seqlen_q,
               max_seqlen_k,
               query_layout_ptr,
               key_layout_ptr,
               mask_mode,
               cmp_ratio,
               metadata);

  return metadata;
}

}  // namespace xllm::kernel::npu

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
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

// QuantLightningIndexerV2 metadata factory, DeepSeek-V4.1 "_va" variant:
// builds the 1024 x int32 per-core schedule buffer consumed by
// quant_lightning_indexer_v2_va. The host side dispatches the
// migrated ACLNN entry point aclnnQuantLightningIndexerV2MetadataVa
// (third_party/xllm_ops, op name QuantLightningIndexerV2MetadataVa), so the
// exported torch interface carries the "_va" suffix as well.
namespace xllm::kernel::npu {

at::Tensor quant_lightning_indexer_v2_metadata_va(
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    int64_t topk,
    int64_t quant_mode,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_k,
    const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_k,
    const c10::optional<at::Tensor>& cmp_residual_k,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    c10::string_view layout_q,
    c10::string_view layout_k,
    int64_t mask_mode,
    int64_t cmp_ratio,
    c10::string_view device) {
  // Every Tensor argument may be omitted, so the output device has to be
  // resolved explicitly: the first present optional Tensor wins, otherwise the
  // caller-provided device string is used. Calling contract: with every
  // Tensor omitted the device string is the only device source and must
  // name a valid device -- the op never silently falls back to CPU.
  if (!cu_seqlens_q.has_value() && !cu_seqlens_k.has_value() &&
      !seqused_q.has_value() && !seqused_k.has_value() &&
      !cmp_residual_k.has_value()) {
    TORCH_CHECK(!device.empty(),
                "quant_lightning_indexer_v2_metadata_va requires a non-empty "
                "device string when no tensor argument carries a device");
  }
  at::Device output_device = at::Device(std::string(device));
  if (cu_seqlens_q.has_value()) {
    output_device = cu_seqlens_q.value().device();
  } else if (cu_seqlens_k.has_value()) {
    output_device = cu_seqlens_k.value().device();
  } else if (seqused_q.has_value()) {
    output_device = seqused_q.value().device();
  } else if (seqused_k.has_value()) {
    output_device = seqused_k.value().device();
  } else if (cmp_residual_k.has_value()) {
    output_device = cmp_residual_k.value().device();
  }

  at::Tensor output =
      at::zeros({kDsaMetadataBufferElements},
                at::TensorOptions().dtype(at::kInt).device(output_device));
  // The generated ACLNN host layer (op_api/aclnn_quant_lightning_indexer_
  // v2_metadata_va.cpp) treats a missing optional input as "absent" only when
  // the pointer is nullptr; a fabricated empty tensor reaches it as a valid
  // aclTensor with a null buffer, which the inner wrapper rejects with
  // AclNN_Parameter_Error(EZ1001). Pass the optionals straight through.

  if (output.device().is_meta()) {
    return output;
  }

  std::string layout_q_str = std::string(layout_q);
  std::string layout_k_str = std::string(layout_k);
  char* layout_q_ptr = const_cast<char*>(layout_q_str.c_str());
  char* layout_k_ptr = const_cast<char*>(layout_k_str.c_str());

  EXEC_NPU_CMD(aclnnQuantLightningIndexerV2MetadataVa,
               cu_seqlens_q,
               cu_seqlens_k,
               seqused_q,
               seqused_k,
               cmp_residual_k,
               num_heads_q,
               num_heads_k,
               head_dim,
               topk,
               quant_mode,
               batch_size,
               max_seqlen_q,
               max_seqlen_k,
               layout_q_ptr,
               layout_k_ptr,
               mask_mode,
               cmp_ratio,
               output);

  return output;
}

}  // namespace xllm::kernel::npu

namespace xllm::kernel::npu {

namespace {

torch::Tensor get_optional_tensor(const std::optional<torch::Tensor>& tensor,
                                  const torch::Device& device) {
  // Absent optionals become a defined empty int32 tensor on purpose for
  // aclnnQuantLightningIndexerV2Metadata: this op was A/B-tested with
  // present zero-size inputs. The main launcher
  // (quant_lightning_indexer_v2.cpp) keeps undefined tensors as null
  // optional inputs for its own op; both paths are EXEC_NPU_CMD-safe with
  // their own sentinel, so the helpers are intentionally not unified.
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

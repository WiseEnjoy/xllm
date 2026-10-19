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

#include "core/runtime/py_runtime_module.h"

#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/runtime/py_attention_metadata.h"
#include "core/runtime/py_input_batch.h"
#include "models/llm/py_causal_lm.h"
#include "platform/layer_synchronizer.h"

#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "core/kernels/npu/utils.h"
#include "platform/npu/npu_layer_synchronizer.h"
#endif

namespace py = pybind11;

namespace xllm {
namespace {

void register_xllm_runtime_module(py::module_& m) {
  register_attention_metadata_views(m);
  register_input_batch_metadata_view(m);
  py::class_<LayerSynchronizer, std::shared_ptr<LayerSynchronizer>>(
      m, "LayerLoadSynchronizer")
      .def(
          "synchronize_layer",
          [](LayerSynchronizer& self, int64_t event_index) {
            CHECK_GE(event_index, 0);
            CHECK_LT(static_cast<uint64_t>(event_index), self.size());
            return self.synchronize_layer(event_index);
          },
          py::call_guard<py::gil_scoped_release>())
      .def("size", &LayerSynchronizer::size);

  m.def("tp_all_reduce", [](torch::Tensor tensor) {
    PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
    if (py_causal_lm != nullptr) {
      py_causal_lm->tp_all_reduce(tensor);
    }
    return tensor;
  });
  m.def("tp_all_gather", [](torch::Tensor tensor, int64_t dim) {
    PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
    if (py_causal_lm != nullptr) {
      return py_causal_lm->tp_all_gather(tensor, dim);
    }
    return tensor;
  });
  m.def("tp_all_to_all",
        [](const torch::Tensor& tensor, int64_t world_size) -> py::object {
          PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
          if (py_causal_lm == nullptr) {
            // No native group is owned by this thread. Let Python select its
            // explicitly initialized process group without attempting an
            // exchange.
            return py::none();
          }
          return py::cast(py_causal_lm->tp_all_to_all(tensor, world_size));
        });
  m.def("dp_all_gather",
        [](torch::Tensor tensor,
           const std::vector<int32_t>& execution_token_counts) {
          PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
          if (py_causal_lm != nullptr) {
            return py_causal_lm->dp_all_gather(tensor, execution_token_counts);
          }
          return tensor;
        });
  m.def("moe_tp_all_reduce", [](torch::Tensor tensor) {
    PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
    if (py_causal_lm != nullptr) {
      py_causal_lm->moe_tp_all_reduce(tensor);
    }
    return tensor;
  });
  m.def("moe_ep_all_reduce", [](torch::Tensor tensor) {
    PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
    if (py_causal_lm != nullptr) {
      py_causal_lm->moe_ep_all_reduce(tensor);
    }
    return tensor;
  });
  // The C++ side owns the MoE EP process group on the V4-family Python path
  // (they manage their own TP/moe_tp/EP splits), so expose its HCCL
  // communicator name for the MC2 guard. An empty string means unavailable.
  m.def("moe_ep_hccl_comm_name", []() {
    PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
    if (py_causal_lm != nullptr) {
      return py_causal_lm->moe_ep_hccl_comm_name();
    }
    return std::string();
  });
  m.def("moe_ep_comm_name", []() {
    PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
    if (py_causal_lm == nullptr) {
      return std::string();
    }
    return py_causal_lm->moe_ep_comm_name();
  });
  // The MC2 dispatch/combine kernels take this rank as ``ep_rank_id``. The
  // V4-family Python models own their EP split in C++, so expose the native
  // rank; a host without an active instance reads as the single-replica 0.
  m.def("moe_ep_rank", []() -> int64_t {
    PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
    if (py_causal_lm == nullptr) {
      return 0;
    }
    return py_causal_lm->moe_ep_rank();
  });
  m.def("eplb_batch_isend_irecv",
        [](py::list operation_types, py::list tensors, py::list remote_ranks) {
          PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
          CHECK(py_causal_lm != nullptr)
              << "EPLB P2P transfer requires an active PyCausalLM.";
          py_causal_lm->eplb_batch_isend_irecv(
              operation_types, tensors, remote_ranks);
        });
  m.def("eplb_wait_batch_isend_irecv", []() {
    PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
    CHECK(py_causal_lm != nullptr)
        << "EPLB P2P transfer requires an active PyCausalLM.";
    py_causal_lm->eplb_wait_batch_isend_irecv();
  });

#if defined(USE_NPU)
  m.def("is_ascend_a3", &kernel::npu::is_ascend_a3);
  py::class_<NPULayerSynchronizerImpl,
             std::shared_ptr<NPULayerSynchronizerImpl>>(m, "LayerSynchronizer")
      .def("record_event",
           [](NPULayerSynchronizerImpl& self, int64_t layer_id) {
             int32_t device_id = static_cast<int32_t>(
                 c10_npu::getCurrentNPUStream().device_index());
             return self.record_event(layer_id, device_id);
           });
#endif
}

}  // namespace

void ensure_xllm_runtime_module() {
  CHECK(Py_IsInitialized())
      << "Python must be initialized before binding xllm_runtime";
  py::gil_scoped_acquire gil;
  py::dict modules = py::module_::import("sys").attr("modules");
  const py::str module_name("xllm_runtime");
  if (modules.contains(module_name)) {
    return;
  }

  py::object module_object =
      py::module_::import("types").attr("ModuleType")(module_name);
  py::module_ module = py::reinterpret_borrow<py::module_>(module_object);
  register_xllm_runtime_module(module);
  modules[module_name] = module;
}

}  // namespace xllm

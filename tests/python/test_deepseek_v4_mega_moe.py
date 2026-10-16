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

"""CPU tests for DeepSeek-V4 MXFP4 MegaMoe wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from xllm.python.models import deepseek_v4
from xllm.python.models.deepseek_v4 import DeepseekV4Config, DeepseekV4MoE


def _config(**overrides: object) -> dict:
    values = {
        "hidden_size": 16,
        "n_layers": 1,
        "n_heads": 2,
        "head_dim": 4,
        "vocab_size": 16,
        "o_lora_rank": 4,
        "o_groups": 1,
        "n_activated_experts": 2,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "moe_intermediate_size": 8,
        "weight_quant_method": "fp8",
        "tp_size": 4,
        "ep_size": 4,
        "ep_rank": 0,
        "moe_tp_size": 1,
        "moe_tp_rank": 0,
        "dp_size": 1,
        "dp_rank": 0,
        "enable_mega_moe": True,
        "megamoe_max_tokens_per_rank": 8,
    }
    values.update(overrides)
    return values


def _moe(**overrides: object) -> DeepseekV4MoE:
    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    moe.cfg = DeepseekV4Config.from_dict(_config(**overrides))
    moe.uses_mx_formats = moe.cfg.uses_mx_formats
    moe.dp_size = moe.cfg.dp_size
    moe.dp_rank = moe.cfg.dp_rank
    moe.ep_size = moe.cfg.ep_size
    moe.ep_rank = moe.cfg.ep_rank
    moe.num_total_experts = moe.cfg.n_routed_experts
    moe.topk = moe.cfg.n_activated_experts
    moe.moe_tp_size = moe.cfg.moe_tp_size
    moe._megamoe_enabled = True
    moe._megamoe_max_rows = moe.cfg.megamoe_max_tokens_per_rank
    moe._megamoe_state = None
    moe._megamoe_prequant = False
    return moe


def test_mega_moe_config_fields_default_to_disabled() -> None:
    config = DeepseekV4Config.from_dict({"hidden_size": 16, "model_type": "deepseek_v4"})

    assert config.enable_mega_moe is False
    assert config.megamoe_max_tokens_per_rank == 512


def test_mega_moe_config_fields_are_parsed() -> None:
    config = DeepseekV4Config.from_dict(_config())

    assert config.enable_mega_moe is True
    assert config.megamoe_max_tokens_per_rank == 8


def test_megamoe_weight_views_keep_storage_and_transpose_scales() -> None:
    moe = _moe()
    moe.experts_w13 = torch.zeros(2, 64, 8, dtype=torch.uint8)
    moe.experts_w2 = torch.zeros(2, 16, 4, dtype=torch.uint8)
    moe.experts_w13_scale = torch.zeros(2, 2, 8, 2, dtype=torch.uint8)
    moe.experts_w2_scale = torch.zeros(2, 16, 2, 2, dtype=torch.uint8)

    w13, w2, w13_sf, w2_sf = moe._prepare_megamoe_weight_views()

    assert w13.shape == (2, 8, 64)
    assert w13.data_ptr() == moe.experts_w13.data_ptr()
    assert w2.shape == (2, 4, 16)
    assert w2.data_ptr() == moe.experts_w2.data_ptr()
    assert w13_sf.shape == (2, 8, 2, 2)
    assert w13_sf.dtype == torch.float8_e8m0fnu
    assert w2_sf.shape == (2, 2, 16, 2)
    assert w2_sf.dtype == torch.float8_e8m0fnu


def test_megamoe_uses_row_cap_and_unmasked_router(monkeypatch: pytest.MonkeyPatch) -> None:
    moe = _moe()
    assert moe._should_use_mega_moe(torch.zeros(8, 16)) is True
    assert moe._should_use_mega_moe(torch.zeros(9, 16)) is False

    fused = MagicMock(return_value=(torch.zeros(2, 16), torch.zeros(2, dtype=torch.int32)))
    moe._megamoe_state = (fused, SimpleNamespace())
    moe._megamoe_w13 = torch.zeros(2, 8, 64, dtype=torch.uint8)
    moe._megamoe_w2 = torch.zeros(2, 4, 16, dtype=torch.uint8)
    moe._megamoe_w13_sf = torch.zeros(2, 8, 2, 2, dtype=torch.uint8)
    moe._megamoe_w2_sf = torch.zeros(2, 2, 16, 2, dtype=torch.uint8)
    fp4_type = object()
    monkeypatch.setattr(deepseek_v4, "torch_npu", SimpleNamespace(float4_e2m1fn_x2=fp4_type))

    hidden = torch.zeros(2, 16)
    router_weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    expert_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    routed = moe._megamoe_routed(hidden, router_weights, expert_ids)

    assert routed.shape == (2, 16)
    args = fused.call_args.args
    assert args[0] is hidden
    assert args[1].dtype == torch.int32
    assert args[2].dtype == torch.bfloat16
    assert args[3] == [moe._megamoe_w13]
    assert args[4] == [moe._megamoe_w2]
    assert args[5] is moe._megamoe_state[1]
    assert fused.call_args.kwargs["l1_weights_sf"] == [moe._megamoe_w13_sf]
    assert fused.call_args.kwargs["l2_weights_sf"] == [moe._megamoe_w2_sf]
    assert fused.call_args.kwargs["weight1_type"] is fp4_type
    assert fused.call_args.kwargs["weight2_type"] is fp4_type
    assert fused.call_args.kwargs["activation_clamp"] == moe.cfg.swiglu_limit


def test_megamoe_prequant_passes_scales_to_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    moe = _moe()
    moe._megamoe_prequant = True
    fused = MagicMock(return_value=(torch.zeros(2, 16), torch.zeros(2, dtype=torch.int32)))
    moe._megamoe_state = (fused, SimpleNamespace())
    moe._megamoe_w13 = torch.zeros(2, 8, 64, dtype=torch.uint8)
    moe._megamoe_w2 = torch.zeros(2, 4, 16, dtype=torch.uint8)
    moe._megamoe_w13_sf = torch.zeros(2, 8, 2, 2, dtype=torch.uint8)
    moe._megamoe_w2_sf = torch.zeros(2, 2, 16, 2, dtype=torch.uint8)

    quantized = torch.zeros(2, 16, dtype=torch.float8_e4m3fn)
    quant_scale = torch.zeros(2, 1, 2, dtype=torch.uint8)

    def dynamic_mx_quant(hidden: torch.Tensor, **kwargs):
        assert hidden.shape == (2, 16)
        assert kwargs["dst_type"] == torch.float8_e4m3fn
        assert kwargs["scale_alg"] == 0
        assert kwargs["round_mode"] == "rint"
        return quantized, quant_scale

    fp4_type = object()
    monkeypatch.setattr(
        deepseek_v4,
        "torch_npu",
        SimpleNamespace(
            npu_dynamic_mx_quant=dynamic_mx_quant,
            float4_e2m1fn_x2=fp4_type,
        ),
    )

    routed = moe._megamoe_routed(torch.zeros(2, 16), torch.ones(2, 2), torch.zeros(2, 2, dtype=torch.int32))

    assert routed.shape == (2, 16)
    args = fused.call_args.args
    assert args[0] is quantized
    assert fused.call_args.kwargs["scales"].dtype == torch.float8_e8m0fnu
    assert fused.call_args.kwargs["scales"].shape == (2, 2)


def test_megamoe_reuses_symmetric_buffer_by_communicator(monkeypatch: pytest.MonkeyPatch) -> None:
    moe = _moe()
    DeepseekV4MoE._megamoe_buffers.clear()
    sym_buffer = SimpleNamespace()
    get_symm_buffer = MagicMock(return_value=sym_buffer)
    mega_moe = MagicMock()
    runtime = SimpleNamespace(
        moe_ep_all_reduce=MagicMock(),
        moe_ep_hccl_comm_name=MagicMock(return_value="ep-comm"),
    )
    monkeypatch.setattr(moe, "_load_megamoe_runtime", lambda: runtime)
    monkeypatch.setattr(moe, "_load_megamoe_glue", lambda: (get_symm_buffer, mega_moe))
    get_rank = MagicMock(return_value=0)
    get_world_size = MagicMock(return_value=4)
    monkeypatch.setattr(torch.distributed, "get_rank", get_rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", get_world_size)

    moe._init_megamoe(torch.device("cpu"))
    first = moe._megamoe_state
    moe._megamoe_state = None
    moe._init_megamoe(torch.device("cpu"))

    assert first == (mega_moe, sym_buffer)
    assert moe._megamoe_state == first
    get_symm_buffer.assert_called_once()
    kwargs = get_symm_buffer.call_args.kwargs
    assert kwargs["num_experts"] == 8
    assert kwargs["num_max_tokens_per_rank"] == 8
    assert kwargs["num_topk"] == 2
    assert kwargs["dispatch_quant_mode"] == 4
    assert kwargs["dispatch_quant_out_dtype"] == torch.float8_e4m3fn
    assert torch.distributed.get_rank is get_rank
    assert torch.distributed.get_world_size is get_world_size


def test_megamoe_combine_skips_expert_parallel_reduce(monkeypatch: pytest.MonkeyPatch) -> None:
    moe = _moe()
    moe.moe_tp_size = 2
    routed = torch.ones(2, 4)
    shared = torch.ones(2, 4)
    ep_reduce = MagicMock()
    moe_tp_reduce = MagicMock()
    monkeypatch.setattr(
        deepseek_v4,
        "distributed",
        SimpleNamespace(moe_ep_all_reduce=ep_reduce, moe_tp_all_reduce=moe_tp_reduce),
    )

    output = moe._reduce_moe_outputs(routed, shared, use_megamoe=True)

    assert torch.equal(output, routed + shared)
    ep_reduce.assert_not_called()
    assert moe_tp_reduce.call_count == 2

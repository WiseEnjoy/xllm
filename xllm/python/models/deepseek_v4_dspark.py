# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek-V4 DSpark draft model for the Python NPU executor.

Reuses the DSV4 decoder layers (which already run sparse_flash_mla +
quant_lightning_indexer_v2 through the DSA attention backend) for the
3 SWA draft stages, plus the DSpark markov/confidence heads. The C++
DSparkWorkerImpl drives this model via PyCausalLM bridges
(write_context_kv / dspark_markov_bias / dspark_confidence_probs).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python.models.base import PyModelBase
from xllm.python.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4DecoderLayer,
    DeepseekV4RotaryEmbedding,
    RMSNorm,
    W8A8WeightLoader,
)
from xllm.python.layers import ColumnParallelLinear


class DSV4DSparkMarkovHead(nn.Module):
    """Low-rank Markov projection: bias(prev) = embed(prev, w1) @ w2^T."""

    def __init__(self, vocab_size: int, markov_rank: int, dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank, dtype=dtype, device=device)
        self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False, dtype=dtype, device=device)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.embed(token_ids))


class DSV4DSparkConfidenceHead(nn.Module):
    """Sigmoid confidence from concat(hidden, markov_embed)."""

    def __init__(self, hidden_size: int, markov_rank: int, with_markov: bool,
                 device: torch.device) -> None:
        super().__init__()
        self.with_markov = with_markov
        input_size = hidden_size + markov_rank if with_markov else hidden_size
        self.proj = nn.Linear(input_size, 1, bias=True, dtype=torch.float32, device=device)

    def forward(self, hidden: torch.Tensor,
                markov_embedding: torch.Tensor | None) -> torch.Tensor:
        if self.with_markov:
            if markov_embedding is None:
                raise ValueError("DSpark confidence head requires markov embeddings")
            hidden = torch.cat((hidden, markov_embedding), dim=-1)
        return torch.sigmoid(self.proj(hidden.float())).squeeze(-1).to(torch.float32)


class DSV4DSparkModel(nn.Module):
    """3 SWA decoder layers + main_proj/norm for context KV injection."""

    def __init__(self, cfg: DeepseekV4Config, dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        capture_count = cfg.dspark_num_layers
        self.main_proj = nn.Linear(
            cfg.hidden_size * capture_count, cfg.hidden_size, bias=False,
            dtype=dtype, device=device)
        self.main_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.rotary = DeepseekV4RotaryEmbedding(
            cfg.qk_rope_head_dim, cfg.max_position_embeddings,
            cfg.rope_scaling_factor, cfg.rope_theta,
            cfg.rope_beta_fast, cfg.rope_beta_slow,
            cfg.max_position_embeddings, dtype=dtype, device=device)
        self.layers = nn.ModuleList([
            DeepseekV4DecoderLayer(cfg, i, dtype, device)
            for i in range(capture_count)
        ])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        from xllm.python.model_executor.forward_context import get_forward_context
        from xllm.python.attention.dsa_attention import _get_layer_cache_tensor
        from xllm.python import kernels as _k
        from xllm.python import dsa_dump

        context = get_forward_context()
        backend = context.attention_backend
        metadata = context.metadata
        backend.reset_forward(metadata)
        positions = positions.to(torch.int64).contiguous()
        cos_sin_cache = self.rotary.cos_sin_cache
        backend.attach_rope_tables(positions, cos_sin_cache, metadata=metadata)
        prepare_dsa = getattr(backend, "prepare_dsa_metadata_for_forward", None)
        if prepare_dsa is not None:
            prepare_dsa(metadata)
        hidden = self.main_proj_input(input_ids, positions)
        residual: torch.Tensor | None = None
        for layer_id, layer in enumerate(self.layers):
            hidden, residual = layer(hidden, residual, positions, cos_sin_cache)
            dsa_dump.snap("layer_output", {"hidden": hidden, "residual": residual},
                          layer=layer_id, kind="dense")
        from xllm.python.model_executor.forward_context import record_layer_event
        hidden = self.norm(hidden, residual)[0]
        return hidden

    def main_proj_input(self, input_ids: torch.Tensor,
                        positions: torch.Tensor) -> torch.Tensor:
        # DSpark draft: embed tokens and expand to hc_mult streams, then run layers.
        # main_proj is applied only in write_context_kv on target hidden states.
        embed = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_size,
                             dtype=self.main_proj.weight.dtype,
                             device=self.main_proj.weight.device)
        return embed(input_ids).unsqueeze(1).expand(-1, self.cfg.hc_mult, -1).contiguous()


class DeepseekV4DSparkForCausalLM(PyModelBase):
    """DSV4 DSpark draft: 3 SWA layers + markov/confidence heads + vocab head."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = DeepseekV4Config.from_dict(config)
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "npu:0"))
        self.model = DSV4DSparkModel(self.cfg, dtype, device)
        self.dtype = dtype
        self.device = device
        self.markov_head = DSV4DSparkMarkovHead(
            self.cfg.vocab_size, self.cfg.markov_rank, dtype, device)
        self.confidence_head = DSV4DSparkConfidenceHead(
            self.cfg.hidden_size, self.cfg.markov_rank,
            with_markov=True, device=device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size, self.cfg.vocab_size // self.cfg.tp_size,
            self.cfg.tp_size, gather_output=True, dtype=dtype, device=device)

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        loader = W8A8WeightLoader(self, state_dicts, tp_rank, tp_size)
        n_layers = self.cfg.dspark_num_layers

        def _copy(ckpt_key: str, param: torch.Tensor) -> None:
            t = loader.load_tensor(ckpt_key)
            if t is not None and t.numel() > 0:
                param.data.copy_(t.to(param.dtype).to(param.device))

        # Draft layers from mtp.<i>.*
        for i in range(n_layers):
            ck = f"mtp.{i}."
            layer = self.model.layers[i]
            attn = layer.self_attn
            _copy(ck + "attn.wq_a.weight", attn.q_a_proj.weight)
            _copy(ck + "attn.wq_a.weight_scale", attn.q_a_proj.weight_scale)
            _copy(ck + "attn.wq_a.weight_offset", attn.q_a_proj.weight_offset)
            _copy(ck + "attn.q_norm.weight", attn.q_a_layernorm.weight)
            _copy(ck + "attn.wq_b.weight", attn.q_b_proj.weight)
            _copy(ck + "attn.wq_b.weight_scale", attn.q_b_proj.weight_scale)
            _copy(ck + "attn.wq_b.weight_offset", attn.q_b_proj.weight_offset)
            _copy(ck + "attn.q_norm_gamma.weight", attn.q_rms_gamma.weight)
            _copy(ck + "attn.wkv.weight", attn.kv_proj.weight)
            _copy(ck + "attn.wkv.weight_scale", attn.kv_proj.weight_scale)
            _copy(ck + "attn.wkv.weight_offset", attn.kv_proj.weight_offset)
            _copy(ck + "attn.kv_norm.weight", attn.kv_a_layernorm.weight)
            _copy(ck + "attn.wo_a.weight", attn.o_a_proj.weight)
            _copy(ck + "attn.wo_b.weight", attn.o_b_proj.weight)
            _copy(ck + "attn.wo_b.weight_scale", attn.o_b_proj.weight_scale)
            _copy(ck + "attn.wo_b.weight_offset", attn.o_b_proj.weight_offset)
            _copy(ck + "attn.attn_sink", attn.attn_sink)
            _copy(ck + "attn_norm.weight", layer.input_layernorm.weight)
            _copy(ck + "ffn_norm.weight", layer.post_attention_layernorm.weight)
            _copy(ck + "hc_attn_fn", layer.hc.hc_attn_fn)
            _copy(ck + "hc_attn_base", layer.hc.hc_attn_base)
            _copy(ck + "hc_attn_scale", layer.hc.hc_attn_scale)
            _copy(ck + "hc_ffn_fn", layer.hc.hc_ffn_fn)
            _copy(ck + "hc_ffn_base", layer.hc.hc_ffn_base)
            _copy(ck + "hc_ffn_scale", layer.hc.hc_ffn_scale)
            mlp = layer.mlp
            _copy(ck + "ffn.gate.weight", mlp.gate.weight)
            _copy(ck + "ffn.gate.bias", mlp.gate.bias)
            _copy(ck + "ffn.shared_experts.w1.weight", mlp.shared_experts.w1.weight)
            _copy(ck + "ffn.shared_experts.w2.weight", mlp.shared_experts.w2.weight)
            _copy(ck + "ffn.shared_experts.w3.weight", mlp.shared_experts.w3.weight)
            _copy(ck + "ffn.shared_experts.w1.weight_scale", mlp.shared_experts.w1.weight_scale)
            _copy(ck + "ffn.shared_experts.w2.weight_scale", mlp.shared_experts.w2.weight_scale)
            _copy(ck + "ffn.shared_experts.w3.weight_scale", mlp.shared_experts.w3.weight_scale)
            _copy(ck + "ffn.experts_w13.weight", mlp.experts_w13.weight)
            _copy(ck + "ffn.experts_w13.weight_scale", mlp.experts_w13.weight_scale)
            _copy(ck + "ffn.experts_w13.weight_offset", mlp.experts_w13.weight_offset)
            _copy(ck + "ffn.experts_w2.weight", mlp.experts_w2.weight)
            _copy(ck + "ffn.experts_w2.weight_scale", mlp.experts_w2.weight_scale)
            _copy(ck + "ffn.experts_w2.weight_offset", mlp.experts_w2.weight_offset)

        # main_proj / main_norm from mtp.0
        _copy("mtp.0.main_proj.weight", self.model.main_proj.weight)
        _copy("mtp.0.main_norm.weight", self.model.main_norm.weight)

        # norm / hc_head / markov_head from last layer
        last = n_layers - 1
        _copy(f"mtp.{last}.norm.weight", self.model.norm.weight)
        _copy(f"mtp.{last}.hc_head_fn", self.model.hc_head_fn)
        _copy(f"mtp.{last}.hc_head_base", self.model.hc_head_base)
        _copy(f"mtp.{last}.hc_head_scale", self.model.hc_head_scale)
        _copy(f"mtp.{last}.markov_head.markov_w1.weight", self.markov_head.markov_w1.weight)
        _copy(f"mtp.{last}.markov_head.markov_w2.weight", self.markov_head.markov_w2.weight)
        _copy(f"mtp.{last}.confidence_head.proj.weight", self.confidence_head.proj.weight)
        _copy(f"mtp.{last}.confidence_head.proj.bias", self.confidence_head.proj.bias)

        # Vocabulary: dedicated mtp.0.embed wins over shared top-level embed.
        _copy("mtp.0.embed.weight", self._embed_weight)
        if self._embed_weight.data.abs().sum() == 0:
            _copy("embed.weight", self._embed_weight)

        # LM head: dedicated mtp.<last>.head wins over shared top-level head.
        _copy(f"mtp.{last}.head.weight", self.lm_head.weight)

    @property
    def _embed_weight(self) -> torch.Tensor:
        return self.model._embed_param

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        hidden = self.model(input_ids, positions)
        return self.compute_logits(hidden, None)

    def compute_logits(self, hidden_states: torch.Tensor,
                       selected_idxes: torch.Tensor | None) -> torch.Tensor:
        logits = self.lm_head(hidden_states.to(self.dtype))
        return logits

    def write_context_kv(self, target_hidden, positions, cache_slots,
                         kv_caches, layer_synchronizer=None):
        """Project captured target hidden and write as shared context KV."""
        # The Python model executor handles this via the PyCausalLM bridge.
        # For now, return None (not implemented yet — handled by the C++ side
        # or a follow-up).
        return None

    def dspark_markov_bias(self, previous_token_ids):
        return self.markov_head.bias(previous_token_ids)

    def dspark_confidence_probs(self, hidden_all, prev_matrix=None):
        markov_embedding = None
        if prev_matrix is not None:
            markov_embedding = self.markov_head.embed(prev_matrix)
        return self.confidence_head(hidden_all, markov_embedding)

    def has_dspark_confidence_head(self):
        return True

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
SWA draft stages, plus the DSpark markov/confidence heads. The C++
DSparkWorkerImpl drives this model via PyCausalLM bridges
(dspark_markov_bias / dspark_confidence_probs).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python.models.base import PyModelBase
from xllm.python.models.deepseek_v32 import W8A8WeightLoader
from xllm.python.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4DecoderLayer,
    DeepseekV4RotaryEmbedding,
    RMSNorm,
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
    """SWA decoder layers + main_proj/norm + hc_head for the DSpark draft."""

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
        self.embed_tokens = nn.Embedding(
            cfg.vocab_size, cfg.hidden_size, dtype=dtype, device=device)
        self.layers = nn.ModuleList([
            DeepseekV4DecoderLayer(cfg, i, dtype, device)
            for i in range(capture_count)
        ])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        # hc_head: merge hc_mult streams (matches C++ hc_head_fn/base/scale).
        self.hc_head_fn = nn.Parameter(
            torch.empty(cfg.hc_mult, cfg.hc_mult * cfg.hidden_size,
                        dtype=torch.float32, device=device),
            requires_grad=False)
        self.hc_head_base = nn.Parameter(
            torch.empty(cfg.hc_mult, dtype=torch.float32, device=device),
            requires_grad=False)
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32, device=device),
            requires_grad=False)

    def hc_head(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.to(torch.float32)
        x_flat = x_float.flatten(-2, -1)
        rsqrt = torch.rsqrt(x_flat.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps)
        mixes = torch.matmul(x_flat, self.hc_head_fn.transpose(0, 1))
        mixes = mixes * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.cfg.hc_eps
        return (pre.unsqueeze(-1) * x_float).sum(-2).to(x.dtype)

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        from xllm.python.model_executor.forward_context import (
            get_forward_context, record_layer_event,
        )
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

        hidden = self.embed_tokens(input_ids)
        hidden = hidden.unsqueeze(1).expand(
            -1, self.cfg.hc_mult, -1).contiguous()
        residual: torch.Tensor | None = None
        for layer_id, layer in enumerate(self.layers):
            hidden, residual = layer(hidden, residual, positions, cos_sin_cache)
            dsa_dump.snap(
                "layer_output", {"hidden": hidden, "residual": residual},
                layer=layer_id, kind="dense")
            record_layer_event(layer_id)
        merged = self.hc_head(residual if residual is not None else hidden)
        normed = self.norm(merged, None)
        return normed[0] if isinstance(normed, tuple) else normed


class DeepseekV4DSparkForCausalLM(PyModelBase):
    """DSV4 DSpark draft: SWA layers + markov/confidence heads + vocab head."""

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
        loader = W8A8WeightLoader(self, state_dicts, tp_size, tp_rank)
        n_layers = self.cfg.dspark_num_layers
        last = n_layers - 1

        def _has(name: str) -> bool:
            return loader.find(name) is not None

        def _cp(ckpt_key: str, param_name: str,
                shard_dim: int | None = None) -> None:
            if not _has(ckpt_key):
                return
            t = loader.load_tensor(ckpt_key)
            if shard_dim is not None:
                t = loader.shard(t, dim=shard_dim)
            loader.copy_in(param_name, t)

        def _w8a8(ckpt_prefix: str, param_prefix: str,
                  shard_dims: dict | None = None) -> None:
            for suffix in ("weight", "weight_scale", "weight_offset"):
                ckpt_key = f"{ckpt_prefix}.{suffix}"
                if not _has(ckpt_key):
                    continue
                t = loader.load_tensor(ckpt_key)
                dim = (shard_dims or {}).get(suffix)
                if dim is not None:
                    t = loader.shard(t, dim=dim)
                import sys
                p = self.get_parameter(f"{param_prefix}.{suffix}") if f"{param_prefix}.{suffix}" in dict(self.named_parameters()) else self.get_buffer(f"{param_prefix}.{suffix}")
                if t.shape != p.shape:
                    print(f"[DSPARK-LOAD] MISMATCH {ckpt_key}: ckpt {list(t.shape)} vs param {list(p.shape)}", file=sys.stderr, flush=True)
                loader.copy_in(f"{param_prefix}.{suffix}", t)

        # Draft layers from mtp.<i>.*
        for i in range(n_layers):
            ck = f"mtp.{i}."
            pm = f"model.layers.{i}."
            layer = self.model.layers[i]
            attn = layer.self_attn
            _w8a8(ck + "attn.wq_a", pm + "self_attn.q_a_proj")
            _w8a8(ck + "attn.wq_b", pm + "self_attn.q_b_proj",
                  {"weight": 0, "weight_scale": 0, "weight_offset": 0})
            _w8a8(ck + "attn.wkv", pm + "self_attn.kv_proj")
            # o_a/o_b are bf16 (unquantized) column/row-parallel weights.
            if _has(ck + "attn.wo_a.weight"):
                loader.copy_in(
                    pm + "self_attn.o_a_proj.weight",
                    loader.shard(loader.load_tensor(ck + "attn.wo_a.weight"), dim=0))
            if _has(ck + "attn.wo_b.weight"):
                loader.copy_in(
                    pm + "self_attn.o_b_proj.weight",
                    loader.shard(loader.load_tensor(ck + "attn.wo_b.weight"), dim=1))
            _cp(ck + "attn.q_norm.weight", pm + "self_attn.q_a_layernorm.weight")
            _cp(ck + "attn.q_norm_gamma.weight", pm + "self_attn.q_rms_gamma.weight")
            _cp(ck + "attn.kv_norm.weight", pm + "self_attn.kv_a_layernorm.weight")
            if _has(ck + "attn.attn_sink"):
                sink = loader.load_tensor(ck + "attn.attn_sink")
                if (sink.dim() == 1 and sink.size(0) == self.cfg.n_heads
                        and self.cfg.tp_size > 1):
                    shard_size = self.cfg.n_heads // self.cfg.tp_size
                    sink = sink.narrow(
                        0, self.cfg.tp_rank * shard_size, shard_size)
                loader.copy_in(pm + "self_attn.attn_sink", sink)
            _cp(ck + "attn_norm.weight", pm + "input_layernorm.weight")
            _cp(ck + "ffn_norm.weight", pm + "post_attention_layernorm.weight")
            _cp(ck + "hc_attn_fn", pm + "hc.hc_attn_fn")
            _cp(ck + "hc_attn_base", pm + "hc.hc_attn_base")
            _cp(ck + "hc_attn_scale", pm + "hc.hc_attn_scale")
            _cp(ck + "hc_ffn_fn", pm + "hc.hc_ffn_fn")
            _cp(ck + "hc_ffn_base", pm + "hc.hc_ffn_base")
            _cp(ck + "hc_ffn_scale", pm + "hc.hc_ffn_scale")
            # MoE: reuse the main model's per-expert loading (w1+w3→w13, EP shard).
            self._load_dspark_moe(loader, ck, pm, layer)

        # main_proj / main_norm from mtp.0
        _cp("mtp.0.main_proj.weight", "model.main_proj.weight")
        _cp("mtp.0.main_norm.weight", "model.main_norm.weight")

        # norm / hc_head / markov / confidence from last layer
        _cp(f"mtp.{last}.norm.weight", "model.norm.weight")
        _cp(f"mtp.{last}.hc_head_fn", "model.hc_head_fn")
        _cp(f"mtp.{last}.hc_head_base", "model.hc_head_base")
        _cp(f"mtp.{last}.hc_head_scale", "model.hc_head_scale")
        _cp(f"mtp.{last}.markov_head.markov_w1.weight", "markov_head.markov_w1.weight")
        _cp(f"mtp.{last}.markov_head.markov_w2.weight", "markov_head.markov_w2.weight")
        _cp(f"mtp.{last}.confidence_head.proj.weight", "confidence_head.proj.weight")
        _cp(f"mtp.{last}.confidence_head.proj.bias", "confidence_head.proj.bias")

        # Vocabulary: dedicated mtp.0.embed wins over shared top-level embed.
        if _has("mtp.0.embed.weight"):
            embed_t = loader.load_tensor("mtp.0.embed.weight")
            if embed_t.size(1) != self.cfg.hidden_size and embed_t.size(0) == self.cfg.hidden_size:
                embed_t = embed_t.t().contiguous()
            if embed_t.size(1) == self.cfg.hidden_size and embed_t.size(1) != self.get_parameter("model.embed_tokens.weight").size(1):
                # checkpoint stores full width; shard dim=1
                embed_t = loader.shard(embed_t, dim=1)
            loader.copy_in("model.embed_tokens.weight", embed_t)
        else:
            loader.copy_in(
                "model.embed_tokens.weight",
                loader.shard(loader.load_tensor("embed.weight"), dim=1),
            )

        # LM head: dedicated mtp.<last>.head wins over shared top-level head.
        head_t = None
        if _has(f"mtp.{last}.head.weight"):
            head_t = loader.load_tensor(f"mtp.{last}.head.weight")
        elif _has("head.weight"):
            head_t = loader.load_tensor("head.weight")
        if head_t is not None:
            if head_t.size(0) == self.cfg.vocab_size:
                head_t = loader.shard(head_t, dim=0)
            loader.copy_in("lm_head.weight", head_t)

        # Post-load weight processing (transpose + scale flatten), matching
        # the main model's process_weights_after_loading call site.
        for module in self.modules():
            if hasattr(module, "process_weights_after_loading"):
                module.process_weights_after_loading()

    def _load_dspark_moe(self, loader, ck: str, pm: str, layer) -> None:
        """Load the draft layer's MoE (per-expert w1+w3->w13, EP sharding)."""
        def _has(name: str) -> bool:
            return loader.find(name) is not None

        mlp = layer.mlp
        if _has(ck + "ffn.gate.weight"):
            loader.copy_in(pm + "mlp.gate.weight",
                          loader.load_tensor(ck + "ffn.gate.weight"))
        if getattr(mlp, "hash_layer", False):
            tid2eid_key = ck + "ffn.gate.tid2eid"
            if not _has(tid2eid_key):
                tid2eid_key += ".weight"
            if _has(tid2eid_key):
                loader.copy_in(pm + "mlp.tid2eid", loader.load_tensor(tid2eid_key))
        else:
            bias_key = ck + "ffn.gate.bias"
            if not _has(bias_key):
                bias_key = ck + "ffn.gate.e_score_correction_bias"
            if _has(bias_key):
                loader.copy_in(pm + "mlp.e_score_correction_bias",
                              loader.load_tensor(bias_key))
        tp = mlp.moe_tp_size
        tp_rank = mlp.moe_tp_rank
        start = mlp.start_expert_id
        nepr = mlp.num_experts_per_rank
        w13 = self.get_parameter(pm + "mlp.experts_w13")
        w2 = self.get_parameter(pm + "mlp.experts_w2")
        w13_scale = self.get_buffer(pm + "mlp.experts_w13_scale")
        w2_scale = self.get_buffer(pm + "mlp.experts_w2_scale")
        for local_idx in range(nepr):
            global_id = start + local_idx
            e = ck + f"ffn.experts.{global_id}."
            w1 = loader.load_tensor(e + "w1.weight")
            w3 = loader.load_tensor(e + "w3.weight")
            w13_j = torch.cat([w1, w3], dim=0)
            w2_j = loader.load_tensor(e + "w2.weight")
            if tp > 1:
                w13_j = loader.shard(w13_j, dim=0, world=tp, rank=tp_rank)
                w2_j = loader.shard(w2_j, dim=1, world=tp, rank=tp_rank)
            w13[local_idx].copy_(w13_j.to(w13.dtype))
            w2[local_idx].copy_(w2_j.to(w2.dtype))
            if _has(e + "w1.weight_scale"):
                s1 = loader.load_tensor(e + "w1.weight_scale")
                s3 = (loader.load_tensor(e + "w3.weight_scale")
                      if _has(e + "w3.weight_scale") else s1)
                s13 = torch.cat([s1, s3], dim=0)
                if tp > 1:
                    s13 = loader.shard(s13, dim=0, world=tp, rank=tp_rank)
                w13_scale[local_idx].copy_(s13[:w13_j.size(0)])
            if _has(e + "w2.weight_scale"):
                w2_scale[local_idx].copy_(loader.load_tensor(e + "w2.weight_scale"))
        se = ck + "ffn.shared_experts."
        if _has(se + "w1.weight"):
            se_w1 = loader.load_tensor(se + "w1.weight")
            se_w3 = loader.load_tensor(se + "w3.weight")
            se_w13 = torch.cat([se_w1, se_w3], dim=0)
            if tp > 1:
                se_w13 = loader.shard(se_w13, dim=0, world=tp, rank=tp_rank)
            loader.copy_in(pm + "mlp.shared_experts.gate_up_proj.weight", se_w13)
            if _has(se + "w1.weight_scale"):
                s1 = loader.load_tensor(se + "w1.weight_scale")
                s3 = loader.load_tensor(se + "w3.weight_scale")
                se_s13 = torch.cat([s1, s3], dim=0)
                if tp > 1:
                    se_s13 = loader.shard(se_s13, dim=0, world=tp, rank=tp_rank)
                loader.copy_in(pm + "mlp.shared_experts.gate_up_proj.weight_scale",
                               se_s13)

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        hidden = self.model(input_ids, positions)
        return self.lm_head(hidden.to(self.dtype))

    def compute_logits(self, hidden_states: torch.Tensor,
                       selected_idxes: torch.Tensor | None) -> torch.Tensor:
        return self.lm_head(hidden_states.to(self.dtype))

    def dspark_markov_bias(self, previous_token_ids):
        return self.markov_head.bias(previous_token_ids)

    def write_context_kv(self, target_hidden, positions, cache_slots,
                         kv_caches, layer_synchronizer=None):
        """Project captured target hidden and write as shared context KV.

        Mirrors C++ DeepseekV4DSparkModelImpl::write_context_kv:
        main_proj(target_hidden) -> main_norm -> RoPE -> scatter into SWA cache.
        """
        from xllm.python import kernels as _k

        projected = self.model.main_proj(target_hidden)
        normed = self.model.main_norm(projected, None)
        if isinstance(normed, tuple):
            normed = normed[0]
        projected = normed

        # Build RoPE cos/sin for the given positions.
        cos_sin = self.model.rotary.cos_sin_cache.index_select(
            0, positions.to(torch.int64))
        half = cos_sin.size(-1) // 2
        cos = cos_sin[..., :half].repeat_interleave(2, dim=-1).contiguous()
        sin = cos_sin[..., half:].repeat_interleave(2, dim=-1).contiguous()

        # Write KV to each layer's SWA cache via the shared slot mapping.
        for i, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            kv = attn.kv_proj(projected)
            kv = attn.kv_a_layernorm(kv)
            kv = kv.view(-1, 1, attn.head_dim)
            _k.npu_inplace_partial_rotary_mul(
                kv, cos, sin, attn.nope_head_dim, attn.rope_head_dim)
            if i < len(kv_caches):
                cache_tuple = kv_caches[i]
                swa_cache = cache_tuple[5] if len(cache_tuple) > 5 else None
                if swa_cache is not None and swa_cache.numel() > 0:
                    from xllm.python.attention.dsa_attention import _scatter_by_slot
                    _scatter_by_slot(swa_cache, cache_slots, kv)
        return projected

    def dspark_confidence_probs(self, hidden_all, prev_matrix=None):
        markov_embedding = None
        if prev_matrix is not None:
            markov_embedding = self.markov_head.embed(prev_matrix)
        return self.confidence_head(hidden_all, markov_embedding)

    def has_dspark_confidence_head(self):
        return True

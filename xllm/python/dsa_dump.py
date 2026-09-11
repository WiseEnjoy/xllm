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

"""Opt-in tensor/metric capture for the DeepSeek-V4 Python graph.

Gated by the ``DSA_DUMP_TAGS`` environment variable (note the three U):
  * unset/empty -> no capture
  * ``all``       -> every forward
  * ``prefill``   -> only prefills
  * ``decode``    -> only decodes
  * ``prefill:2`` -> tag ``prefill`` and ``max_layer >= 2``
  * ``dense``     -> layer-kinded capture for dense layers
  * ``moe``       -> layer-kinded capture for MoE layers
``DSA_DUMP_ROOT`` overrides the output root (default ``/tmp/dsa_dump``).

Each forward is one directory ``S{tick}_{stage}/``. :func:`snap` writes full
tensors as ``<name>.pt`` plus one aggregated ``manifest.json`` on
:func:`end_fwd`; every entry also carries shape/dtype/NaN/mean/min/max so a
diff can start from the manifest without loading the tensors.

The module has no xllm_ops dependency; tensor captures use ``torch.save`` only.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

import torch

_ENV_TAGS = "DSA_DUMP_TAGS"
_ENV_ROOT = "DSA_DUMP_ROOT"

_HEAD_N = 16

_STATE: dict[str, Any] = {
    "root": os.environ.get(_ENV_ROOT) or "/tmp/dsa_dump",
    "tags": "",
    "tick": 0,
    "manifest": [],
}
_CTX = threading.local()


def _logger():
    try:
        from scripts.logger import logger
    except Exception:
        return None
    return logger


def _tags() -> str:
    return _STATE["tags"]


def enabled() -> bool:
    return bool(_tags())


def refresh() -> None:
    """Re-read env tags at forward start so a long-lived worker picks up
    operator-configured dump flags without a restart is NOT possible here;
    this is kept because the worker forks before importing and env is stable.
    """
    _STATE["tags"] = os.environ.get(_ENV_TAGS, "") or ""


# -- forward lifecycle -------------------------------------------------------


def start_fwd(
    stage: str,
    *,
    tick: int = -1,
    max_layer: int = -1,
    ntokens: int = 0,
    meta: dict[str, Any] | None = None,
) -> bool:
    refresh()
    if not enabled():
        return False
    if tick >= 0:
        _STATE["tick"] = tick
    else:
        _STATE["tick"] += 1
    _STATE["manifest"] = []
    ctx_set(
        {
            "stage": stage,
            "tick": _STATE["tick"],
            "max_layer": max_layer,
            "ntokens": ntokens,
            "meta": meta or {},
        }
    )
    write_meta()
    return True


def end_fwd() -> None:
    if not enabled():
        return
    manifest = _STATE["manifest"]
    if not manifest:
        return
    path = os.path.join(step_dir(), "manifest.json")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=_json_default)
    except Exception:
        pass
    _STATE["manifest"] = []


# -- context -----------------------------------------------------------------


def ctx_set(ctx: dict[str, Any] | None) -> None:
    _CTX.data = ctx


def ctx_get() -> dict[str, Any] | None:
    return getattr(_CTX, "data", None)


# -- predicates --------------------------------------------------------------


def shall_dump(*, layer: int | None = None, kind: str | None = None) -> bool:
    """Whether the current capture context matches the configured tags.

    ``layer``/``kind`` may refine the gathered context (used mid-forward so
    each operator records which layer it belongs to).
    """
    if not enabled():
        return False
    ctx = ctx_get()
    if ctx is None:
        return False
    if layer is not None and isinstance(layer, int):
        ctx["layer"] = layer
    if kind:
        ctx["layer_kind"] = kind
    return _match(ctx)


def ctx_layer(layer: int | None, kind: str | None = None) -> None:
    """Set the current layer/kind on the active forward context.

    Called by the model loop before each decoder layer so operator-level
    ``snap`` calls inherit the layer identity without passing it explicitly.
    """
    if not enabled():
        return
    ctx = ctx_get()
    if ctx is None:
        return
    if layer is not None and isinstance(layer, int):
        ctx["layer"] = layer
    if kind:
        ctx["layer_kind"] = kind


def stage() -> str:
    ctx = ctx_get()
    return (ctx or {}).get("stage", "fwd")


def _match(ctx: dict[str, Any]) -> bool:
    tags = _tags()
    if not tags:
        return False
    if tags == "all":
        return True
    cur = ctx.get("stage") or "fwd"
    for raw in tags.split(","):
        raw = raw.strip()
        if not raw:
            continue
        tag, _, cond = raw.partition(":")
        if tag not in ("all", cur):
            continue
        if _cond_ok(cond, ctx):
            return True
    return False


def _cond_ok(cond: str, ctx: dict[str, Any]) -> bool:
    if not cond:
        return True
    if cond == "dense":
        return ctx.get("layer_kind") == "dense"
    if cond == "moe":
        return ctx.get("layer_kind") == "moe"
    try:
        return int(cond) <= int(ctx.get("max_layer", -1))
    except (TypeError, ValueError):
        return False


# -- directory helpers -------------------------------------------------------


def _rank_from_ctx(ctx: dict[str, Any]) -> str | None:
    rank = ctx.get("rank")
    if rank is not None:
        return str(rank)
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get("DEVICE")
    if visible is not None:
        visible = visible.strip()
        if visible:
            return visible
    meta = ctx.get("meta") or {}
    dev = meta.get("device") if isinstance(meta, dict) else None
    if isinstance(dev, str) and ":" in dev:
        try:
            return dev.rsplit(":", 1)[-1]
        except ValueError:
            return dev
    return dev


def _env_filter_ok(layer: int | None, ctx: dict[str, Any] | None) -> bool:
    """Optional env-gated narrowing: layers / tick range / ranks."""
    layers_env = os.environ.get("DSA_DUMP_LAYERS", "").strip()
    if layers_env:
        eff_layer = layer if layer is not None else (ctx or {}).get("layer")
        if eff_layer is not None:
            allow = {int(x) for x in layers_env.split(",") if x.strip()}
            if eff_layer not in allow:
                return False
    steps_env = os.environ.get("DSA_DUMP_STEPS", "").strip()
    if steps_env:
        parts = steps_env.split(":")
        lo = int(parts[0]) if parts and parts[0] else 0
        hi = int(parts[1]) if len(parts) > 1 and parts[1] else 1 << 62
        tick = int((ctx or {}).get("tick", _STATE.get("tick", 0)) or 0)
        if tick < lo or tick > hi:
            return False
    ranks_env = os.environ.get("DSA_DUMP_RANKS", "").strip()
    if ranks_env:
        allow = {x.strip() for x in ranks_env.split(",") if x.strip()}
        if _rank_from_ctx(ctx) not in allow:
            return False
    return True


def step_dir() -> str:
    ctx = ctx_get() or {}
    tick = ctx.get("tick", _STATE["tick"])
    rank = _rank_from_ctx(ctx)
    base = os.path.join(_STATE["root"], f"S{tick}_{ctx.get('stage', 'fwd')}")
    if rank is None:
        return base
    return os.path.join(base, f"r{rank}")


def write_meta() -> None:
    if not enabled():
        return
    ctx = ctx_get()
    if ctx is None:
        return
    try:
        os.makedirs(step_dir(), exist_ok=True)
        payload = {k: v for k, v in ctx.items() if k != "ctx"}
        with open(os.path.join(step_dir(), "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=_json_default)
    except Exception:
        pass


# -- capture -----------------------------------------------------------------


def snap(
    task: str,
    tensors: dict[str, torch.Tensor | None],
    *,
    layer: int | None = None,
    kind: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Capture named tensors plus scalar context for one operator invocation."""
    if not shall_dump(layer=layer, kind=kind):
        return
    _allowed = {"sparse_attn_sharedkv", "sparse_attn_sharedkv_out",
                "sparse_flash_mla", "sparse_flash_mla_out",
                "sparse_flash_mla_metadata",
                "quant_lightning_indexer_metadata",
                "logits",
                "quant_lightning_indexer_v2_out", "quant_lightning_indexer_out",
                "quant_lightning_indexer_v2", "quant_lightning_indexer",
                "graph_bt_0", "graph_bt_1", "graph_bt_2", "graph_bt_3",
                "graph_slots_0", "graph_meta", "layer_output",
                "ctx_kv", "draft_final", "aux_capture",
                "decoder_input", "hc_pre_attn", "input_layernorm_out",
                "attn_output", "hc_post_attn", "hc_pre_ffn",
                "post_attention_layernorm_out", "mlp_output", "hc_post_ffn"}
    if task not in _allowed:
        return
    if not _env_filter_ok(layer, ctx_get()):
        return
    d = step_dir()
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        return
    ctx = ctx_get() or {}
    layer_tag = ""
    layer_id = layer if layer is not None else ctx.get("layer")
    if isinstance(layer_id, int) and layer_id >= 0:
        layer_tag = f"L{layer_id}_"
    entry: dict[str, Any] = {"task": task}
    if layer_id is not None:
        entry["layer"] = layer_id
    if kind:
        entry["kind"] = kind
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            entry[name] = {"scalar": tensor}
            continue
        desc = _describe(tensor)
        entry[name] = desc
        if tensor.device.type != "cpu":
            try:
                torch.save(
                    tensor.detach().cpu(),
                    os.path.join(d, f"{layer_tag}{name}.pt"),
                )
            except Exception:
                pass
    if extra:
        entry["extra"] = extra
    _STATE["manifest"].append(entry)

    log = _logger()
    if log is not None:
        brief = " ".join(
            f"{k}={_brief(entry[k])}" for k in tensors if k in entry
        )
        log.info("DSA_DUMP %s | %s", task, brief)


def _describe(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None:
        return {"empty": True}
    if tensor.numel() == 0:
        return {"empty": True, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    arr = tensor.detach()
    desc: dict[str, Any] = {
        "empty": False,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    if arr.is_floating_point():
        try:
            f = arr.float()
            desc["nan"] = int(torch.isnan(f).sum().item())
            desc["inf"] = int(torch.isinf(f).sum().item())
            desc["mean"] = float(f.mean())
            desc["std"] = float(f.std())
            desc["absmax"] = float(f.abs().max())
        except Exception:
            pass
    if arr.dtype in (torch.int32, torch.int64, torch.int8, torch.uint8):
        try:
            desc["min"] = int(arr.min())
            desc["max"] = int(arr.max())
        except Exception:
            pass
    try:
        desc["head"] = arr.cpu().flatten()[: _HEAD_N].tolist()
    except Exception:
        pass
    return desc


def _brief(desc: dict[str, Any]) -> str:
    if "scalar" in desc:
        return f"scalar={desc['scalar']}"
    if desc.get("empty"):
        return "empty"
    out = [str(desc.get("dtype")), "x".join(map(str, desc.get("shape", [])))]
    for key in ("nan", "mean", "std", "min", "max"):
        if key in desc:
            out.append(f"{key}={desc[key]}")
    return ",".join(out)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)

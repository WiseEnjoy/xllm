"""Unit tests for the MegaMoe Python integration.

Runs without NPU hardware: validates the scale encoding, config plumbing,
and the model's MegaMoe branch selection logic.
"""

from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TestScaleEncoding(unittest.TestCase):
    """W8A8 scale fp32 -> int64 bit-cast encoding."""

    def test_encode_matches_cpp(self) -> None:
        """Verify the Python encoding matches C++ convert_fp32_scale_to_int64.

        C++ (fused_moe.cpp:232):
            scale.to(kFloat32).view(kInt32).to(kInt64)
        """
        scale = torch.tensor([0.5, 1.0, 2.0, 0.125], dtype=torch.float32)
        encoded = scale.view(torch.int32).to(torch.int64)
        self.assertEqual(encoded.dtype, torch.int64)
        self.assertEqual(encoded.shape, scale.shape)
        # Round-trip: int64 -> int32 -> float32 recovers the original
        decoded = encoded.to(torch.int32).view(torch.float32)
        torch.testing.assert_close(decoded, scale)

    def test_encode_preserves_bit_pattern(self) -> None:
        """Each fp32's raw bits must survive as the low 32 bits of int64."""
        for val in [0.0, 1.0, -1.0, 0.5, 255.0, 1e-10, 1e10]:
            scale = torch.tensor([val], dtype=torch.float32)
            encoded = scale.view(torch.int32).to(torch.int64)
            raw_bits = struct.unpack("<I", struct.pack("<f", val))[0]
            self.assertEqual(int(encoded[0]) & 0xFFFFFFFF, raw_bits)

    def test_zero_scale(self) -> None:
        """Zero scale encodes to zero int64."""
        scale = torch.zeros(4, dtype=torch.float32)
        encoded = scale.view(torch.int32).to(torch.int64)
        torch.testing.assert_close(encoded, torch.zeros(4, dtype=torch.int64))


class TestConfigPlumbing(unittest.TestCase):
    """DeepseekV4Config parses enable_mega_moe from the config dict."""

    def test_default_off(self) -> None:
        from xllm.python.models.deepseek_v4 import DeepseekV4Config

        cfg = DeepseekV4Config.from_dict({
            "hidden_size": 4096,
            "model_type": "deepseek_v4",
        })
        self.assertFalse(cfg.enable_mega_moe)

    def test_enabled(self) -> None:
        from xllm.python.models.deepseek_v4 import DeepseekV4Config

        cfg = DeepseekV4Config.from_dict({
            "hidden_size": 4096,
            "model_type": "deepseek_v4",
            "enable_mega_moe": True,
        })
        self.assertTrue(cfg.enable_mega_moe)


class TestMegaMoeBranchSelection(unittest.TestCase):
    """MoE forward dispatches to MegaMoe when enabled and EP > 1."""

    def _make_moe(self, enable: bool, ep_size: int = 4) -> MagicMock:
        """Build a DeepseekV4MoE with mock weights (no NPU needed)."""
        from xllm.python.models.deepseek_v4 import DeepseekV4Config, DeepseekV4MoE

        cfg = DeepseekV4Config.from_dict({
            "hidden_size": 64,
            "model_type": "deepseek_v4",
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "ep_size": ep_size,
            "ep_rank": 0,
            "tp_size": ep_size,
            "tp_rank": 0,
            "n_shared_experts": 1,
            "enable_mega_moe": enable,
        })
        device = torch.device("cpu")
        moe = DeepseekV4MoE(cfg, 3, device=device, dtype=torch.float32)
        return moe

    def test_disabled_uses_standard_path(self) -> None:
        moe = self._make_moe(enable=False)
        self.assertFalse(moe.enable_mega_moe)

    def test_enabled_sets_flag(self) -> None:
        moe = self._make_moe(enable=True)
        self.assertTrue(moe.enable_mega_moe)
        self.assertFalse(moe._mega_moe_prepared)

    def test_ep1_disables(self) -> None:
        """MegaMoe requires EP > 1 even when config enables it."""
        moe = self._make_moe(enable=True, ep_size=1)
        self.assertFalse(moe.enable_mega_moe)


class TestMegaMoeKernelSignature(unittest.TestCase):
    """The kernels_npu mega_moe wrapper matches the C++ operator signature."""

    def test_function_exists(self) -> None:
        from xllm.python.kernels_npu import moe as moe_kernels

        self.assertTrue(hasattr(moe_kernels, "mega_moe"))
        self.assertTrue(hasattr(moe_kernels, "encode_w8a8_scale_int64"))
        self.assertTrue(hasattr(moe_kernels, "has_mega_moe"))

    def test_has_mega_moe_returns_bool(self) -> None:
        from xllm.python.kernels_npu.moe import has_mega_moe

        result = has_mega_moe()
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()

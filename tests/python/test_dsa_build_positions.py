"""Unit tests for the vectorized DsaMetadataBuilder._build_positions.

Runs without NPU hardware: verifies the vectorized per-sequence RoPE-position
gather produces bit-identical c4/c128 padded positions to the reference nested
loop it replaces.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from xllm.python.attention.dsa_metadata import DsaMetadata, DsaMetadataBuilder


def _reference_positions(
    total_tokens: int,
    kv_seq_lens: list[int],
    q_lens: list[int],
    enable_graph: bool,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The pre-vectorization nested-loop reference."""
    c4: list[int] = []
    c128: list[int] = []
    for seq, kv_len in enumerate(kv_seq_lens):
        q_len = min(q_lens[seq], kv_len)
        start_pos = kv_len - q_len
        for i in range(q_len):
            next_pos = start_pos + i + 1
            if next_pos % 4 == 0:
                c4.append(next_pos - 4)
            if next_pos % 128 == 0:
                c128.append(next_pos - 128)

    def _pad(positions: list[int], ratio: int) -> torch.Tensor:
        if enable_graph:
            target = total_tokens
        else:
            target = min(total_tokens, total_tokens // ratio + len(kv_seq_lens))
        out = torch.zeros(target, dtype=dtype)
        for idx, p in enumerate(positions):
            if idx >= target:
                break
            out[idx] = p
        return out

    return _pad(c4, 4), _pad(c128, 128)


def _make_dsa(total_tokens: int) -> DsaMetadata:
    return DsaMetadata(
        layer_id=0,
        seq_lens=torch.empty(0),
        seq_lens_q=torch.empty(0),
        actual_seq_lengths_kv=torch.empty(0),
        actual_seq_lengths_query=torch.empty(0),
        kv_cu_seq_lens=torch.empty(0),
        max_seqlen_kv=torch.empty(0),
        max_seqlen_q=torch.empty(0),
        max_query_len=0,
        max_seq_len=0,
        input_positions=torch.zeros(total_tokens, dtype=torch.int64),
        c4_pad_positions=torch.empty(0, dtype=torch.int64),
        c128_pad_positions=torch.empty(0, dtype=torch.int64),
        start_pos=torch.empty(0),
    )


class TestBuildPositionsVectorized(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = DsaMetadataBuilder(caches_info=[], group_infos=[])

    def _run_case(self, kv_seq_lens, q_lens, enable_graph, total_tokens=None):
        total_tokens = total_tokens if total_tokens is not None else sum(
            min(q, kv) for q, kv in zip(q_lens, kv_seq_lens)
        )
        dsa = _make_dsa(total_tokens)
        self.builder._build_positions(dsa, kv_seq_lens, q_lens, enable_graph)
        ref_c4, ref_c128 = _reference_positions(
            total_tokens, kv_seq_lens, q_lens, enable_graph, torch.int64
        )
        torch.testing.assert_close(dsa.c4_pad_positions, ref_c4)
        torch.testing.assert_close(dsa.c128_pad_positions, ref_c128)

    def test_single_decode_step(self) -> None:
        self._run_case([129], [1], enable_graph=False)

    def test_single_decode_step_graph(self) -> None:
        self._run_case([129], [1], enable_graph=True)

    def test_prefill_long_context(self) -> None:
        self._run_case([1024], [1024], enable_graph=False)

    def test_prefill_long_context_graph(self) -> None:
        self._run_case([1024], [1024], enable_graph=True)

    def test_multi_sequence_mixed_lens(self) -> None:
        self._run_case([500, 1000, 128], [500, 1000, 128], enable_graph=False)

    def test_multi_sequence_mixed_lens_graph(self) -> None:
        self._run_case([500, 1000, 128], [500, 1000, 128], enable_graph=True)

    def test_decode_multi_sequence(self) -> None:
        self._run_case([300, 600, 900], [1, 1, 1], enable_graph=False)

    def test_empty_batch(self) -> None:
        self._run_case([], [], enable_graph=False, total_tokens=0)

    def test_boundary_ratio_crossing(self) -> None:
        # Context lengths exactly at ratio boundaries force c4/c128 emissions.
        self._run_case([255, 256, 257], [255, 256, 257], enable_graph=False)


if __name__ == "__main__":
    unittest.main()

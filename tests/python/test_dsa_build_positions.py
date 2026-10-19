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

"""Tests for vectorized compressed-position metadata construction."""

from __future__ import annotations

import torch

from xllm.python.attention.dsa_metadata import DsaMetadata, DsaMetadataBuilder


def _metadata(total_tokens: int) -> DsaMetadata:
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


def _reference_positions(
    total_tokens: int,
    kv_seq_lens: list[int],
    q_lens: list[int],
    enable_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the original sequence-loop semantics for comparison."""
    c4: list[int] = []
    c128: list[int] = []
    for sequence, kv_len in enumerate(kv_seq_lens):
        q_len = min(q_lens[sequence], kv_len)
        start_pos = kv_len - q_len
        for offset in range(q_len):
            next_pos = start_pos + offset + 1
            if next_pos % 4 == 0:
                c4.append(next_pos - 4)
            if next_pos % 128 == 0:
                c128.append(next_pos - 128)

    def padded(values: list[int], ratio: int) -> torch.Tensor:
        if enable_graph:
            target = total_tokens
        else:
            target = min(total_tokens, total_tokens // ratio + len(kv_seq_lens))
        output = torch.zeros(target, dtype=torch.int64)
        for index, value in enumerate(values):
            if index >= target:
                break
            output[index] = value
        return output

    return padded(c4, 4), padded(c128, 128)


def _assert_positions_match_reference(
    kv_seq_lens: list[int],
    q_lens: list[int],
    *,
    enable_graph: bool,
    total_tokens: int | None = None,
) -> None:
    if total_tokens is None:
        total_tokens = sum(min(q_len, kv_len) for q_len, kv_len in zip(q_lens, kv_seq_lens))
    metadata = _metadata(total_tokens)
    builder = DsaMetadataBuilder(caches_info=[], group_infos=[])
    builder._build_positions(metadata, kv_seq_lens, q_lens, enable_graph)
    expected_c4, expected_c128 = _reference_positions(total_tokens, kv_seq_lens, q_lens, enable_graph)

    torch.testing.assert_close(metadata.c4_pad_positions, expected_c4)
    torch.testing.assert_close(metadata.c128_pad_positions, expected_c128)


def test_swa_group_missing_table_rows_keep_per_sequence_output_offsets() -> None:
    """A batch sequence without a block-table row must not shift the gathered
    tokens of the sequences before it."""
    builder = DsaMetadataBuilder(caches_info=[], group_infos=[])
    _, slots = builder._process_swa_group(
        torch.tensor([[3]], dtype=torch.int32),
        4,
        [2, 2],
        [1, 1],
        2,
        2,
        2,
        None,
    )
    # Sequence 0's only query token (block 3, offset 1) must land at output
    # index 0; sequence 1 has no block-table row, so its output offset keeps
    # the -1 padding. The old cursor advanced past missing rows before the
    # gathered tokens were written, shifting the real token to index 1.
    assert slots.tolist() == [13, -1]


def test_decode_positions_match_sequence_loop() -> None:
    _assert_positions_match_reference([129], [1], enable_graph=False)


def test_decode_positions_match_sequence_loop_in_graph_mode() -> None:
    _assert_positions_match_reference([129], [1], enable_graph=True)


def test_prefill_positions_match_sequence_loop() -> None:
    _assert_positions_match_reference([1024], [1024], enable_graph=False)


def test_mixed_sequence_positions_match_sequence_loop() -> None:
    _assert_positions_match_reference([500, 1000, 128], [500, 1000, 128], enable_graph=False)


def test_empty_batch_has_empty_position_tables() -> None:
    _assert_positions_match_reference([], [], enable_graph=False, total_tokens=0)


def test_ratio_boundaries_are_kept() -> None:
    _assert_positions_match_reference([255, 256, 257], [255, 256, 257], enable_graph=False)

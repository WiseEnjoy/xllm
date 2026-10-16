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

"""Tests for the NPU ACL decode-graph runner."""

import logging
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.dsa_metadata import DsaMetadataBuilder, build_cache_specs
from xllm.python.attention.expanded_decode_metadata import ExpandedDecodeMetadata
from xllm.python.model_executor.runners.decode_acl_graph import (
    DecodeAclGraphRunner,
)
from xllm.python.models.engram_config import EngramConfig


def _runner() -> DecodeAclGraphRunner:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    return DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
    )


def _metadata(linear_state_indices: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        slot_mapping=torch.arange(4, dtype=torch.int32),
        paged_kv_indptr=torch.arange(5, dtype=torch.int32),
        paged_kv_indices=torch.tensor([10, 20, 30, 40], dtype=torch.int32),
        paged_kv_last_page_len=torch.arange(1, 5, dtype=torch.int32),
        block_table=torch.tensor(
            [[10, 0], [20, 0], [30, 0], [40, 0]],
            dtype=torch.int32,
        ),
        kv_seq_lens=torch.arange(1, 5, dtype=torch.int32),
        kv_seq_lens_host_values=[1, 2, 3, 4],
        kv_cu_seq_lens=torch.tensor([0, 1, 3, 6, 10], dtype=torch.int32),
        q_cu_seq_lens=torch.arange(5, dtype=torch.int32),
        linear_state_indices=linear_state_indices,
        num_accepted_tokens=torch.tensor([1, 2, 1, 2], dtype=torch.int32),
        expanded_decode_metadata=None,
        multi_block_tables=(),
        new_cache_slots_host_values=[0, 1, 2, 3],
        is_prefill=False,
        is_chunked_prefill=False,
    )


def test_spec_verify_rows_expand_to_token_row_bucket() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=2,
        max_model_len=8,
        num_decoding_tokens=6,
        spec_verify_rows=6,
    )
    metadata = _metadata(torch.arange(2, dtype=torch.int32))

    assert runner.sequence_batch_capacity == 2
    assert runner.max_batch == 16
    assert runner._engram_graph_token_capacity() == 16
    with patch.object(runner, "_has_compatible_decode_metadata", return_value=True):
        assert runner.can_execute(torch.zeros(12, dtype=torch.int32), metadata)


def test_plain_decode_does_not_inherit_spec_verify_capacity() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=2,
        max_model_len=8,
        num_decoding_tokens=6,
    )
    metadata = _metadata(torch.arange(2, dtype=torch.int32))

    assert runner.sequence_batch_capacity == 2
    assert runner.max_batch == 2
    assert runner._engram_graph_token_capacity() == 16
    with patch.object(runner, "_has_compatible_decode_metadata", return_value=True):
        assert runner.can_execute(torch.zeros(12, dtype=torch.int32), metadata) is False


@pytest.mark.parametrize(
    "limit,sequences,expected",
    [(16, 1, True), (16, 2, True), (16, 3, False), (24, 3, False), (32, 3, True), (32, 4, True), (32, 5, False)],
)
def test_eplv2_graph_admission_uses_padded_token_rows_for_dflash2(limit: int, sequences: int, expected: bool) -> None:
    runner = _runner()
    runner.max_batch = 128
    runner.num_decoding_tokens = 8
    runner._eplv2_graph_token_limit = limit
    metadata = _metadata(torch.arange(3, dtype=torch.int32))
    with (
        patch.object(runner, "_has_compatible_decode_metadata", return_value=True),
        patch(
            "xllm.python.model_executor.runners.decode_acl_graph.resolve_expanded_decode_metadata",
            return_value=object(),
        ),
    ):
        # Three 8-row sequences capture a padded 32-row graph. A 24-row
        # communication cap is insufficient even though the active rows fit.
        metadata.linear_state_indices = torch.arange(sequences)
        assert runner.can_execute(torch.arange(sequences * 8), metadata) is expected


def test_accepted_tokens_use_live_per_sequence_graph_buffer() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.tensor([3, 3, 7, 7], dtype=torch.int32))
    metadata.num_accepted_tokens = torch.tensor([4, 2], dtype=torch.int32)
    entry = runner._allocate_entry(8, input_ids, positions, metadata)
    static_counts = entry.static_metadata.num_accepted_tokens
    address = static_counts.data_ptr()
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)
        assert static_counts.tolist() == [4, 2, 1, 1, 1, 1, 1, 1]
        metadata.num_accepted_tokens.copy_(torch.tensor([1, 3]))
        runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)
        assert static_counts.tolist() == [1, 3, 1, 1, 1, 1, 1, 1]
        metadata.num_accepted_tokens = torch.tensor([2], dtype=torch.int32)
        runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)
        assert static_counts.tolist() == [2, 1, 1, 1, 1, 1, 1, 1]
        metadata.num_accepted_tokens = None
        metadata.is_dummy = True
        runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)
    assert static_counts.data_ptr() == address
    assert static_counts.tolist() == [1] * 8

    plain_metadata = _metadata(torch.tensor([3, 7, 11, 15], dtype=torch.int32))
    plain_metadata.num_accepted_tokens = None
    plain_entry = runner._allocate_entry(8, input_ids, positions, plain_metadata)
    assert plain_entry.static_metadata.num_accepted_tokens is None


def test_verify_graph_key_tracks_width_not_active_sequence_count() -> None:
    runner = _runner()
    captured_keys = []

    def capture_key(key: tuple) -> None:
        captured_keys.append(key)
        raise LookupError("graph key recorded")

    runner._graphs = SimpleNamespace(get=capture_key)
    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.resolve_expanded_decode_metadata",
        return_value=object(),
    ):
        for sequence_count, width in ((3, 4), (4, 4), (8, 2)):
            input_ids = torch.arange(sequence_count * width, dtype=torch.int32)
            metadata = SimpleNamespace(linear_state_indices=torch.arange(sequence_count))
            with pytest.raises(LookupError, match="graph key recorded"):
                runner.execute(input_ids, input_ids, metadata)
    assert captured_keys[0] == captured_keys[1]
    assert captured_keys[1] != captured_keys[2]


@pytest.mark.parametrize(
    "accepted_counts,error_type",
    [
        (None, RuntimeError),
        (torch.ones(2, 2, dtype=torch.int32), ValueError),
        (torch.ones(0, dtype=torch.int32), ValueError),
        (torch.ones(5, dtype=torch.int32), ValueError),
    ],
)
def test_accepted_tokens_reject_missing_or_invalid_replay_metadata(
    accepted_counts: torch.Tensor | None, error_type: type[Exception]
) -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.tensor([3, 3, 7, 7], dtype=torch.int32))
    metadata.num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    entry = runner._allocate_entry(8, input_ids, positions, metadata)
    entry.static_metadata.is_spec_verify = True
    metadata.num_accepted_tokens = accepted_counts
    with (
        patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True),
        pytest.raises(error_type, match="accepted-token metadata"),
    ):
        runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)


def test_slice_output_preserves_aux_hidden_tuple() -> None:
    hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    aux_hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    output = DecodeAclGraphRunner._slice_output((hidden, aux_hidden), 3)

    assert isinstance(output, tuple)
    torch.testing.assert_close(output[0], hidden[:3])
    torch.testing.assert_close(output[1], aux_hidden[:3])


def test_linear_state_graph_buffers_are_stable() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.tensor([3, 7, 11, 15], dtype=torch.int32))
    metadata.kpool_query_lens = [4]
    entry = runner._allocate_entry(
        padded_batch_size=8,
        input_ids=input_ids,
        positions=positions,
        metadata=metadata,
    )
    static_indices = entry.static_metadata.linear_state_indices
    static_num_accepted_tokens = entry.static_metadata.num_accepted_tokens
    data_ptr = static_indices.data_ptr()
    accepted_data_ptr = static_num_accepted_tokens.data_ptr()
    assert entry.static_metadata.kpool_query_lens == (4, 4)
    assert entry.static_metadata.kpool_query_lens_device.tolist() == [4, 4]
    assert DecodeAclGraphRunner._graph_key(
        8,
        True,
        None,
        entry.static_metadata.kpool_query_lens,
    ) != DecodeAclGraphRunner._graph_key(8, True, None, (2, 2, 2, 2))

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ):
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )
        assert static_indices.tolist() == [3, 7, 11, 15, 0, 0, 0, 0]
        metadata.linear_state_indices = torch.tensor(
            [4, 8, 12, 16],
            dtype=torch.int32,
        )
        metadata.num_accepted_tokens = torch.tensor(
            [2, 1, 2, 1],
            dtype=torch.int32,
        )
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )

    assert static_indices.data_ptr() == data_ptr
    assert static_indices.tolist() == [4, 8, 12, 16, 0, 0, 0, 0]
    assert static_num_accepted_tokens.data_ptr() == accepted_data_ptr
    assert static_num_accepted_tokens.tolist() == [2, 1, 2, 1, 1, 1, 1, 1]


def test_dflash_proposal_mode_uses_a_distinct_graph_key() -> None:
    ordinary_key = DecodeAclGraphRunner._graph_key(8, False, None)
    proposal_key = DecodeAclGraphRunner._graph_key(
        8,
        False,
        None,
        is_dflash_proposal=True,
    )

    assert ordinary_key != proposal_key


@pytest.mark.parametrize("verify_width", [1, 2])
def test_capture_preserves_live_state_until_first_replay(monkeypatch: pytest.MonkeyPatch, verify_width: int) -> None:
    npu = MagicMock()
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    runner = _runner()
    slots = torch.arange(1, 5, verify_width)
    metadata = _metadata(slots)
    metadata.has_initial_state = torch.ones_like(slots)
    metadata.num_accepted_tokens = torch.full_like(slots, verify_width, dtype=torch.int32)
    if verify_width > 1:
        metadata.is_spec_verify = True
        metadata.expanded_decode_metadata = ExpandedDecodeMetadata(
            kv_seq_lens=metadata.kv_seq_lens,
            block_table=metadata.block_table,
            paged_kv_indptr=metadata.paged_kv_indptr,
            paged_kv_indices=metadata.paged_kv_indices,
            paged_kv_last_page_len=metadata.paged_kv_last_page_len,
            paged_attention_tiling_data=None,
            kv_seq_lens_host=None,
            kv_seq_lens_host_values=metadata.kv_seq_lens_host_values,
        )
    conv = torch.arange(8, dtype=torch.float32)
    ssm = torch.arange(8 * verify_width, dtype=torch.float32)
    initial_conv, initial_ssm = conv.clone(), ssm.clone()
    runner.bind_layer_caches([LayerCache(None, None, conv=conv, ssm=ssm)])
    output = torch.empty(4, 1)
    prepared = []

    def prepare(static: object, *, graph_mode: bool) -> None:
        indices = static.linear_state_indices[::verify_width]
        checkpoints = indices[:, None] * verify_width + torch.arange(verify_width)
        if static.linear_state_checkpoint_indices is None:
            static.linear_state_checkpoint_indices = torch.empty_like(checkpoints)
        static.linear_state_checkpoint_indices.copy_(checkpoints)
        prepared.append(static)

    def advance(_entry: object = None) -> torch.Tensor:
        static = prepared[-1]
        indices = static.linear_state_indices[::verify_width]
        checkpoints = static.linear_state_checkpoint_indices
        accepted = static.num_accepted_tokens[: indices.numel()].long()
        values = ssm[checkpoints[torch.arange(indices.numel()), accepted - 1]] + 1
        conv.index_copy_(0, indices, conv[indices] + 1)
        ssm.index_copy_(0, checkpoints.flatten(), values.repeat_interleave(verify_width))
        output.copy_(values.repeat_interleave(verify_width).view(-1, 1))
        if npu.NPUGraph.return_value.replay.call_count == 0:
            torch.testing.assert_close(conv[1:], initial_conv[1:])
            torch.testing.assert_close(ssm[verify_width:], initial_ssm[verify_width:])
        return output

    runner.attention_backend.prepare = prepare
    monkeypatch.setattr(runner, "_forward_static", advance)
    npu.NPUGraph.return_value.replay.side_effect = advance
    tokens = torch.arange(4, dtype=torch.int32)
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        result = runner.execute(tokens, tokens, metadata)
    expected = initial_ssm[slots * verify_width + verify_width - 1] + 1
    torch.testing.assert_close(result.flatten(), expected.repeat_interleave(verify_width))
    torch.testing.assert_close(conv[slots], initial_conv[slots] + 1)
    checkpoints = slots[:, None] * verify_width + torch.arange(verify_width)
    torch.testing.assert_close(ssm[checkpoints], expected[:, None].expand_as(checkpoints))
    torch.testing.assert_close(metadata.linear_state_indices, slots)
    assert metadata.has_initial_state.tolist() == [1] * slots.numel()
    assert metadata.num_accepted_tokens.tolist() == [verify_width] * slots.numel()


def test_explicit_verify_reuses_row_aligned_paging_metadata() -> None:
    runner = _runner()
    metadata = _metadata(torch.tensor([1, 2], dtype=torch.int32))
    metadata.expanded_decode_metadata = ExpandedDecodeMetadata(
        kv_seq_lens=metadata.kv_seq_lens,
        block_table=metadata.block_table,
        paged_kv_indptr=metadata.paged_kv_indptr,
        paged_kv_indices=metadata.paged_kv_indices,
        paged_kv_last_page_len=metadata.paged_kv_last_page_len,
        paged_attention_tiling_data=None,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=metadata.kv_seq_lens_host_values,
    )
    metadata.is_spec_verify = True
    metadata.is_chunked_prefill = True
    metadata.num_accepted_tokens = torch.tensor([4, 2], dtype=torch.int32)
    metadata.q_seq_lens = torch.tensor([2, 2], dtype=torch.int32)
    metadata.q_cu_seq_lens = torch.tensor([0, 2, 4], dtype=torch.int32)
    metadata.kv_seq_lens = torch.tensor([2, 4], dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [2, 4]
    metadata.kv_cu_seq_lens = torch.tensor([0, 2, 6], dtype=torch.int32)
    metadata.block_table = metadata.block_table[1::2]

    with patch.object(
        runner,
        "_build_row_aligned_paged_kv_metadata",
        side_effect=AssertionError("row-aligned paging must come from C++"),
    ):
        resolved = runner._decode_metadata(metadata)

    assert resolved[3] is metadata.paged_kv_indptr
    assert resolved[4] is metadata.paged_kv_indices
    assert resolved[5] is metadata.paged_kv_last_page_len
    input_ids = torch.arange(4, dtype=torch.int32)
    assert runner.can_execute(input_ids, metadata)
    entry = runner._allocate_entry(8, input_ids, input_ids, metadata)
    assert entry.static_metadata.is_spec_verify
    assert entry.static_metadata.q_seq_lens.tolist() == [2, 2, 2, 2]
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        runner._fill_entry(entry, input_ids, input_ids, metadata, batch_size=4, input_embedding=None)
    assert entry.static_metadata.num_accepted_tokens.tolist() == [4, 2, 1, 1, 1, 1, 1, 1]
    assert entry.static_metadata.linear_state_indices.tolist() == [1, 1, 2, 2, 0, 0, 0, 0]


def test_verify_graph_requires_explicit_metadata() -> None:
    runner = _runner()
    metadata = _metadata(torch.tensor([1, 2], dtype=torch.int32))
    assert not runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)


@pytest.mark.parametrize("dp_size", [1, 2])
def test_fill_entry_refreshes_mega_moe_mask_in_place(dp_size: int) -> None:
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
        dp_size=dp_size,
        enable_mega_moe_token_mask=True,
    )
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    entry = runner._allocate_entry(
        padded_batch_size=8,
        input_ids=input_ids,
        positions=positions,
        metadata=metadata,
    )
    mask = entry.static_metadata.mega_moe_token_mask
    data_ptr = mask.data_ptr()

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ):
        for remote_count in (3, 1):
            metadata.dp_execution_token_counts = (4, remote_count)[:dp_size]
            runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)
            expected = [True] * 4 + [False] * 4
            if dp_size == 2:
                expected += [True] * remote_count + [False] * (8 - remote_count)
            assert mask.tolist() == expected
            assert entry.static_metadata.mega_moe_token_mask.data_ptr() == data_ptr


def test_dsa_graph_tables_use_compressed_block_counts() -> None:
    _, group_infos = build_cache_specs([1, 4, 128], 128, 3)
    attention_backend = SimpleNamespace(
        page_size=128,
        is_mla=False,
        group_infos=group_infos,
    )
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=8,
        max_model_len=32768,
    )
    runner._max_blocks_per_sequence = 256

    tables = runner._build_static_multi_block_tables(4, torch.device("cpu"))

    assert [tuple(table.shape) for table in tables] == [(4, 256), (4, 16), (4, 16)]
    assert all(torch.all(table == 0) for table in tables)

    block_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    indptr, indices, last_page_len = runner._build_row_aligned_paged_kv_metadata(
        block_table,
        [129, 132],
    )
    assert indptr.tolist() == [0, 2, 4]
    assert indices.tolist() == [10, 11, 20, 21]
    assert last_page_len.tolist() == [1, 4]

    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.kv_seq_lens_host_values = None
    metadata.paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32)
    metadata.paged_kv_last_page_len = torch.tensor([1], dtype=torch.int32)
    assert not runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)


def test_dsa_graph_refreshes_every_manager_and_clears_tails() -> None:
    _, group_infos = build_cache_specs([1, 4, 128], 128, 3)
    attention_backend = SimpleNamespace(
        page_size=128,
        is_mla=False,
        group_infos=group_infos,
    )
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=4,
        max_model_len=512,
    )
    static_metadata = SimpleNamespace(
        multi_block_tables=(
            torch.full((4, 4), 99, dtype=torch.int32),
            torch.full((4, 2), 99, dtype=torch.int32),
            torch.full((4, 1), 99, dtype=torch.int32),
        )
    )
    metadata = SimpleNamespace(
        multi_block_tables=(
            torch.tensor([[10, 11], [12, 13]], dtype=torch.int32),
            torch.tensor([[20], [21]], dtype=torch.int32),
            torch.tensor([[30], [31]], dtype=torch.int32),
        )
    )

    runner._fill_dsa_block_tables(
        static_metadata,
        metadata,
        torch.empty(0, dtype=torch.int32),
        batch_size=2,
    )

    assert static_metadata.multi_block_tables[0].tolist() == [
        [10, 11, 0, 0],
        [12, 13, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert static_metadata.multi_block_tables[1].tolist() == [
        [20, 0],
        [21, 0],
        [0, 0],
        [0, 0],
    ]
    assert static_metadata.multi_block_tables[2].tolist() == [[30], [31], [0], [0]]

    metadata.multi_block_tables = (
        torch.tensor([[40]], dtype=torch.int32),
        torch.tensor([[50]], dtype=torch.int32),
        torch.tensor([[60]], dtype=torch.int32),
    )
    runner._fill_dsa_block_tables(
        static_metadata,
        metadata,
        torch.empty(0, dtype=torch.int32),
        batch_size=1,
    )

    assert static_metadata.multi_block_tables[0].tolist() == [
        [40, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert static_metadata.multi_block_tables[1].tolist() == [
        [50, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
    assert static_metadata.multi_block_tables[2].tolist() == [[60], [0], [0], [0]]


def test_dsa_graph_clamps_target_tables_to_draft_groups() -> None:
    _, group_infos = build_cache_specs([1], 128, 1)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=128, is_mla=False, group_infos=group_infos),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=512,
    )
    static_metadata = SimpleNamespace(multi_block_tables=(torch.full((4, 2), 99, dtype=torch.int32),))
    metadata = SimpleNamespace(
        multi_block_tables=(
            torch.tensor([[10], [11]], dtype=torch.int32),
            torch.tensor([[20], [21]], dtype=torch.int32),
            torch.tensor([[30], [31]], dtype=torch.int32),
        )
    )

    runner._fill_dsa_block_tables(
        static_metadata,
        metadata,
        torch.empty(0, dtype=torch.int32),
        batch_size=2,
    )

    assert static_metadata.multi_block_tables[0].tolist() == [
        [10, 0],
        [11, 0],
        [0, 0],
        [0, 0],
    ]


def test_dsa_graph_padding_uses_reserved_single_kv_length() -> None:
    _, group_infos = build_cache_specs([1, 4, 128], 128, 3)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(
            page_size=128,
            is_mla=False,
            group_infos=group_infos,
        ),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=512,
    )
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=SimpleNamespace(kv_seq_lens_host_values=[1, 1, 1, 1]),
    )

    runner._fill_host_metadata(entry, [9, 17], batch_size=2)

    assert entry.static_metadata.kv_seq_lens_host_values == [9, 17, 1, 1]


def test_dsa_graph_pads_new_cache_slots_to_graph_bucket() -> None:
    entry = SimpleNamespace(
        batch_size=16,
        static_metadata=SimpleNamespace(new_cache_slots_host_values=[]),
    )
    metadata = SimpleNamespace(new_cache_slots_host_values=[301, 302, 401, 402])

    DecodeAclGraphRunner._fill_new_cache_slots_host_metadata(
        entry,
        metadata,
        batch_size=4,
    )

    assert entry.static_metadata.new_cache_slots_host_values == [301, 302, 401, 402] + [0] * 12


def test_dsa_graph_padding_preserves_scheduler_resolved_swa_slots() -> None:
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=SimpleNamespace(new_cache_slots_host_values=[]),
    )
    metadata = SimpleNamespace(
        new_cache_slots_host_values=[1280, 2560, 3840],
    )
    DecodeAclGraphRunner._fill_new_cache_slots_host_metadata(
        entry,
        metadata,
        batch_size=3,
    )

    caches_info, group_infos = build_cache_specs([0, 4, 128, 4], 128, 4)
    builder = DsaMetadataBuilder(caches_info, group_infos)
    dsa = builder.build(
        multi_block_tables=[
            torch.tensor(
                [
                    [10, 11, 0, 0],
                    [20, 21, 0, 0],
                    [30, 31, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=torch.int32,
            ),
            torch.zeros((4, 4), dtype=torch.int32),
            torch.zeros((4, 4), dtype=torch.int32),
        ],
        kv_seq_lens=[257, 257, 257, 1],
        q_seq_lens=[1, 1, 1, 1],
        positions=torch.tensor([256, 256, 256, 0], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
        new_cache_slots=entry.static_metadata.new_cache_slots_host_values,
        enable_graph=True,
        graph_block_table_capacity_cols=4,
    )

    assert dsa.slot_mappings[0][0].tolist() == [1280, 2560, 3840, 0]


def test_dsa_graph_allows_dummy_without_scheduler_cache_slots() -> None:
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=SimpleNamespace(new_cache_slots_host_values=[99]),
    )
    metadata = SimpleNamespace(new_cache_slots_host_values=[], is_dummy=True)

    DecodeAclGraphRunner._fill_new_cache_slots_host_metadata(
        entry,
        metadata,
        batch_size=1,
    )

    assert entry.static_metadata.new_cache_slots_host_values == []


def test_dsv4_decode_admits_scheduler_slots_when_top_level_mapping_is_empty() -> None:
    runner = _runner()
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.slot_mapping = torch.empty(0, dtype=torch.int32)
    metadata.block_table = None
    metadata.multi_block_tables = (torch.tensor([[10, 0], [20, 0], [30, 0], [40, 0]], dtype=torch.int32),)
    metadata.new_cache_slots_host_values = [301, 302, 401, 402]

    assert runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)
    assert runner._effective_slot_mapping(metadata, 4, torch.device("cpu")).tolist() == [
        301,
        302,
        401,
        402,
    ]


def test_decode_rejects_mismatched_linear_state_validity_mask() -> None:
    runner = _runner()
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.has_initial_state = torch.ones(3, dtype=torch.int32)

    assert not runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)


def test_decode_rejects_mismatched_linear_state_indices_without_cache_probe() -> None:
    runner = _runner()
    metadata = _metadata(torch.arange(3, dtype=torch.int32))

    assert not runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)


def test_fill_entry_uses_scheduler_slots_for_dsv4_decode() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.slot_mapping = torch.empty(0, dtype=torch.int32)
    metadata.new_cache_slots_host_values = [301, 302, 401, 402]
    entry = runner._allocate_entry(8, input_ids, positions, metadata)

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ) as update_metadata:
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )

    assert update_metadata.call_args.args[2].tolist() == [301, 302, 401, 402]
    assert entry.static_metadata.new_cache_slots_host_values == [301, 302, 401, 402, 0, 0, 0, 0]


def test_dp_real_and_dummy_decode_share_graph_admission() -> None:
    def make_metadata(*, is_dummy: bool) -> SimpleNamespace:
        metadata = _metadata(torch.tensor([0], dtype=torch.int32))
        metadata.slot_mapping = torch.tensor([0], dtype=torch.int32) if is_dummy else torch.empty(0, dtype=torch.int32)
        metadata.block_table = None
        metadata.multi_block_tables = (torch.tensor([[0]], dtype=torch.int32),)
        metadata.kv_seq_lens = torch.tensor([1], dtype=torch.int32)
        metadata.kv_seq_lens_host_values = [1]
        metadata.kv_cu_seq_lens = torch.tensor([0, 1], dtype=torch.int32)
        metadata.q_cu_seq_lens = torch.tensor([0, 1], dtype=torch.int32)
        metadata.paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32)
        metadata.paged_kv_indices = torch.tensor([0], dtype=torch.int32)
        metadata.paged_kv_last_page_len = torch.tensor([1], dtype=torch.int32)
        metadata.new_cache_slots_host_values = [] if is_dummy else [301]
        metadata.dp_execution_token_counts = (1, 1)
        metadata.dp_is_decode = (1, 1)
        metadata.is_dummy = is_dummy
        return metadata

    real_runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
    )
    dummy_runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=8,
        dp_size=2,
        dp_rank=1,
    )
    input_ids = torch.tensor([1], dtype=torch.int32)

    assert real_runner.can_execute(input_ids, make_metadata(is_dummy=False))
    assert dummy_runner.can_execute(input_ids, make_metadata(is_dummy=True))


def test_dsa_graph_positions_are_refreshed_from_current_input() -> None:
    runner = _runner()
    entry = SimpleNamespace(
        batch_size=4,
        static_positions=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        static_metadata=SimpleNamespace(dsa_positions=None),
    )

    runner._fill_graph_dsa_positions(
        entry,
        torch.tensor([20, 21], dtype=torch.int32),
    )

    assert entry.static_metadata.dsa_positions.tolist() == [20, 21, 0, 0]


def _dp_metadata(
    token_counts: tuple[int, int],
    dp_is_decode: tuple[int, int] = (1, 1),
) -> SimpleNamespace:
    return SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        dp_execution_token_counts=tuple(1 if count == 0 else count for count in token_counts),
        dp_is_decode=dp_is_decode,
    )


def test_dp_empty_rank_uses_group_wide_acl_graph_bucket() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=1,
    )

    with patch.object(
        runner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runner.can_execute(
            torch.zeros(1, dtype=torch.int32),
            _dp_metadata((5, 0)),
        )


def test_dp_mtp_target_dummy_rank_uses_shared_graph_bucket() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    active_runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
        num_decoding_tokens=4,
    )
    empty_runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=1,
        num_decoding_tokens=4,
    )
    draft_runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
        num_decoding_tokens=1,
    )
    metadata = _dp_metadata((4, 0))

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert active_runner.can_execute(torch.zeros(4, dtype=torch.int32), metadata)
        assert empty_runner.can_execute(torch.zeros(1, dtype=torch.int32), metadata)
        assert draft_runner.can_execute(torch.zeros(4, dtype=torch.int32), metadata)


def test_dp_expanded_verify_and_dummy_rank_share_graph_decision() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=3,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=4,
        )
        for rank in range(2)
    ]
    active = _expanded_verify_metadata(4)
    active.dp_execution_token_counts = (4, 1)
    active.raw_dp_execution_token_counts = (4, 0)
    active.is_spec_verify = True
    active.is_chunked_prefill = True
    dummy = _dp_metadata((4, 0))
    dummy.raw_dp_execution_token_counts = (4, 0)
    dummy.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0].can_execute(torch.zeros(4, dtype=torch.int32), active)
        assert runners[1].can_execute(torch.zeros(1, dtype=torch.int32), dummy)
        busy_plan = runners[0]._shared_graph_plan(active, 4)
        empty_plan = runners[1]._shared_graph_plan(dummy, 1)
        assert busy_plan == empty_plan == (True, 4, 1)

    wide_runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=16,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=4,
        )
        for rank in range(2)
    ]
    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert wide_runners[0].can_execute(torch.zeros(4, dtype=torch.int32), active)
        assert wide_runners[1].can_execute(torch.zeros(1, dtype=torch.int32), dummy)


def test_dp_plain_decode_with_empty_rank_keeps_width_one() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=3,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=4,
        )
        for rank in range(2)
    ]
    active = _dp_metadata((4, 0))
    active.raw_dp_execution_token_counts = (4, 0)
    dummy = _dp_metadata((4, 0))
    dummy.raw_dp_execution_token_counts = (4, 0)

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0]._shared_graph_plan(active, 4) == (False, 1, 4)
        assert runners[1]._shared_graph_plan(dummy, 1) == (False, 1, 4)
        assert runners[0].can_execute(torch.zeros(4, dtype=torch.int32), active) is False
        assert runners[1].can_execute(torch.zeros(1, dtype=torch.int32), dummy) is False


def test_dp_empty_spec_rank_matches_busy_verify_graph_key() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=3,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=4,
        )
        for rank in range(2)
    ]
    active = _expanded_verify_metadata(4)
    active.dp_execution_token_counts = (4, 1)
    active.raw_dp_execution_token_counts = (4, 0)
    active.is_spec_verify = True
    dummy = _dp_metadata((4, 0))
    dummy.raw_dp_execution_token_counts = (4, 0)
    dummy.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        busy_plan = runners[0]._shared_graph_plan(active, 4)
        empty_plan = runners[1]._shared_graph_plan(dummy, 1)
        assert busy_plan == (True, 4, 1)
        assert empty_plan == busy_plan
        busy_key = DecodeAclGraphRunner._graph_key(4, busy_plan[0], None, (), verify_width=busy_plan[1])
        empty_key = DecodeAclGraphRunner._graph_key(4, empty_plan[0], None, (), verify_width=empty_plan[1])
        assert busy_key == empty_key
        assert runners[0].can_execute(torch.zeros(4, dtype=torch.int32), active)
        assert runners[1].can_execute(torch.zeros(1, dtype=torch.int32), dummy)

        unmarked = _expanded_verify_metadata(4)
        unmarked.dp_execution_token_counts = (4, 1)
        unmarked.raw_dp_execution_token_counts = (4, 0)
        unmarked.is_spec_verify = False
        with pytest.raises(RuntimeError, match="group-wide spec verify"):
            runners[0]._shared_graph_plan(unmarked, 4)


def test_dflash_draft_graph_capacity_uses_sequence_count() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=8,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=1,
            is_spec_draft=True,
            draft_query_width=8,
        )
        for rank in range(2)
    ]
    active = _dp_metadata((16, 0))
    active.raw_dp_execution_token_counts = (16, 0)
    dummy = _dp_metadata((16, 0))
    dummy.raw_dp_execution_token_counts = (16, 0)
    misaligned = _dp_metadata((10, 0))
    misaligned.raw_dp_execution_token_counts = (10, 0)

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0]._shared_graph_plan(active, 16) == (False, 1, 2)
        assert runners[1]._shared_graph_plan(dummy, 1) == (False, 1, 2)
        assert runners[0].can_execute(torch.zeros(16, dtype=torch.int32), active)
        assert runners[1].can_execute(torch.zeros(1, dtype=torch.int32), dummy)
        assert runners[0].can_execute(torch.zeros(10, dtype=torch.int32), misaligned) is False
        assert runners[1].can_execute(torch.zeros(1, dtype=torch.int32), misaligned) is False


def test_dp_uneven_sequences_stay_in_graph_with_full_batch_capacity() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=8,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=6,
        )
        for rank in range(2)
    ]
    active = _dp_metadata((48, 0))
    active.raw_dp_execution_token_counts = (48, 0)
    active.is_spec_verify = True
    dummy = _dp_metadata((48, 0))
    dummy.raw_dp_execution_token_counts = (48, 0)
    dummy.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0].max_batch == 8
        assert runners[0].can_execute(torch.zeros(48, dtype=torch.int32), active)
        assert runners[1].can_execute(torch.zeros(1, dtype=torch.int32), dummy)


def test_dp_spec_verify_width_six_pads_to_bucket_eight() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=64,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=6,
        )
        for rank in range(2)
    ]
    active = _expanded_verify_metadata(6)
    active.dp_execution_token_counts = (6, 6)
    active.raw_dp_execution_token_counts = (6, 6)
    active.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0]._shared_graph_plan(active, 6) == (True, 6, 1)
        assert runners[0].can_execute(torch.zeros(6, dtype=torch.int32), active)
        assert runners[1].can_execute(torch.zeros(6, dtype=torch.int32), active)


def test_dp_spec_verify_refuses_bucket_not_divisible_by_width() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=8,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=3,
        )
        for rank in range(2)
    ]
    active = _dp_metadata((2, 0))
    active.raw_dp_execution_token_counts = (2, 0)
    active.is_spec_verify = True
    dummy = _dp_metadata((2, 0))
    dummy.raw_dp_execution_token_counts = (2, 0)
    dummy.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0]._shared_graph_plan(active, 2) == (True, 3, None)
        assert runners[1]._shared_graph_plan(dummy, 1) == (True, 3, None)
        assert runners[0].can_execute(torch.zeros(2, dtype=torch.int32), active) is False
        assert runners[1].can_execute(torch.zeros(1, dtype=torch.int32), dummy) is False


def test_dp_spec_verify_uses_num_decoding_tokens_on_every_rank() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=16,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=8,
        )
        for rank in range(2)
    ]
    active = _expanded_verify_metadata(8)
    active.dp_execution_token_counts = (8, 1)
    active.raw_dp_execution_token_counts = (8, 0)
    active.is_spec_verify = True
    dummy = _dp_metadata((8, 0))
    dummy.raw_dp_execution_token_counts = (8, 0)
    dummy.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0]._shared_graph_plan(active, 8) == (True, 8, 1)
        assert runners[1]._shared_graph_plan(dummy, 1) == (True, 8, 1)


def test_dp_spec_verify_rejects_measured_width_mismatch() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
        num_decoding_tokens=8,
    )
    active = _expanded_verify_metadata(4)
    active.dp_execution_token_counts = (4, 1)
    active.raw_dp_execution_token_counts = (4, 0)
    active.is_spec_verify = True

    with pytest.raises(RuntimeError, match="num_decoding_tokens"):
        runner._shared_graph_plan(active, 4)


def test_dp_spec_verify_stays_eager_when_active_count_is_not_a_multiple() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=16,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=4,
        )
        for rank in range(2)
    ]
    active = _dp_metadata((6, 0))
    active.raw_dp_execution_token_counts = (6, 0)
    active.is_spec_verify = True
    dummy = _dp_metadata((6, 0))
    dummy.raw_dp_execution_token_counts = (6, 0)
    dummy.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0].can_execute(torch.zeros(6, dtype=torch.int32), active) is False
        assert runners[1].can_execute(torch.zeros(1, dtype=torch.int32), dummy) is False


def test_dp_kpool_spans_must_match_shared_bucket_layout() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
        num_decoding_tokens=8,
    )
    runner.layer_caches = [SimpleNamespace(kpool_tail=torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16))]
    aligned = _dp_metadata((8, 8))
    aligned.is_spec_verify = True
    mismatched = _dp_metadata((8, 8))
    mismatched.is_spec_verify = True
    mismatched.kpool_query_lens = (1,) * 8

    with patch.object(DecodeAclGraphRunner, "_has_compatible_decode_metadata", return_value=True):
        assert runner.can_execute(torch.zeros(8, dtype=torch.int32), aligned)
        with pytest.raises(RuntimeError, match="shared bucket layout"):
            runner.can_execute(torch.zeros(8, dtype=torch.int32), mismatched)
    assert runner._graph_kpool_query_lens(aligned, 8, 8, 8) == (8,)
    assert runner._graph_kpool_query_lens(aligned, 1, 8, 8) == (8,)

    uneven = _dp_metadata((16, 16))
    uneven.is_spec_verify = True
    uneven.kpool_query_lens = (10, 6)
    short = _dp_metadata((16, 16))
    short.is_spec_verify = True
    short.kpool_query_lens = (8,)
    nonpositive = _dp_metadata((8, 8))
    nonpositive.is_spec_verify = True
    nonpositive.kpool_query_lens = (0, 8)
    with (
        patch.object(DecodeAclGraphRunner, "_has_compatible_decode_metadata", return_value=True),
        pytest.raises(RuntimeError, match="must be uniform"),
    ):
        runner.can_execute(torch.zeros(16, dtype=torch.int32), uneven)
    with (
        patch.object(DecodeAclGraphRunner, "_has_compatible_decode_metadata", return_value=True),
        pytest.raises(RuntimeError, match="positive lengths"),
    ):
        runner.can_execute(torch.zeros(16, dtype=torch.int32), short)
    with (
        patch.object(DecodeAclGraphRunner, "_has_compatible_decode_metadata", return_value=True),
        pytest.raises(RuntimeError, match="positive lengths"),
    ):
        runner.can_execute(torch.zeros(8, dtype=torch.int32), nonpositive)


def _aligned_dp_decode_metadata(rows: int) -> SimpleNamespace:
    metadata = _dp_metadata((rows, rows))
    metadata.is_spec_verify = True
    metadata.slot_mapping = torch.arange(rows, dtype=torch.int32)
    metadata.block_table = torch.zeros((rows, 2), dtype=torch.int32)
    metadata.kv_seq_lens = torch.ones(rows, dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [1] * rows
    metadata.kv_cu_seq_lens = torch.arange(rows + 1, dtype=torch.int32)
    metadata.q_cu_seq_lens = torch.arange(rows + 1, dtype=torch.int32)
    metadata.paged_kv_indptr = torch.arange(rows + 1, dtype=torch.int32)
    metadata.paged_kv_indices = torch.zeros(rows, dtype=torch.int32)
    metadata.paged_kv_last_page_len = torch.ones(rows, dtype=torch.int32)
    metadata.linear_state_indices = torch.zeros(rows, dtype=torch.int32)
    metadata.expanded_decode_metadata = None
    return metadata


def test_dp_all_decode_row_mismatch_raises() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
        num_decoding_tokens=8,
    )
    aligned = _aligned_dp_decode_metadata(8)
    mismatched = _aligned_dp_decode_metadata(8)
    mismatched.block_table = torch.zeros((1, 2), dtype=torch.int32)
    mismatched.kv_seq_lens = torch.ones(1, dtype=torch.int32)
    mismatched.kv_seq_lens_host_values = [1]
    mismatched.slot_mapping = torch.zeros(1, dtype=torch.int32)

    assert runner.can_execute(torch.zeros(8, dtype=torch.int32), aligned)
    assert runner.can_execute(torch.zeros(8, dtype=torch.int32), aligned, torch.zeros(8, 4))
    narrow = _aligned_dp_decode_metadata(8)
    narrow.paged_kv_indptr = None
    narrow.paged_kv_indices = None
    narrow.paged_kv_last_page_len = None
    narrow.block_table = torch.zeros((8, 1), dtype=torch.int32)
    narrow.kv_seq_lens = torch.full((8,), 5, dtype=torch.int32)
    narrow.kv_seq_lens_host_values = [5] * 8
    with pytest.raises(RuntimeError, match="cannot hold the KV length"):
        runner.can_execute(torch.zeros(8, dtype=torch.int32), narrow)
    fits = _aligned_dp_decode_metadata(8)
    fits.paged_kv_indptr = None
    fits.paged_kv_indices = None
    fits.paged_kv_last_page_len = None
    fits.block_table = torch.zeros((8, 1), dtype=torch.int32)
    fits.kv_seq_lens = torch.full((8,), 4, dtype=torch.int32)
    fits.kv_seq_lens_host_values = [4] * 8
    assert runner.can_execute(torch.zeros(8, dtype=torch.int32), fits)
    with pytest.raises(RuntimeError, match="input_embedding does not match"):
        runner.can_execute(torch.zeros(8, dtype=torch.int32), aligned, torch.zeros(3, 4))
    with pytest.raises(RuntimeError, match="does not match token rows"):
        runner.can_execute(torch.zeros(8, dtype=torch.int32), mismatched)

    bad_q_cu = _aligned_dp_decode_metadata(8)
    bad_q_cu.q_cu_seq_lens = torch.zeros(3, dtype=torch.int32)
    bad_indices = _aligned_dp_decode_metadata(8)
    bad_indices.linear_state_indices = torch.zeros(3, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="q_cu_seq_lens does not match"):
        runner.can_execute(torch.zeros(8, dtype=torch.int32), bad_q_cu)
    with pytest.raises(RuntimeError, match="linear_state_indices does not match"):
        runner.can_execute(torch.zeros(8, dtype=torch.int32), bad_indices)

    bad_initial_state = _aligned_dp_decode_metadata(8)
    bad_initial_state.has_initial_state = torch.ones(3, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="has_initial_state does not match"):
        runner.can_execute(torch.zeros(8, dtype=torch.int32), bad_initial_state)

    runner.layer_caches = [SimpleNamespace(conv=object(), kpool_tail=None)]
    missing_indices = _aligned_dp_decode_metadata(8)
    missing_indices.linear_state_indices = None
    missing_indices.raw_dp_execution_token_counts = (8, 0)
    with pytest.raises(RuntimeError, match="missing linear_state_indices"):
        runner.can_execute(torch.zeros(8, dtype=torch.int32), missing_indices)

    empty_runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=1,
        num_decoding_tokens=8,
    )
    empty_runner.layer_caches = [SimpleNamespace(conv=object(), kpool_tail=None)]
    placeholder = _aligned_dp_decode_metadata(1)
    placeholder.dp_execution_token_counts = (8, 1)
    placeholder.raw_dp_execution_token_counts = (8, 0)
    placeholder.linear_state_indices = None
    assert empty_runner.can_execute(torch.zeros(1, dtype=torch.int32), placeholder)


def test_dp_kpool_spans_keep_complete_groups_only() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=16,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=6,
        )
        for rank in range(2)
    ]
    for runner in runners:
        runner.layer_caches = [SimpleNamespace(kpool_tail=torch.zeros((1, 2, 1, 1), dtype=torch.bfloat16))]
    active = _dp_metadata((6, 6))
    active.is_spec_verify = True
    active.kpool_query_lens = (6,)
    empty = _dp_metadata((6, 6))
    empty.is_spec_verify = True

    with patch.object(DecodeAclGraphRunner, "_has_compatible_decode_metadata", return_value=True):
        assert runners[0].can_execute(torch.zeros(6, dtype=torch.int32), active)
        assert runners[1].can_execute(torch.zeros(6, dtype=torch.int32), empty)
    assert runners[0]._graph_kpool_query_lens(active, 6, 8, 6) == (6,)
    assert runners[1]._graph_kpool_query_lens(empty, 6, 8, 6) == (6,)
    assert runners[0]._graph_kpool_query_lens(empty, 8, 16, 8) == (8, 8)


def test_empty_rank_kpool_spans_follow_shared_verify_width() -> None:
    runner = _runner()
    runner.layer_caches = [SimpleNamespace(kpool_tail=torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16))]
    metadata = SimpleNamespace()

    busy = runner._padded_kpool_query_lens(metadata, 8, 16, verify_width=8)
    empty = runner._padded_kpool_query_lens(metadata, 1, 16, verify_width=8)

    assert busy == (8, 8)
    assert empty == busy


def test_dp_expanded_verify_without_placeholder_folds_to_sequences() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=3,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=4,
        )
        for rank in range(2)
    ]
    metadata = _expanded_verify_metadata(8)
    metadata.dp_execution_token_counts = (8, 8)
    metadata.raw_dp_execution_token_counts = (8, 8)
    metadata.linear_state_indices = torch.tensor([0, 1], dtype=torch.int32)
    metadata.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        for runner in runners:
            assert runner.can_execute(torch.zeros(8, dtype=torch.int32), metadata)


def _expanded_verify_metadata(rows: int) -> SimpleNamespace:
    metadata = _dp_metadata((rows, rows))
    metadata.slot_mapping = torch.arange(rows, dtype=torch.int32)
    metadata.linear_state_indices = torch.tensor([0], dtype=torch.int32)
    metadata.expanded_decode_metadata = SimpleNamespace(
        enabled=True,
        kv_seq_lens=torch.ones(rows, dtype=torch.int32),
        block_table=torch.zeros((rows, 1), dtype=torch.int32),
        paged_kv_indptr=None,
        paged_kv_indices=None,
        paged_kv_last_page_len=None,
        paged_attention_tiling_data=None,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=None,
    )
    return metadata


@pytest.mark.parametrize(
    ("width", "disable_verify_graph"),
    [
        pytest.param(4, False, id="kda-verify-default"),
        pytest.param(4, True, id="verify-graph-disabled"),
        pytest.param(3, False, id="width-three-pads-inside-bucket"),
    ],
)
def test_dp_mtp_target_fallback_is_group_wide(
    width: int,
    disable_verify_graph: bool,
) -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=16,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=width,
        )
        for rank in range(2)
    ]
    metadata = _expanded_verify_metadata(width)
    metadata.is_spec_verify = True
    admitted = not disable_verify_graph

    with (
        patch.dict(
            os.environ,
            {"XLLM_NO_VERIFY_GRAPH": "1" if disable_verify_graph else "0"},
        ),
        patch.object(
            DecodeAclGraphRunner,
            "_has_compatible_decode_metadata",
            return_value=True,
        ),
    ):
        for runner in runners:
            assert runner.can_execute(torch.zeros(width, dtype=torch.int32), metadata) is admitted


def test_kda_verify_graph_is_enabled_by_default() -> None:
    runner = _runner()
    metadata = _metadata(torch.tensor([1, 3], dtype=torch.int32))
    metadata.q_cu_seq_lens = torch.arange(5, dtype=torch.int32)
    metadata.kv_seq_lens = torch.tensor([5, 6, 9, 10], dtype=torch.int32)
    with (
        patch.dict(os.environ, {"XLLM_NO_VERIFY_GRAPH": "0"}),
        patch.object(runner, "_has_compatible_decode_metadata", return_value=True),
    ):
        assert runner.can_execute(torch.zeros(4, dtype=torch.int32), metadata)


def test_dp_mixed_step_does_not_enter_acl_decode_graph() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
    )

    with patch.object(
        runner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert not runner.can_execute(
            torch.zeros(3, dtype=torch.int32),
            _dp_metadata((3, 2), dp_is_decode=(0, 1)),
        )
        missing_step_type = _dp_metadata((3, 2))
        del missing_step_type.dp_is_decode
        with pytest.raises(RuntimeError, match="requires dp_is_decode"):
            runner.can_execute(torch.zeros(3, dtype=torch.int32), missing_step_type)
        assert not runner.can_execute(
            torch.zeros(1, 3, dtype=torch.int32),
            _dp_metadata((3, 2), dp_is_decode=(0, 1)),
        )
        with pytest.raises(RuntimeError, match="must be one-dimensional"):
            runner.can_execute(
                torch.zeros(1, 8, dtype=torch.int32),
                _dp_metadata((8, 8)),
            )


def test_dp_acl_graph_requires_group_wide_token_counts() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
    )
    metadata = _dp_metadata((3, 2))
    metadata.dp_execution_token_counts = (3,)

    with (
        patch.object(
            runner,
            "_has_compatible_decode_metadata",
            return_value=True,
        ),
        pytest.raises(RuntimeError, match="valid dp_execution_token_counts"),
    ):
        runner.can_execute(torch.zeros(3, dtype=torch.int32), metadata)


def test_xfia_graph_accepts_device_only_lengths() -> None:
    runner = _runner()
    runner.attention_backend.requires_host_kv_lengths = False
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.kv_seq_lens_host_values = None
    assert runner._has_compatible_decode_metadata(torch.arange(4), metadata)
    table, lengths, host_lengths, *_ = runner._decode_metadata(metadata)
    assert table is metadata.block_table
    assert lengths is metadata.kv_seq_lens
    assert host_lengths is None
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=SimpleNamespace(kv_seq_lens_host_values=[1] * 4),
    )
    runner._fill_host_metadata(entry, None, batch_size=4)
    assert entry.static_metadata.kv_seq_lens_host_values == [1] * 4


# ---------------------------------------------------------------------------
# Engram graph-visible staging capacity hint
# ---------------------------------------------------------------------------


def test_acl_runner_pre_sizes_engram_graph_staging_capacity() -> None:
    """The runner tells the model its largest decode bucket before capture.

    Capture is lazy and buckets warm up ascending, so without this hint the
    engram graph-visible block would be fixed at the first (smallest) bucket.
    """
    from xllm.python.model_executor.runners.decode_cuda_graph import _decode_bucket

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hints: list[int] = []

        def configure_engram_graph_capacity(self, max_tokens: int) -> bool:
            self.hints.append(int(max_tokens))
            return True

    model = _Model()
    DecodeAclGraphRunner(
        model,
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=64,
        max_model_len=8,
        num_decoding_tokens=2,
    )
    # 64 sequences x 2 token rows, folded through the decode bucket.
    assert model.hints == [_decode_bucket(64 * 2)]
    assert model.hints == [128]

    # ``decode_batch_size_limit`` clamps the captured buckets, so the hint too.
    limited = _Model()
    DecodeAclGraphRunner(
        limited,
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=64,
        max_model_len=8,
        decode_batch_size_limit=32,
        num_decoding_tokens=1,
    )
    assert limited.hints == [_decode_bucket(32)]

    # A spec-draft runner emits up to ``draft_query_width`` query rows per
    # sequence, which can exceed ``num_decoding_tokens``; the graph-visible
    # engram block must cover that wider bucket, not the narrower MTP one.
    draft = _Model()
    DecodeAclGraphRunner(
        draft,
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=64,
        max_model_len=8,
        num_decoding_tokens=1,
        is_spec_draft=True,
        draft_query_width=8,
    )
    assert draft.hints == [_decode_bucket(64 * 8)]

    # A model without the hook is untouched.
    DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
    )


def test_acl_runner_warns_once_when_engram_staging_returns_false() -> None:
    """A False staging result on an offload-active model: capture warns, replay raises.

    The pre-graph staging hook's return value gates whether a replay re-reads
    the engram staging block. On a capture (``capturing=True``) this helper
    returns False and latches a single per-runner warning as a companion
    diagnostic; the runner's ``_engram_first_capture_gate`` then refuses the
    capture before it is cached (see the gate tests below), so no Engram-less
    graph is stored. On a replay (``capturing=False``) the graph was already
    captured *with* engram, so continuing would serve the previous step's stale
    rows as a silent wrong output (the eager path skips engram in the same
    condition); that is a fail-fast ``RuntimeError``.

    Discriminating: removing the capture warning makes ``len(warnings) == 1``
    fail; removing the replay raise makes ``pytest.raises`` fail; dropping the
    ``engram_offload_active`` gate makes the non-offload model warn/raise. The
    warning is asserted through the shared ``xllm`` logger (``scripts.logger``),
    not a per-module logger. That logger sets ``propagate = False``, so a
    root-attached ``caplog`` handler cannot see its records. Rather than flip the
    global ``propagate`` (which would leak into other tests, including ones
    running concurrently), the test attaches its own scoped handler straight to
    the ``xllm`` logger and removes it in ``finally``.
    """

    class _Model(nn.Module):
        def __init__(self, *, staged: bool, offload_active: bool) -> None:
            super().__init__()
            self._staged = staged
            self.engram_offload_active = offload_active
            self.layers: list = []
            self.calls = 0

        def stage_engram_rows_outside_graph(self, *args: object) -> bool:
            self.calls += 1
            return self._staged

    def make_runner(model: nn.Module) -> DecodeAclGraphRunner:
        return DecodeAclGraphRunner(
            model,
            SimpleNamespace(page_size=4, is_mla=False),
            torch.device("cpu"),
            max_batch=8,
            max_model_len=8,
        )

    # Scoped handler on the shared ``xllm`` logger: collect WARNINGs without
    # touching the logger's global ``propagate`` (see the docstring).
    xllm_logger = logging.getLogger("xllm")
    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _Capture(level=logging.WARNING)
    xllm_logger.addHandler(handler)
    try:
        # Capture: a not-staged offload model warns exactly once and never
        # raises, even though a capture walks several buckets (the latch holds).
        failing = make_runner(_Model(staged=False, offload_active=True))
        assert failing._stage_engram_rows_outside_graph(None, None, None, capturing=True) is False
        assert failing._stage_engram_rows_outside_graph(None, None, None, capturing=True) is False
        assert failing._engram_stage_skip_warned is True
        warnings = [record for record in captured if "staged nothing" in record.getMessage()]
        assert len(warnings) == 1, "a capture-stage failure must warn exactly once"
        assert "capturing=True" in warnings[0].getMessage()

        # Replay: the same condition is a hard error, not a stale-row replay, and
        # it must not also latch the capture warning.
        captured.clear()
        replay = make_runner(_Model(staged=False, offload_active=True))
        with pytest.raises(RuntimeError, match="previous step's rows"):
            replay._stage_engram_rows_outside_graph(None, None, None, capturing=False)
        assert not [record for record in captured if "staged nothing" in record.getMessage()]

        # Legitimate no-ops: a successful stage and a non-offload model stay
        # silent (neither warning nor raise) on both capture and replay.
        captured.clear()
        for model in (_Model(staged=True, offload_active=True), _Model(staged=False, offload_active=False)):
            runner = make_runner(model)
            assert runner._stage_engram_rows_outside_graph(None, None, None, capturing=True) == model._staged
            assert runner._stage_engram_rows_outside_graph(None, None, None, capturing=False) == model._staged
        assert not [record for record in captured if "staged nothing" in record.getMessage()]
    finally:
        xllm_logger.removeHandler(handler)


def _engram_gate_runner(*, offload_active: bool) -> DecodeAclGraphRunner:
    """A CPU ACL runner whose model toggles ``engram_offload_active``."""

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.engram_offload_active = offload_active

    return DecodeAclGraphRunner(
        _Model(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
    )


def test_acl_runner_engram_first_capture_gate_refuses_when_not_staged() -> None:
    """A first-capture staging failure must fail fast before anything is cached.

    The gate runs before ``_capture`` and before the entry is registered, so a
    ``False`` stage on an offload-active model must raise and leave ``_graphs``
    untouched (there is no entry to drop). Discriminating: deleting the raise
    lets the Engram-less graph be captured and cached.
    """

    runner = _engram_gate_runner(offload_active=True)

    with pytest.raises(RuntimeError) as excinfo:
        runner._engram_first_capture_gate(staged=False)

    assert runner._graphs == {}, "the gate refuses before anything is cached"
    message = str(excinfo.value)
    assert "first ACL graph capture" in message
    assert "Engram-less" in message
    assert "engram_config.cpu_offload=false" in message
    # The hint must name the copy-pasteable CLI form, and that exact JSON must
    # parse with both switches off (the HBM path the message points at).
    example = '{"cpu_offload":false,"dp_shared_memory":false}'
    assert f"--engram-config '{example}'" in message
    assert EngramConfig.from_value(example) == EngramConfig(cpu_offload=False, dp_shared_memory=False)


def test_acl_runner_engram_first_capture_gate_passes_when_staged() -> None:
    """A first capture that staged engram is allowed through unchanged.

    ``staged=True`` short-circuits the gate, so a healthy capture proceeds and
    the gate is a pure pass-through (no raise, no cache mutation).
    """

    runner = _engram_gate_runner(offload_active=True)
    runner._engram_first_capture_gate(staged=True)
    assert runner._graphs == {}


def test_acl_runner_engram_first_capture_gate_passes_when_offload_inactive() -> None:
    """A non-offload model's ``False`` hook must not trip the gate.

    Non-engram / non-offload models legitimately return ``False`` from the
    staging hook (it is a no-op), so gating only on ``staged`` would raise for
    them; the ``engram_offload_active`` guard keeps them silent.
    """

    runner = _engram_gate_runner(offload_active=False)
    runner._engram_first_capture_gate(staged=False)
    assert runner._graphs == {}


class _StoppedAfterCaptureError(Exception):
    """Sentinel raised at the replay boundary, after a successful capture."""


class _StopAtReplayStream:
    """``self._stream`` stand-in that stops ``execute`` at the replay.

    ``execute`` only reaches ``self._stream.wait_stream(...)`` after the first
    capture has completed and been registered, so raising from ``wait_stream``
    keeps the test device-free while still exercising the capture-then-register
    ordering.
    """

    def wait_stream(self, *args: object, **kwargs: object) -> None:
        raise _StoppedAfterCaptureError


def _engram_execute_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    staged: bool,
    capture_error: Exception | None = None,
) -> tuple[DecodeAclGraphRunner, Any, list[Any], Any]:
    """Wire a runner for the real ``DecodeAclGraphRunner.execute`` path.

    Only ``_allocate_entry`` / ``_fill_entry`` / ``_capture`` and the backend
    ``prepare`` are stubbed; ``execute`` itself -- the gate call, the capture
    call and the registration that follows it -- runs for real. ``_stream``
    raises :class:`_StoppedAfterCaptureError` from ``wait_stream`` so a
    successful capture stops right after registration, before the replay block
    (no device needed); when ``capture_error`` is set the stubbed ``_capture``
    raises it instead. ``torch.npu`` is stubbed to the single entry point used
    on that path so the test does not require an NPU.
    """

    class _EngramModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.engram_offload_active = True
            self.layers: list = []
            self.staged = staged
            self.stage_calls = 0

        def stage_engram_rows_outside_graph(self, *args: object) -> bool:
            self.stage_calls += 1
            return self.staged

    class _KeyRecordingDict(dict):
        def __init__(self) -> None:
            super().__init__()
            self.keys_written: list[Any] = []

        def __setitem__(self, key: Any, value: Any) -> None:
            self.keys_written.append(key)
            super().__setitem__(key, value)

    model = _EngramModel()
    backend = SimpleNamespace(page_size=4, is_mla=False, prepare=lambda *args, **kwargs: None)
    runner = DecodeAclGraphRunner(model, backend, torch.device("cpu"), max_batch=8, max_model_len=8)
    runner._stream = _StopAtReplayStream()
    graphs = _KeyRecordingDict()
    runner._graphs = graphs

    def fake_allocate_entry(padded_batch_size, input_ids, positions, metadata, input_batch=None, verify_width=1):
        return SimpleNamespace(
            static_metadata=SimpleNamespace(),
            static_input_ids=input_ids,
            static_positions=positions,
            execution_state=object(),
            execution_contexts={},
            eplb=None,
            graph=None,
        )

    capture_calls: list[Any] = []

    def fake_capture(entry):
        capture_calls.append(entry)
        if capture_error is not None:
            raise capture_error

    monkeypatch.setattr(runner, "_allocate_entry", fake_allocate_entry)
    monkeypatch.setattr(runner, "_fill_entry", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_capture", fake_capture)
    # The only ``torch.npu`` call reached on this path is the replay-boundary
    # ``current_stream`` lookup, right before ``_stream.wait_stream`` raises.
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda: None), raising=False)
    return runner, model, capture_calls, graphs


def _engram_execute_inputs() -> tuple[torch.Tensor, torch.Tensor, Any]:
    input_ids = torch.arange(4, dtype=torch.int64)
    positions = torch.arange(4, dtype=torch.int64)
    metadata = SimpleNamespace(linear_state_indices=torch.arange(4, dtype=torch.int32))
    return input_ids, positions, metadata


def test_acl_runner_execute_refuses_first_capture_when_engram_staging_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """execute() wiring: the gate refuses the first capture before it is cached.

    Drives the real ``execute`` path with a ``False`` stage. Deleting the
    ``_engram_first_capture_gate`` call turns this red: ``_capture`` would run
    and the replay-boundary sentinel would surface instead of the expected
    ``RuntimeError``. The entry must never be written to ``_graphs``.
    """

    runner, model, capture_calls, graphs = _engram_execute_harness(monkeypatch, staged=False)
    input_ids, positions, metadata = _engram_execute_inputs()

    with pytest.raises(RuntimeError) as excinfo:
        runner.execute(input_ids, positions, metadata)

    assert "first ACL graph capture" in str(excinfo.value)
    assert model.stage_calls == 1
    assert capture_calls == [], "_capture must not run when the gate refuses"
    assert graphs.keys_written == [], "a refused capture must not register anything"
    assert runner._graphs == {}


def test_acl_runner_execute_registers_only_after_successful_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """execute() registers the entry only once ``_capture`` has succeeded.

    With staging recovered the first step captures, then registers the entry, so
    the graph is available for replay. Registering *before* ``_capture`` would
    cache a half-built entry whenever the capture fails.
    """

    runner, model, capture_calls, graphs = _engram_execute_harness(monkeypatch, staged=True)
    input_ids, positions, metadata = _engram_execute_inputs()

    with pytest.raises(_StoppedAfterCaptureError):
        runner.execute(input_ids, positions, metadata)

    assert len(capture_calls) == 1
    assert len(graphs.keys_written) == 1
    graph_key = graphs.keys_written[0]
    assert runner._graphs.get(graph_key) is capture_calls[0]
    assert len(runner._graphs) == 1


def test_acl_runner_execute_leaves_no_entry_when_capture_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``_capture`` failure must not leave a ``graph is None`` entry cached.

    Staging succeeds but ``_capture`` raises; because the entry is registered
    only after ``_capture`` returns, nothing is cached and the next step can
    re-capture. Registering before the capture would cache the partial entry
    (``entry.graph is None``) and the next step's replay branch would dereference
    ``entry.graph`` -> ``AttributeError``.
    """

    class _CaptureFailedError(Exception):
        """Sentinel raised by the stubbed ``_capture``."""

    runner, model, capture_calls, graphs = _engram_execute_harness(
        monkeypatch,
        staged=True,
        capture_error=_CaptureFailedError(),
    )
    input_ids, positions, metadata = _engram_execute_inputs()

    with pytest.raises(_CaptureFailedError):
        runner.execute(input_ids, positions, metadata)

    assert len(capture_calls) == 1
    assert graphs.keys_written == [], "a failed capture must not register anything"
    assert runner._graphs == {}


def test_acl_runner_execute_recaptures_after_engram_staging_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """execute() recovery: once staging succeeds the bucket is re-captured.

    The failing first step must register nothing, so the next step takes the
    first-capture path again (a fresh staging attempt) and captures the
    Engram-baked graph. Registering before the capture would leave the refused
    entry cached, so the second step would take the replay branch and never
    re-capture.
    """

    runner, model, capture_calls, graphs = _engram_execute_harness(monkeypatch, staged=False)
    input_ids, positions, metadata = _engram_execute_inputs()

    with pytest.raises(RuntimeError):
        runner.execute(input_ids, positions, metadata)
    assert capture_calls == []
    assert graphs.keys_written == []

    model.staged = True
    with pytest.raises(_StoppedAfterCaptureError):
        runner.execute(input_ids, positions, metadata)

    assert len(capture_calls) == 1
    graph_key = graphs.keys_written[0]
    assert graph_key in runner._graphs
    assert len(runner._graphs) == 1
    assert graphs.keys_written == [graph_key]

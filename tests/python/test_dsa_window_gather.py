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

"""Unit tests for the SWA window gather ring addressing.

Regression tests for the placeholder-stable block-table indexing: the SWA
manager releases slid-out leading blocks but keeps them in the table as
invalid (-1) placeholders so ``(pos // block_size) % table_width`` keeps
resolving each position to its owning column. The window gather must use
that full-width modulo; ranking among valid columns instead shifts the
mapping once the first block is released
(cached_tokens >= window + num_spec_tokens + block_size) and the window
reads the wrong physical blocks.

Pure-Python: does not load compiled operators.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from xllm.python.attention.dsa_attention import DsaAttentionBackend
from xllm.python.attention.dsa_metadata import build_cache_specs

BLOCK_SIZE = 128
WINDOW = 128


def _make_backend(dspark_block_size: int = 0) -> DsaAttentionBackend:
    compress_ratios = [1]
    build_cache_specs(compress_ratios, WINDOW, 1)
    return DsaAttentionBackend(
        compress_ratios=compress_ratios,
        window_size=WINDOW,
        n_layers=1,
        num_heads=8,
        attn_head_dim=512,
        index_topk=512,
        index_n_heads=64,
        index_head_dim=128,
        rope_head_dim=64,
        dspark_block_size=dspark_block_size,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )


def _expected_slot(block_table_row: torch.Tensor, pos: int) -> int:
    width = block_table_row.numel()
    block_id = int(block_table_row[(pos // BLOCK_SIZE) % width])
    assert block_id >= 0, "in-window position must not hit a placeholder"
    return block_id * BLOCK_SIZE + pos % BLOCK_SIZE


def _window_src(backend: DsaAttentionBackend, block_table: torch.Tensor,
                kv_len: int) -> torch.Tensor:
    """Invoke _window_src_for_group with a one-sequence table."""
    dsa_metadata = SimpleNamespace(block_tables=[[block_table]])
    mapping = backend._resolve_cache_mapping(0, 1)
    kv = torch.tensor([kv_len], dtype=torch.int64)
    k = kv.clamp(max=backend.window_size)
    token = kv.unsqueeze(1) - k.unsqueeze(1) + torch.arange(
        backend.window_size).unsqueeze(0)
    valid = torch.arange(backend.window_size).unsqueeze(0) < k.unsqueeze(1)
    return backend._window_src_for_group(
        dsa_metadata, 0, mapping.ori_cache_idx, token, valid, n_seqs=1
    )


def test_window_src_stable_after_first_block_release() -> None:
    """cached=260: block 0 slid out and stays as a -1 placeholder."""
    backend = _make_backend()
    # Columns: released block (positions 0-127), live block A (128-255),
    # live block B (256-383).
    block_table = torch.tensor([[-1, 10, 20]], dtype=torch.int64)
    src = _window_src(backend, block_table, kv_len=260)

    row = block_table[0]
    for j in range(backend.window_size):
        pos = 260 - 128 + j
        assert int(src[0, j]) == _expected_slot(row, pos), (
            f"window slot {j} (pos {pos}) misrouted: "
            f"{int(src[0, j])} != {_expected_slot(row, pos)}"
        )
    # The pre-release mapping (no placeholders) must agree on the overlap.
    clean_table = torch.tensor([[9, 10, 20]], dtype=torch.int64)
    clean_src = _window_src(backend, clean_table, kv_len=260)
    assert torch.equal(src[:, 132 - 132:], clean_src[:, 132 - 132:])


def test_window_src_stable_after_multiple_block_releases() -> None:
    """cached=400: blocks 0 and 1 slid out; only one release parity left."""
    backend = _make_backend()
    # Columns: two released blocks, live block C (256-383), live block D
    # (384-511).
    block_table = torch.tensor([[-1, -1, 30, 40]], dtype=torch.int64)
    src = _window_src(backend, block_table, kv_len=400)

    row = block_table[0]
    for j in range(backend.window_size):
        pos = 400 - 128 + j
        assert int(src[0, j]) == _expected_slot(row, pos), (
            f"window slot {j} (pos {pos}) misrouted: "
            f"{int(src[0, j])} != {_expected_slot(row, pos)}"
        )


def test_decode_window_compact_eager_after_block_release() -> None:
    """The eager compact gathers the newest window through the ring."""
    backend = _make_backend()
    block_table = torch.tensor([[-1, 10, 20]], dtype=torch.int32)
    # Cache where every slot row encodes its own slot id.
    num_slots = 32 * BLOCK_SIZE
    head_dim = 8
    ori_kv = torch.zeros(num_slots, 1, head_dim, dtype=torch.float32)
    ori_kv[:, 0, 0] = torch.arange(num_slots, dtype=torch.float32)
    seq_kv = torch.tensor([260], dtype=torch.int32)

    compact_kv, _bt, seqused = backend._decode_window_compact_eager(
        ori_kv, block_table, seq_kv, layer_id=0
    )
    assert int(seqused[0]) == WINDOW
    row = block_table[0].to(torch.int64)
    for j in range(WINDOW):
        pos = 260 - WINDOW + j
        expected = _expected_slot(row, pos)
        assert float(compact_kv[0, j, 0, 0]) == float(expected), (
            f"compact[{j}] (pos {pos}) holds slot "
            f"{float(compact_kv[0, j, 0, 0])}, expected {expected}"
        )


def test_dspark_draft_window_widening() -> None:
    """DSpark draft rows attend window + block tokens (reference SAS)."""
    plain = _make_backend()
    assert plain.attn_win_left == WINDOW - 1
    assert plain.attn_win_capacity == WINDOW

    draft = _make_backend(dspark_block_size=5)
    assert draft.attn_win_left == WINDOW + 5 - 1
    assert draft.attn_win_capacity == WINDOW + 5

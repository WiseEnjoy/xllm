# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Env-gated decode-step segment timing for latency profiling.

Enable with XLLM_STEP_TIMER=<interval>, where <interval> is the number of
finished steps between summary log lines (for example 200). An unset
variable or a value below 1 disables the timer; the disabled path costs a
cached-None lookup per phase.

A speculative-decoding step alternates two Python executor calls on the
compute stream: the draft model forward (eager) and the target verify
forward (ACL graph replay). NPU events recorded after each enqueue split
the device timeline of a step into two slots:

  draft_slot  = previous replay end -> draft enqueue end: the previous
                sampler tail, device idle bubbles, and the draft execution
  glue_slot   = draft enqueue end -> verify fill enqueue start: the C++
                orchestration host gap the device stream waits through
  replay_slot = verify fill enqueue start -> replay completion: the graph
                input fill H2D, the refresh host gap, and the verify
                replay execution

Host phases measure the serial Python time inside the calls:
draft_host (with its metadata-prepare split eager_prepare_host), fill_host,
refresh_host, replay_host for graph verify steps, and target_eager_host for
target prefill or eager-fallback forwards. A step contributes to the slot
averages only when its event sequence is exactly draft -> replay without a
graph capture or an interleaved target eager forward. Slot events are
processed one step later, after the worker's per-step stream
synchronization has certainly completed them, so the timer never blocks
the host on the compute stream.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

import torch

PHASE_DRAFT_HOST = "draft_host"
PHASE_EAGER_PREPARE_HOST = "eager_prepare_host"
PHASE_TARGET_EAGER_HOST = "target_eager_host"
PHASE_FILL_HOST = "fill_host"
PHASE_REFRESH_HOST = "refresh_host"
PHASE_REPLAY_HOST = "replay_host"

EVENT_DRAFT = "draft"
EVENT_VERIFY_FILL = "verify_fill"
EVENT_REPLAY = "replay"
EVENT_TARGET_EAGER = "target_eager"

_SLOT_DRAFT_MS = "draft_slot_ms"
_SLOT_GLUE_MS = "glue_slot_ms"
_SLOT_REPLAY_MS = "replay_slot_ms"
_CLEAN_SEQUENCE = [EVENT_DRAFT, EVENT_VERIFY_FILL, EVENT_REPLAY]


class _PendingStep:
    """Events parked until the next step, when they are surely complete."""

    def __init__(self, events: list[tuple[str, torch.npu.Event]],
                 captured: bool) -> None:
        self.events = events
        self.captured = captured


class _StepTimer:
    """Accumulates per-step host phases and device slot statistics."""

    def __init__(self, interval: int) -> None:
        self.interval = interval
        self._events: list[tuple[str, torch.npu.Event]] = []
        self._captured_step = False
        self._pending: _PendingStep | None = None
        self._prev_replay: torch.npu.Event | None = None
        self._step_wall_start = 0.0
        self._interval_steps = 0
        self._interval_wall_ms = 0.0
        self._interval_skipped = 0
        self._phase_ms: dict[str, float] = defaultdict(float)
        self._slots_ms: dict[str, list[float]] = defaultdict(list)

    def add_phase(self, name: str, elapsed_ms: float) -> None:
        self._phase_ms[name] += elapsed_ms

    def mark_event(self, name: str) -> None:
        event = torch.npu.Event(enable_timing=True)
        event.record(torch.npu.current_stream())
        self._events.append((name, event))

    def note_capture(self) -> None:
        self._captured_step = True

    def finish_step(self) -> None:
        self._process_pending()
        now = time.perf_counter()
        if self._step_wall_start > 0.0:
            self._interval_wall_ms += (now - self._step_wall_start) * 1e3
            self._interval_steps += 1
        self._step_wall_start = now
        self._pending = _PendingStep(self._events, self._captured_step)
        self._events = []
        self._captured_step = False
        if self._interval_steps >= self.interval:
            self._log_summary()

    def _process_pending(self) -> None:
        pending = self._pending
        self._pending = None
        if pending is None:
            return
        events = pending.events
        names = [name for name, _ in events]
        replay_event = next(
            (event for name, event in reversed(events)
             if name == EVENT_REPLAY), None)
        # Chain anchor from the previous step; register this step's replay
        # (clean or not) so the next step's draft slot stays unbroken.
        prev_replay = self._prev_replay
        if replay_event is not None:
            self._prev_replay = replay_event
        clean_step = (not pending.captured
                      and names == _CLEAN_SEQUENCE)
        if not clean_step:
            self._interval_skipped += 1
            return
        draft_event = events[0][1]
        fill_event = events[1][1]
        if not (draft_event.query() and fill_event.query()
                and replay_event is not None and replay_event.query()):
            self._interval_skipped += 1
            return
        if prev_replay is not None and prev_replay.query():
            self._slots_ms[_SLOT_DRAFT_MS].append(
                prev_replay.elapsed_time(draft_event))
        # glue slot: draft enqueue completion -> target fill enqueue start
        # (C++ orchestration host gap the device stream waits through);
        # replay slot: fill start -> replay completion (fill H2D + refresh
        # host gap + verify replay execution).
        self._slots_ms[_SLOT_GLUE_MS].append(
            draft_event.elapsed_time(fill_event))
        self._slots_ms[_SLOT_REPLAY_MS].append(
            fill_event.elapsed_time(replay_event))

    def _log_summary(self) -> None:
        if self._interval_steps == 0:
            return
        from scripts.logger import logger

        steps = self._interval_steps
        phases = " ".join(
            f"{name}={total / steps:.2f}ms"
            for name, total in sorted(self._phase_ms.items()))

        def slot_summary(name: str, values: list[float]) -> str:
            if not values:
                return f"{name}=n/a"
            return f"{name}={sum(values) / len(values):.2f}/{max(values):.2f}ms"

        slots = " ".join(
            slot_summary(name, values)
            for name, values in sorted(self._slots_ms.items()))
        logger.info(
            f"step_timer steps={steps} "
            f"wall={self._interval_wall_ms / steps:.2f}ms "
            f"skipped={self._interval_skipped} | {phases} | {slots}")
        self._interval_steps = 0
        self._interval_wall_ms = 0.0
        self._interval_skipped = 0
        self._phase_ms = defaultdict(float)
        self._slots_ms = defaultdict(list)


_TIMER: _StepTimer | None = None
_INITIALIZED = False


def _get_timer() -> _StepTimer | None:
    global _TIMER, _INITIALIZED
    if not _INITIALIZED:
        _INITIALIZED = True
        raw_value = os.environ.get("XLLM_STEP_TIMER", "")
        try:
            interval = int(raw_value) if raw_value else 0
        except ValueError:
            interval = 0
        if interval >= 1:
            _TIMER = _StepTimer(interval)
    return _TIMER


@contextmanager
def host_phase(name: str) -> Iterator[None]:
    """Time a host block; pass-through when the timer is disabled."""
    timer = _get_timer()
    if timer is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        timer.add_phase(name, (time.perf_counter() - start) * 1e3)


def mark_event(name: str) -> None:
    """Record an NPU event on the current stream; no-op when disabled."""
    timer = _get_timer()
    if timer is not None:
        timer.mark_event(name)


def note_capture() -> None:
    """Flag the current step as a graph-capture step (excluded)."""
    timer = _get_timer()
    if timer is not None:
        timer.note_capture()


def finish_step() -> None:
    """Close the current decode step and maybe log the interval summary."""
    timer = _get_timer()
    if timer is not None:
        timer.finish_step()

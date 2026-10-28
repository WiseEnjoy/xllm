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

"""CPU-side tests for the PyPTO-Pro mhcpresinkhorn_rmsnorm host wrapper.

The kernel itself needs ``torch_npu`` + ``pypto_pro`` (see
``test_mhcpresinkhorn_rmsnorm_npu.py``).  This module only covers the parts
that must hold on any machine:

* the tiling / layout constants stay mutually consistent,
* every argument-contract violation raises ``ValueError`` before any device
  work is enqueued,
* the host-side workspace cache keys, reuses and retires entries correctly,
* the kernel is launched with the documented buffer contract.

``torch_npu``, ``pypto_pro`` and the tensors are replaced by lightweight
stubs so the module can be imported without an Ascend device.  A real CPU-only
torch build cannot allocate ``npu`` storage, and ``Tensor.device`` is a
read-only C descriptor, so tensors are represented by ``_StubTensor``: the
host wrapper only reads ``dtype``, ``shape``, ``dim()``, ``device``,
``is_contiguous()``, ``data_ptr()``, ``view()`` and ``reshape()``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

_MODULE_NAME = "xllm.python.kernels_npu.pypto_pro_mhcpresinkhorn_rmsnorm"
_SOURCE = (Path(__file__).parents[2] / "xllm" / "python" / "kernels_npu"
           / "pypto-pro" / "mhcpresinkhorn_rmsnorm.py")

# ``torch.device("npu:0")`` is only parsed once the privateuse1 backend is
# renamed; ``torch_npu`` does that when it is imported for real.
torch.utils.backend_registration._rename_privateuse1_backend("npu")


class _StubTensor:
    """Stand-in for a device tensor: shape/dtype/device/contiguity only."""

    _next_ptr = 1

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype,
                 device: torch.device, *, contiguous: bool = True,
                 view_of: "_StubTensor | None" = None) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self._contiguous = contiguous
        # Views share the storage (and therefore data_ptr) of their base.
        self._view_of = view_of
        if view_of is None:
            self._ptr = _StubTensor._next_ptr
            _StubTensor._next_ptr += 1
        else:
            self._ptr = view_of.data_ptr()

    # ---- the surface the host wrapper touches ----
    def dim(self) -> int:
        return len(self.shape)

    def is_contiguous(self) -> bool:
        return self._contiguous

    def data_ptr(self) -> int:
        return self._ptr

    def view(self, *shape: int) -> "_StubTensor":
        return _StubTensor(shape, self.dtype, self.device,
                           contiguous=self._contiguous, view_of=self)

    def reshape(self, *shape: int) -> "_StubTensor":
        return _StubTensor(shape, self.dtype, self.device,
                           contiguous=self._contiguous, view_of=self)

    def t(self) -> "_StubTensor":
        return _StubTensor(tuple(reversed(self.shape)), self.dtype, self.device,
                           contiguous=False, view_of=self)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_StubTensor({self.shape}, {self.dtype}, {self.device})"


def _tensor(shape: tuple[int, ...], dtype: torch.dtype,
            device: torch.device, *, contiguous: bool = True) -> _StubTensor:
    return _StubTensor(shape, dtype, device, contiguous=contiguous)


def _npu_device(index: int = 0) -> torch.device:
    return torch.device("npu", index)


class _FakeStream:
    def __init__(self, stream_id: int) -> None:
        self.stream_id = stream_id
        self.synchronized = 0

    def synchronize(self) -> None:
        self.synchronized += 1


class _FakeEvent:
    """Event stub whose ``query()`` is driven by the test."""

    def __init__(self, done: bool = True) -> None:
        self.done = done
        self.recorded = 0
        self.synchronized = 0
        self.raise_on_query = False
        self.raise_on_record = False

    def query(self) -> bool:
        if self.raise_on_query:
            raise RuntimeError("broken event")
        return self.done

    def record(self, stream: Any) -> None:
        if self.raise_on_record:
            raise RuntimeError("record failed")
        self.recorded += 1

    def synchronize(self) -> None:
        self.synchronized += 1
        self.done = True


class _FakeNpu:
    """Minimal ``torch.npu`` stand-in used by the host wrapper."""

    def __init__(self) -> None:
        self.current_device_id = 0
        self.streams: dict[int, _FakeStream] = {}
        self.synchronized: list[Any] = []
        self.next_event_done = True
        self.capturing = False
        self.allocations: list[tuple[tuple[int, ...], torch.dtype, Any]] = []

    def current_stream(self, device: torch.device | None = None) -> _FakeStream:
        # Mirror torch.npu: no device means "the current one".
        if device is None or device.index is None:
            index = self.current_device_id
        else:
            index = device.index
        return self.streams.setdefault(index, _FakeStream(index))

    def current_device(self) -> int:
        return self.current_device_id

    def is_current_stream_capturing(self) -> bool:
        return self.capturing

    def Event(self) -> _FakeEvent:  # noqa: N802 - mirrors the torch.npu API
        return _FakeEvent(done=self.next_event_done)

    def synchronize(self, device: Any = None) -> None:
        self.synchronized.append(device)

    def empty(self, shape: tuple[int, ...], dtype: torch.dtype,
              device: torch.device) -> _StubTensor:
        self.allocations.append((tuple(shape), dtype, device))
        return _tensor(tuple(shape), dtype, device)


class _FakeKernel:
    """Records kernel launches so the wrapper's call contract is checkable."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, tuple[Any, ...]]] = []
        self.grid: Any = None

    def __getitem__(self, grid: Any) -> "_FakeKernel":
        self.grid = grid
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((self.grid, args))


class _FakeAnnotation:
    """Anything ``pypto_pro.language`` exposes: subscriptable and callable."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __getitem__(self, item: Any) -> "_FakeAnnotation":
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # ``pl.jit(...)`` / ``pl.vector_function`` decorate a function.
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return _FakeAnnotation(self.name)

    def __getattr__(self, name: str) -> "_FakeAnnotation":
        if name.startswith("__"):
            raise AttributeError(name)
        return _FakeAnnotation(f"{self.name}.{name}")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_FakeAnnotation({self.name})"


class _FakeLanguage(types.ModuleType):
    """``pypto_pro.language`` stub: every attribute resolves to a fake."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return _FakeAnnotation(name)


@pytest.fixture()
def op(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import the operator module against stubbed ``torch_npu``/``pypto_pro``."""
    fake_npu = _FakeNpu()
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(torch, "empty", fake_npu.empty, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))

    language = _FakeLanguage("pypto_pro.language")
    language.Vf = _FakeLanguage("pypto_pro.language.Vf")
    pypto_pro = types.ModuleType("pypto_pro")
    pypto_pro.language = language
    monkeypatch.setitem(sys.modules, "pypto_pro", pypto_pro)
    monkeypatch.setitem(sys.modules, "pypto_pro.language", language)
    monkeypatch.setitem(sys.modules, "pypto_pro.language.Vf", language.Vf)

    # The source lives in a directory whose name ("pypto-pro") is not a valid
    # module name, so it is loaded straight from the file path.
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    module._WS_CACHE.clear()
    module._WS_RETIRED.clear()
    module._fake_npu = fake_npu
    module._fake_kernel = _FakeKernel()
    module.mhcpresinkhorn_rmsnorm_kernel_k = module._fake_kernel
    yield module
    sys.modules.pop(_MODULE_NAME, None)


def _valid_args(device: torch.device, t: int = 4) -> dict[str, Any]:
    return dict(
        x=_tensor((t, 4, 4096), torch.bfloat16, device),
        hc_fn=_tensor((24, 16384), torch.bfloat16, device),
        hc_scale=_tensor((3,), torch.float32, device),
        hc_base=_tensor((24,), torch.float32, device),
        gamma=_tensor((4096,), torch.bfloat16, device),
    )


def test_module_constants_are_mutually_consistent(op: Any) -> None:
    assert op.HC == 4
    assert op.D == 4096
    assert op.HC_D == op.HC * op.D == 16384
    assert op.MIX_HC == (2 + op.HC) * op.HC == 24
    assert op.SINKHORN_ITERS == 20
    assert op.EPS == 1e-6
    assert op.RMS_EPS == 1e-6
    assert op.INV_HC_D == 1.0 / op.HC_D
    assert op.INV_D == 1.0 / op.D
    assert op.NDC == op.D // op.DC == 8
    assert op.NDC * op.DC == op.D
    assert op.Y_SEGMENT == op.DC // 4 == 128
    assert op.SSQ_DK == op.HC_D // op.SSQ_SLICES == 512
    assert op.SSQ_SLICES * op.SSQ_DK == op.HC_D
    assert op.KLAST == op.HC_D - (op.NB - 1) * op.KA == 400
    assert (op.NB - 1) * op.KA + op.KLAST == op.HC_D
    assert op.KA % 16 == 0 and op.KLAST % 16 == 0
    assert op.MIXC >= op.MIX_HC
    # cbt holds the 4x4 comb matrix as 16 rows of SINK_ROW_W fp32 elements.
    assert op.SINK_ROW_W * op.HC * op.HC == 16 * op.SINK_ROW_W
    assert op.SINK_ROW_W >= op.HC


def test_ub_layout_does_not_overlap(op: Any) -> None:
    """Every UB buffer must start after the previous one ends."""
    layout = [
        ("XSQ_UB", op.TG * op.SSQ_DK * 2),
        ("SSQST_UB", op.TG * 64 * 4),
        ("SSQTT_UB", 64 * op.TG * 4),
        ("SLAB_UB", op.NB * op.MIXC * 4),
        ("MIXR_UB", 1 * op.MIXC * 4),
        ("SSQL_UB", op.SSQ_SLICES * op.TG * 4),
        ("SSQR_UB", 1 * op.TG * 4),
        ("SCALE3_UB", 1 * 8 * 4),
        ("SCALE24_UB", 1 * 32 * 4),
        ("PMUL24_UB", 1 * 32 * 4),
        ("EPS24_UB", 1 * 32 * 4),
        ("BASE_UB", 1 * 32 * 4),
        ("ZST_UB", 8 * op.MIX_HC * 4),
        ("ZSD_UB", op.MIX_HC * op.TG * 4),
        ("CBT_UB", 16 * 8 * 4),
        ("CBTT_UB", 8 * 16 * 4),
        ("PST_UB", 8 * 8 * 4),
        ("PSTT_UB", 8 * 8 * 4),
        ("PRE_UB", 1 * 8 * 4),
        ("XSUB_UB", 2 * op.HC * op.DC * 2),
        ("YUB_UB", 2 * 1 * op.DC * 2),
        ("YROW_UB", 1 * op.D * 2),
        ("GAMMA_UB", 1 * op.D * 2),
        ("RSTD_UB", 1 * 64 * 4),
    ]
    for (name, size), (next_name, _) in zip(layout, layout[1:]):
        assert getattr(op, name) + size <= getattr(op, next_name), \
            f"{name} overlaps {next_name}"
    # The whole UB span stays inside the 248 KiB vector buffer.
    last_name, last_size = layout[-1]
    assert getattr(op, last_name) + last_size <= 248 * 1024


def test_l1_layout_does_not_overlap(op: Any) -> None:
    """The two matmul operand shapes must not share L1 lines."""
    assert op.LHS_SRC_L1 + op.TG * op.KA * 2 <= op.LHS_SRC2_L1
    assert op.LHS_SRC2_L1 + op.TG * op.KLAST * 2 <= op.RHS_SRC_L1
    assert op.RHS_SRC_L1 + op.KA * op.MIXC * 2 <= op.RHS_SRC2_L1
    # L0B (both shapes) must fit the 64 KiB right matrix buffer.
    assert op.KA * op.MIXC * 2 + op.KLAST * op.MIXC * 2 <= 64 * 1024


def test_ws_key_uses_device_index_and_stream(op: Any) -> None:
    device = _npu_device(1)
    op._fake_npu.streams[1] = _FakeStream(7)
    assert op._ws_key(32, device) == (32, 1, 7)
    op._fake_npu.streams[1] = _FakeStream(9)
    assert op._ws_key(32, device) == (32, 1, 9)


def test_ws_key_falls_back_to_current_device(op: Any) -> None:
    # ``torch.device("npu")`` carries no index, so the key must ask the device.
    op._fake_npu.current_device_id = 3
    op._fake_npu.streams[3] = _FakeStream(1)
    assert op._ws_key(16, torch.device("npu")) == (16, 3, 1)


def test_ws_key_prefers_explicit_device_index(op: Any) -> None:
    # An explicit index wins over the ambient current device.
    op._fake_npu.current_device_id = 3
    op._fake_npu.streams[1] = _FakeStream(5)
    assert op._ws_key(16, _npu_device(1)) == (16, 1, 5)


def test_get_workspaces_shapes_and_views(op: Any) -> None:
    device = _npu_device()
    event, (pmix3, pmix3f, ssq_ws) = op._get_workspaces(40, device)
    ng = (40 + op.TG - 1) // op.TG
    assert isinstance(event, _FakeEvent)
    assert tuple(pmix3.shape) == (op.NB, ng * op.TG, op.MIXC)
    assert tuple(pmix3f.shape) == (op.NB, ng * op.TG * op.MIXC)
    assert tuple(ssq_ws.shape) == (op.SSQ_SLICES, ng * op.TG)
    assert pmix3.dtype == ssq_ws.dtype == torch.float32
    # pmix3f is a flattened view of pmix3, i.e. the same storage.
    assert pmix3f._view_of is pmix3
    assert pmix3.device == device and ssq_ws.device == device
    # ssq_ws is allocated first, then pmix3; both use the rounded-up row count
    # that the kernel indexes with.
    assert op._fake_npu.allocations == [
        ((op.SSQ_SLICES, ng * op.TG), torch.float32, device),
        ((op.NB, ng * op.TG, op.MIXC), torch.float32, device),
    ]


def test_get_workspaces_reuses_completed_entry(op: Any) -> None:
    device = _npu_device()
    first = op._get_workspaces(16, device)
    op._mark_workspaces_done(16, device, first)
    op._fake_npu.next_event_done = True
    # A completed entry is handed back to the caller, not reallocated.
    assert op._get_workspaces(16, device) is first
    assert op._WS_CACHE == {}


def test_get_workspaces_skips_in_flight_entry(op: Any) -> None:
    device = _npu_device()
    first = op._get_workspaces(16, device)
    op._mark_workspaces_done(16, device, first)
    first[0].done = False
    second = op._get_workspaces(16, device)
    assert second is not first
    assert second[1][0].data_ptr() != first[1][0].data_ptr()


def test_mark_workspaces_done_records_and_publishes(op: Any) -> None:
    device = _npu_device()
    entry = op._get_workspaces(16, device)
    op._mark_workspaces_done(16, device, entry)
    assert entry[0].recorded == 1
    assert op._WS_CACHE[op._ws_key(16, device)] is entry


def test_mark_workspaces_done_retires_previous_entry(op: Any) -> None:
    device = _npu_device()
    first = op._get_workspaces(16, device)
    op._mark_workspaces_done(16, device, first)
    second = op._get_workspaces(32, device)
    # Same t/device/stream key: publishing the second entry evicts the first.
    op._mark_workspaces_done(16, device, second)
    assert op._WS_CACHE[op._ws_key(16, device)] is second
    # The evicted entry already finished, so it is dropped instead of parked.
    assert op._WS_RETIRED == []


def test_mark_workspaces_done_parks_in_flight_entries(op: Any) -> None:
    device = _npu_device()
    first = op._get_workspaces(16, device)
    op._mark_workspaces_done(16, device, first)
    first[0].done = False
    second = op._get_workspaces(16, device)
    op._mark_workspaces_done(16, device, second)
    assert op._WS_RETIRED == [first]


def test_mark_workspaces_done_bounds_cache_size(op: Any) -> None:
    device = _npu_device()
    for step in range(op._WS_CACHE_MAX + 2):
        t = (step + 1) * op.TG
        op._mark_workspaces_done(t, device, op._get_workspaces(t, device))
    assert len(op._WS_CACHE) == op._WS_CACHE_MAX
    # The oldest keys were evicted first.
    assert op._ws_key(op.TG, device) not in op._WS_CACHE
    assert op._ws_key(2 * op.TG, device) not in op._WS_CACHE


def test_reclaim_retired_drops_completed_and_parks_pending(op: Any) -> None:
    done = (_FakeEvent(done=True), (_tensor((1,), torch.float32, _npu_device()),) * 3)
    pending = (_FakeEvent(done=False),
               (_tensor((1,), torch.float32, _npu_device()),) * 3)
    op._reclaim_retired([done, pending])
    assert op._WS_RETIRED == [pending]


def test_reclaim_retired_synchronizes_broken_event(op: Any) -> None:
    broken = _FakeEvent(done=False)
    broken.raise_on_query = True
    entry = (broken, (_tensor((1,), torch.float32, _npu_device()),) * 3)
    op._reclaim_retired([entry])
    assert op._WS_RETIRED == []
    assert len(op._fake_npu.synchronized) == 1


def test_reclaim_retired_bounds_parked_entries(op: Any) -> None:
    pending = [(_FakeEvent(done=False),
                (_tensor((1,), torch.float32, _npu_device()),) * 3)
               for _ in range(op._WS_CACHE_MAX + 2)]
    op._reclaim_retired(pending)
    assert len(op._WS_RETIRED) == op._WS_CACHE_MAX
    # The two oldest entries were drained (synchronized) and dropped.
    assert pending[0][0].synchronized == 1
    assert pending[1][0].synchronized == 1
    assert pending[2][0].synchronized == 0


def test_mark_workspaces_done_synchronizes_when_record_fails(op: Any) -> None:
    device = _npu_device()
    entry = op._get_workspaces(16, device)
    entry[0].raise_on_record = True
    with pytest.raises(RuntimeError, match="record failed"):
        op._mark_workspaces_done(16, device, entry)
    assert op._fake_npu.current_stream(device).synchronized == 1
    assert op._WS_CACHE == {}


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"hc_mult": 2}, "hc_mult"),
        ({"sinkhorn_iters": 10}, "sinkhorn_iters"),
        ({"eps": 1e-5}, "eps"),
    ],
)
def test_rejects_unsupported_scalar_options(op: Any, kwargs: dict[str, Any],
                                            message: str) -> None:
    device = _npu_device()
    with pytest.raises(ValueError, match=message):
        op.mhcpresinkhorn_rmsnorm(**_valid_args(device), **kwargs)
    # Nothing was enqueued: no workspace was allocated nor kernel launched.
    assert op._fake_npu.allocations == []
    assert op._fake_kernel.calls == []


@pytest.mark.parametrize(
    "field, shape, dtype, message",
    [
        ("x", (4, 4, 4096), torch.float32, "x must be"),
        ("x", (4, 2, 4096), torch.bfloat16, "x must be"),
        ("x", (4, 4, 2048), torch.bfloat16, "x must be"),
        ("hc_fn", (24, 16384), torch.float32, "hc_fn must be"),
        ("hc_fn", (23, 16384), torch.bfloat16, "hc_fn must be"),
        ("hc_scale", (3,), torch.bfloat16, "hc_scale must be"),
        ("hc_scale", (4,), torch.float32, "hc_scale must be"),
        ("hc_base", (24,), torch.bfloat16, "hc_base must be"),
        ("hc_base", (23,), torch.float32, "hc_base must be"),
        ("gamma", (4096,), torch.float32, "gamma must be"),
        ("gamma", (2048,), torch.bfloat16, "gamma must be"),
    ],
)
def test_rejects_bad_tensor_contract(op: Any, field: str,
                                     shape: tuple[int, ...],
                                     dtype: torch.dtype, message: str) -> None:
    device = _npu_device()
    args = _valid_args(device)
    args[field] = _tensor(shape, dtype, device)
    with pytest.raises(ValueError, match=message):
        op.mhcpresinkhorn_rmsnorm(**args)
    assert op._fake_kernel.calls == []


def test_rejects_bad_x_rank(op: Any) -> None:
    device = _npu_device()
    args = _valid_args(device)
    args["x"] = _tensor((4, 16384), torch.bfloat16, device)
    with pytest.raises(ValueError, match="x must be"):
        op.mhcpresinkhorn_rmsnorm(**args)
    assert op._fake_kernel.calls == []


def test_rejects_non_contiguous_tensors(op: Any) -> None:
    device = _npu_device()
    # Shape/dtype stay valid so the contiguity check is what fires.
    for field in ("x", "hc_fn", "hc_scale", "hc_base", "gamma"):
        args = _valid_args(device)
        args[field] = _tensor(args[field].shape, args[field].dtype, device,
                              contiguous=False)
        with pytest.raises(ValueError, match=f"{field} must be contiguous"):
            op.mhcpresinkhorn_rmsnorm(**args)
    assert op._fake_kernel.calls == []


def test_rejects_cpu_input(op: Any) -> None:
    with pytest.raises(ValueError, match="must be on an npu device"):
        op.mhcpresinkhorn_rmsnorm(**_valid_args(torch.device("cpu")))
    assert op._fake_kernel.calls == []


def test_rejects_device_mismatch(op: Any) -> None:
    device = _npu_device()
    args = _valid_args(device)
    args["gamma"] = _tensor((4096,), torch.bfloat16, _npu_device(1))
    with pytest.raises(ValueError, match="gamma is on"):
        op.mhcpresinkhorn_rmsnorm(**args)
    assert op._fake_kernel.calls == []


def test_launches_kernel_with_expected_buffers(op: Any) -> None:
    device = _npu_device()
    args = _valid_args(device, t=20)
    y, rstd, post, comb = op.mhcpresinkhorn_rmsnorm(**args)

    assert (tuple(y.shape), y.dtype) == ((20, op.D), torch.bfloat16)
    assert (tuple(rstd.shape), rstd.dtype) == ((20, 1), torch.float32)
    assert (tuple(post.shape), post.dtype) == ((20, op.HC), torch.float32)
    assert (tuple(comb.shape), comb.dtype) == ((20, op.HC, op.HC), torch.float32)
    for tensor in (y, rstd, post, comb):
        assert tensor.device == device

    assert len(op._fake_kernel.calls) == 1
    grid, call_args = op._fake_kernel.calls[0]
    # One cube core per K slice.
    assert grid == (None, op.NB)
    assert len(call_args) == 13
    (x_2d, x, hc_fn, hc_scale, hc_base, gamma, out_y, out_rstd, out_post,
     comb16, pmix3, pmix3f, ssq_ws) = call_args
    assert x is args["x"]
    assert tuple(x_2d.shape) == (20, op.HC_D)
    assert hc_fn is args["hc_fn"]
    assert hc_scale.shape == (1, 3) and hc_base.shape == (1, op.MIX_HC)
    assert gamma.shape == (1, op.D)
    assert out_y is y and out_rstd is rstd and out_post is post
    assert tuple(comb16.shape) == (20, op.HC * op.HC)
    assert comb16.data_ptr() == comb.data_ptr()
    ng = (20 + op.TG - 1) // op.TG
    assert tuple(pmix3.shape) == (op.NB, ng * op.TG, op.MIXC)
    assert tuple(pmix3f.shape) == (op.NB, ng * op.TG * op.MIXC)
    assert tuple(ssq_ws.shape) == (op.SSQ_SLICES, ng * op.TG)
    # The workspace is published for reuse once the launch is recorded.
    assert op._WS_CACHE[op._ws_key(20, device)][0].recorded == 1


def test_marks_workspaces_done_when_kernel_raises(op: Any) -> None:
    device = _npu_device()

    class _BoomKernel(_FakeKernel):
        def __call__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("kernel failed")

    op.mhcpresinkhorn_rmsnorm_kernel_k = _BoomKernel()
    with pytest.raises(RuntimeError, match="kernel failed"):
        op.mhcpresinkhorn_rmsnorm(**_valid_args(device))
    # Scratch is retained even for a failed launch, so the partial work is
    # drained before the buffers can be handed out again.
    assert op._WS_CACHE[op._ws_key(4, device)][0].recorded == 1

def test_capture_does_not_query_or_publish_eager_workspace_events(op: Any) -> None:
    device = _npu_device()
    entry = op._get_workspaces(4, device)
    entry[0].raise_on_query = True
    op._WS_CACHE[op._ws_key(4, device)] = entry
    op._WS_RETIRED.append(entry)
    op._fake_npu.capturing = True

    for _ in range(16):
        op.mhcpresinkhorn_rmsnorm(**_valid_args(device))

    assert op._WS_CACHE[op._ws_key(4, device)] is entry
    assert op._WS_RETIRED == [entry]
    assert entry[0].recorded == entry[0].synchronized == 0
    assert op._fake_npu.synchronized == []
    assert len(op._fake_kernel.calls) == 16

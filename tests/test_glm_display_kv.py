"""The display-reservation KV span (``display_kv``, Mia's 0072 + 0087) on the CPU: the setting, the refusal while a
display is connected, how many latent planes the span takes, the span's mapping and unwinding (a fake for every DRM
and CUDA driver call), the planes carved from a span, the pool's row copies of spanned planes, and 0087's backend
switch (a fake dispram client). Ported from Mia's ``tools/display_kv_check.py`` (its CPU half); the GPU half is
``tests/cuda/test_glm_display_kv_gpu.py``."""

from __future__ import annotations

import inspect
import sys

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.glm5_next.cuda import display_kv as dk  # noqa: E402
from tensorfold.families.glm5_next.cuda import kv8  # noqa: E402
from tensorfold.families.glm5_next.cuda.pool import Arena, Plane  # noqa: E402

pytestmark = pytest.mark.torch

MIB = 2 ** 20
D, O = 1792 * MIB, 256 * MIB
BASE = 1 << 44


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in (dk.ENV, dk.CARD_ENV, dk.BACKEND_ENV):
        monkeypatch.delenv(name, raising=False)
    dk._OWNERS.clear()
    yield
    dk._OWNERS.clear()


# -- the setting ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw, want", [(None, 0), ("", 0), ("0", 0), ("16", 16), ("1792", 1792), ("2032", 2032),
                                       (" 256 ", 256)])
def test_setting(monkeypatch, raw, want):
    if raw is not None:
        monkeypatch.setenv(dk.ENV, raw)
    assert dk.mib() == want
    assert dk.credit() == want * MIB


@pytest.mark.parametrize("raw", ["2048", "17", "-16", "abc", "1.5"])
def test_setting_refused(monkeypatch, raw):
    monkeypatch.setenv(dk.ENV, raw)
    with pytest.raises(ValueError, match=dk.ENV):
        dk.mib()


# -- how many latent planes the span takes ----------------------------------------------------------------------------

def test_span_planes():
    assert dk.span_planes(17, 10 ** 9, 1792 * MIB) == 2
    assert dk.span_planes(17, 2 * 10 ** 9, 1792 * MIB) == 1
    assert dk.span_planes(4, 1792 * MIB // 2, 1792 * MIB) == 2          # exact fit
    with pytest.raises(ValueError, match="latent planes"):
        dk.span_planes(1, 10 ** 9, 1792 * MIB)


# -- connected outputs ------------------------------------------------------------------------------------------------

def _sysfs(root, states):
    for name, state in states.items():
        (root / name).mkdir()
        (root / name / "status").write_text(state + "\n")
    return str(root)


def test_connected_outputs(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = _sysfs(tmp_path / "a",
               {"card0-HDMI-A-1": "disconnected", "card0-Unknown-1": "disconnected", "card1-DP-1": "connected"})
    assert dk.connected_outputs(a, "card0") == []
    b = _sysfs(tmp_path / "b", {"card0-HDMI-A-1": "connected", "card0-DP-1": "disconnected"})
    assert dk.connected_outputs(b, "card0") == ["HDMI-A-1"]
    assert dk.connected_outputs(str(tmp_path / "missing"), "card0") == []    # no KMS connectors: headless


# -- mapping the span: every driver call through the seam, unwound on failure ----------------------------------------

class FakeSystem:
    fail = None             # (call, nth): the nth call of that name raises
    dumb_bytes = None       # None: what was asked
    delta = None            # None: the right one
    last = None

    def __init__(self):
        self.calls = []
        FakeSystem.last = self

    def _do(self, name, *args):
        self.calls.append((name, *args))
        if FakeSystem.fail and FakeSystem.fail == (name, sum(c[0] == name for c in self.calls)):
            raise OSError(f"{name} failed on purpose")

    def open_card(self):
        self._do("open_card")
        return 7

    def create_dumb(self, fd, size):
        self._do("create_dumb", size)
        return 3, size if FakeSystem.dumb_bytes is None else FakeSystem.dumb_bytes

    def map_offset(self, fd, handle):
        self._do("map_offset")
        return 1 << 40

    def reserve(self, size):
        self._do("reserve", size)
        return BASE

    def map_ordinary(self, address, size):
        self._do("map_ordinary", address, size)

    def map_display(self, address, size, fd, offset):
        self._do("map_display", address, size)

    def register(self, address, size, *, io_memory):
        self._do("register", address, size, io_memory)

    def device_pointer(self, address):
        self._do("device_pointer", address)
        if FakeSystem.delta is not None and address != BASE:
            return BASE + FakeSystem.delta
        return address

    def unregister(self, address):
        self._do("unregister", address)

    def unmap(self, address, size):
        self._do("unmap", address, size)

    def close_gem(self, fd, handle):
        self._do("close_gem")

    def close_fd(self, fd):
        self._do("close_fd")


def _names(system):
    return [c[0] for c in system.calls]


@pytest.fixture
def fake(monkeypatch, tmp_path):
    FakeSystem.fail = FakeSystem.dumb_bytes = FakeSystem.delta = FakeSystem.last = None
    monkeypatch.setattr(dk, "SYSTEM", FakeSystem)
    monkeypatch.setattr(dk, "SYSFS", _sysfs(tmp_path, {"card0-DP-2": "disconnected"}))
    yield FakeSystem
    FakeSystem.fail = FakeSystem.dumb_bytes = FakeSystem.delta = FakeSystem.last = None


def test_connected_display_refused_before_any_driver_call(fake, monkeypatch, tmp_path):
    root = tmp_path / "on"
    root.mkdir()
    monkeypatch.setattr(dk, "SYSFS", _sysfs(root, {"card0-DP-2": "connected"}))
    with pytest.raises(RuntimeError, match="card0-DP-2"):
        dk.map_span(O, D)
    assert fake.last is None


def test_span_layout(fake):
    span = dk.map_span(O, D)
    calls = fake.last.calls
    assert (span.pointer, span.size) == (BASE, O + D)
    assert span.__cuda_array_interface__ == {"shape": (O + D,), "typestr": "|u1", "data": (BASE, False), "version": 3}
    assert ("create_dumb", D) in calls and ("reserve", O + D) in calls
    assert ("map_ordinary", BASE, O) in calls
    assert ("map_display", BASE + O, D) in calls                         # right above the ordinary half
    assert ("register", BASE, O, False) in calls and ("register", BASE + O, D, True) in calls


def test_no_ordinary_half_when_the_planes_are_the_reservation_exactly(fake):
    span = dk.map_span(0, D)
    assert "map_ordinary" not in _names(fake.last) and span.size == D
    assert [c for c in fake.last.calls if c[0] == "register"] == [("register", BASE, D, True)]


STEPS = [("open_card", 1), ("create_dumb", 1), ("map_offset", 1), ("reserve", 1), ("map_ordinary", 1),
         ("map_display", 1), ("register", 1), ("register", 2), ("device_pointer", 1), ("device_pointer", 2)]


@pytest.mark.parametrize("step, nth", STEPS)
def test_a_failing_step_raises_and_unwinds(fake, step, nth):
    fake.fail = (step, nth)
    with pytest.raises(RuntimeError, match=dk.ENV):
        dk.map_span(O, D)
    s = fake.last
    names = _names(s)

    def succeeded(name):
        return name in names and name != step

    registered = [c[1] for c in s.calls if c[0] == "register"][:-1 if step == "register" else None]
    assert all(("unregister", a) in s.calls for a in registered)
    assert len(registered) == sum(c[0] == "unregister" for c in s.calls)
    assert (("unmap", BASE, O + D) in s.calls) == succeeded("reserve")
    assert ("close_gem" in names) == succeeded("create_dumb")
    assert ("close_fd" in names) == succeeded("open_card")


def test_short_display_reserve_refused(fake):
    fake.dumb_bytes = D - 4096
    with pytest.raises(RuntimeError, match="display reserve gave"):
        dk.map_span(O, D)
    assert "close_gem" in _names(fake.last) and "close_fd" in _names(fake.last)


def test_halves_apart_on_the_device_refused_and_unwound(fake):
    fake.delta = O + 4096
    with pytest.raises(RuntimeError, match="apart"):
        dk.map_span(O, D)
    assert ("unmap", BASE, O + D) in fake.last.calls and "close_fd" in _names(fake.last)


# -- latent planes carved from one span -------------------------------------------------------------------------------

@pytest.fixture
def carved(monkeypatch):
    spans = []

    def map_span(ordinary, display):
        spans.append((ordinary, display))
        return dk.Span(0, ordinary + display)

    monkeypatch.setattr(dk, "map_span", map_span)
    monkeypatch.setattr(dk, "as_tensor", lambda span, device: torch.full((span.size,), 7, dtype=torch.uint8))
    return spans


@pytest.mark.parametrize("kind", ["fp8", "bf16"])
def test_latent_planes(carved, kind):
    rows, width, display = 1 << 20, 512, 1792 * MIB
    want = kv8.zeros(rows, width, kind, "meta")
    plane_bytes = want.numel() * want.element_size()
    assert plane_bytes == rows * kv8.row_bytes(width, kind)
    planes = dk.latent_planes(17, rows, width, kind, "cpu", display)
    k = len(planes)
    assert k == dk.span_planes(17, plane_bytes, display)
    assert carved == [(k * plane_bytes - display, display)]             # the reservation is the span's top
    assert all(p.shape == want.shape and p.dtype == want.dtype for p in planes)
    assert all(int(p.count_nonzero()) == 0 for p in planes)
    base = planes[0].data_ptr()
    assert all(p.data_ptr() == base + i * plane_bytes for i, p in enumerate(planes))
    with pytest.raises(RuntimeError, match="already"):
        dk.latent_planes(17, rows, width, kind, "cpu", display)


def test_fp8_rows_hold_about_twice_the_tokens(carved):
    """The same reservation carries ~2x the tokens with FP8 latent rows (kv_lora + 16 B) as with BF16 (2 x kv_lora)."""
    rows, width = 1 << 16, 512
    fp8 = kv8.row_bytes(width, "fp8")
    bf16 = kv8.row_bytes(width, "bf16")
    assert (fp8, bf16) == (width + kv8.PAD, 2 * width)
    assert 1.9 < bf16 / fp8 < 2.0
    assert len(dk.latent_planes(78, rows, width, "fp8", "cpu", 256 * MIB)) == -(-256 * MIB // (rows * fp8))


# -- copies between pool rows of spanned planes (0072's pool.py hunk: serial lane) ------------------------------------

SPANNED_POOL = "spanned" in inspect.signature(Plane.__init__).parameters


@pytest.mark.skipif(not SPANNED_POOL, reason="pool.Plane(spanned=) is 0072's pool.py hook (serial lane), not landed")
@pytest.mark.parametrize("dtype, shape", [(torch.uint8, (64, 528)), (torch.bfloat16, (64, 512))])
@pytest.mark.parametrize("src, dst, n", [(0, 32, 16), (4, 10, 30), (30, 2, 25), (0, 0, 8), (60, 1, 4)])
def test_spanned_rows_copy_as_stock_by_a_kernel(monkeypatch, dtype, shape, src, dst, n):
    gen = torch.Generator().manual_seed(src * 97 + dst * 7 + n)
    base = (torch.randint(0, 255, shape, generator=gen, dtype=torch.uint8) if dtype == torch.uint8
            else torch.randn(shape, generator=gen).to(dtype))
    pooled = torch.randint(0, 255, (64 // 4 + 2, 144), generator=gen, dtype=torch.uint8)
    stock = Arena(64, [Plane(base.clone()), Plane(pooled.clone(), 4, 2)])
    spanned = Arena(64, [Plane(base.clone(), spanned=True), Plane(pooled.clone(), 4, 2, spanned=True)])
    kernel = torch.bitwise_or
    calls = []
    monkeypatch.setattr(torch, "bitwise_or", lambda *a, **k: calls.append(1) or kernel(*a, **k))
    stock.copy(src, dst, n)
    stock_calls = len(calls)
    spanned.copy(src, dst, n)
    assert all(torch.equal(a.tensor, b.tensor) for a, b in zip(stock.planes, spanned.planes))
    assert stock_calls == 0 and (len(calls) > 0) == (n > 0 and src != dst)


# -- patch 0087: the reservation from kindling's dispramd (TF_GLM_DISPLAY_KV_BACKEND=dispram) -------------------------

@pytest.mark.parametrize("raw, want", [(None, "drm"), ("drm", "drm"), ("dispram", "dispram"), (" DisPram ", "dispram")])
def test_backend(monkeypatch, raw, want):
    if raw is not None:
        monkeypatch.setenv(dk.BACKEND_ENV, raw)
    assert dk.backend() == want


def test_backend_refuses_anything_else(monkeypatch):
    monkeypatch.setenv(dk.BACKEND_ENV, "dumb")
    with pytest.raises(ValueError, match="drm or dispram"):
        dk.backend()


class FakeDispram:
    """kindling's client: available() is the daemon answering; map_glued(nbytes, device) -> (pointer, the ordinary
    bytes, the reservation bytes lent above them)."""

    def __init__(self, up=True, lend=None):
        self.up, self.lend, self.calls = up, lend, []

    def available(self):
        return self.up

    def map_glued(self, nbytes, index):
        self.calls.append((nbytes, index))
        return 0x7000_0000_0000, nbytes - self.lend, self.lend


@pytest.fixture
def dispram(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)

    def install(client):
        if client is None:
            monkeypatch.setitem(sys.modules, "dispram", None)          # import raises ImportError
        else:
            monkeypatch.setitem(sys.modules, "dispram", client)
        return client

    return install


def test_dispram_refusals(dispram):
    dispram(None)
    with pytest.raises(RuntimeError, match="not importable"):
        dk._map_span_dispram(64 * MIB, D)
    dispram(FakeDispram(up=False, lend=D))
    with pytest.raises(RuntimeError, match="does not answer"):
        dk._map_span_dispram(64 * MIB, D)
    dispram(FakeDispram(lend=D - 16 * MIB))
    with pytest.raises(RuntimeError, match="short of"):
        dk._map_span_dispram(64 * MIB, D)


def test_dispram_span_is_one_range_and_map_span_never_takes_drm(dispram, monkeypatch):
    client = dispram(FakeDispram(lend=D))
    span = dk._map_span_dispram(64 * MIB, D)
    assert span.size == 64 * MIB + D and client.calls == [(64 * MIB + D, 0)]
    monkeypatch.setenv(dk.BACKEND_ENV, "dispram")
    monkeypatch.setattr(dk, "connected_outputs", lambda sysfs, card: [])

    def no_drm():
        raise AssertionError("the DRM path ran")

    monkeypatch.setattr(dk, "SYSTEM", no_drm)
    client = dispram(FakeDispram(lend=D))
    span = dk.map_span(64 * MIB, D)
    assert span.size == 64 * MIB + D and len(client.calls) == 1
    monkeypatch.setattr(dk, "connected_outputs", lambda sysfs, card: ["HDMI-A-1"])
    with pytest.raises(RuntimeError, match="display connected"):
        dk.map_span(64 * MIB, D)

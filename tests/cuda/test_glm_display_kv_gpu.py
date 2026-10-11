"""The display-reservation KV span (``display_kv``) on a real GB10: latent planes carved from a span whose top is the
UEFI display reservation, written by the GPU and read back across both halves, and a row copy across the halves'
boundary by a kernel (CUDA refuses a memcpy that crosses the two registrations). Mia's ``tools/display_kv_check.py
--gpu``, ported.

Skips unless the hardware precondition holds: the DRM card (``TF_GLM_DISPLAY_KV_CARD``, default /dev/dri/card0)
exists and this user can open it read-write, none of its outputs has a display connected, and the driver hands out a
dumb buffer of the test's size (nvidia_drm loaded with modeset=1, the reservation not held by a running server).
With ``TF_GLM_DISPLAY_KV_BACKEND=dispram`` the span comes from kindling's dispramd instead, and the test skips unless
its client imports and the daemon answers. Run it under the GPU lock. The span is small (``TF_GLM_DISPLAY_KV_TEST_MIB``,
default 256) and lives until the process exits (the module maps one span a process)."""

from __future__ import annotations

import errno
import os

import pytest
import torch

from tensorfold.families.glm5_next.cuda import display_kv as dk
from tensorfold.families.glm5_next.cuda.pool import Arena, Plane

MIB = 2 ** 20
TEST_MIB = int(os.environ.get("TF_GLM_DISPLAY_KV_TEST_MIB", "256"))


def _precondition() -> str | None:
    """Why the real span cannot be mapped on this node, or None when it can."""
    if not torch.cuda.is_available():
        return "no CUDA device"
    try:
        backend = dk.backend()
    except ValueError as exc:
        return str(exc)
    card_path = os.environ.get(dk.CARD_ENV, "/dev/dri/card0")
    card = os.path.basename(card_path)
    if busy := dk.connected_outputs(dk.SYSFS, card):
        return f"{card} has a display connected ({', '.join(busy)}): its reservation is in use"
    if backend == "dispram":
        try:
            import dispram
        except ImportError:
            return "TF_GLM_DISPLAY_KV_BACKEND=dispram, but the dispram client is not importable"
        if not dispram.available():
            return "TF_GLM_DISPLAY_KV_BACKEND=dispram, but dispramd does not answer"
        return None
    if not os.path.exists(card_path):
        return f"{card_path} is absent"
    if not os.access(card_path, os.R_OK | os.W_OK):
        return f"{card_path} is not read-write for this user (needs the video group, or root in the container)"
    system = dk.System.__new__(dk.System)        # only the DRM calls: no CUDA context, no libcuda
    system.card = card_path
    try:
        fd = system.open_card()
    except OSError as exc:
        return f"open({card_path}) failed: {exc}"
    try:
        handle, got = system.create_dumb(fd, TEST_MIB * MIB)
        system.close_gem(fd, handle)
    except OSError as exc:
        why = {errno.ENOSYS: "nvidia_drm without modeset=1", errno.EOPNOTSUPP: "nvidia_drm without modeset=1",
               errno.ENOMEM: "the reservation is held (a server started with TF_GLM_DISPLAY_KV_MIB?)"}
        return f"DRM_IOCTL_MODE_CREATE_DUMB({TEST_MIB} MiB) failed: {exc} ({why.get(exc.errno, 'no dumb buffers')})"
    finally:
        os.close(fd)
    if got < TEST_MIB * MIB:
        return f"the display reserve gave {got} B, short of {TEST_MIB} MiB"
    return None


REASON = _precondition()
pytestmark = pytest.mark.skipif(REASON is not None, reason=f"display reservation precondition: {REASON}")


def _pattern(i: int, n: int) -> torch.Tensor:
    return (torch.arange(n, device="cuda", dtype=torch.int32) + 7 * i).remainder_(251).to(torch.uint8)


def _kernel_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    torch.bitwise_or(src.view(torch.uint8), 0, out=dst.view(torch.uint8))


def test_real_span_round_trip_and_copies_across_the_halves(monkeypatch):
    monkeypatch.setenv(dk.ENV, str(TEST_MIB))
    display = dk.credit()
    rows, width = 1 << 16, 512                   # 33 MiB FP8 planes: the span takes several
    planes = dk.latent_planes(78, rows, width, "fp8", "cuda", display)
    k = len(planes)
    flat = [p.view(-1) for p in planes]
    plane_bytes = flat[0].numel()
    total = k * plane_bytes
    assert k == dk.span_planes(78, plane_bytes, display) and total >= display
    assert all(p.is_cuda and int(p.count_nonzero()) == 0 for p in planes)

    # written by a kernel, as the server writes them (a memcpy may not cross the halves)
    for i, f in enumerate(flat):
        _kernel_copy(f, _pattern(i, f.numel()))
    torch.cuda.synchronize()
    assert all(torch.equal(f, _pattern(i, f.numel())) for i, f in enumerate(flat))
    assert torch.equal(flat[-1][-MIB:].cpu(), _pattern(k - 1, plane_bytes)[-MIB:].cpu())   # the CPU sees the top

    # a latent plane in the span holds exactly what a plain device tensor holds
    plain = torch.randint(0, 255, planes[-1].shape, dtype=torch.uint8, device="cuda")
    _kernel_copy(planes[-1], plain)
    torch.cuda.synchronize()
    assert torch.equal(planes[-1], plain)

    # row copies whose source / destination cross from the ordinary half into the display half
    ordinary = total - display
    j = ordinary // plane_bytes
    row_bytes = planes[j].shape[1]
    edge = (ordinary - j * plane_bytes) // row_bytes          # the row the display half starts in
    lo, hi = edge - 500, edge + 2700
    if lo < 0 or hi > rows:
        pytest.skip(f"boundary row {edge} of plane {j} too near its edge for the copy test")
    before = torch.empty_like(planes[j])
    _kernel_copy(before, planes[j])
    expect = before.clone()
    expect[edge + 1000:edge + 1600] = before[edge - 300:edge + 300]
    expect[edge - 500:edge + 200] = expect[edge + 2000:edge + 2700].clone()
    if "spanned" in Plane.__init__.__code__.co_varnames:   # 0072's pool hook, once the serial lane lands it
        arena = Arena(rows, [Plane(planes[j], spanned=True)])
        arena.copy(edge - 300, edge + 1000, 600)
        arena.copy(edge + 2000, edge - 500, 700)
    else:
        for a, b, n in ((edge - 300, edge + 1000, 600), (edge + 2000, edge - 500, 700)):
            staged = torch.empty_like(planes[j][a:a + n])          # by kernels both ways (clone() is a memcpy)
            _kernel_copy(staged, planes[j][a:a + n])
            _kernel_copy(planes[j][b:b + n], staged)
    torch.cuda.synchronize()
    assert torch.equal(planes[j], expect)

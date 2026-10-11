"""Mia's chunked (WY) KDA prompt kernel (her 0012, 0014, 0039; port-prefill-8s), TF_GLM_KDA_CHUNKED=1, off by default.

Contracts: the kernel follows its float64 algorithm (``kda_chunked.chunked_reference``) and lands within the serial
kernel's error of the float64 delta rule; a row's bits depend only on its 32-row sub-chunk and the entering state, so
prompt chunks split on the 32-row grid give the whole chunk's bits (and splits off the grid need not); in the engine,
prompt chunks of any size on the 64-token grid leave the same state, a resumed prompt equals a fresh prefill, drafted
replies equal serial ones, and the states stay within tolerance of the serial kernel's."""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402


def _inputs(R: int, H: int = 2, seed: int = 3):
    gen = torch.Generator().manual_seed(seed)
    C = 3 * H * 128
    width = C + 256 + H
    P = (torch.randn((R, width), generator=gen) * 0.5).to(torch.bfloat16).cuda()
    A = (torch.randn((R, H * 128), generator=gen)).to(torch.bfloat16).cuda()
    G = (torch.randn((R, H * 128), generator=gen)).to(torch.bfloat16).cuda()
    cw = (torch.randn((C, 4), generator=gen) * 0.3).to(torch.bfloat16).cuda()
    cs0 = (torch.randn((3, C), generator=gen) * 0.5).to(torch.bfloat16).cuda()
    st0 = (torch.randn((H, 128, 128), generator=gen) * 0.05).cuda()
    a_log = torch.randn((H,), generator=gen).cuda()
    dt = torch.randn((H * 128,), generator=gen).cuda()
    nw = (1 + 0.1 * torch.randn((128,), generator=gen)).to(torch.bfloat16).cuda()
    return dict(P=P, A=A, G=G, cw=cw, cs0=cs0, st0=st0, a_log=a_log, dt=dt, nw=nw, b_off=C + 256, H=H, C=C)


def _chunked(x, lo: int, hi: int, cs, st, pos: int):
    from tensorfold.families.glm5_next.cuda import kda_chunked

    n = hi - lo
    out = torch.empty((n, x["H"] * 128), dtype=torch.bfloat16, device="cuda")
    s_out = torch.empty_like(st)
    kda_chunked.chain(x["P"][lo:hi], x["b_off"], x["A"][lo:hi], x["G"][lo:hi], cs, x["cw"], st, x["a_log"], x["dt"],
                      x["nw"], 1e-5, -5.0, n, out, s_out, pos)
    return out, s_out


def _conv_after(x, cs, lo: int, hi: int):
    return torch.cat([cs, x["P"][lo:hi, :x["C"]]])[-3:].contiguous()


@pytest.mark.parametrize("H", [2, 32])
def test_kernel_follows_its_algorithm_and_the_delta_rule(H):
    from tensorfold.families.glm5_next.cuda import kda, kda_chunked

    R = 100
    x = _inputs(R, H)
    for pos in (0, 7):
        out, s_out = _chunked(x, 0, R, x["cs0"], x["st0"], pos)
        args = (x["P"], x["b_off"], x["A"], x["G"], x["cs0"], x["cw"], x["st0"], x["a_log"], x["dt"], x["nw"], 1e-5,
                -5.0, R)
        ref_c, S_c, _ = kda_chunked.chain_reference(*args, pos=pos, chunked=True, dtype=torch.float64)
        ref_r, S_r, _ = kda_chunked.chain_reference(*args, chunked=False, dtype=torch.float64)
        assert float((S_c - S_r).abs().max()) <= 1e-8 * max(1.0, float(S_r.abs().max()))      # the same math
        serial = kda.KDAScratch(R, H, "cuda")
        s_ser = torch.empty_like(x["st0"])
        ser = kda.chain(*args, serial, s_ser).clone()
        # state: within fp32 rounding of the float64 rule, as the serial kernel is
        e_new = float((s_out - S_r).abs().max())
        e_ser = float((s_ser - S_r).abs().max())
        assert e_new <= max(4 * e_ser, 1e-5), (e_new, e_ser)
        # outputs: bf16 rows, a rounding step apart at most
        d = (out.float() - ser.float()).abs().max()
        assert float(d) <= 0.05 * float(ser.float().abs().max()), float(d)
        assert float((out.float() - ref_r.float()).abs().max()) <= 0.05 * float(ref_r.float().abs().max())


def test_rows_depend_only_on_their_sub_chunk():
    R = 160
    x = _inputs(R, 2, seed=5)
    whole, s_whole = _chunked(x, 0, R, x["cs0"], x["st0"], 0)
    for cut in (32, 64, 96, 128):                  # on the 32-row grid: the whole chunk's bits
        a, s_a = _chunked(x, 0, cut, x["cs0"], x["st0"], 0)
        b, s_b = _chunked(x, cut, R, _conv_after(x, x["cs0"], 0, cut), s_a, cut)
        assert torch.equal(torch.cat([a, b]), whole), cut
        assert torch.equal(s_b, s_whole), cut
    # off the grid: still close (another sum), which is why the engine keeps prompt states on a grid
    a, s_a = _chunked(x, 0, 50, x["cs0"], x["st0"], 0)
    b, s_b = _chunked(x, 50, R, _conv_after(x, x["cs0"], 0, 50), s_a, 50)
    assert torch.equal(a[:32], whole[:32])
    assert float((s_b - s_whole).abs().max()) <= 1e-4 * float(s_whole.abs().max())


def test_prompt_grid_settings():
    from tensorfold.families.glm5_next.cuda.engine import grid_point, prompt_grid

    assert prompt_grid(2048, False) == 0 and prompt_grid(2048, True) == 64
    with pytest.raises(ValueError):
        prompt_grid(2000, True)
    # off / on-grid prompts keep the end state (Mia's 0008 + 0042 replay); else the last grid multiple past the cut
    assert grid_point(100, 0, 0) == 100 and grid_point(1, 0, 0) == 1
    assert grid_point(100, 0, 64) == 64 and grid_point(129, 64, 64) == 128 and grid_point(128, 64, 64) == 128
    assert grid_point(64, 0, 64) == 64 and grid_point(100, 64, 64) is None


@pytest.fixture(scope="module")
def chunked_engines(tmp_path_factory):
    """A chunked and a serial engine over the synthetic checkpoint with the DFlash2 drafter."""

    from test_glm_engine import _TwoCopies, _checkpoint, _drafter

    from tensorfold.families.glm5_next.cuda import kda
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    path = tmp_path_factory.mktemp("glm_chunked")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    old = kda.CHUNKED
    kda.CHUNKED = True
    try:
        e = GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", comm=_TwoCopies(),
                      context=2600)
        assert e.grid == 64
        yield e, path
    finally:
        kda.CHUNKED = old


def test_chunked_prompt_chunks_on_the_grid_leave_the_same_state(chunked_engines):
    from test_glm_engine import _state

    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill

    e, _ = chunked_engines
    prompt = [int(t) for t in np.random.default_rng(8).integers(0, 1000, size=2300)]
    runs = []
    for rows in (2048, 64, 128):
        eng = Engine(e.w, capacity=2600, max_rows=8, prefill_rows=rows, long_context=True)
        runs.append((prefill(eng, prompt, None), [t.clone() for t in _state(eng)]))
        del eng
    for first, st in runs[1:]:
        assert first == runs[0][0] and all(torch.equal(a, b) for a, b in zip(st, runs[0][1]))


@pytest.mark.parametrize("sampling", [Sampling(77, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_chunked_resumes_like_a_fresh_prefill_and_drafted_equals_serial(chunked_engines, sampling):
    from test_glm_engine import _forget, _generate

    e, _ = chunked_engines
    first = [int(t) for t in np.random.default_rng(9).integers(0, 1000, size=333)]
    _forget(e)
    reply, _ = _generate(e, first, sampling, tokens=20)
    assert [len(s.ids) for s in e.cache] == [320]            # the kept state sits on the 64-token grid
    after = first + reply + [5, 6, 7]
    warm, stats = _generate(e, after, sampling, tokens=20)
    assert stats["cached"] == 320
    _forget(e)
    cold, stats = _generate(e, after, sampling, tokens=20)
    assert stats["cached"] == 0 and warm == cold
    _forget(e)
    serial, _ = _generate(e, after, sampling, draft=False, tokens=20)
    assert serial == cold


def test_chunked_stays_close_to_the_serial_kernel(chunked_engines):
    from test_glm_engine import _state

    from tensorfold.families.glm5_next.cuda import kda
    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill

    e, _ = chunked_engines
    prompt = [int(t) for t in np.random.default_rng(10).integers(0, 1000, size=700)]
    eng = Engine(e.w, capacity=2600, max_rows=8, prefill_rows=2048, long_context=True)
    prefill(eng, prompt, None, mtp=False)
    a = [t.clone() for t in _state(eng)]
    la = eng.pbuf.logits[:1].float().clone()
    kda.CHUNKED = False
    try:
        prefill(eng, prompt, None, mtp=False)
    finally:
        kda.CHUNKED = True
    b = _state(eng)
    lb = eng.pbuf.logits[:1].float()
    rec = float((a[0] - b[0]).abs().max())
    assert 0 < rec <= 2e-2 * float(b[0].abs().max()), rec      # other bits, the same state within tolerance
    assert float(torch.nn.functional.cosine_similarity(la, lb, dim=1)) > 0.999

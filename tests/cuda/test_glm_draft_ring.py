"""DFlash2's windowed block attention and context ring against the loop from tile 0 and a flat buffer, bitwise."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.geometry import draft_ring_rows  # noqa: E402
from test_glm_engine import DRAFT, D, V, _drafter  # noqa: E402


@pytest.mark.parametrize("window", [2047, 48])
@pytest.mark.parametrize("causal", [False, True])
def test_the_window_loop_and_the_ring_keep_the_full_loops_bits(window, causal):
    from tensorfold.families.glm5_next.cuda.dflash2 import _dattn_kernel

    gen = torch.Generator(device="cuda").manual_seed(window)
    H, KV, N, HD = 16, 4, 8, 128
    ring = draft_ring_rows(window, N)
    flat_cap = 13000
    kc = torch.randn((KV, flat_cap, HD), generator=gen, device="cuda").bfloat16()
    vc = torch.randn((KV, flat_cap, HD), generator=gen, device="cuda").bfloat16()
    q = torch.randn((H, N, HD), generator=gen, device="cuda").bfloat16()

    def run(k, v, cap, s, ring_mode, skip=True):
        out = torch.empty((N, H * HD), dtype=torch.bfloat16, device="cuda")
        _dattn_kernel[(KV,)](q, k, v, out, torch.tensor([s], device="cuda"), window, HD ** -0.5, N=N, G=H // KV,
                             NH=H, HD=HD, CAP=cap, BK=64, CAUSAL=causal, RING=ring_mode, SKIP=skip, num_warps=4)
        return out

    starts = sorted({0, 1, 7, 63, 64, 65, window - 1, window, window + 1, window + 63, window + 64, ring - N,
                     ring, ring + 1, 2 * ring + 5, 4095, 4096, 5000, 12345, flat_cap - N})
    for s in starts:
        want = run(kc, vc, flat_cap, s, False, skip=False)       # the old loop: every tile from 0
        assert torch.equal(run(kc, vc, flat_cap, s, False), want), s         # the loop from the window's tile
        rk = torch.randn((KV, ring, HD), generator=gen, device="cuda").bfloat16() * 100      # stale rows
        rv = torch.randn((KV, ring, HD), generator=gen, device="cuda").bfloat16() * 100
        pos = torch.arange(max(0, s + N - ring), s + N, device="cuda")
        rk[:, pos % ring], rv[:, pos % ring] = kc[:, pos], vc[:, pos]
        assert torch.equal(run(rk, rv, ring, s, True), want), s


def _weights():
    from tensorfold.families.glm5_next.cuda import qmm

    g = torch.Generator(device="cuda").manual_seed(5)
    embed = (torch.randn((V, D), generator=g, device="cuda") * 0.05).bfloat16()
    head = qmm.quantize4((torch.randn((V, D), generator=g, device="cuda") * 0.03).bfloat16())
    return SimpleNamespace(device=torch.device("cuda"), rank=0, world=1, comm=None, embed=embed, head=head,
                           draft_head=None, vocab_offset=0)


def _engine_stub():
    """What ``take_snapshot`` and ``restore`` touch of an engine besides the drafter."""

    st = SimpleNamespace(cur=[], rec=[torch.zeros(1, device="cuda")], conv=torch.zeros(1, device="cuda"), mtp_len=0,
                         mtp_drafted=0, set_pos=lambda n: None, set_mtp_len=lambda n: None)
    return SimpleNamespace(st=st)


@pytest.fixture(scope="module", params=[2048, 49], ids=["window2047", "window48"])
def pair(request, tmp_path_factory):
    from tensorfold.families.glm5_next.cuda.dflash2 import Drafter

    path = tmp_path_factory.mktemp(f"dring{request.param}")
    _drafter(path)
    (path / "config.json").write_text(json.dumps(dict(DRAFT, sliding_window=request.param)))
    w = _weights()
    flat = Drafter(path, w, capacity=12000, ring=False)
    ring = Drafter(path, w, capacity=12000, ring=True)
    assert flat.ring == 0 and ring.ring == draft_ring_rows(request.param - 1, flat.block) and ring.cap == ring.ring
    assert ring.nbytes() < flat.nbytes()
    yield flat, ring
    del flat, ring
    torch.cuda.empty_cache()


def _drive(pair, seed: int) -> int:
    """Rounds far past the window on both drafters, a kept state taken, overwritten, restored; candidates compared."""

    from tensorfold.families.glm5_next.cuda import decode

    flat, ring = pair
    rng = np.random.default_rng(seed)
    width = flat.tap_in.shape[1]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    e = _engine_stub()
    rounds = 0

    def taps(n: int) -> torch.Tensor:
        return (torch.randn((n, width), generator=gen, device="cuda") * 0.5).bfloat16()

    def both_add(n: int) -> None:
        t = taps(n)
        flat.add_taps(t)
        ring.add_taps(t)
        assert flat.context_end == ring.context_end

    def compare(k: int) -> None:
        nonlocal rounds
        for _ in range(k):
            pending, depth = int(rng.integers(0, 1000)), int(rng.integers(1, flat.block))
            a, b = flat.candidates(pending, depth), ring.candidates(pending, depth)
            assert all(np.array_equal(x, y) for x, y in zip(a, b)), (flat.context_end, rounds)
            rounds += 1
            both_add(int(rng.integers(1, 9)))

    for d in (flat, ring):
        d.reset()
    for n in (300, 64, 1000, 17, 2100):                  # a prompt in chunks: 3,481 rows
        both_add(n)
    compare(40)
    n = flat.context_end
    snaps = [decode.take_snapshot(e, list(range(n)), None, mtp=False, drafter=d) for d in (flat, ring)]
    assert snaps[0].drafter_rows is None and snaps[1].drafter_rows is not None
    both_add(3000)                                       # another conversation, past the ring many times over
    compare(20)
    for d, s in zip((flat, ring), snaps):
        decode.restore(e, s, d)
    compare(60)                                          # resumed: the kept window came back
    for n in (64, 1500):                                 # a longer prompt resumed from there, then decode on
        both_add(n)
    compare(40)
    return rounds


def test_the_ring_drafts_the_flat_buffers_bits_eager(pair):
    assert _drive(pair, 1) == 160


def test_the_ring_drafts_the_flat_buffers_bits_in_cuda_graphs(pair):
    for d in pair:
        d.capture()
    try:
        assert _drive(pair, 2) == 160
    finally:
        for d in pair:
            d.block_graph, d.tap_graphs = None, {}


@pytest.fixture(scope="module")
def engine_ring(tmp_path_factory):
    """The synthetic GLM engine with DFlash2 in a 128-row ring (window 48) and 128-row prompt chunks."""

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from test_glm_engine import _TwoCopies, _checkpoint

    path = tmp_path_factory.mktemp("glm_ring")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    (path / "dflash2" / "config.json").write_text(json.dumps(dict(DRAFT, sliding_window=49)))
    engine = GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", comm=_TwoCopies(),
                       prefill_rows=128)
    assert engine.drafter.ring == draft_ring_rows(48, engine.drafter.block) == 128
    yield engine
    del engine
    torch.cuda.empty_cache()


def _rounds(drafter, pending: int, seed: int, k: int = 12) -> list:
    """k rounds of the drafter's candidates, the same taps between them for every caller with this seed."""

    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for _ in range(k):
        out.append(drafter.candidates(pending, int(rng.integers(1, drafter.block))))
        n = int(rng.integers(1, 9))
        drafter.add_taps((torch.randn((n, drafter.tap_in.shape[1]), generator=gen, device="cuda") * 0.5).bfloat16())
        pending = int(rng.integers(0, 1000))
    return out


def _same(a: list, b: list) -> bool:
    return len(a) == len(b) and all(all(np.array_equal(x, y) for x, y in zip(r, s)) for r, s in zip(a, b))


def test_a_kept_prompt_point_resumes_the_rings_drafts_round_by_round(engine_ring):
    """A prompt's state kept at an interior point (as the engine keeps a shared-prefix point: prefill to it with
    ``sample=False`` then ``take_snapshot``) in a ring wrapped past its window: a next turn resumed from it drafts,
    round by round, what a fresh prefill of the turn drafts; the ring's window came along with the snapshot."""

    from tensorfold.families.glm5_next.cuda import decode

    e, drafter = engine_ring.e, engine_ring.drafter
    prompt = [int(t) for t in np.random.default_rng(41).integers(0, 1000, size=300)]      # 128 + 128 + 44 rows
    point = len(prompt) - 1
    decode.prefill(e, prompt[:point], None, drafter=drafter, sample=False)
    snap = decode.take_snapshot(e, prompt[:point], e.last_hidden, mtp=True, drafter=drafter)
    assert snap.drafter_end == point and snap.drafter_rows is not None
    turn = prompt[:point] + [271, 77, 78]
    first = decode.prefill(e, turn, None, drafter=drafter, resume=snap)
    resumed = _rounds(drafter, first, 2)
    assert decode.prefill(e, turn, None, drafter=drafter) == first                             # next turn, fresh
    assert _same(_rounds(drafter, first, 2), resumed)


def test_a_kept_interior_point_always_keeps_the_rings_window(engine_ring):
    """With the sample=False keep flow a prompt stops exactly at the kept point, so the ring still holds its window
    at any point (the removed mid-chunk cut could write past it and lose the window)."""

    from tensorfold.families.glm5_next.cuda import decode

    e, drafter = engine_ring.e, engine_ring.drafter
    rows = e.prefill_rows
    prompt = [int(t) for t in np.random.default_rng(43).integers(0, 1000, size=rows + 30)]
    for point in (rows - 5, rows, rows + 20):
        decode.prefill(e, prompt[:point], None, drafter=drafter, sample=False)
        snap = decode.take_snapshot(e, prompt[:point], e.last_hidden, mtp=True, drafter=drafter)
        assert snap.drafter_end == point and snap.drafter_rows is not None, point


def test_identical_resends_and_a_turn_keep_the_rings_window_through_the_engine(engine_ring):
    """Each identical resend keeps the state it resumed from again (the same snapshot), which must keep its window."""

    from test_glm_engine import _forget, _generate

    _forget(engine_ring)
    prompt = [int(t) for t in np.random.default_rng(47).integers(0, 1000, size=300)]
    fresh, stats = _generate(engine_ring, prompt, None, policy="fc5:0.3", tokens=24)
    want = (fresh, stats["rounds"], stats["min_rows"])
    for _ in range(3):
        again, stats = _generate(engine_ring, prompt, None, policy="fc5:0.3", tokens=24)
        assert stats["cached"] == len(prompt) and (again, stats["rounds"], stats["min_rows"]) == want  # a replay
    turn = prompt[:-1] + [271, 77, 78]
    resumed, stats = _generate(engine_ring, turn, None, policy="fc5:0.3", tokens=24)
    assert stats["cached"] == 0                           # a different ending shares no kept state (no shared prefix)
    got = (resumed, stats["rounds"], stats["min_rows"])
    _forget(engine_ring)
    fresh, stats = _generate(engine_ring, turn, None, policy="fc5:0.3", tokens=24)
    assert stats["cached"] == 0 and (fresh, stats["rounds"], stats["min_rows"]) == got


@pytest.fixture(scope="module", params=[1, 2], ids=["one_rank", "rank0_of_two"])
def drafters(request, tmp_path_factory):
    """A flat and a ring drafter (window 48: the ring wraps every 128 rows), on one rank or as rank 0 of two (every
    all-gather returns this rank's rows twice)."""

    from tensorfold.families.glm5_next.cuda.dflash2 import Drafter
    from test_glm_engine import _TwoCopies

    path = tmp_path_factory.mktemp(f"mdraft{request.param}")
    _drafter(path)
    (path / "config.json").write_text(json.dumps(dict(DRAFT, sliding_window=49)))
    w = _weights()
    if request.param == 2:
        w.world, w.comm = 2, _TwoCopies()
    pair = [Drafter(path, w, capacity=4000, ring=ring, tap_rows=16) for ring in (False, True)]
    assert pair[0].ring == 0 and pair[1].ring == 128 and pair[0].gathered == request.param
    yield pair
    del pair
    torch.cuda.empty_cache()


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
@pytest.mark.parametrize("ring", [False, True], ids=["flat", "ring"])
def test_batched_drafts_equal_each_streams_own(drafters, ring, graphs):
    """TF_GLM_MULTI_DRAFT=batched against serial: three streams' ``MultiDrafter`` slots and three solo contexts fed
    the same taps (prompt chunks alone, a round's kept rows together) propose the same chains and candidates, bit
    for bit, in block passes of 1, 2 and 3 streams in any order: mixed policies (keyed samples, confidence and
    noise-aware stops), a stream with no context yet, a stream that copied (keeps rows, asks nothing), a kept
    window taken and put back over another conversation's rows."""

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.glm5_next.cuda.dflash2_multi import (DraftRequest, MultiDrafter, context, put_window,
                                                                  take_window)

    d = drafters[int(ring)]
    refs = [context(d, capture=graphs) for _ in range(3)]
    md = MultiDrafter(d, 3)
    if graphs:
        md.capture()
        assert sorted(md.block_graphs) == [1, 2, 3] and md.tap_graphs
    ctxs = md.contexts
    rng = np.random.default_rng(int(ring) * 2 + int(graphs))
    gen = torch.Generator(device="cuda").manual_seed(7)
    width = d.tap_in.shape[1]
    policies = [dict(sampling=Sampling(1234, 1.0, 20, 0.95), confidence=0.0),
                dict(sampling=None, confidence=0.3),
                dict(sampling=Sampling(77, 0.8, 0, 0.9), confidence=0.3, beta=2.0),
                dict(sampling=None, confidence=0.0, beta=2.0, round_ms=(55.0, 57.0, 58.5, 60.0, 61.0, 62.5, 64.0, 66.0))]

    def taps(n: int) -> torch.Tensor:
        return (torch.randn((n, width), generator=gen, device="cuda") * 0.5).bfloat16()

    def alone(s: int, n: int) -> None:                   # a prompt chunk of stream s
        t = taps(n)
        refs[s].add_taps(t)
        ctxs[s].add_taps(t)

    def ends() -> list[int]:
        got = [c.context_end for c in ctxs]
        assert got == [c.context_end for c in refs] == md.pos[:3].tolist()
        return got

    alone(0, 300)
    alone(1, 77)
    asking = [[0], [0, 1, 2], [2, 0], [1], [0, 2]]       # stream 1 copies in the last pattern: it keeps, asks nothing
    drafted = sizes = 0
    for r in range(30):
        who = asking[r % 5]
        reqs = []
        for s in who:
            p = policies[(s + r) % 4]
            reqs.append(DraftRequest(ctxs[s], int(rng.integers(0, 1000)), int(rng.integers(1, d.block + 2)), **p))
        got = md.propose(reqs)
        want = [refs[s].propose(q.pending, q.depth, q.sampling, q.confidence, beta=q.beta, round_ms=q.round_ms)
                for s, q in zip(who, reqs)]
        assert got == want, (r, who)
        drafted += sum(len(g) for g in got)
        live = [(s, q) for s, q in zip(who, reqs) if refs[s].context_end > 0]
        if live:
            cand = md.candidates([(ctxs[s], q.pending, min(q.depth, d.block - 1)) for s, q in live])
            sizes = max(sizes, len(live))
            for (s, q), c in zip(live, cand):
                assert all(np.array_equal(x, y)
                           for x, y in zip(c, refs[s].candidates(q.pending, min(q.depth, d.block - 1)))), (r, s)
        keep = [int(rng.integers(1, d.block + 1)) for _ in range(3)]
        if r % 5 == 4:
            keep[1] = 16                                # copy drafts: a 16-row window kept whole
        items = []
        for s in range(3):
            if refs[s].context_end == 0 and r < 10:
                continue                                # no prompt yet
            t = taps(keep[s])
            refs[s].add_taps(t)
            items.append((ctxs[s], t))
        md.commit(items)
        ends()
        if r == 10:
            alone(2, 200)                               # stream 2's prompt arrives
        if r == 20:                                     # stream 1's state kept, another conversation, put back
            n = ends()[1]
            rows = take_window(refs[1], n)
            assert all(torch.equal(a, b) for a, b in zip(rows, take_window(ctxs[1], n)))
            for c in (refs[1], ctxs[1]):
                c.reset()
            alone(1, 333)
            for c in (refs[1], ctxs[1]):
                put_window(c, n, rows)
            ends()
    assert drafted > 0 and sizes == 3

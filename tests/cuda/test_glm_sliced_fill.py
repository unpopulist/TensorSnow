"""Mia's sliced prompt fills (her 0062; port-prefill-8s), TF_GLM_FILL_BUDGET_MS > 0 under --parallel: a prompt chunk
runs in layer slices with decode work between them. Contract: the sliced chunk makes the unsliced chunk's calls, so
every bit is the same (states, taps, head), whatever decode windows of other streams run between its slices, alone
or as a multi-prompt chunk, split between two ranks or not; through the scheduler every reply equals its solo one,
rank 1's replay of the FILLPART / MFILLPART messages agrees, and a filling stream whose client left ends."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_engine import _checkpoint, _TwoCopies  # noqa: E402
from test_glm_multi_prefill import _states, _view, _Taps  # noqa: E402


def _prompts(seed, sizes):
    rng = np.random.default_rng(seed)
    return [[int(t) for t in rng.integers(0, 1000, size=n)] for n in sizes]


def _run(w, prompts, *, sliced: bool, many: bool, cap=2600, rows=1024, long_context=True, split=None):
    """Each prompt's first chunk (``rows``) alone or as one multi-prompt chunk, sliced (a decode window of another
    stream between every two layers) or not; then the states, taps and heads."""

    from tensorfold.families.glm5_next.cuda import multi_prefill
    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill_chunk, prefill_chunk_steps
    from tensorfold.families.glm5_next.cuda.forward import commit, forward

    n = len(prompts)
    e = Engine(w, capacity=cap, max_rows=8, prefill_rows=rows * n, long_context=long_context, streams=n + 1,
               pool_rows=(n + 1) * cap, taps=(0, 1), split=split)
    sts = _states(e, cap, n + 1)
    other = sts[-1]                             # a stream decoding between slices
    other.reset()
    prev = e.use(other)
    prefill_chunk(e, _prompts(99, (40,))[0], 0, 40)
    e.use(prev)

    def decode_step(i):
        forward(w, other, e.buf, [i % 1000, (i * 7) % 1000, 3])
        commit(w, other, e.buf, 3, 2)

    taps = [_Taps() for _ in prompts]
    heads = []
    for st in sts[:n]:
        st.reset()
    chunks = [list(q[:rows]) for q in prompts]
    if many:
        pieces = [multi_prefill.Piece(st, c, drafter=t, head=len(c) == len(q))
                  for st, c, t, q in zip(sts, chunks, taps, prompts)]
        if sliced:
            gen = multi_prefill.prefill_piece_steps(e, pieces)
            try:
                i = 0
                while True:
                    next(gen)
                    decode_step(i)
                    i += 1
            except StopIteration as stop:
                heads = stop.value
        else:
            heads = multi_prefill.prefill_pieces(e, pieces)
    else:
        for st, c, t, q in zip(sts, chunks, taps, prompts):
            prev = e.use(st)
            try:
                if sliced:
                    gen = prefill_chunk_steps(e, q, 0, len(c), drafter=t, head=len(c) == len(q))
                    try:
                        i = 0
                        while True:
                            next(gen)
                            decode_step(i)
                            i += 1
                    except StopIteration as stop:
                        heads.append(stop.value)
                else:
                    heads.append(prefill_chunk(e, q, 0, len(c), drafter=t, head=len(c) == len(q)))
            finally:
                e.use(prev)
    torch.cuda.synchronize()
    return [(h, t.rows, _view(st)) for h, t, st in zip(heads, taps, sts[:n])]


def _same(a, b):
    for k, ((ha, ta, sa), (hb, tb, sb)) in enumerate(zip(a, b)):
        assert (ha is None) == (hb is None), k
        if ha is not None:
            assert torch.equal(ha[:1], hb[:1]), k
        assert len(ta) == len(tb) and all(torch.equal(x, y) for x, y in zip(ta, tb)), k
        assert len(sa) == len(sb)
        for i, (x, y) in enumerate(zip(sa, sb)):
            assert torch.equal(x, y), (k, i)


@pytest.fixture(scope="module", params=["mlx", "exl3"])
def weights_one(request, tmp_path_factory):
    from tensorfold.families.glm5_next.cuda import weights

    path = tmp_path_factory.mktemp("glm_sf_" + request.param)
    _checkpoint(path, exl3=request.param == "exl3", mtp=False)
    w = weights.load(path, rank=0, device="cuda", mtp=False, world=2)
    w.comm = _TwoCopies()
    return w, path


@pytest.mark.parametrize("many", [False, True], ids=["alone", "pieces"])
def test_sliced_chunks_equal_unsliced(weights_one, many):
    w, _ = weights_one
    prompts = _prompts(5, (2300, 500, 90))
    want = _run(w, prompts, sliced=False, many=many)
    got = _run(w, prompts, sliced=True, many=many)
    _same(want, got)


def test_sliced_chunks_with_the_split_between_two_ranks(weights_one):
    from glm_two_ranks import both, load_pair

    from tensorfold.families.glm5_next.cuda.hcsplit import SplitSettings

    _, path = weights_one
    ws, hub = load_pair(path)
    prompts = _prompts(6, (301, 130))
    split = SplitSettings(True, True, 2, fronts=True)
    for many in (False, True):
        res = both(hub, *[lambda r=r: (_run(ws[r], prompts, sliced=False, many=many, cap=2560, rows=512,
                                            long_context=False, split=split),
                                       _run(ws[r], prompts, sliced=True, many=many, cap=2560, rows=512,
                                            long_context=False, split=split)) for r in (0, 1)])
        for want, got in res:
            _same(want, got)


def test_layer_costs_slice_within_the_budget():
    from tensorfold.families.glm5_next.cuda.sliced_fill import LayerCosts, budget_ms

    c = LayerCosts({64: [1.0, 2.0, 1.0, 3.0], 1024: [4.0, 8.0, 4.0, 12.0]})
    assert c.stop(64, 0, 3.0) == 2 and c.stop(64, 2, 100.0) == 4 and c.stop(64, 0, 0.5) == 1
    assert c.costs(544) == pytest.approx((2.5, 5.0, 2.5, 7.5))
    assert budget_ms("") == 0 and budget_ms("12.5") == 12.5
    with pytest.raises(ValueError):
        budget_ms("-1")


# -- through the scheduler ---------------------------------------------------------------------------------------
def _env_engine(models, env, **kw):
    from test_glm_multi import _engine, _with_env

    return _with_env(env, lambda: _engine(models / "model", drafter=models / "dflash2", parallel=3,
                                          multi_options={"draft_graphs": False}, **kw))


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    from test_glm_engine import _drafter

    path = tmp_path_factory.mktemp("glm_sf_models")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    return path


@pytest.mark.parametrize("group", ["0", "1"], ids=["fill", "multi-prefill"])
def test_sliced_fills_through_the_scheduler(models, group):
    from test_glm_multi import SAMPLINGS, _concurrent, _engine, _solo

    from tensorfold.families.glm5_next.cuda.multi import FILLPART, MFILLPART

    solo = _engine(models / "model", drafter=models / "dflash2")
    env = {"TF_GLM_FILL_BUDGET_MS": "0.01", "TF_GLM_MULTI_PREFILL": group, "TF_GLM_MULTI_VERIFY": "batched",
           "TF_GLM_FILL_ROWS": "64"}
    a = _env_engine(models, env)
    b = _env_engine(models, env)
    try:
        b.scheduler.close()
        assert a.multi.fill_budget == 0.01 and a.multi.fill_costs is not None
        a.multi.log = []
        prompts = _prompts(7, (40, 300, 150, 260))
        samplings = SAMPLINGS + [None]
        jobs = [(p, s, n) for p, s, n in zip(prompts, samplings, (300, 24, 24, 24))]   # the first decodes long
        want = [_solo(solo, p, s, n) for p, s, n in jobs]
        got, _ = _concurrent(a, jobs, delays=[0.0, 0.1, 0.12, 0.2])
        assert got == want
        ops = [op for msg in a.multi.log for op, _ in a.multi.parse(msg)]
        assert FILLPART in ops or MFILLPART in ops
        d1 = b.multi
        seen: list[list[int]] = []
        finish = d1._finish

        def record(sid):
            if sid in d1.lanes and not (d1.partial is not None and d1.partial.contains(sid)):
                seen.append(list(d1.lanes[sid].s.out))
            finish(sid)

        d1._finish = record
        for msg in a.multi.log:
            d1.apply(msg)
        assert sorted(seen) == sorted(got)
    finally:
        a.scheduler.close()


def test_a_filling_stream_whose_client_left_ends(models):
    from test_glm_multi import SAMPLINGS, _engine, _solo

    solo = _engine(models / "model", drafter=models / "dflash2")
    a = _env_engine(models, {"TF_GLM_FILL_BUDGET_MS": "0.01", "TF_GLM_FILL_ROWS": "64",
                             "TF_GLM_MULTI_VERIFY": "batched"})
    try:
        import threading

        from tensorfold.cuda.streams import Stream

        d = a.multi
        p0, p1 = _prompts(8, (30, 900))
        s0 = Stream(list(p0), 30, SAMPLINGS[0], draft=True, stop_eos=False)
        s1 = Stream(list(p1), 10, None, draft=True, stop_eos=False)
        left = threading.Event()
        s1.gone = left.is_set
        for s in (s0, s1):
            s.emit = lambda new: False
            s.glm = {}
        d.admit(s0)
        d.admit(s1)
        finished = []
        for i in range(400):
            done = d.round()
            d.finish(done)
            finished += done
            if i == 6:
                left.set()                      # s1's client leaves while its prompt fills in slices
            if s0.done and s1.done:
                break
        assert s1.done and len(s1.out) == 0
        assert s0.out == _solo(solo, p0, SAMPLINGS[0], 30)
        assert not d.lanes and d.partial is None
    finally:
        a.scheduler.close()

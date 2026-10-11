"""GLM-5.3-Flash's concurrent streams (--parallel N, ``multi.py``) on the tiny synthetic checkpoint of
``test_glm_engine``: every stream's reply equals the reply its request gets alone from a --parallel 1 engine with
``draft=False`` (the serial reference), token for token, whatever else decodes beside it; arrivals staggered or at
once; a stream cancelled mid-reply; a kept prompt resumed; rank 1's replay of rank 0's messages gives the same streams.

One GPU plays rank 0 of two (``_TwoCopies``), as in ``test_glm_engine``."""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_engine import _checkpoint, _drafter, _TwoCopies  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

os.environ.setdefault("TF_GLM_CACHE_GIB", "0.05")
os.environ.setdefault("TF_GLM_MULTI_VERIFY", "serial")    # the reference seam unless a test asks for batched     # the --parallel pool: a few thousand tokens of the tiny model

SAMPLINGS = [Sampling(1234, 1.0, 20, 0.95), Sampling(77, 0.8, 0, 0.9), None]


def _engine(path, **kw):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies(), **kw)


def _with_env(env: dict, make):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return make()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_multi")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    _checkpoint(path / "exl3", exl3=True)
    return path


@pytest.fixture(scope="module")
def solo(models):
    """The serial reference: a --parallel 1 engine (MTP head and drafter as the solo tests run them)."""
    return _engine(models / "model", drafter=models / "dflash2")


@pytest.fixture(scope="module", params=["serial", "batched"])
def multi(models, request):
    """--parallel 3 with the DFlash2 drafter, each verify seam (contexts without graphs to keep the test short)."""
    e = _with_env({"TF_GLM_MULTI_VERIFY": request.param},
                  lambda: _engine(models / "model", drafter=models / "dflash2", parallel=3,
                                  multi_options={"draft_graphs": False}))
    yield e
    e.scheduler.close()


def _solo(engine, prompt, sampling, tokens):
    out: list[int] = []
    engine.request.policy = None
    engine.request.stop_eos = False
    engine.generate(list(prompt), tokens, sampling, lambda new: out.extend(new), draft=False)
    return out


def _concurrent(engine, jobs, *, delays=None, stop_after=None, policy=None):
    """jobs [(prompt, sampling, tokens)] submitted from threads (``delays``: seconds before each); ``stop_after``:
    {job: n} the client stops after n tokens. Returns each job's tokens and stats."""

    outs: list[list[int]] = [[] for _ in jobs]
    stats: list[dict | None] = [None] * len(jobs)
    errors: list[BaseException] = []

    def run(k):
        prompt, sampling, tokens = jobs[k]
        if delays:
            time.sleep(delays[k])
        engine.request.policy = policy
        engine.request.stop_eos = False

        def emit(new):
            outs[k].extend(new)
            return stop_after is not None and k in stop_after and len(outs[k]) >= stop_after[k]

        try:
            stats[k] = engine.generate(list(prompt), tokens, sampling, emit)
        except BaseException as exc:      # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(k,)) for k in range(len(jobs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(600)
    assert not errors, errors
    return outs, stats


def _prompts(seed, sizes):
    rng = np.random.default_rng(seed)
    return [[int(t) for t in rng.integers(0, 1000, size=n)] for n in sizes]


def test_streams_equal_their_solo_replies(solo, multi):
    prompts = _prompts(31, (37, 80, 23))
    jobs = [(p, s, 24) for p, s in zip(prompts, SAMPLINGS)]
    want = [_solo(solo, p, s, n) for p, s, n in jobs]
    d = multi.multi
    widths: list[int] = []
    inner = d._round
    d._round = lambda lanes, depths=None, **kw: (widths.append(len(lanes)), inner(lanes, depths, **kw))[1]
    try:
        got, stats = _concurrent(multi, jobs)
    finally:
        del d._round
    assert got == want
    assert max(widths) == 3                                  # rounds really shared by all three streams
    assert all(st["parallel"] == 3 for st in stats)
    assert sum(st["drafted"] for st in stats) > 0           # DFlash2 drafted (the synthetic drafter rarely agrees)


def test_more_requests_than_slots_and_staggered_arrivals(solo, multi):
    prompts = _prompts(32, (41, 19, 66, 30, 52))
    samplings = SAMPLINGS + SAMPLINGS[:2]
    jobs = [(p, s, 20) for p, s in zip(prompts, samplings)]
    want = [_solo(solo, p, s, n) for p, s, n in jobs]
    got, _ = _concurrent(multi, jobs, delays=[0.0, 0.05, 0.4, 0.1, 0.9])
    assert got == want


def test_serial_policy_and_drafted_policies_side_by_side(solo, multi):
    prompts = _prompts(33, (29, 44))
    jobs = [(p, SAMPLINGS[0], 20) for p in prompts]
    want = [_solo(solo, p, s, n) for p, s, n in jobs]
    for policy in ("0", "f3", "2", "auto"):
        got, stats = _concurrent(multi, jobs, policy=policy)
        assert got == want, policy
        if policy == "0":
            assert all(st["drafted"] == 0 for st in stats)


def test_a_cancelled_stream_leaves_the_others_exact(solo, multi):
    prompts = _prompts(34, (35, 47, 28))
    jobs = [(p, s, 30) for p, s in zip(prompts, SAMPLINGS)]
    want = [_solo(solo, p, s, n) for p, s, n in jobs]
    got, _ = _concurrent(multi, jobs, stop_after={1: 5})
    assert got[0] == want[0] and got[2] == want[2]
    assert 5 <= len(got[1]) < 30 and got[1] == want[1][:len(got[1])]
    assert not multi.multi.lanes                                     # every slot free again


def test_a_kept_prompt_resumes_like_a_fresh_prefill(solo, multi):
    sampling = Sampling(5, 1.0, 20, 0.95)
    first = _prompts(35, (60,))[0]
    reply, stats = _concurrent(multi, [(first, sampling, 16)])
    after = first + reply[0] + [7, 8, 9]
    got, stats = _concurrent(multi, [(after, sampling, 16), (_prompts(36, (25,))[0], None, 16)])
    assert stats[0]["cached"] == len(first)             # resumes from first's own end state (Mia's 0008)
    assert got[0] == _solo(solo, after, sampling, 16)


def test_rank1_replays_rank0s_messages(models):
    """The ops rank 0 sent, applied by another decoder in order (as rank 1's ``follow`` does), leave it with the
    same streams' tokens: everything rank 1 derives (drafts, samples, placement checks) agrees. Both decoders start
    from empty pools."""

    a = _engine(models / "model", drafter=models / "dflash2", parallel=3, multi_options={"draft_graphs": False})
    b = _engine(models / "model", drafter=models / "dflash2", parallel=3, multi_options={"draft_graphs": False})
    try:
        b.scheduler.close()
        a.multi.log = []
        prompts = _prompts(37, (33, 51, 26))
        jobs = [(p, s, 18) for p, s in zip(prompts, SAMPLINGS)]
        got, _ = _concurrent(a, jobs, delays=[0.0, 0.2, 0.3])
        first = a.multi.kept[-1].ids if a.multi.kept else None
        got2, _ = _concurrent(a, [(prompts[1] + got[1] + [3, 4], SAMPLINGS[0], 12)])   # resumes a kept prompt
        d1 = b.multi
        seen: list[list[int]] = []
        finish = d1._finish

        def record(sid):
            seen.append(list(d1.lanes[sid].s.out))
            finish(sid)

        d1._finish = record
        for msg in a.multi.log:
            d1.apply(msg)
        assert sorted(seen) == sorted(got + got2)
        assert first is not None and [len(c.ids) for c in d1.kept] == [len(c.ids) for c in a.multi.kept]
        assert [(x.base, x.size) for x in d1.pool.extents] == [(x.base, x.size) for x in a.multi.pool.extents]
    finally:
        a.scheduler.close()


def test_exl3_checkpoint_and_no_draft_model(models):
    """An EXL3 checkpoint (trellis experts) under --parallel 2 without a draft model: one token a round for every
    stream, each reply its solo serial one."""

    solo_x = _engine(models / "exl3", drafter=models / "dflash2")
    want_jobs = [(p, s, 16) for p, s in zip(_prompts(38, (45, 31)), SAMPLINGS)]
    want = [_solo(solo_x, p, s, n) for p, s, n in want_jobs]
    del solo_x
    m = _engine(models / "exl3", parallel=2)
    try:
        assert m.multi.contexts is None
        got, stats = _concurrent(m, want_jobs)
        assert got == want and all(st["drafted"] == 0 for st in stats)
    finally:
        m.scheduler.close()


@pytest.mark.parametrize("graphs", ["1", "0"], ids=["graphs", "eager"])
def test_batched_verify_equals_solo(models, solo, graphs):
    """TF_GLM_MULTI_VERIFY=batched: one forward over every stream's rows (segmented KDA and DSA kernels), captured
    as a graph a window size or eager; every reply its solo serial one."""

    from tensorfold.families.glm5_next.cuda.verify_batched import BatchedVerify

    m = _with_env({"TF_GLM_MULTI_VERIFY": "batched", "TF_GLM_MULTI_GRAPHS": graphs},
                  lambda: _engine(models / "model", drafter=models / "dflash2", parallel=3,
                                  multi_options={"draft_graphs": False}))
    try:
        v = m.multi.verify
        assert isinstance(v, BatchedVerify) and bool(v.graphs) == (graphs == "1")
        prompts = _prompts(39, (37, 80, 23, 51))
        samplings = SAMPLINGS + [Sampling(9, 0.7, 40, 1.0, 0.05)]
        jobs = [(p, s, 24) for p, s in zip(prompts, samplings)]
        want = [_solo(solo, p, s, n) for p, s, n in jobs]
        got, _ = _concurrent(m, jobs, delays=[0.0, 0.0, 0.0, 0.3])
        assert got == want
        for policy in ("0", "f3"):
            got, _ = _concurrent(m, jobs[:3], policy=policy)
            assert got == want[:3], policy
        assert v.replays["graph" if graphs == "1" else "eager"] > 0
    finally:
        m.scheduler.close()


def test_batched_drafter_streams_equal_solo(models, solo):
    """TF_GLM_MULTI_DRAFT=batched (the default) with its CUDA graphs: --parallel 3 replies equal their solo serial
    ones, and every stream's rounds, drafts and accepted drafts equal TF_GLM_MULTI_DRAFT=serial's (each stream's
    proposals are its own context's), under the default policy and the noise-aware fnc7:0.3; block passes of
    several streams ran."""

    from tensorfold.families.glm5_next.cuda.dflash2_multi import MultiDrafter

    env = {"TF_GLM_MULTI_VERIFY": "batched"}
    b = _with_env({**env, "TF_GLM_MULTI_DRAFT": "batched"},
                  lambda: _engine(models / "model", drafter=models / "dflash2", parallel=3))
    s = _with_env({**env, "TF_GLM_MULTI_DRAFT": "serial"},
                  lambda: _engine(models / "model", drafter=models / "dflash2", parallel=3,
                                  multi_options={"draft_graphs": False}))
    try:
        md = b.multi.drafts
        assert isinstance(md, MultiDrafter) and sorted(md.block_graphs) == [1, 2, 3] and s.multi.drafts is None
        sizes: list[int] = []
        inner = md.candidates
        md.candidates = lambda reqs: (sizes.append(len(reqs)), inner(reqs))[1]
        jobs = [(p, sm, 24) for p, sm in zip(_prompts(43, (41, 90, 27)), SAMPLINGS)]
        want = [_solo(solo, p, sm, n) for p, sm, n in jobs]
        for policy in (None, "fnc7:0.3"):
            got, stats = _concurrent(b, jobs, policy=policy)
            ref, ref_stats = _concurrent(s, jobs, policy=policy)
            assert got == want and ref == want, policy
            # a stream's rounds follow from its own proposals alone here: depths are per-stream policies and the
            # windows (3 x 8 rows) never reach the 32-row cap that ``trim`` cuts by round composition
            assert [(st["rounds"], st["drafted"], st["accepted"]) for st in stats] == \
                [(st["rounds"], st["drafted"], st["accepted"]) for st in ref_stats], policy
        assert sum(st["drafted"] for st in stats) > 0 and max(sizes) >= 2
    finally:
        b.scheduler.close()
        s.scheduler.close()


def test_batched_verify_exl3(models):
    """The EXL3 checkpoint's trellis experts in batched windows (up to 32 rows) keep each stream's bits."""

    solo_x = _engine(models / "exl3", drafter=models / "dflash2")
    jobs = [(p, s, 20) for p, s in zip(_prompts(40, (45, 31, 60)), SAMPLINGS)]
    want = [_solo(solo_x, p, s, n) for p, s, n in jobs]
    del solo_x
    m = _with_env({"TF_GLM_MULTI_VERIFY": "batched"},
                  lambda: _engine(models / "exl3", drafter=models / "dflash2", parallel=3,
                                  multi_options={"draft_graphs": False}))
    try:
        got, _ = _concurrent(m, jobs)
        assert got == want
    finally:
        m.scheduler.close()


def test_batched_verify_past_the_dense_limit(models):
    """Streams past 2,051 tokens attend to DSA-selected tokens: the segmented selection and attention give each
    row the bits of its solo window (2 index heads, not GLM-5.3-Flash's 32)."""

    solo_l = _engine(models / "model", context=2600)
    prompts = _prompts(41, (2100, 2060, 300))
    jobs = [(p, s, 24) for p, s in zip(prompts, SAMPLINGS)]
    want = [_solo(solo_l, p, s, n) for p, s, n in jobs]
    del solo_l
    m = _with_env({"TF_GLM_MULTI_VERIFY": "batched", "TF_GLM_CACHE_GIB": "0.2"},
                  lambda: _engine(models / "model", context=2600, parallel=3))
    try:
        got, _ = _concurrent(m, jobs)
        assert got == want
    finally:
        m.scheduler.close()


def test_a_full_pool_gives_the_youngest_stream_back(models, solo):
    """A pool of two 2,048-token extents (TF_GLM_CACHE_GIB=0, TF_GLM_POOL_STREAMS=1): two long prompts both need a second extent near
    their ends, neither can grow, so the youngest goes back to the queue and replays after the other; a third
    request waits for room. Every reply is still its solo one (what a replay already sent is checked, not resent)."""

    # TF_GLM_POOL_STREAMS=1: the shared pool of one window (by default it holds every stream's full window)
    m = _with_env({"TF_GLM_CACHE_GIB": "0", "TF_GLM_POOL_STREAMS": "1"}, lambda: _engine(models / "model", parallel=3))
    try:
        assert m.multi.pool.rows == 4096
        prompts = _prompts(42, (2000, 2003, 40))
        jobs = [(p, s, 45) for p, s in zip(prompts, SAMPLINGS)]
        want = [_solo(solo, p, s, n) for p, s, n in jobs]
        d = m.multi
        d.log = []
        # Admit both long prompts in the SAME iteration so they really contend for the pool (a 0.5 s timer raced:
        # the worker could admit and grow the first before the second was queued, and no give-back fired). Gate the
        # first _admit until both requests are queued; later admits (the requeued stream's replay) are not gated.
        sched = m.scheduler
        orig_admit, once = sched._admit, [False]

        def gated_admit(first=None, until=None):
            if not once[0]:
                once[0] = True
                deadline = time.monotonic() + 30
                while sched.waiting.qsize() + (first is not None) < 2 and time.monotonic() < deadline:
                    time.sleep(0.005)
            return orig_admit(first, until)

        sched._admit = gated_admit
        try:
            got, stats = _concurrent(m, jobs[:2])
        finally:
            sched._admit = orig_admit
        got += _concurrent(m, jobs[2:])[0]
        assert got == want
        assert m.scheduler.yields >= 1                       # a stream was given back and replayed
        assert not m.multi.lanes
    finally:
        m.scheduler.close()


def test_a_queued_request_whose_client_left_is_dropped(models, solo, multi):
    """A request still waiting in the queue when its client leaves raises RequestCancelled without being admitted;
    the others are untouched."""

    from tensorfold.server.cancellation import RequestCancelled

    d = multi.multi
    gate = threading.Event()
    inner = d.round
    d.round = lambda: (gate.wait(30), inner())[1]
    admitted: list[int] = []
    admit = d.admit
    d.admit = lambda s: (admitted.append(len(s.prompt)), admit(s))[1]
    prompt = _prompts(43, (33,))[0]
    out: list[int] = []
    errors: list[BaseException] = []

    def leaver():
        multi.request.policy, multi.request.stop_eos = None, False
        multi.request.cancelled = lambda: True
        try:
            multi.generate([5] * 77, 10, None, lambda new: False)
        except BaseException as exc:     # noqa: BLE001
            errors.append(exc)

    def stayer():
        multi.request.policy, multi.request.stop_eos = None, False
        multi.request.cancelled = lambda: False
        multi.generate(prompt, 16, SAMPLINGS[0], lambda new: out.extend(new))

    try:
        a = threading.Thread(target=stayer)
        a.start()
        time.sleep(0.2)                  # the worker holds `stayer` (admitted, its round gated)
        b = threading.Thread(target=leaver)
        b.start()
        b.join(10)
        assert not b.is_alive() and len(errors) == 1 and isinstance(errors[0], RequestCancelled)
        gate.set()
        a.join(60)
        assert 77 not in admitted
        assert out == _solo(solo, prompt, SAMPLINGS[0], 16)
    finally:
        gate.set()
        del d.round, d.admit


@pytest.mark.parametrize("verify", ["serial", "batched"])
def test_multi_prefill_groups_prompts_and_stays_exact(models, solo, verify):
    """TF_GLM_MULTI_PREFILL=1 (Mia's 0049): prompts arriving together share prompt chunks (an idle server waits
    TF_GLM_MULTI_PREFILL_WAIT_MS for them); each reply is still its solo serial one, a kept prompt resumes, and rank
    1's replay of the MFILL messages gives the same streams."""

    env = {"TF_GLM_MULTI_PREFILL": "1", "TF_GLM_MULTI_PREFILL_WAIT_MS": "400", "TF_GLM_MULTI_VERIFY": verify}
    a = _with_env(env, lambda: _engine(models / "model", drafter=models / "dflash2", parallel=3,
                                       multi_options={"draft_graphs": False}))
    b = _with_env(env, lambda: _engine(models / "model", drafter=models / "dflash2", parallel=3,
                                       multi_options={"draft_graphs": False}))
    try:
        b.scheduler.close()
        assert a.multi.group and a.scheduler.gather_s == 0.4
        a.multi.log = []
        prompts = _prompts(41, (37, 150, 23))
        jobs = [(p, s, 20) for p, s in zip(prompts, SAMPLINGS)]
        want = [_solo(solo, p, s, n) for p, s, n in jobs]
        got, _ = _concurrent(a, jobs, delays=[0.0, 0.05, 0.1])
        assert got == want
        assert a.multi.grouped["chunks"] > 0 and a.multi.grouped["pieces"] >= 2
        after = prompts[1] + got[1] + [3, 4]
        got2, stats = _concurrent(a, [(after, SAMPLINGS[0], 12), (_prompts(42, (90,))[0], None, 12)])
        assert stats[0]["cached"] == len(prompts[1])    # resumes from prompts[1]'s own end state (Mia's 0008)
        assert got2[0] == _solo(solo, after, SAMPLINGS[0], 12)
        d1 = b.multi
        seen: list[list[int]] = []
        finish = d1._finish

        def record(sid):
            seen.append(list(d1.lanes[sid].s.out))
            finish(sid)

        d1._finish = record
        for msg in a.multi.log:
            d1.apply(msg)
        assert sorted(seen) == sorted(got + got2)
        assert [len(c.ids) for c in d1.kept] == [len(c.ids) for c in a.multi.kept]
    finally:
        a.scheduler.close()


@pytest.mark.parametrize("kind, window", [("model", "32"), ("exl3", "32"), ("model", "64")])
def test_eight_streams_batched(models, kind, window):
    """--parallel 8 (Mia's 0069): eight streams in one batched window (segment tables of 8), the default 32-row and
    a 64-row TF_GLM_MULTI_WINDOW; every reply is its solo serial one and a round really held all eight. The pool holds
    every stream's full window by default (TF_GLM_POOL_STREAMS)."""

    solo_k = _engine(models / kind, drafter=models / "dflash2")
    prompts = _prompts(50, (37, 80, 23, 51, 66, 29, 44, 70))
    samplings = (SAMPLINGS * 3)[:8]
    jobs = [(p, s, 20) for p, s in zip(prompts, samplings)]
    want = [_solo(solo_k, p, s, n) for p, s, n in jobs]
    del solo_k
    m = _with_env({"TF_GLM_MULTI_VERIFY": "batched", "TF_GLM_MULTI_WINDOW": window},
                  lambda: _engine(models / kind, drafter=models / "dflash2", parallel=8,
                                  multi_options={"draft_graphs": False}))
    try:
        d = m.multi
        assert d.window == int(window) and d.verify.segs == 8 and d.verify.b.rows == int(window)
        assert m.pool_streams == 8 and d.pool.rows >= 8 * 2048
        widths: list[int] = []
        inner = d._round
        d._round = lambda lanes, depths=None, **kw: (widths.append(len(lanes)), inner(lanes, depths, **kw))[1]
        gate = threading.Event()
        rnd = d.round

        def held():                                      # the worker starts once all eight requests are admitted
            gate.wait(30)
            return rnd()

        d.round = held
        threading.Timer(1.0, gate.set).start()
        try:
            got, stats = _concurrent(m, jobs)
        finally:
            del d._round, d.round
        assert got == want
        assert max(widths) == 8
        assert all(st["parallel"] == 8 for st in stats)
    finally:
        m.scheduler.close()


def test_parallel_refuses_a_context_n_streams_cannot_reach(models, monkeypatch):
    """Before any weight loads: --parallel 8 at a --context whose eight full windows do not fit the budget refuses
    on every rank with a message naming --parallel and TF_GLM_POOL_STREAMS; TF_GLM_POOL_STREAMS=1 (a shared pool)
    admits the same context in the same budget."""

    from tensorfold.cuda import capacity
    from tensorfold.families.glm5_next.cuda import engine as eng
    from tensorfold.families.glm5_next.cuda import weights

    calls = []
    real_load = weights.load

    def no_load(*a, **kw):
        calls.append(1)
        return real_load(*a, **kw)

    monkeypatch.setattr(weights, "load", no_load)
    text = __import__("tensorfold.cuda.capacity", fromlist=["config"]).config(models / "model")
    one = eng.multi_geometry(text, 2, 8, streams=1, window=32)
    eight = eng.multi_geometry(text, 2, 8, streams=8, window=32)
    ctx = 65536
    assert eight.bytes_at(ctx) - one.bytes_at(ctx) > 7 * eng.pool_cache_bytes(text, 2, ctx)
    from tensorfold.cuda.geometry import split_weights
    from tensorfold.families.glm5_next.cuda.split import geometry_rule

    w_est = capacity.estimate_weights(models / "model", eng.without_mtp(split_weights(geometry_rule, 2), 2)).resident
    # a budget between the one-window and eight-window needs (the drafter's share is small beside the gap)
    budget = int(w_est + (one.needed(ctx) + eight.needed(ctx)) // 2 + (64 << 20))
    monkeypatch.setattr(capacity, "available_bytes", lambda torch: budget)
    with pytest.raises(ValueError) as err:
        _engine(models / "model", parallel=8, context=ctx)
    assert "--parallel 8" in str(err.value) and "TF_GLM_POOL_STREAMS" in str(err.value)
    assert not calls                                     # refused before the weights load
    m = _with_env({"TF_GLM_POOL_STREAMS": "1"}, lambda: _engine(models / "model", parallel=8, context=ctx))
    try:
        assert calls and m.pool_streams == 1
    finally:
        m.scheduler.close()


def test_shared_prefix_streams_resume_and_an_identical_resend_replays(models, solo):
    """TF_GLM_SHARED_PREFIX under --parallel: a second chat keeps a state where its prompt stops sharing the first's
    (the system block); a third chat sharing only that prefix resumes from it (cached == the shared point) and its
    reply equals its solo one; an identical resend of it replays the kept head (cached == len(prompt))."""

    e = _with_env({"TF_GLM_SHARED_PREFIX": "32"},
                  lambda: _engine(models / "model", drafter=models / "dflash2", parallel=2,
                                  multi_options={"draft_graphs": False}))
    try:
        system = _prompts(61, (59,))[0]
        a = system + _prompts(62, (40,))[0]
        b = system + _prompts(63, (30,))[0]
        _concurrent(e, [(a, SAMPLINGS[0], 16)])                   # chat A, kept end state
        _concurrent(e, [(b, SAMPLINGS[1], 16)])                   # chat B keeps a shared state at the system block
        assert 59 in {len(c.ids) for c in e.multi.kept}           # the shared-prefix state at the system block
        c = system + _prompts(64, (20,))[0]                       # chat C shares only the system block
        gc, sc = _concurrent(e, [(c, SAMPLINGS[2], 16)])
        assert sc[0]["cached"] == 59
        assert gc[0] == _solo(solo, c, SAMPLINGS[2], 16)
        gc2, sc2 = _concurrent(e, [(c, SAMPLINGS[2], 16)])        # an identical resend replays the kept head
        assert sc2[0]["cached"] == len(c) and gc2[0] == gc[0]
    finally:
        e.scheduler.close()

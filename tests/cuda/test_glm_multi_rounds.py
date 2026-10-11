"""Mia's 0035 multi rounds (``multi_tune``) wired into --parallel: the packed sampler, the joint and scaled draft
depths, async rank messages, and a lone stream on the one-stream CUDA graphs (SLOT: its slot state, index ring rows
and DFlash2 context moved to slot 0, its extent to the pool's first rows). None of them may change a reply: every
stream still equals its solo serial reply (``draft=False`` on a --parallel 1 engine), and rank 1's replay of rank
0's messages (SLOT included) gives the same streams. The packed sampler equals ``sample_streams`` on two real ranks
(``glm_two_ranks``: each rank a thread with its own weights half and real logits).

One GPU plays rank 0 of two (``_TwoCopies``), as in test_glm_multi."""

from __future__ import annotations

import os
import threading

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_engine import _checkpoint, _drafter  # noqa: E402
from test_glm_multi import SAMPLINGS, _concurrent, _engine, _prompts, _solo, _with_env  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

os.environ.setdefault("TF_GLM_CACHE_GIB", "0.05")

MIN_P = Sampling(9, 0.7, 40, 1.0, 0.05)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_multi_rounds")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    return path


def _settings(**kw):
    from tensorfold.families.glm5_next.cuda.multi_tune import MultiSettings

    return MultiSettings(**{"lone": False, **kw})


def test_round_settings_leave_every_reply_unchanged(models, capsys):
    """Packed sampler, joint depth on the startup-timed batched window (``_time_rows``), async messages, the profile
    (``_compare_paths``) and lone mode together on the batched seam: every reply (greedy, top-k, nucleus, min_p)
    equals its solo serial one, a stream alone and streams together."""

    solo = _engine(models / "model", drafter=models / "dflash2")
    jobs = [(p, s, 24) for p, s in zip(_prompts(51, (37, 80, 23, 51)), SAMPLINGS + [MIN_P])]
    want = [_solo(solo, p, s, n) for p, s, n in jobs]
    del solo
    tune = _settings(sampler="packed", depth="joint", async_msg=True, profile=8, lone=True)
    m = _with_env({"TF_GLM_MULTI_VERIFY": "batched"},
                  lambda: _engine(models / "model", drafter=models / "dflash2", parallel=3,
                                  multi_options={"draft_graphs": False, "tune": tune}))
    try:
        d = m.multi
        assert d.row_ms is not None and len(d.row_ms) == d.window
        assert "one-stream graphs vs batched graphs" in capsys.readouterr().out
        got, stats = _concurrent(m, jobs[:1])
        assert got == want[:1]
        got, stats = _concurrent(m, jobs, delays=[0.0, 0.0, 0.1, 0.3])
        assert got == want
        assert sum(st["drafted"] for st in stats) > 0
    finally:
        m.scheduler.close()


def _ops(log, op):
    from tensorfold.families.glm5_next.cuda.multi import MultiDecoder

    return [p for msg in log for o, p in MultiDecoder.parse(msg) if o == op]


def _gated(engine, jobs, gate):
    """jobs[0] and jobs[1] at once, jobs[2] once ``gate`` is set; each job's tokens."""

    outs: list[list[int]] = [[] for _ in jobs]
    errors: list[BaseException] = []

    def run(k):
        prompt, sampling, tokens = jobs[k]
        if k == 2:
            assert gate.wait(120)
        engine.request.policy = None
        engine.request.stop_eos = False
        try:
            engine.generate(list(prompt), tokens, sampling, lambda new: outs[k].extend(new))
        except BaseException as exc:      # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(k,)) for k in range(len(jobs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(600)
    assert not errors, errors
    return outs


def _lone_run(models, solo_engine, *, kv=None, context=None, sizes=(30, 44, 36), counts=(6, 90, 130)):
    """A (short) and B (long) together, B goes on alone (homes: SLOT from slot 1, MOVE to base 0), C joins it on
    the batched path, B finishes, C goes on alone (homes again). Every reply equals its solo one; rank 1's replay of
    the messages (another decoder) gives the same streams."""

    from tensorfold.families.glm5_next.cuda.multi import MOVE, SLOT

    extra = {} if kv is None else {"kv": kv}
    if context is not None:
        extra["context"] = context
    env = {"TF_GLM_MULTI_VERIFY": "batched", **({"TF_GLM_CACHE_GIB": "0.2"} if context else {})}
    make = lambda: _engine(models / "model", drafter=models / "dflash2", parallel=3,   # noqa: E731
                           multi_options={"draft_graphs": False, "tune": _settings(lone=True)}, **extra)
    a = _with_env(env, make)
    b = _with_env(env, make)
    try:
        b.scheduler.close()
        prompts = _prompts(53, sizes)
        samplings = [SAMPLINGS[0], SAMPLINGS[2], SAMPLINGS[1]]
        jobs = [(p, s, n) for p, s, n in zip(prompts, samplings, counts)]
        want = [_solo(solo_engine, p, s, n) for p, s, n in jobs]
        d = a.multi
        d.log = []
        before = dict(a.e.replays)
        rounds = {"lone": 0, "batched": 0}
        inner = d._round

        def spy(lanes, depths=None, **kw):
            rounds["lone" if kw.get("lone") else "batched"] += 1
            return inner(lanes, depths, **kw)

        d._round = spy
        lone = threading.Event()

        def spy_lone(lanes, depths=None, **kw):
            if kw.get("lone") and rounds["lone"] >= 5:
                lone.set()                                 # C arrives once B has decoded alone at home a while
            return spy(lanes, depths, **kw)

        d._round = spy_lone
        moved = []
        reslot = d._reslot

        def checked(lane, slot):
            """The lane's slot state before and after the move: KDA states, conv windows, its ring rows (the pools
            still filling), its DFlash2 context (the same object, now at ``slot``)."""
            st = lane.st
            before = [st.rec.clone(), st.conv.clone()] + [t.clone() for i in (st.index or []) for t in i[:2]]
            ctx = d.contexts[lane.slot]
            reslot(lane, slot)
            st = lane.st
            after = [st.rec, st.conv] + [t for i in (st.index or []) for t in i[:2]]
            assert lane.slot == slot and st.slot == slot and d.contexts[slot] is ctx
            assert len(before) == len(after) and all(torch.equal(x, y) for x, y in zip(before, after))
            moved.append(len(before))

        d._reslot = checked
        try:
            got = _gated(a, jobs, lone)
        finally:
            del d._round, d._reslot
        assert moved and all(n == moved[0] for n in moved)
        if a.e.caches.rings:
            assert moved[0] > 2                            # ring rows were compared
        assert got == want
        slots, moves = _ops(d.log, SLOT), _ops(d.log, MOVE)
        assert len(slots) >= 2, f"B and then C move into slot 0: {slots}"
        assert rounds["lone"] > 0 and rounds["batched"] > 0
        main = a.e.replays["main"] - before.get("main", 0) + a.e.replays.get("sparse", 0) - before.get("sparse", 0)
        assert main > 0                                    # lone windows replayed the one-stream graphs
        d1 = b.multi
        seen: list[list[int]] = []
        finish = d1._finish

        def record(sid):
            seen.append(list(d1.lanes[sid].s.out))
            finish(sid)

        d1._finish = record
        for msg in d.log:
            d1.apply(msg)
        assert sorted(seen) == sorted(got)
        assert [(x.base, x.size) for x in d1.pool.extents] == [(x.base, x.size) for x in d.pool.extents]
        return a, slots, moves, rounds
    finally:
        a.scheduler.close()


def test_a_lone_stream_past_the_dense_limit_with_fp8_index_rings_and_chunked_kda(models, monkeypatch):
    """The launcher's --parallel setting: TF_GLM_KV=fp8 (index rings on) and TF_GLM_KDA_CHUNKED=1 (prompt chunks and
    kept states on the 64-token grid). A stream past the dense limit moves slots, its ring rows (the pools still
    filling) with it, and its sparse selection stays its solo (chunked, fp8) one."""

    from tensorfold.families.glm5_next.cuda import kda

    monkeypatch.setattr(kda, "CHUNKED", True)
    solo_l = _engine(models / "model", drafter=models / "dflash2", context=2600, kv="fp8")
    try:
        assert solo_l.grid == 64
        a, slots, moves, rounds = _lone_run(models, solo_l, kv="fp8", context=2600, sizes=(2090, 2110, 2075),
                                            counts=(6, 80, 120))
        assert a.e.caches.rings, "fp8 caches without index rings"
        assert a.multi.grid == 64
        print(f"[measure] fp8 lone run: SLOT {slots}, MOVE {moves}, rounds {rounds}")
    finally:
        del solo_l


def test_packed_equals_streams_on_two_real_ranks(models, monkeypatch):
    """``sample_packed`` == ``sample_streams`` token for token on two ranks (threads, each its weights half and
    its real logits shard of a few decode windows), greedy, top-k, nucleus and min_p, nucleus union on and off; the
    packed sampler makes fewer gathers."""

    from glm_two_ranks import both, load_pair

    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill_chunk
    from tensorfold.families.glm5_next.cuda.multi import sample_streams
    from tensorfold.families.glm5_next.cuda.multi_tune import sample_packed

    ws, hub = load_pair(models / "model")
    prompts = _prompts(54, (40, 23, 61, 33))
    samplings = [None, SAMPLINGS[0], SAMPLINGS[1], MIN_P, Sampling(11, 1.0, 0, 0.5)]

    def logits(w):
        e = Engine(w, capacity=256, max_rows=8, prefill_rows=128)
        out = []
        for i, p in enumerate(prompts):
            e.reset()
            prefill_chunk(e, p, 0, len(p), head=False)
            out.append(e.forward([int(t) for t in np.arange(1 + i % 4) * 7 + 3]).clone())
        return out

    rows = both(hub, *[lambda r=r: logits(ws[r]) for r in (0, 1)])
    for union in ("0", "1"):
        monkeypatch.setenv("TENSORFOLD_NUCLEUS_UNION", union)
        for shift in (0,):
            def run(r):
                w = ws[r]
                parts = [(x, [100 + j for j in range(x.shape[0])], samplings[(k + shift) % len(samplings)])
                         for k, x in enumerate(rows[r])]
                c0 = w.comm.calls
                a = sample_streams(w, parts)
                c1 = w.comm.calls
                b = sample_packed(w, parts)
                return a, b, c1 - c0, w.comm.calls - c1

            (a0, b0, s0, p0), (a1, b1, s1, p1) = both(hub, lambda: run(0), lambda: run(1))
            assert a0 == b0 == a1 == b1, (union, shift)
            assert p0 == p1 and s0 == s1 and p0 <= s0

"""TF_GLM_KDA_CHUNKED=1 (Mia's chunked KDA prompt kernel) under --parallel: prompt chunks start on the 64-token grid
and kept prompt states sit on it, so with the setting on, every stream equals its solo reply on a --parallel 1 engine
with the same setting (``draft=False``: drafted == serial), a stream resumed from a kept or shared-prefix point equals
a fresh prefill, an identical resend replays its kept head, and rank 1's replay of rank 0's messages agrees; FP8 KV
on. One scenario covers a multi-prompt chunk (MFILL), a sliced fill (FILLPART) and both kinds of kept point.

``kda.CHUNKED`` is process-wide and read at call time: this module's engines are built and run with it set, and it
is restored at teardown, so no engine of another module runs under it."""

from __future__ import annotations

import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_engine import _checkpoint, _drafter  # noqa: E402
from test_glm_multi import SAMPLINGS, _engine, _prompts, _solo, _with_env  # noqa: E402

os.environ.setdefault("TF_GLM_CACHE_GIB", "0.05")

ENV = {"TF_GLM_MULTI_PREFILL": "1", "TF_GLM_FILL_BUDGET_MS": "0.01", "TF_GLM_FILL_ROWS": "64",
       "TF_GLM_SHARED_PREFIX": "32", "TF_GLM_MULTI_VERIFY": "batched"}


@pytest.fixture(scope="module")
def engines(tmp_path_factory):
    """(solo, a, b): a chunked --parallel 1 engine and two chunked --parallel 3 ones (rank 0 and its replayer)."""

    from tensorfold.families.glm5_next.cuda import kda

    path = tmp_path_factory.mktemp("glm_multi_chunked")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    old = kda.CHUNKED
    kda.CHUNKED = True
    made = []
    try:
        solo = _engine(path / "model", drafter=path / "dflash2", kv="fp8")
        for _ in range(2):
            made.append(_with_env(ENV, lambda: _engine(path / "model", drafter=path / "dflash2", parallel=3,
                                                       kv="fp8", multi_options={"draft_graphs": False})))
        assert solo.grid == 64 and all(e.multi.grid == 64 for e in made)
        yield solo, made[0], made[1]
    finally:
        for e in made:
            e.scheduler.close()
        kda.CHUNKED = old


def _stream(prompt, count, sampling):
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.glm5_next.cuda.engine import encode_policy

    s = Stream(list(prompt), count, sampling, draft=True, stop_eos=False)
    s.emit = lambda new: False
    s.glm = {"code": encode_policy("auto")}          # DFlash2 drafts under --parallel
    return s


def _drive(d, streams, *, late=None, rounds=2000):
    """Rank 0's iterations until every stream is done; ``late``: (stream, n) admitted after n iterations."""

    for s in streams:
        d.admit(s)
    pending = list(late or [])
    for i in range(rounds):
        for s, at in [p for p in pending if p[1] == i]:
            d.admit(s)
            pending.remove((s, at))
        d.finish(d.round())
        if not pending and all(s.done for s in streams + [s for s, _ in late or []]):
            return
    raise AssertionError("the streams did not finish")


def test_chunked_kda_streams_equal_solo_resume_on_the_grid_and_ranks_agree(engines):
    from tensorfold.families.glm5_next.cuda.multi import FILLPART, MFILL, MFILLPART

    solo, a, b = engines
    d = a.multi
    d.log = []
    system = _prompts(71, (140,))[0]
    p0 = _prompts(72, (40,))[0]                        # off-grid end, nothing kept: decodes long beside the others
    p1 = system + _prompts(73, (60,))[0]               # 200: keeps 192 (its last grid point)
    p2 = system + _prompts(74, (110,))[0]              # 250, arrives while others decode: shared point 128 (140 -> grid)
    p3 = system + _prompts(75, (52,))[0]               # 192: resumes at the shared 128, keeps its end + head
    s0, s1 = _stream(p0, 120, SAMPLINGS[0]), _stream(p1, 16, SAMPLINGS[1])
    s2 = _stream(p2, 16, SAMPLINGS[2])
    _drive(d, [s0, s1], late=[(s2, 3)])
    s3 = _stream(p3, 16, SAMPLINGS[0])
    _drive(d, [s3])
    s4 = _stream(p3, 16, SAMPLINGS[0])                 # an identical resend: replays the kept head
    _drive(d, [s4])
    streams = [s0, s1, s2, s3, s4]
    jobs = [(p0, SAMPLINGS[0], 120), (p1, SAMPLINGS[1], 16), (p2, SAMPLINGS[2], 16), (p3, SAMPLINGS[0], 16)]
    want = [_solo(solo, p, s, n) for p, s, n in jobs]
    assert [s.out for s in streams] == want + [want[3]]
    assert [s.cached for s in streams] == [0, 0, 0, 128, 192]
    assert d.kept and all(len(c.ids) % 64 == 0 for c in d.kept)
    ops = {op for msg in d.log for op, _ in d.parse(msg)}
    assert MFILL in ops and (FILLPART in ops or MFILLPART in ops)     # a multi-prompt chunk and a sliced fill ran
    assert d.grouped["pieces"] >= 2 and sum(s.drafted for s in streams) > 0

    # resume == fresh, bit for bit: p3's end state (resumed at 128, filled to 192) and its kept head row are a fresh
    # one-chunk prefill's (an off-grid resume would pass the token checks above on this tiny model, not this one)
    from tensorfold.families.glm5_next.cuda.decode import prefill_chunk

    kept = next(c for c in d.kept if c.ids == p3)
    e = solo.e
    prev = e.use(e.home)
    try:
        e.reset()
        head = prefill_chunk(e, p3, 0, len(p3), head=True)
        st = e.st
        assert torch.equal(kept.rec, st.rec[st.cur[0]]) and torch.equal(kept.conv, st.conv)
        assert torch.equal(kept.head, head[:1])
    finally:
        e.reset()
        e.use(prev)

    d1 = b.multi                                       # rank 1: the same messages give the same streams
    seen: list[list[int]] = []
    finish = d1._finish

    def record(sid):
        seen.append(list(d1.lanes[sid].s.out))
        finish(sid)

    d1._finish = record
    for msg in d.log:
        d1.apply(msg)
    assert sorted(seen) == sorted(s.out for s in streams)
    assert [len(c.ids) for c in d1.kept] == [len(c.ids) for c in d.kept]


def test_chunked_kda_pieces_equal_chunks_of_their_own(engines):
    """A multi-prompt chunk's pieces on the 64-row grid get the bits of chunks of their own (states, taps, heads)."""

    from test_glm_multi_prefill import _same, _solo_and_grouped

    solo, _, _ = engines
    prompts = _prompts(76, (150, 300, 23))
    _same(*_solo_and_grouped(solo.w, prompts, cap=2560, rows=64, long_context=False))

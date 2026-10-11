"""Image prompts under --parallel (Mia's 0030 vision feed, ``multi.py``) on the tiny synthetic checkpoint of
``test_glm_engine``: an image prompt's reply equals the one it gets alone from a --parallel 1 engine with
``draft=False``, beside text streams, with multi-prompt chunks (0049) and sliced fills (0062); rank 1's replay of
rank 0's messages builds the same feeds from the ADMIT's positions.

The tower is a stub giving fixed feature rows; one GPU plays rank 0 of two (``_TwoCopies``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_multi import SAMPLINGS, _concurrent, _engine, _prompts, _solo, _with_env, models, solo  # noqa: E402,F401
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402


def _tower(engine, seed: int = 9):
    """A stub tower on ``engine``: every picture row gets a fixed random feature row (the same on every engine)."""
    hidden = engine.w.cfg.hidden
    g = torch.Generator().manual_seed(seed)
    table = torch.randn(4096, hidden, generator=g).to(torch.bfloat16)

    def features(prepared):
        return table[torch.tensor(prepared.positions)].to("cuda")

    engine.vision = SimpleNamespace(features=features)
    return table


def _picture(prompt, positions):
    return SimpleNamespace(positions=list(positions), token_ids=list(prompt))


def _solo_image(engine, prompt, positions, sampling, tokens):
    out: list[int] = []
    engine.request.policy = None
    engine.request.stop_eos = False
    engine.generate(list(prompt), tokens, sampling, lambda new: out.extend(new), draft=False,
                    vision=_picture(prompt, positions))
    return out


def _concurrent_images(engine, jobs, visions, **kw):
    """``_concurrent`` with ``visions[k]`` (positions or None) passed as job k's pictures."""
    inner = engine.generate

    def generate(prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None, **rest):
        k = next((i for i, (p, _, _) in enumerate(jobs) if list(p) == list(prompt)), None)
        vision = _picture(prompt, visions[k]) if k is not None and visions[k] is not None else None
        return inner(prompt, max_tokens, sampling, on_tokens, draft, constraint, vision=vision, **rest)

    engine.generate = generate
    try:
        return _concurrent(engine, jobs, **kw)
    finally:
        del engine.generate


@pytest.fixture(scope="module")
def solo_tower(solo):
    _tower(solo)
    yield solo
    solo.vision = None


@pytest.mark.parametrize("verify", ["serial", "batched"])
def test_image_prompt_beside_text_streams_equals_its_solo_reply(models, solo_tower, verify):
    a = _with_env({"TF_GLM_MULTI_VERIFY": verify},
                  lambda: _engine(models / "model", drafter=models / "dflash2", parallel=3,
                                  multi_options={"draft_graphs": False}))
    try:
        _tower(a)
        prompts = _prompts(51, (70, 45, 33))
        visions = [list(range(10, 26)), None, [3, 4, 5, 20]]
        jobs = [(p, s, 20) for p, s in zip(prompts, SAMPLINGS)]
        want = [_solo_image(solo_tower, p, v, s, n) if v is not None else _solo(solo_tower, p, s, n)
                for (p, s, n), v in zip(jobs, visions)]
        assert want[0] != _solo(solo_tower, prompts[0], SAMPLINGS[0], 20)     # the rows really change the reply
        got, stats = _concurrent_images(a, jobs, visions, delays=[0.0, 0.05, 0.1])
        assert got == want
        assert all(c.ids != prompts[0][:len(c.ids)] for c in a.multi.kept)    # image prompts keep nothing
        # a later turn of the image conversation does not resume a kept prefix past its first picture row
        after = prompts[0] + got[0] + [7, 8]
        got2, stats2 = _concurrent_images(a, [(after, SAMPLINGS[0], 12)], [visions[0]])
        assert stats2[0]["cached"] == 0
        assert got2[0] == _solo_image(solo_tower, after, visions[0], SAMPLINGS[0], 12)
    finally:
        a.scheduler.close()


def test_image_prompts_with_multi_prefill_and_sliced_fills(models, solo_tower):
    env = {"TF_GLM_MULTI_PREFILL": "1", "TF_GLM_MULTI_PREFILL_WAIT_MS": "300", "TF_GLM_MULTI_VERIFY": "batched",
           "TF_GLM_FILL_BUDGET_MS": "0.01", "TF_GLM_FILL_ROWS": "64"}
    a = _with_env(env, lambda: _engine(models / "model", drafter=models / "dflash2", parallel=3,
                                       multi_options={"draft_graphs": False}))
    try:
        _tower(a)
        prompts = _prompts(52, (150, 40, 130))
        visions = [list(range(60, 90)), None, list(range(100, 120))]
        jobs = [(p, s, 16) for p, s in zip(prompts, SAMPLINGS)]
        want = [_solo_image(solo_tower, p, v, s, n) if v is not None else _solo(solo_tower, p, s, n)
                for (p, s, n), v in zip(jobs, visions)]
        d = a.multi
        sliced: list[bool] = []
        inner = d._fill

        def fill(lane, stop, *, layers=None):
            if lane.feed is not None:
                sliced.append(layers is not None)
            return inner(lane, stop, layers=layers)

        d._fill = fill
        try:
            got, _ = _concurrent_images(a, jobs, visions, delays=[0.0, 0.05, 0.1])
        finally:
            del d._fill
        assert got == want
        assert any(sliced)                       # an image lane's chunk really ran in layer slices (inject per slice)
    finally:
        a.scheduler.close()


def test_rank1_replay_builds_the_same_feeds(models, monkeypatch):
    """Rank 1 applies rank 0's ADMITs: an image stream gets a feed with the same positions and no rows of its own;
    its chunks take rank 0's rows (here from the stub's table, as the all-gather would), so its tokens agree."""

    from tensorfold.families.glm5_next.cuda import engine as eng

    a = _engine(models / "model", drafter=models / "dflash2", parallel=3, multi_options={"draft_graphs": False})
    b = _engine(models / "model", drafter=models / "dflash2", parallel=3, multi_options={"draft_graphs": False})
    try:
        b.scheduler.close()
        table = _tower(a)
        a.multi.log = []
        prompts = _prompts(53, (60, 38))
        visions = [list(range(5, 21)), None]
        jobs = [(p, s, 14) for p, s in zip(prompts, SAMPLINGS)]
        got, _ = _concurrent_images(a, jobs, visions, delays=[0.0, 0.05])

        chunk = eng.VisionFeed.chunk
        feeds = []

        def rank0_rows(self, start, count):
            if self.rows is None:                   # rank 1's feed: the rows rank 0 sends through the gather
                feeds.append(list(self.positions))
                self.rows = table[torch.tensor(self.positions)].to("cuda")
            return chunk(self, start, count)

        monkeypatch.setattr(eng.VisionFeed, "chunk", rank0_rows)
        d1 = b.multi
        seen: list[list[int]] = []
        finish = d1._finish

        def record(sid):
            seen.append(list(d1.lanes[sid].s.out))
            finish(sid)

        d1._finish = record
        for msg in a.multi.log:
            d1.apply(msg)
        assert sorted(seen) == sorted(got)
        assert feeds == [visions[0]]
    finally:
        a.scheduler.close()



def _forget_cache(engine) -> None:
    engine.cache.clear()
    engine.live = []


class _KeyedPrepared:
    """A GlmPrepared stand-in whose one picture carries a content key (Mia's 0054)."""

    def __init__(self, prompt, positions, key):
        self.token_ids = list(prompt)
        self.positions = list(positions)
        self.item_keys = [int(key)]
        self.item_rows = [len(positions)]

    def from_item(self, k: int) -> "_KeyedPrepared":
        p = _KeyedPrepared(self.token_ids, self.positions[sum(self.item_rows[:k]):], 0)
        p.item_keys, p.item_rows = self.item_keys[k:], self.item_rows[k:]
        return p

    def keyed_ids(self) -> list[int]:
        ids = list(self.token_ids)
        for pos in self.positions:
            ids[pos] = -1 - self.item_keys[0]
        return ids


def _gen_image(engine, prompt, prepared, sampling, tokens, draft=True):
    out: list[int] = []
    engine.request.policy = None
    engine.request.stop_eos = False
    stats = engine.generate(list(prompt), tokens, sampling, lambda new: out.extend(new), draft=draft,
                            vision=prepared)
    return out, stats


def test_serial_image_prompt_resumes_by_picture_content(models, solo_tower):
    """A second turn of an image conversation (same picture, more text) resumes the kept keyed state and gives the
    reply a fresh prefill gives; a different picture at the same token positions resumes nothing (Mia's 0054)."""

    e = solo_tower
    _forget_cache(e)
    pos = list(range(10, 26))
    prompt1 = _prompts(71, (40,))[0]
    sampling = Sampling(5, 1.0, 20, 0.95)
    r1, _ = _gen_image(e, prompt1, _KeyedPrepared(prompt1, pos, 111), sampling, 12)
    turn2 = prompt1 + r1 + [7, 8, 9]                         # the picture sits inside prompt1, before the resume point
    warm, stats = _gen_image(e, turn2, _KeyedPrepared(turn2, pos, 111), sampling, 12)
    assert stats["cached"] == len(prompt1)                   # resumed from the kept image state
    _forget_cache(e)
    cold, stats = _gen_image(e, turn2, _KeyedPrepared(turn2, pos, 111), sampling, 12)
    assert stats["cached"] == 0 and warm == cold
    _forget_cache(e)
    _gen_image(e, prompt1, _KeyedPrepared(prompt1, pos, 111), sampling, 12)
    diff = prompt1 + [7, 8, 9]
    _, stats = _gen_image(e, diff, _KeyedPrepared(diff, pos, 222), sampling, 12)
    assert stats["cached"] == 0                              # a different picture never matches the kept key
    _forget_cache(e)


def test_parallel_image_prompt_resumes_by_picture_content(models, solo_tower):
    """Under --parallel, a second turn of an image conversation resumes its kept keyed state and equals its solo."""

    a = _engine(models / "model", drafter=models / "dflash2", parallel=2, multi_options={"draft_graphs": False})
    try:
        _tower(a)
        pos = list(range(10, 26))
        prompt1 = _prompts(72, (40,))[0]
        sampling = Sampling(7, 1.0, 20, 0.95)

        def run(prompt, prep, tokens):
            inner = a.generate

            def generate(p, mt, s, ot, draft=True, constraint=None, **rest):
                return inner(p, mt, s, ot, draft, constraint, vision=prep, **rest)

            a.generate = generate
            try:
                return _concurrent(a, [(prompt, sampling, tokens)])
            finally:
                a.generate = inner

        (r1,), _ = run(prompt1, _KeyedPrepared(prompt1, pos, 111), 12)
        turn2 = prompt1 + r1 + [7, 8, 9]
        (got,), stats = run(turn2, _KeyedPrepared(turn2, pos, 111), 12)
        assert stats[0]["cached"] == len(prompt1)
        want, _ = _gen_image(solo_tower, turn2, _KeyedPrepared(turn2, pos, 111), sampling, 12, draft=False)
        assert got == want
    finally:
        a.scheduler.close()
        _forget_cache(solo_tower)

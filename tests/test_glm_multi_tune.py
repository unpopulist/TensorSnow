"""``glm5_next.cuda.multi_tune`` on the CPU (Mia's 0035): the settings' parsing and rank code, the joint draft-depth
allocation, and the packed sampler token for token against ``multi.sample_streams`` (one rank, and two ranks
simulated by threads over a fake all-gather)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda.multi import sample_streams
from tensorfold.families.glm5_next.cuda.multi_tune import (MultiSettings, RoundProfile, allocate, reach_of,
                                                           sample_packed, scaled_confidence)


# -- MultiSettings ----------------------------------------------------------------------------------------------------
def test_defaults():
    s = MultiSettings.from_env({})
    assert (s.sampler, s.depth, s.alpha, s.overhead_ms, s.async_msg, s.lone, s.profile) == \
        ("streams", "policy", 0.0, 12.0, False, True, 0)
    assert s == MultiSettings()
    assert s.code() == [0, 0, 0, 12000, 0, 1, 0]


def test_parse_every_setting():
    s = MultiSettings.from_env({"TF_GLM_MULTI_SAMPLER": " Packed ", "TF_GLM_MULTI_DEPTH": "scale:0.5",
                                "TF_GLM_MULTI_OVERHEAD_MS": "7.25", "TF_GLM_MULTI_ASYNC": "1",
                                "TF_GLM_MULTI_LONE": "0", "TF_GLM_MULTI_PROFILE": "50"})
    assert (s.sampler, s.depth, s.alpha, s.overhead_ms, s.async_msg, s.lone, s.profile) == \
        ("packed", "scale", 0.5, 7.25, True, False, 50)
    assert s.code() == [1, 2, 500, 7250, 1, 0, 1]
    assert MultiSettings.from_env({"TF_GLM_MULTI_DEPTH": "JOINT"}).depth == "joint"
    assert MultiSettings.from_env({"TF_GLM_MULTI_DEPTH": "scale:4"}).alpha == 4.0
    assert MultiSettings.from_env({"TF_GLM_MULTI_OVERHEAD_MS": "0"}).overhead_ms == 0.0   # 0 is allowed


@pytest.mark.parametrize("key,value", [
    ("TF_GLM_MULTI_SAMPLER", "fast"),
    ("TF_GLM_MULTI_DEPTH", "deep"),
    ("TF_GLM_MULTI_DEPTH", "scale:0"),
    ("TF_GLM_MULTI_DEPTH", "scale:5"),
    ("TF_GLM_MULTI_DEPTH", "scale:x"),
    ("TF_GLM_MULTI_DEPTH", "scale:-1"),
    ("TF_GLM_MULTI_OVERHEAD_MS", "-1"),
    ("TF_GLM_MULTI_OVERHEAD_MS", "2000"),
    ("TF_GLM_MULTI_OVERHEAD_MS", "abc"),
    ("TF_GLM_MULTI_ASYNC", "2"),
    ("TF_GLM_MULTI_ASYNC", "yes"),
    ("TF_GLM_MULTI_LONE", "x"),
    ("TF_GLM_MULTI_PROFILE", "-1"),
    ("TF_GLM_MULTI_PROFILE", "abc"),
    ("TF_GLM_MULTI_PROFILE", "1.5"),
])
def test_rejects(key, value):
    with pytest.raises(ValueError, match=key):
        MultiSettings.from_env({key: value})


def test_from_env_reads_os_environ(monkeypatch):
    monkeypatch.setenv("TF_GLM_MULTI_SAMPLER", "packed")
    monkeypatch.setenv("TF_GLM_MULTI_DEPTH", "joint")
    s = MultiSettings.from_env()
    assert (s.sampler, s.depth) == ("packed", "joint")


def test_code_tells_every_rank_setting_apart():
    """Every setting that changes collectives or drafts changes the code; the profile only by on/off (its period is
    rank 0's logging, which rank 1 need not share)."""

    base = MultiSettings.from_env({})
    variants = [{"TF_GLM_MULTI_SAMPLER": "packed"}, {"TF_GLM_MULTI_DEPTH": "joint"},
                {"TF_GLM_MULTI_DEPTH": "scale:1"}, {"TF_GLM_MULTI_DEPTH": "scale:2"},
                {"TF_GLM_MULTI_OVERHEAD_MS": "11.999"}, {"TF_GLM_MULTI_ASYNC": "1"}, {"TF_GLM_MULTI_LONE": "0"},
                {"TF_GLM_MULTI_PROFILE": "1"}]
    codes = [tuple(MultiSettings.from_env(v).code()) for v in variants] + [tuple(base.code())]
    assert len(set(codes)) == len(codes)
    assert all(len(c) == 7 and all(isinstance(x, int) for x in c) for c in codes)
    assert MultiSettings.from_env({"TF_GLM_MULTI_PROFILE": "1"}).code() == \
        MultiSettings.from_env({"TF_GLM_MULTI_PROFILE": "50"}).code()
    # the same settings from different spellings give the same code (both ranks agree)
    assert MultiSettings.from_env({"TF_GLM_MULTI_SAMPLER": "STREAMS", "TF_GLM_MULTI_OVERHEAD_MS": "12.0"}).code() == \
        base.code()


def test_code_round_trip():
    """The code is what the engine compares across ranks: settings rebuilt from its fields code alike."""

    for env in ({}, {"TF_GLM_MULTI_SAMPLER": "packed", "TF_GLM_MULTI_DEPTH": "scale:1.25",
                     "TF_GLM_MULTI_OVERHEAD_MS": "3.5", "TF_GLM_MULTI_ASYNC": "1", "TF_GLM_MULTI_LONE": "0",
                     "TF_GLM_MULTI_PROFILE": "9"}):
        s = MultiSettings.from_env(env)
        c = s.code()
        back = MultiSettings(sampler=("streams", "packed")[c[0]], depth=("policy", "joint", "scale")[c[1]],
                             alpha=c[2] / 1000, overhead_ms=c[3] / 1000, async_msg=bool(c[4]), lone=bool(c[5]),
                             profile=c[6])
        assert back.code() == c


def test_describe():
    assert MultiSettings.from_env({}).describe() == (
        "sampler streams, depth policy, async messages off, lone stream on the one-stream graphs on")
    d = MultiSettings.from_env({"TF_GLM_MULTI_SAMPLER": "packed", "TF_GLM_MULTI_DEPTH": "joint",
                                "TF_GLM_MULTI_OVERHEAD_MS": "8", "TF_GLM_MULTI_ASYNC": "1",
                                "TF_GLM_MULTI_LONE": "0", "TF_GLM_MULTI_PROFILE": "20"}).describe()
    for part in ("sampler packed", "depth joint (overhead 8 ms)", "async messages on", "graphs off",
                 "profile every 20 rounds"):
        assert part in d
    assert "depth scale:0.5," in MultiSettings.from_env({"TF_GLM_MULTI_DEPTH": "scale:0.5"}).describe()


# -- depth -----------------------------------------------------------------------------------------------------------
def test_scaled_confidence():
    assert scaled_confidence(0.3, 1, 1.0) == 0.3                   # one stream: as alone
    assert scaled_confidence(0.0, 4, 1.0) == 0.0                   # no threshold stays none
    assert scaled_confidence(0.3, 4, 0.0) == 0.3
    assert scaled_confidence(0.1, 4, 0.5) == pytest.approx(0.2)
    assert scaled_confidence(0.5, 8, 1.0) == 0.95                  # capped


def test_reach_of():
    assert reach_of([]) == []
    assert reach_of([0.5, 0.5, 1.0, 0.2]) == pytest.approx([0.5, 0.25, 0.25, 0.05])
    assert all(isinstance(x, float) for x in reach_of([0.9]))


def _linear(per_row: float = 1.0, base: float = 10.0):
    return lambda rows: base + per_row * rows


def test_allocate_basics():
    assert allocate([], 3, _linear(), 12.0, 32) == []
    # certain drafts on a flat cost: every one is worth its row, up to the cap
    take = allocate([[1.0] * 10, [1.0] * 10], 2, lambda r: 10.0, 12.0, 8)
    assert take == [6, 0]                                         # ties: the lower stream first
    # hopeless drafts on a steep cost: none
    assert allocate([[0.01] * 5, [0.01] * 5], 2, _linear(5.0), 12.0, 32) == [0, 0]
    # never past a chain's own length
    assert allocate([[1.0], [1.0] * 3], 2, lambda r: 1.0, 0.0, 32) == [1, 3]


def test_allocate_best_reach_first_and_ties_low_index():
    reach = [reach_of([0.5, 0.5, 0.5]), reach_of([0.9, 0.9, 0.9])]
    assert allocate(reach, 2, lambda r: 10.0 + 0.1 * r, 0.0, 3) == [0, 1]
    assert allocate([[0.8, 0.8], [0.8, 0.8]], 2, lambda r: 10.0, 0.0, 3) == [1, 0]


def test_allocate_deterministic_and_bounded():
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = int(rng.integers(1, 6))
        reach = [reach_of(rng.uniform(0.2, 1.0, int(rng.integers(0, 9))).tolist()) for _ in range(n)]
        fixed = n + int(rng.integers(0, 3))
        cap = int(rng.integers(n, 33))
        cost = _linear(float(rng.uniform(0.05, 2.0)), float(rng.uniform(5, 40)))
        over = float(rng.uniform(0, 30))
        take = allocate(reach, fixed, cost, over, cap)
        assert take == allocate([list(r) for r in reach], fixed, cost, over, cap)
        assert len(take) == n and all(0 <= t <= len(r) for t, r in zip(take, reach))
        assert fixed + sum(take) <= max(cap, fixed)
        # a stream's rows are a prefix of its chain, taken best reach first: a stream never takes a draft while a
        # better untaken one of another stream waits (reach is non-increasing along each chain)
        if any(take):
            worst_taken = min(r[t - 1] for r, t in zip(reach, take) if t)
            best_left = max((r[t] for r, t in zip(reach, take) if t < len(r)), default=-1.0)
            assert best_left <= worst_taken
        # a larger fixed overhead makes a round's rows cheaper relative to it: never fewer drafts
        assert sum(allocate(reach, fixed, cost, over + 50.0, cap)) >= sum(take)
        # a larger cap never takes fewer
        assert sum(allocate(reach, fixed, cost, over, cap + 4)) >= sum(take)


# -- the packed sampler ------------------------------------------------------------------------------------------------
def _parts(vocab: int, seed: int = 0):
    """A round's parts mixing every kind, interleaved (greedy, top-k, nucleus, min_p, short and long rows), in bf16 as
    the head gives them (ties included). Rows: peaked (the nucleus inside the candidates) and flat (past them)."""

    g = torch.Generator().manual_seed(seed)

    def rows(n, peaked):
        x = torch.randn((n, vocab), generator=g) * (2.0 if peaked else 0.05)
        if peaked:
            x[:, torch.randint(0, vocab, (3,), generator=g)] += 12.0
        return x.to(torch.bfloat16)

    S = Sampling
    specs = [
        (rows(3, True), None),
        (rows(4, True), S(seed=11, temperature=0.8, top_k=20, top_p=0.95)),
        (rows(2, True), S(seed=12, temperature=1.0, top_k=0, top_p=0.9)),
        (rows(1, False), S(seed=13, temperature=1.0, top_k=40, top_p=1.0, min_p=0.05)),
        (rows(3, False), S(seed=14, temperature=0.7, top_k=0, top_p=0.95, min_p=0.0)),
        (rows(1, True), S(seed=15, temperature=0.0, top_k=20)),            # temperature 0: greedy
        (rows(2, True), S(seed=16, temperature=1.2, top_k=0, top_p=1.0, min_p=0.1)),
        (rows(5, False), S(seed=17, temperature=1.0, top_k=0, top_p=0.5)),
    ]
    out, pos = [], 100
    for logits, s in specs:
        out.append((logits, list(range(pos, pos + logits.shape[0])), s))
        pos += 1000
    return out


def _w(comm=None, world=1, offset=0):
    return SimpleNamespace(comm=comm, world=world, vocab_offset=offset)


@pytest.mark.parametrize("vocab", [300, 5000])
@pytest.mark.parametrize("union", ["0", "1"])
def test_packed_equals_streams_one_rank(monkeypatch, vocab, union):
    monkeypatch.setenv("TENSORFOLD_NUCLEUS_UNION", union)
    parts = _parts(vocab, seed=vocab)
    want = sample_streams(_w(), parts)
    got = sample_packed(_w(), parts)
    assert got == want
    assert [len(r) for r in got] == [p[0].shape[0] for p in parts]
    # one part at a time as well (the offsets of a lone part)
    for p, w in zip(parts, want):
        assert sample_packed(_w(), [p]) == [w]


def test_packed_empty():
    assert sample_packed(_w(), []) == []


class _Ranks:
    """Two ranks as threads: ``all_gather(src, dst)`` fills dst with every rank's src, rank order."""

    def __init__(self, world: int) -> None:
        self.world = world
        self.bar = threading.Barrier(world)
        self.slots = [None] * world
        self.calls = [0] * world

    def comm(self, rank: int):
        ranks = self

        class Comm:
            world = ranks.world

            def all_gather(self, src, dst):
                ranks.calls[rank] += 1
                ranks.slots[rank] = src.clone()
                ranks.bar.wait()
                dst.copy_(torch.cat([s.reshape(-1) for s in ranks.slots]))
                ranks.bar.wait()

        c = Comm()
        c.rank = rank
        return c

    def run(self, fn):
        out, err = [None] * self.world, []

        def go(r):
            try:
                out[r] = fn(r)
            except BaseException as exc:          # noqa: BLE001 - surfaced below
                err.append(exc)
                self.bar.abort()

        threads = [threading.Thread(target=go, args=(r,)) for r in range(self.world)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if err:
            raise err[0]
        return out


@pytest.mark.parametrize("union", ["0", "1"])
def test_packed_equals_streams_two_ranks(monkeypatch, union):
    """Vocabulary split in two shards: the packed sampler gives every rank the same tokens as ``sample_streams`` over
    the same shards and as one rank over the whole vocabulary, in no more all-gathers than ``sample_streams``; with
    TENSORFOLD_NUCLEUS_UNION=1 and peaked rows (each nucleus inside the ranks' candidates together) in exactly two.
    Without the union rule a rank whose own candidates do not reach the whole row's nucleus mass reads whole shards
    (a third gather), exactly as ``nucleus_rows`` does alone."""

    monkeypatch.setenv("TENSORFOLD_NUCLEUS_UNION", union)
    vocab = 6000
    parts = _parts(vocab, seed=3)
    whole = sample_streams(_w(), parts)
    half = vocab // 2

    def shard(r):
        return [(l[:, r * half:(r + 1) * half].contiguous(), p, s) for l, p, s in parts]

    calls = {}
    for fn in (sample_streams, sample_packed):
        ranks = _Ranks(2)
        got = ranks.run(lambda r: fn(_w(ranks.comm(r), 2, r * half), shard(r)))
        assert got[0] == got[1] == whole, fn.__name__
        calls[fn.__name__] = ranks.calls[0]
    assert calls["sample_packed"] < calls["sample_streams"]

    # peaked rows only: two gathers for the whole round when the union rule covers every nucleus
    peaked = [p for p in parts if p[0].float().max() > 5]
    ranks = _Ranks(2)
    got = ranks.run(lambda r: sample_packed(_w(ranks.comm(r), 2, r * half),
                                            [(l[:, r * half:(r + 1) * half].contiguous(), p, s)
                                             for l, p, s in peaked]))
    assert got[0] == got[1] == sample_streams(_w(), peaked)
    assert ranks.calls[0] == ranks.calls[1]
    assert ranks.calls[0] >= 2 and (union == "0" or ranks.calls[0] == 2)


def test_round_profile_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)     # host timing only: no CUDA events
    lines = []
    prof = RoundProfile(2, log=lines.append, label="batched")
    for _ in range(2):
        prof.begin()
        for stage in RoundProfile.STAGES:
            prof.mark(stage)
        prof.gpu_start()
        prof.gpu_stop()
        prof.end(streams=2, rows=6, tokens=4, drafted=4)
    prof.fill(0.01)
    assert len(lines) == 1 and "multi profile (batched), 2 rounds: 2.00 streams, 6.0 rows (4.0 drafted), 4.00 tokens a round" in lines[0]
    assert prof.rounds == 0

"""TENSORFOLD_NUCLEUS_UNION=1 (Mia's patch 0034): ``_draw`` tests top_p coverage on every rank's candidates together.
It must give exactly the draws (token and share of the mass) the default per-rank test gives, while reading whole
shards less often. Ported from Mia's ``tools/test_nucleus_union.py`` (numpy emulation of ``_shares`` over 2-4 ranks),
plus a threaded two-rank check through the real ``nucleus_rows``."""

import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import sampling as cs  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402


def _np_shares(scaled, mass, count, ranks):
    """``_shares`` on numpy: per rank the top ``count`` (value, id, mass) per row, padded, plus shard sums and widths."""

    rows, vocab = scaled.shape
    cuts = np.linspace(0, vocab, ranks + 1).astype(int)
    vs, ids, ms, sums, wid = [], [], [], [], []
    for k in range(ranks):
        lo, hi = cuts[k], cuts[k + 1]
        v = np.full((rows, count), -np.inf)
        i = np.full((rows, count), -1, np.int64)
        m = np.zeros((rows, count), np.int64)
        for r in range(rows):
            order = np.lexsort((np.arange(lo, hi), -scaled[r, lo:hi]))[:count]   # topk (ties: lowest id first)
            n = len(order)
            v[r, :n], i[r, :n], m[r, :n] = scaled[r, lo:hi][order], order + lo, mass[r, lo:hi][order]
        vs.append(v), ids.append(i), ms.append(m)
        sums.append(mass[:, lo:hi].sum(axis=1)), wid.append(np.full(rows, hi - lo))
    return np.stack(vs), np.stack(ids), np.stack(ms), np.stack(sums), np.stack(wid)


def _np_nucleus(scaled, positions, smp, ranks):
    """``nucleus_rows``'s retry ladder on the emulated shards; returns the draws and the number of gathers."""

    top = scaled.max(axis=1)
    mass = np.floor(np.exp(scaled - top[:, None]) * cs.MASS).astype(np.int64)
    got = _np_shares(scaled, mass, cs.NUCLEUS, ranks)
    drawn, trips = cs._draw(got, positions, smp), 1
    if drawn is None and cs.union_cover() and int(got[4].max()) > cs.WIDER:
        drawn, trips = cs._draw(_np_shares(scaled, mass, cs.WIDER, ranks), positions, smp), 2
    if drawn is None:
        drawn, trips = cs._draw(_np_shares(scaled, mass, int(got[4].max()), ranks), positions, smp), trips + 1
    return drawn, trips


def test_union_cover_reads_the_env(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_NUCLEUS_UNION", raising=False)
    assert not cs.union_cover()
    monkeypatch.setenv("TENSORFOLD_NUCLEUS_UNION", "1")
    assert cs.union_cover()
    monkeypatch.setenv("TENSORFOLD_NUCLEUS_UNION", "0")
    assert not cs.union_cover()


def test_union_draws_what_the_per_rank_test_draws(monkeypatch):
    """Mia's emulation: random, concentrated, flat, tie-heavy and broad rows over 2-4 ranks, top_p and min_p."""

    rng = np.random.default_rng(7)
    vocab, cases = 20000, 40
    trips_off = trips_on = n = 0
    for case in range(cases):
        ranks = int(rng.integers(2, 5))
        kind = case % 5
        logits = rng.normal(size=(3, vocab)) * (1.0, 4.0, 0.3, 0.0, 1.0)[kind]
        if kind == 3:
            logits = np.round(logits, 0)
            logits[:, rng.integers(0, vocab, 8)] = 2.0                   # a few tied peaks
        if kind == 0:
            logits[:, :40] += 12.0                                       # concentrated
        if kind == 4:                                                    # ~4000 broad candidates over the ranks
            for r in range(3):
                logits[r, rng.choice(vocab, 4000, replace=False)] += 7.0
        temp = float((1.0, 0.7, 1.3)[case % 3])
        top_p = float((0.95, 0.9, 0.6, 0.99)[case % 4])
        min_p = float((0.0, 0.0, 0.05)[case % 3])
        smp = Sampling(int(rng.integers(1, 10**6)), temp, 0, top_p, min_p)
        scaled = logits / temp
        positions = [int(x) for x in rng.integers(0, 10**5, 3)]
        monkeypatch.setenv("TENSORFOLD_NUCLEUS_UNION", "0")
        a, ta = _np_nucleus(scaled, positions, smp, ranks)
        monkeypatch.setenv("TENSORFOLD_NUCLEUS_UNION", "1")
        b, tb = _np_nucleus(scaled, positions, smp, ranks)
        assert a == b, f"case {case} differs: {a} vs {b}"
        n += len(a)
        trips_off += ta
        trips_on += tb
    assert n == 3 * cases
    assert trips_on < trips_off, (trips_on, trips_off)                   # the point of the patch: fewer gathers


def test_union_covers_when_the_nucleus_spans_both_ranks():
    """Each rank alone cannot reach top_p inside its candidates, the two together can (the per-rank test says None)."""

    # two ranks, 4 candidates each out of a 6-wide shard; the nucleus is the top 4 values overall, 2 on each rank
    vals = np.array([[[5.0, 4.0, 1.0, 0.5]], [[5.0, 4.0, 1.0, 0.5]]])
    ids = np.array([[[0, 1, 2, 3]], [[6, 7, 8, 9]]])
    mass = np.array([[[100, 50, 2, 1]], [[100, 50, 2, 1]]], dtype=np.int64)
    sums = np.array([[154], [154]], dtype=np.int64)                     # one unsent token of mass 1 per rank
    widths = np.array([[6], [6]])
    need = 290                                                           # ~0.94 of 308: all four 100/50 tokens
    assert cs._union_covers(vals, ids, mass, widths, 0, need)
    assert not cs._union_covers(vals, ids, mass, widths, 0, 307)         # the unsent tokens are needed: not covered


def test_union_covers_whole_shards():
    vals = np.array([[[1.0, 0.0]], [[1.0, -np.inf]]])
    ids = np.array([[[0, 1]], [[2, -1]]])
    mass = np.array([[[10, 1]], [[10, 0]]], dtype=np.int64)
    widths = np.array([[2], [1]])                                        # both ranks sent every token
    assert cs._union_covers(vals, ids, mass, widths, 0, 21)


class Ranks:
    """Two ranks as threads: each ``gather`` stacks both ranks' words, rank 0 first."""

    def __init__(self):
        self.slots, self.barrier = [None, None], threading.Barrier(2, timeout=60)

    def gather(self, rank):
        def gather(words):
            self.slots[rank] = words.clone()
            self.barrier.wait()
            both = torch.stack(self.slots)
            self.barrier.wait()
            return both

        return gather


def _two_ranks(logits, positions, sampling, split, probs):
    ranks, out = Ranks(), [None, None]
    shards = (logits[:, :split], logits[:, split:])

    def run(r):
        out[r] = cs.nucleus_rows(shards[r], positions, sampling, offset=0 if r == 0 else split,
                                 gather=ranks.gather(r), probs=probs if r == 0 else None)

    threads = [threading.Thread(target=run, args=(r,)) for r in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out[0] == out[1]
    return out[0]


@pytest.mark.parametrize("top_p", [0.6, 0.95])
@pytest.mark.parametrize("min_p", [0.0, 0.05])
def test_real_nucleus_rows_two_ranks_union_on_and_off(monkeypatch, top_p, min_p):
    """Through ``nucleus_rows`` on two threaded ranks: union on draws what union off (and one rank) draws."""

    reads = []
    real = cs._shares
    monkeypatch.setattr(cs, "_shares", lambda *a: reads.append(a[3]) or real(*a))
    whole_off = whole_on = 0
    for seed in range(8):
        g = torch.Generator().manual_seed(seed)
        logits = (torch.randn(4, 40000, generator=g) * (0.5 if seed % 2 else 2.0)).to(torch.bfloat16)
        s = Sampling(seed * 31 + 1, 1.0, 0, top_p, min_p)
        positions = [10 * seed + r for r in range(4)]
        monkeypatch.setenv("TENSORFOLD_NUCLEUS_UNION", "0")
        one = cs.nucleus_rows(logits, positions, s)
        reads.clear()
        p_off = []
        off = _two_ranks(logits, positions, s, 20000, p_off)
        whole_off += reads.count(20000)
        monkeypatch.setenv("TENSORFOLD_NUCLEUS_UNION", "1")
        reads.clear()
        p_on = []
        on = _two_ranks(logits, positions, s, 20000, p_on)
        whole_on += reads.count(20000)
        assert on == off == one
        assert p_on == p_off
    assert whole_on <= whole_off

"""The kept-prompt cap under --parallel (Mia's 0063 and 0077, ``multi.cap_victim``): past TF_GLM_CACHE_ENTRIES a
conversation's superseded (shorter) states go first, then the least recently used; a shared-prefix state is never
dropped as superseded, only by recency. CPU only: ``MultiDecoder._keep`` on the real ``Pool`` with a stubbed snapshot,
in the style of Mia's tools/kept_cap_check.py. Below, her kept_cap_check.py (0097: ``engine.kept_entries``,
``multi.known_key``) and kept_state_check.py (0089: TF_GLM_KEEP_PER_CHAT, TF_GLM_KEPT_BYTES_GIB, and which drops the
spill tier writes) as tests."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
import torch

import tensorfold.families.glm5_next.cuda.decode as D
import tensorfold.families.glm5_next.cuda.multi as M
from tensorfold.families.glm5_next.cuda.pool import ALIGN, Pool, align_up


def _c(extent, n, shared=False):
    return NS(extent=extent, ids=list(range(n)), shared=shared)


def test_superseded_states_go_first_oldest_first():
    a, b = object(), object()
    kept = [_c(a, 100), _c(b, 50), _c(a, 40), _c(a, 200), _c(b, 80)]
    assert M.cap_victim(kept) is kept[0]                # a's 100 < a's 200: superseded, the oldest of those
    kept = [_c(a, 300), _c(b, 50), _c(a, 40), _c(b, 80)]
    assert M.cap_victim(kept) is kept[1]                # b's 50 < 80, before a's 40 (later in the list)


def test_without_superseded_states_the_least_recently_used():
    kept = [_c(object(), n) for n in (90, 10, 500)]
    assert M.cap_victim(kept) is kept[0]


def test_a_shared_prefix_state_goes_by_recency_only():
    a, b = object(), object()
    # 0077: a's system-block state (shared) is shorter than a's own state, yet not dropped as superseded
    kept = [_c(b, 70), _c(a, 30, shared=True), _c(a, 400)]
    assert M.cap_victim(kept) is kept[0]                # nothing superseded but the shared state: LRU, b's
    kept = [_c(a, 30, shared=True), _c(b, 70), _c(a, 400)]
    assert M.cap_victim(kept) is kept[0]                # it is the least recently used: it goes, by recency
    kept = [_c(a, 30, shared=True), _c(a, 100), _c(a, 400)]
    assert M.cap_victim(kept) is kept[1]                # a's earlier own turn goes before its shared state


def test_deterministic_in_order_and_extents():
    a, b = object(), object()
    kept = [_c(a, 10), _c(b, 10), _c(a, 20), _c(b, 20)]
    assert all(M.cap_victim(list(kept)) is kept[0] for _ in range(3))


# -- MultiDecoder._keep on the real pool -------------------------------------------------------------------------------
@pytest.fixture
def decoder(monkeypatch):
    def fake_take(e, ids, pending, mtp, drafter):
        return NS(ids=list(ids), rec=torch.zeros(1), conv=torch.zeros(1), pending=None, drafter_rows=None, head=None,
                  tail=None, rows=None, nbytes=0, mtp_len=-1)

    monkeypatch.setattr(D, "take_snapshot", fake_take)

    def make(entries):
        m = object.__new__(M.MultiDecoder)
        m.pool = Pool(400 * ALIGN)
        m.kept, m.lanes, m.partial, m.ops = [], {}, None, []
        m.e = NS(use=lambda st: None)
        m.g = NS(cache_entries=entries)
        m.rank, m.next_kid = 0, 0
        m._emit = lambda op, p: m.ops.append((op, list(p)))
        m._ctx = lambda lane: None
        return m

    return make


_at = [0]


def _lane(m, chat, toks, shared=()):
    x = m.pool.add(_at[0] * ALIGN * 2 % m.pool.rows, align_up(toks))
    _at[0] += 1
    x.kept, x.owner = [], None
    return NS(st=None, s=NS(prompt=list(range(chat * 10000, chat * 10000 + toks + 10))), dflash=False, extent=x,
              shared=list(shared), key=None, images=False, chat=chat)


def test_keep_drops_by_cap_victim(decoder):
    _at[0] = 0
    m = decoder(4)
    a = _lane(m, 1, 300, shared=[64])
    m._keep(a, 64)                                    # a's system block (shared)
    m._keep(a, 100)                                   # a's turns
    m._keep(a, 200)
    b = _lane(m, 2, 300)
    m._keep(b, 150)
    assert len(m.kept) == 4
    m._keep(a, 300)                                   # past the cap: a's earliest own turn (100) goes, not b's
    assert sorted((c.ids[0] // 10000, len(c.ids)) for c in m.kept) == [(1, 64), (1, 200), (1, 300), (2, 150)]
    m._keep(b, 250)                                   # a's 200 (superseded by a's 300) is older than b's 150
    assert sorted((c.ids[0] // 10000, len(c.ids)) for c in m.kept) == [(1, 64), (1, 300), (2, 150), (2, 250)]
    m._keep(a, 310)                                   # b's 150 then; the shared 64 stays while others are superseded
    assert sorted((c.ids[0] // 10000, len(c.ids)) for c in m.kept) == [(1, 64), (1, 300), (1, 310), (2, 250)]
    assert [c.shared for c in m.kept if len(c.ids) == 64] == [True]
    assert not m.ops                                  # the cap drops on both ranks alike: no EVICT is sent


def test_keep_with_only_latest_states_drops_the_least_recently_used(decoder):
    _at[0] = 0
    m = decoder(32)
    for i in range(60):                               # 60 finished one-shot conversations (kept_cap_check's case)
        lane = _lane(m, i, 64)
        m._keep(lane, 64)
    assert len(m.kept) == 32
    assert [c.ids[0] // 10000 for c in m.kept] == list(range(28, 60))


# -- Mia's tools/kept_cap_check.py (0097) ------------------------------------------------------------------------------
def test_kept_entries_share_and_ceiling():
    from tensorfold.families.glm5_next.cuda.engine import kept_entries

    mb, gib = 1 << 20, 1 << 30
    state, budget = 45 * mb, int(12.5 * gib)
    assert kept_entries(32, 0, budget, state) == (32, "")                       # off: the configured count, no note
    n, note = kept_entries(32, 25, budget, state)
    assert n == 71 and note and n * state <= budget * 0.25                      # 25%: a quarter of the budget
    assert kept_entries(128, 5, budget, state)[0] == 128                        # never below the configured count
    assert kept_entries(32, 100, budget, state)[0] == int(budget * 0.75 // state)    # past 75% read as 75%
    n, note = kept_entries(400, 0, budget, state)
    assert n == int(budget * 0.75 // state) and "three quarters" in note        # an explicit count past 3/4 is cut
    assert kept_entries(32, 0, 10 * mb, state)[0] == 1                          # a small budget still keeps one


def test_known_key_is_built_once():
    import numpy as np

    c = NS(key=list(range(1000)))
    first = M.known_key(c)
    assert np.array_equal(first, np.asarray(c.key, dtype=np.int64)) and M.known_key(c) is first


# -- Mia's tools/kept_state_check.py (0089) ----------------------------------------------------------------------------
@pytest.fixture
def stated(monkeypatch):
    """``MultiDecoder`` stubs whose kept states own 10 MiB each and whose spill tier records what it is asked."""

    mb = 1 << 20

    def fake_take(e, ids, pending, mtp, drafter):
        return NS(ids=list(ids), rec=torch.zeros(10 * mb // 4, dtype=torch.float32), conv=torch.zeros(1),
                  pending=None, drafter_rows=None, head=None, tail=None, rows=None, nbytes=0, mtp_len=-1)

    monkeypatch.setattr(D, "take_snapshot", fake_take)

    def make(entries=32, cap=0, rank=0):
        m = object.__new__(M.MultiDecoder)
        m.pool = Pool(1000 * ALIGN)
        m.kept, m.lanes, m.partial, m.ops, m.spilled = [], {}, None, [], []
        m.e = NS(use=lambda st: None)
        m.g = NS(cache_entries=entries)
        m.rank, m.next_kid, m.kept_bytes_cap = rank, 0, cap
        m.disk = object()                                # the spill tier "on": ``_spill`` records its kept prompt
        m._spill = lambda c, leaving=False, keep=frozenset(): m.spilled.append(c.kid) or True
        m._emit = lambda op, p: m.ops.append((op, list(p)))
        m._ctx = lambda lane: None
        return m

    return make


def test_quota_keeps_a_chats_newest_own_states_unspilled(stated, monkeypatch):
    assert M.KEEP_PER_CHAT == 0 and M.KEPT_BYTES_CAP == 0           # both off by default
    _at[0] = 0
    m = stated()
    a = _lane(m, 1, 64)
    for i in range(4):
        m._keep(a, 64 + i, own=True)
    assert len(m.kept) == 4 and not m.ops and not m.spilled          # off: nothing dropped
    monkeypatch.setattr(M, "KEEP_PER_CHAT", 2)
    m = stated()
    a, b = _lane(m, 1, 64), _lane(m, 2, 64)
    for i in range(5):
        m._keep(a, 64 + i, own=True)
    m._keep(b, 64)                                                  # shared-prefix states do not count
    m._keep(b, 66)
    assert sorted(len(c.ids) for c in m.kept if c.own) == [67, 68] and sum(not c.own for c in m.kept) == 2
    assert not m.spilled and not m.ops                              # quota drops: both ranks alike, never written


def test_byte_cap_evicts_on_rank_0_unspilled_and_entry_cap_still_spills(stated):
    _at[0] = 0
    m = stated(cap=25 << 20)
    for i in range(6):
        m._keep(_lane(m, i, 64), 64 + i)
    m._cap_bytes()                                                  # rank 0, at its next iteration
    assert m._held_bytes() <= 25 << 20 and len(m.kept) == 2
    assert [p for op, p in m.ops if op == M.EVICT] == [[k, 0] for k in range(4)]   # an EVICT a victim, no spill
    assert not m.spilled
    m = stated(entries=2)
    for i in range(4):
        m._keep(_lane(m, 10 + i, 64), 64 + i)
    assert len(m.kept) == 2 and m.spilled == [0, 1]                 # the entry cap's drops are still written
    r = stated(rank=1)                                              # rank 1 applies EVICT [kid, 0]: no write
    r._keep(_lane(r, 20, 64), 64)
    r.apply([M.EVICT, 2, r.kept[0].kid, 0])
    assert not r.kept and not r.spilled

"""Patch 0100 (MiaAI-Lab's glm-draft-prelaunch, by BadAd84) on the host side, no GPU: the part of Mia's
``tools/draft_prelaunch_check.py`` that needs none. ``MultiDecoder._prelaunch`` decides from state every rank holds
(``Stream.done``, set by rank 0's emit alone, changes nothing), leaves out the streams that will not ask next round,
leaves the copies' ``proposed`` as it was, and ``_copy_proposal`` reuses its proposal only while the copies are
unchanged; every change of a DFlash2 context bumps its generation (a pass ahead's key). The drafts and replies being
the same with the pass ahead is the GPU test ``tests/cuda/test_glm_copy_drafts.py``'s parallel case."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tensorfold.cuda.streams import Stream
from tensorfold.families.glm5_next.cuda import multi
from tensorfold.families.glm5_next.cuda.copy_drafts import CopyDrafts
from tensorfold.families.glm5_next.cuda.dflash2_multi import DraftContext, MultiDrafter, put_window

EOS = 9


def test_prelaunch_reads_only_every_ranks_state_and_reuses_unchanged_copies():
    calls = []
    dec = object.__new__(multi.MultiDecoder)
    dec.eos = (EOS,)
    dec.drafts = SimpleNamespace(block=8, prelaunch=lambda reqs: calls.append([(c.slot, p) for c, p in reqs]))
    dec.contexts = [SimpleNamespace(slot=i, context_end=10) for i in range(6)]

    def lane(slot, out, *, count=50, depth=3, dflash=True, copies=None):
        s = Stream(list(range(100, 120)), count)
        s.out = list(out)
        return multi.Lane(s=s, sid=slot, slot=slot, extent=None, st=None, order=slot, code=[], dflash=dflash,
                          depth=depth, copies=copies)

    repeat = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3]
    copied = CopyDrafts(list(range(100, 120)) + repeat, 3, 8, prompt=20)  # its suffix "1 2 3" occurred: copies
    plain = CopyDrafts(list(range(100, 120)) + [40, 41, 42], 3, 8, prompt=20)  # no earlier occurrence: DFlash2's
    lanes = [lane(0, [40, 41, 42], copies=plain), lane(1, [5, 6], count=2), lane(2, [5, EOS]),
             lane(3, [5, 6], depth=0), lane(4, [5, 6], dflash=False), lane(5, repeat, copies=copied)]
    lanes[0].s.done = True                            # rank 0's stop string: rank 1 never sees it, so it changes nothing
    copied.proposed = plain.proposed = 4              # (a stand-in for a proposal not yet settled by extend)
    dec._prelaunch(lanes)
    assert calls == [[(0, 42)]]                       # at its count / after an end / depth 0 / no DFlash2 / copied: out
    assert copied.proposed == 4 and plain.proposed == 4

    fresh = CopyDrafts(copied.tokens(), 3, 8, prompt=20)
    want = fresh.propose(dec._copy_room(lanes[5]))
    assert want and dec._copy_proposal(lanes[5]) == want and copied.proposed == fresh.proposed
    copied.proposed = 0
    lanes[5].next_copy = ((copied.length, dec._copy_room(lanes[5])), [77], 1)
    lanes[5].s.out.append(4)                          # the room changed: proposed again, not reused
    assert dec._copy_proposal(lanes[5]) == copied.propose(dec._copy_room(lanes[5]))


def test_every_context_change_bumps_its_generation():
    d = SimpleNamespace(ring=0, window=16, block=8, capacity=64, tap_layers=(), cap=72, kvh=1, hd=4)
    owner = SimpleNamespace(d=d, cap=72, kc=[torch.zeros((1, 2 * 72, 4))], vc=[torch.zeros((1, 2 * 72, 4))],
                            pos=torch.zeros((3,), dtype=torch.int64))
    a, b = DraftContext(owner, 0), DraftContext(owner, 1)
    key = MultiDrafter._key([(a, 5, 3), (b, 6, 3)])
    a.context_end = a.context_end                     # any write, even of the same end (a restore under the same end)
    assert MultiDrafter._key([(a, 5, 3), (b, 6, 3)]) != key
    key = MultiDrafter._key([(a, 5, 3)])
    put_window(b, 4, [torch.zeros((1, 4, 4)), torch.zeros((1, 4, 4))])   # another slot's restore: a's key holds
    assert MultiDrafter._key([(a, 5, 3)]) == key
    b.reset()
    assert b.context_end == 0 and MultiDrafter._key([(a, 5, 3)]) == key

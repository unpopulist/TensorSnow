"""The GLM multi-stream cache pool's host bookkeeping (``pool.Pool``) and row copies (``pool.Arena``), on the CPU."""

from __future__ import annotations

import pytest
import torch

from tensorfold.families.glm5_next.cuda.pool import ALIGN, Arena, Plane, Pool, align_up


def test_placement_is_lowest_base_first_fit():
    p = Pool(8 * ALIGN)
    a = p.alloc(ALIGN + 1)                      # two blocks
    b = p.alloc(ALIGN)
    c = p.alloc(3 * ALIGN)
    assert (a.base, a.size, b.base, c.base) == (0, 2 * ALIGN, 2 * ALIGN, 3 * ALIGN)
    assert p.free_rows() == 2 * ALIGN
    p.remove(b)
    assert p.gaps() == [(2 * ALIGN, ALIGN), (6 * ALIGN, 2 * ALIGN)]
    assert p.place(2 * ALIGN) == 6 * ALIGN and p.place(ALIGN) == 2 * ALIGN
    assert p.room_after(a) == ALIGN
    p.resize(a, 3 * ALIGN)
    with pytest.raises(ValueError):
        p.resize(a, 4 * ALIGN)                  # c sits right after
    assert p.place(ALIGN, ignore=[c]) == 3 * ALIGN
    old = p.move(c, 5 * ALIGN)
    assert old == 3 * ALIGN and p.gaps() == [(3 * ALIGN, 2 * ALIGN)]
    with pytest.raises(ValueError):
        p.add(4 * ALIGN, 2 * ALIGN)             # overlaps c
    assert align_up(1) == ALIGN and align_up(ALIGN) == ALIGN


def test_two_pools_making_the_same_calls_stay_equal():
    calls = [("alloc", 3000), ("alloc", 5000), ("alloc", 100), ("remove", 1), ("alloc", 2048), ("resize", 0, 1000)]

    def run():
        p, xs = Pool(16 * ALIGN), []
        for c in calls:
            if c[0] == "alloc":
                xs.append(p.alloc(c[1]))
            elif c[0] == "remove":
                p.remove(xs[c[1]])
            else:
                p.resize(xs[c[1]], c[2])
        return [(x.eid, x.base, x.size) for x in p.extents]

    assert run() == run()


def test_arena_views_and_overlapping_copies():
    rows = 4 * ALIGN
    lat = torch.arange(rows, dtype=torch.float32)[:, None].repeat(1, 3)
    pk = torch.arange(rows // 4 + 2, dtype=torch.float32)[:, None].repeat(1, 2)
    arena = Arena(rows, [Plane(lat.clone(), 1, 0), Plane(pk.clone(), 4, 2)])
    v = arena.view(1, ALIGN, ALIGN)
    assert v.shape[0] == ALIGN // 4 + 2 and float(v[0, 0]) == ALIGN // 4      # the pad reads the next extent
    with pytest.raises(ValueError):
        arena.view(0, 3 * ALIGN, 2 * ALIGN)
    # overlapping copies: down and up by less than the length
    for src, dst in ((ALIGN, ALIGN // 2), (0, 100)):
        arena = Arena(rows, [Plane(lat.clone(), 1, 0), Plane(pk.clone(), 4, 2)])
        n = ALIGN + 37
        arena.copy(src, dst, n)
        assert torch.equal(arena.planes[0].tensor[dst:dst + n], lat[src:src + n])
        m = -(-n // 4)
        assert torch.equal(arena.planes[1].tensor[dst // 4:dst // 4 + m], pk[src // 4:src // 4 + m])
    with pytest.raises(ValueError):
        Arena(rows, [Plane(torch.zeros(rows + 1, 1), 1, 0)])

"""GLM's L2 prefetch ranges (``glm5_next/cuda/l2pf.py``) over this tree's native EXL3 weights, on the CPU: an ``X3``
gives its scales and its decode kernel's warp-stream heads (never the shared scratch ``X3.sc`` or the counters), a
``Stack`` its parts in order, plain BF16 / Q4 weights Mia's ranges; budget and alignment hold. No GPU, no kernels."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.glm5_next.cuda import exl3_dense, l2pf, qmm


def _x3(k: int, n: int, bits: int = 4, bias: bool = False, sc: exl3_dense.Scratch | None = None) -> exl3_dense.X3:
    g = torch.Generator().manual_seed(k * 7 + n + bits)
    trellis = torch.randint(-2**15, 2**15, (k // 16, n // 16, 16 * bits), dtype=torch.int16, generator=g)
    suh = torch.randn(k, generator=g).half()
    svh = torch.randn(n, generator=g).half()
    b = torch.randn(n, generator=g).half() if bias else None
    x = exl3_dense.x3(trellis, suh, svh, "mul1", bias=b, device="cpu", sc=sc or exl3_dense.SCRATCH)
    x.lin.counters                                   # the lazy semaphores exist, so a walk could see them
    return x


def _scratch() -> exl3_dense.Scratch:
    """A scratch with real buffers (decode xh/z/tmp), so collecting it would show up."""

    sc = exl3_dense.Scratch()
    for name, numel, dt in (("xh", 1 << 16, torch.float16), ("z", 1 << 16, torch.float32),
                            ("tmp", 1 << 16, torch.bfloat16)):
        sc.get(name, numel, dt, "cpu")
    return sc


def _extent(t: torch.Tensor) -> tuple[int, int]:
    return t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()


def _inside(spans, tensors) -> bool:
    ext = [_extent(t) for t in tensors]
    # _span rounds a start down / an end up to 16 bytes, so allow that slack around each tensor
    return all(any(lo - 16 < a and a + n <= hi + 16 for lo, hi in ext) for a, n in spans)


def _disjoint(spans, tensors) -> bool:
    ext = [_extent(t) for t in tensors]
    return all(a + n <= lo or a >= hi for a, n in spans for lo, hi in ext)


def _scratch_tensors(sc: exl3_dense.Scratch) -> list[torch.Tensor]:
    return list(sc.buffers.values()) + [t for t in vars(sc.prefill).values() if isinstance(t, torch.Tensor)]


def test_tensors_skip_scratch_and_counters():
    sc = _scratch()
    x = _x3(1024, 256, bias=True, sc=sc)
    got = l2pf.tensors(x)
    lin = x.lin
    assert [t.data_ptr() for t in got] == [t.data_ptr() for t in (lin.suh, lin.words, lin.svh, lin.bias)]
    # also through a plain container (a layer's block holding an X3 and a Stack)
    st = exl3_dense.Stack([x, qmm.make_b16(torch.randn(128, 1024))], sc)
    block = SimpleNamespace(o=x, proj=st, norm=torch.randn(64))
    ptrs = {t.data_ptr() for t in l2pf.tensors(block)}
    banned = {t.data_ptr() for t in _scratch_tensors(sc)} | {lin.counters.data_ptr()}
    assert ptrs and not ptrs & banned


def test_x3_heads_follow_the_decode_kernel():
    x = _x3(1024, 256)
    lin = x.lin
    sk, wk = lin.split
    assert (sk, wk) == (2, 4)                         # plan(1024, 256): the shape this test's offsets assume
    kt, nb = lin.k // 16, lin.n // 128
    per_warp = kt // sk // wk
    tile = 8 * 4 * lin.k2 * 4                         # bytes of one 128-column k tile (8 tiles of TW words)
    heads = l2pf.x3_heads(lin, 1 << 30)               # budget to spare: whole streams
    assert len(heads) == nb * sk * wk
    base = lin.words.data_ptr()
    want = []
    for p in range(nb * sk):
        col, split = p % nb, p // nb
        for w in range(wk):
            kt0 = split * per_warp * wk + w * per_warp
            want.append((base + (col * kt + kt0) * tile, per_warp * tile))
            # the head starts on the words of tile (col, kt0) in the strips layout
            assert lin.words[col, kt0].data_ptr() == want[-1][0]
    assert heads == want
    # the streams cover the matrix exactly once
    assert sum(n for _, n in heads) == lin.words.numel() * 4
    # a small budget: the same starts, equal 16-byte multiples of at least 512 bytes
    few = l2pf.x3_heads(lin, 16 << 10)
    assert [a for a, _ in few] == [a for a, _ in want]
    assert len({n for _, n in few}) == 1 and few[0][1] % 16 == 0 and 512 <= few[0][1] <= (16 << 10) // len(few)
    assert l2pf.x3_heads(lin, 4 << 10) == []         # under 512 bytes a stream: none


def test_x3_heads_first_wave_only():
    x = _x3(256, 128 * 200)                           # 200 column blocks, one split: more than one wave
    lin = x.lin
    assert lin.split[0] == 1
    heads = l2pf.x3_heads(lin, 1 << 30)
    assert len(heads) == l2pf.ONE_WAVE * lin.split[1]
    stride_nb = lin.strides[1] * 4
    assert {(a - lin.words.data_ptr()) // stride_nb for a, _ in heads} == set(range(l2pf.ONE_WAVE))


def test_stored_layout_falls_back_to_leading_bytes():
    x = _x3(1024, 256)
    from tensorfold.cuda.exl3.linear import Exl3Linear

    g = torch.Generator().manual_seed(3)
    lin = Exl3Linear.from_tensors(torch.randint(-100, 100, (64, 16, 64), dtype=torch.int16, generator=g),
                                  x.lin.suh, x.lin.svh, "mul1", device="cpu", layout="stored")
    assert l2pf.x3_heads(lin, 1 << 30) == []
    spans = l2pf.ranges([exl3_dense.X3(lin)], 8 << 10)
    assert spans[-1][0] == lin.words.data_ptr()       # small scales first, then the words' leading bytes


@pytest.mark.parametrize("budget", [8 << 10, 64 << 10, 8 << 20])
def test_ranges_x3_stack_plain(budget):
    sc = _scratch()
    a = _x3(1024, 256, bias=True, sc=sc)              # e.g. a shared expert's down / an o_proj
    b = _x3(1024, 384, bits=5, sc=sc)
    plain = qmm.make_b16(torch.randn(128, 1024))      # a Stack's plain part (KDA f_a|g_a|b)
    stack = exl3_dense.Stack([b, plain], sc)
    norm = torch.randn(1024).bfloat16()
    q = SimpleNamespace(weight=torch.zeros(256, 1024 // 8, dtype=torch.int32),
                        scales=torch.zeros(256, 16).bfloat16(), biases=torch.zeros(256, 16).bfloat16(), n=256, k=1024)
    groups = [norm, a, stack, q]
    spans = l2pf.ranges(groups, budget)
    assert spans
    assert sum(n for _, n in spans) <= budget
    assert all(x % 16 == 0 and n % 16 == 0 and n > 0 for x, n in spans)
    weights = [norm, plain.weight, q.weight, q.scales, q.biases]
    for x in (a, b):
        weights += [x.lin.words, x.lin.suh, x.lin.svh] + ([x.lin.bias] if x.lin.bias is not None else [])
    assert _inside(spans, weights)
    assert _disjoint(spans, _scratch_tensors(sc) + [a.lin.counters, b.lin.counters])
    # order: the groups as given, a Stack's parts in order (first span of each object that got any)

    def first(ts):
        idx = [i for i, (x, _) in enumerate(spans) if any(lo - 16 < x < hi for lo, hi in map(_extent, ts))]
        return idx[0] if idx else None

    order = [first([norm]), first([a.lin.suh, a.lin.svh, a.lin.bias, a.lin.words]),
             first([b.lin.suh, b.lin.svh, b.lin.words]), first([plain.weight]),
             first([q.weight, q.scales, q.biases])]
    seen = [i for i in order if i is not None]
    assert seen == sorted(seen) and seen[0] == 0
    if budget >= 8 << 20:                             # everything fits: every object contributes
        assert None not in order
        # a's words came as warp heads (starts inside the words, more than one), not one leading span
        lo, hi = _extent(a.lin.words)
        assert sum(1 for x, _ in spans if lo <= x < hi) == len(l2pf.x3_heads(a.lin, 1 << 30))


def test_prefetch_plan_on_exl3_layers(monkeypatch):
    """Prefetch's site plan over a stand-in two-layer model of X3 / Stack weights: built on the CPU (no side stream),
    every site's spans inside weights and none in the scratch."""

    sc = _scratch()
    D = 1024
    mlp = lambda: SimpleNamespace(gu=exl3_dense.Stack([_x3(D, 256, sc=sc), _x3(D, 256, sc=sc)], sc),   # noqa: E731
                                  down=_x3(256 * 2, D, sc=sc), width=256)
    hc = lambda: SimpleNamespace(fn=torch.randn(24, 64), base=torch.randn(24), scale=torch.randn(3))  # noqa: E731

    def layer(i):
        kda = SimpleNamespace(proj=exl3_dense.Stack([_x3(D, 384, sc=sc), qmm.make_b16(torch.randn(128, D))], sc),
                              fb=qmm.make_b16(torch.randn(128, 64)), gb=qmm.make_b16(torch.randn(128, 64)),
                              conv=torch.randn(384, 4), a_log=torch.randn(4), dt_bias=torch.randn(512),
                              o=_x3(512, D, sc=sc))
        moe = SimpleNamespace(router=torch.randn(16, D), bias=torch.randn(16), shared=mlp())
        return SimpleNamespace(index=i, attn_hc=hc(), ffn_hc=hc(), in_norm=torch.randn(D), post_norm=torch.randn(D),
                               kda=kda, dsa=None, mlp=None, moe=moe)

    w = SimpleNamespace(layers=[layer(0), layer(1)], norm=torch.randn(D), head=_x3(D, 1024, bits=8, sc=sc),
                        device=torch.device("cpu"))
    s = l2pf.Settings.from_env({"TF_GLM_L2PF": "1", "TF_GLM_L2PF_MB": "0.0625"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)   # no side stream, no build: the table alone
    pf = l2pf.Prefetch(w, s)
    assert set(pf.sites) == {(i, n) for i in (0, 1) for n in "afo"}
    assert all(n <= (64 << 10) for n in pf.bytes.values())
    banned = _scratch_tensors(sc)
    rows = pf.table.tolist()
    assert _disjoint([tuple(r) for r in rows], banned)
    assert "a 2 sites" in pf.summary()

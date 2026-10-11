"""DFlash2 for GLM-5.3-Flash's concurrent streams: a context per stream slot over the drafter's weights.

TF_GLM_MULTI_DRAFT=batched (the default; Mia's 0027 ``MultiDrafter`` with 0092's checked candidates): every stream's
context is a slot of one shared pool (``DraftContext``), and a round's block passes and context updates run as ONE
launch sequence for all its streams (``MultiDrafter.propose`` / ``candidates`` / ``commit``), so the drafter's weights
and the output head are read once a round rather than once a stream. Each stream's candidates are the bits of its own
solo ``Drafter``: every kernel of a block pass is row-independent (the 4-bit and EXL3 decode matmuls run the same
groups and K slices at any row count; norms, rotary, ``_prep``; the dynamic convolutions per block of rows, SEG),
except the block attention, which reads each stream's own slot at its own position (``_dattn_seg_kernel``, the solo
kernel's loop and arithmetic), and the two ops whose kernels could depend on the row count (cuBLAS's selector
projection and ``torch.topk``), which run per stream on that stream's rows alone. Context updates stack every
stream's kept rows; each row carries its own position and destination. The batched pass's all-gathers are S times a
stream's (the row-parallel partials and the candidates), so both ranks must agree on the setting (the engine's
two-rank settings comparison carries it) and run the same streams in the same order.

TF_GLM_MULTI_DRAFT=serial: each stream drafts on its own ``context`` (a ``dflash2.Drafter`` of its own buffers), one
after another: the reference. Drafts only propose (every draft is verified against the serial sample), and both
settings propose the same drafts, so neither changes a reply nor a round."""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tensorfold.engine.exact_sampling import Sampling

from . import glue
from .dflash2 import NO_LIMIT, BadCandidates, Drafter, _dconv, _mm, merge_candidates

DRAFT_KINDS = ("batched", "serial")


def draft_kind(value: str | None = None) -> str:
    """TF_GLM_MULTI_DRAFT: batched (the default: one block pass and one context update for every stream of a round)
    or serial (each stream's context on its own, one after another)."""

    value = (os.environ.get("TF_GLM_MULTI_DRAFT", "") if value is None else value).strip().lower() or "batched"
    if value not in DRAFT_KINDS:
        raise ValueError(f"TF_GLM_MULTI_DRAFT: batched or serial, not {value!r}")
    return value


def context(drafter, *, capture: bool = False):
    """A new context over ``drafter``'s weights: fresh caches and static inputs, no captured graphs unless
    ``capture`` (both ranks must capture alike: the block pass all-gathers)."""

    c = copy.copy(drafter)
    c.kc = [torch.zeros_like(t) for t in drafter.kc]
    c.vc = [torch.zeros_like(t) for t in drafter.vc]
    c.pos_dev = torch.zeros_like(drafter.pos_dev)
    c.context_end = 0
    c.ids = torch.full_like(drafter.ids, drafter.mask_id)
    c.ids_host = torch.zeros((1,), dtype=torch.int32, pin_memory=torch.cuda.is_available())
    c.tap_in = torch.zeros_like(drafter.tap_in)
    # a block pass's candidates and projected rows (Mia's 0013: one device buffer, one pinned read-back): the
    # context's own, so no two streams' passes or read-backs share them
    cuda = torch.cuda.is_available()
    c.cand = torch.zeros_like(drafter.cand)
    c.cand_host = torch.zeros(drafter.cand_host.shape, dtype=drafter.cand_host.dtype, pin_memory=cuda)
    c.cand_ready = torch.cuda.Event() if cuda else None
    c.packed = c.cand[:drafter.cand_n].view(drafter.packed.shape)
    c.proj = c.cand[drafter.cand_n:].view(drafter.proj.shape)
    c.pool = None
    c.block_graph = None
    c.tap_graphs = {}
    if capture:
        c.capture()
    return c


def context_bytes(drafter) -> int:
    """Device bytes one more context takes (its caches and static inputs)."""

    t = list(drafter.kc) + list(drafter.vc) + [drafter.tap_in, drafter.ids, drafter.pos_dev, drafter.cand]
    return sum(x.numel() * x.element_size() for x in t)


def window_slots(ctx, n: int) -> torch.Tensor:
    """The context rows a block pass at context end n reads (positions n - window - 1 .. n - 1), ring or flat."""

    lo = max(0, n - ctx.window - 1)
    idx = torch.arange(lo, n, device=ctx.kc[0].device)
    return idx % ctx.ring if ctx.ring else idx


def take_window(ctx, n: int) -> list[torch.Tensor]:
    """A copy of a context's window rows before n, in position order (a kept prompt's DFlash2 state)."""

    idx = window_slots(ctx, n)
    return [c.index_select(1, idx) for c in ctx.kc] + [c.index_select(1, idx) for c in ctx.vc]


def put_window(ctx, n: int, rows: list[torch.Tensor]) -> None:
    """Put a kept prompt's window rows back and end the context at n."""

    idx = window_slots(ctx, n)
    caches = list(ctx.kc) + list(ctx.vc)
    if len(rows) != len(caches) or any(r.shape[1] != idx.numel() for r in rows):
        raise ValueError("a kept state's DFlash2 window does not match the context")
    for c, r in zip(caches, rows):
        c.index_copy_(1, idx, r)
    ctx.context_end = n
    ctx.pos_dev.fill_(n)


# -- batched (TF_GLM_MULTI_DRAFT=batched) -------------------------------------------------------------------------------
@triton.jit
def _dattn_seg_kernel(Q, K, V, OUT, POS, SLOT, window, scale, L, SLOT_ROWS,
                      N: tl.constexpr, G: tl.constexpr, NH: tl.constexpr, HD: tl.constexpr, CAP: tl.constexpr,
                      BK: tl.constexpr, CAUSAL: tl.constexpr, RING: tl.constexpr, SKIP: tl.constexpr = True):
    """``dflash2._dattn_kernel`` for block ``seg`` (program 1) of L rows side by side: its queries at rows seg * N of
    Q [heads, L, HD], its context in pool slot SLOT[seg] (rows slot * CAP .. of K / V [kv heads, SLOT_ROWS, HD]) at
    position POS[seg]. The solo kernel's loop (from the window's tile with SKIP) and arithmetic, so each block gets
    its solo bits."""

    kvh = tl.program_id(0)
    seg = tl.program_id(1)
    M: tl.constexpr = G * N
    m = tl.arange(0, M)
    qh = kvh * G + m // N
    qr = m % N
    d = tl.arange(0, HD)
    q = tl.load(Q + (qh[:, None] * L + seg * N + qr[:, None]) * HD + d[None, :])
    s = tl.load(POS + seg).to(tl.int32)
    base = kvh.to(tl.int64) * SLOT_ROWS + tl.load(SLOT + seg).to(tl.int64) * CAP
    klen = s + N
    qpos = s + qr
    m_i = tl.full([M], -1e30, tl.float32)
    l_i = tl.zeros([M], tl.float32)
    acc = tl.zeros([M, HD], tl.float32)
    lo = 0
    if SKIP:
        lo = tl.maximum(s - window, 0) // BK * BK
    for start in range(lo, klen, BK):
        kk = start + tl.arange(0, BK)
        kin = kk < klen
        if RING:
            row = kk % CAP
        else:
            row = kk
        k = tl.load(K + (base + row[:, None]) * HD + d[None, :], mask=kin[:, None], other=0.0)
        v = tl.load(V + (base + row[:, None]) * HD + d[None, :], mask=kin[:, None], other=0.0)
        sc = tl.dot(q, tl.trans(k)) * scale
        ok = kin[None, :] & (((kk[None, :] < s) & (qpos[:, None] - kk[None, :] <= window)) | (kk[None, :] >= s))
        if CAUSAL:
            ok = ok & (kk[None, :] <= qpos[:, None])
        sc = tl.where(ok, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(OUT + (seg * N + qr[:, None]) * (NH * HD) + qh[:, None] * HD + d[None, :], out.to(tl.bfloat16))


class DraftContext:
    """One stream's DFlash2 context: its slot of the shared pool (``kc`` / ``vc`` per layer are views [kv heads, ring
    or capacity + block rows, head_dim]) and where its committed rows end. Stands in for a solo ``Drafter`` wherever
    one stream's context is handled on its own (a prompt chunk's ``add_taps``, ``take_window`` / ``put_window``,
    ``reset``). ``multi._reslot`` swaps whole context objects between stream slots: each keeps its pool slot."""

    def __init__(self, owner: "MultiDrafter", slot: int) -> None:
        d = owner.d
        self.owner, self.slot = owner, slot
        self.ring, self.window, self.block, self.capacity = d.ring, d.window, d.block, d.capacity
        self.cap = owner.cap
        self.tap_layers = d.tap_layers
        cap = owner.cap
        self.kc = [c[:, slot * cap:(slot + 1) * cap] for c in owner.kc]
        self.vc = [c[:, slot * cap:(slot + 1) * cap] for c in owner.vc]
        self.pos_dev = owner.pos[slot:slot + 1]          # the device mirror of context_end (put_window fills it)
        self.context_end = 0

    def reset(self) -> None:
        self.context_end = 0
        self.pos_dev.zero_()

    def add_taps(self, taps: torch.Tensor) -> None:
        """This stream's committed rows' taps alone (a prompt's chunks): ``MultiDrafter.commit`` of one stream."""
        self.owner.commit([(self, taps)])

    def candidates(self, pending: int, depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.owner.candidates([(self, pending, depth)])[0]

    def chain(self, *args, **kw) -> list[int]:
        return self.owner.d.chain(*args, **kw)

    def propose(self, pending: int, depth: int, sampling: Sampling | None, confidence: float = 0.0, *,
                beta: float = 0.0, round_ms: tuple[float, ...] | None = None) -> list[int]:
        return self.owner.propose([DraftRequest(self, pending, depth, sampling, confidence, beta, round_ms)])[0]


@dataclass
class DraftRequest:
    """One stream's draft for a round: ``Drafter.propose``'s arguments with the stream's context."""

    ctx: DraftContext
    pending: int
    depth: int
    sampling: Sampling | None = None
    confidence: float = 0.0
    beta: float = 0.0
    round_ms: tuple[float, ...] | None = None


class MultiDrafter:
    """Up to ``streams`` DFlash2 contexts over one ``Drafter``'s weights (its own context buffers stay unused).

    The pool per layer is [kv heads, streams * cap + trash, head_dim]: slot i's rows at i * cap .. (the drafter's
    ring, or a flat capacity + block when it has none), then ``trash`` rows that padded context updates (a captured
    row bucket larger than the rows kept) and the captures write to."""

    def __init__(self, drafter: Drafter, streams: int, *, tap_rows: int | None = None) -> None:
        d = self.d = drafter
        if streams < 1:
            raise ValueError("a multi-stream drafter needs at least one stream")
        self.streams = streams
        self.dev = d.dev
        self.block = d.block
        self.cap = d.cap
        n = self.block
        # context updates: rows per launch (every stream's kept rows stacked), padded to a multiple of the block
        rows = max(64, streams * (tap_rows or d.tap_rows))
        self.t_max = -(-rows // n) * n
        self.trash = self.t_max
        self.slot_rows = streams * self.cap + self.trash
        KV, hd = d.kvh, d.hd
        self.kc = [torch.zeros((KV, self.slot_rows, hd), dtype=torch.bfloat16, device=self.dev) for _ in d.layers]
        self.vc = [torch.zeros((KV, self.slot_rows, hd), dtype=torch.bfloat16, device=self.dev) for _ in d.layers]
        self.pos = torch.zeros((streams + 1,), dtype=torch.int64, device=self.dev)   # + the trash entry
        self.contexts = [DraftContext(self, i) for i in range(streams)]
        pinned = torch.cuda.is_available()
        # block passes: [pending x streams | slot x streams | position x streams]
        self.b_meta = torch.zeros((3 * streams,), dtype=torch.int64, device=self.dev)
        self.b_host = torch.zeros((3 * streams,), dtype=torch.int64, pin_memory=pinned)
        self.ids = torch.full((streams * n,), d.mask_id, dtype=torch.int32, device=self.dev)
        self.ar = torch.arange(max(64, n), device=self.dev)
        # context updates: [position x t_max | destination row x t_max | slot x streams | end x streams]
        T = self.t_max
        self.t_meta = torch.zeros((2 * T + 2 * streams,), dtype=torch.int64, device=self.dev)
        self.t_host = torch.zeros((2 * T + 2 * streams,), dtype=torch.int64, pin_memory=pinned)
        self.t_ready = torch.cuda.Event() if pinned else None
        self.tap_in = torch.zeros((T, d.tap_in.shape[1]), dtype=torch.bfloat16, device=self.dev)
        m = n - 1
        self.cand_n = d.gathered * streams * m * 2 * d.top_k
        self.cand = torch.zeros((self.cand_n + streams * m * d.hproj.shape[0],), dtype=torch.float32,
                                device=self.dev)
        self.cand_host = torch.zeros(self.cand.shape, dtype=torch.float32, pin_memory=pinned)
        self.cand_ready = torch.cuda.Event() if pinned else None
        self.proj = self.cand[self.cand_n:].view(streams * m, d.hproj.shape[0])
        self.pool = None
        self.block_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.tap_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.buckets = list(range(n, T + 1, n))

    def nbytes(self) -> int:
        """Device bytes of the pool and the static inputs (the weights are the drafter's; graphs not counted)."""
        t = self.kc + self.vc + [self.pos, self.b_meta, self.ids, self.ar, self.t_meta, self.tap_in, self.cand]
        return sum(x.numel() * x.element_size() for x in t)

    def reset(self) -> None:
        for c in self.contexts:
            c.reset()

    # -- pieces -------------------------------------------------------------------------------------------------
    def _rotary(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``Drafter._rotary`` for rows at the positions ``pos`` (int64): the same float32 ops, element for element."""
        phase = pos.to(torch.float32)[:, None] * self.d.inv_freq[None, :]
        return phase.cos().contiguous(), phase.sin().contiguous()

    def _local(self, pos: torch.Tensor) -> torch.Tensor:
        return pos % self.d.ring if self.d.ring else pos

    def _layer(self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, idx: torch.Tensor, S: int,
               pos: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """``Drafter._layer`` over S blocks side by side; the attention reads each block's own slot."""

        d = self.d
        L = d.layers[i]
        n = self.block
        rows = x.shape[0]
        normed, xs = d._norm(x, L.in_norm)
        dyn = _mm(normed, L.a_kp, xs)
        q, k, v = d._prep(_mm(_dconv(normed, dyn, L.a_base, 0, d.gs, seg=n), L.qkv), L, cos, sin, d.heads)
        self.kc[i].index_copy_(1, idx, k)
        self.vc[i].index_copy_(1, idx, v)
        out = torch.empty((rows, d.heads * d.hd), dtype=torch.bfloat16, device=self.dev)
        _dattn_seg_kernel[(d.kvh, S)](q, self.kc[i], self.vc[i], out, pos, slot, d.window, d.hd ** -0.5, rows,
                                      self.slot_rows, N=n, G=d.heads // d.kvh, NH=d.heads, HD=d.hd, CAP=self.cap,
                                      BK=64, CAUSAL=d.causal, RING=bool(d.ring), num_warps=4)
        x = _dconv(d._row(out, L.o), dyn, L.a_base, 1, d.gs, x, seg=n)
        normed, xs = d._norm(x, L.post_norm)
        dyn = _mm(normed, L.m_kp, xs)
        gu = _mm(_dconv(normed, dyn, L.m_base, 0, d.gs, seg=n), L.gu)
        act = torch.empty((rows, d.inter), dtype=torch.bfloat16, device=self.dev)
        axs = torch.empty((rows, d.inter // 64), dtype=torch.float32, device=self.dev)
        glue.swiglu(gu, act, axs, NO_LIMIT)
        return _dconv(d._row(act, L.down, axs), dyn, L.m_base, 1, d.gs, x, seg=n)

    def _block_compute(self, S: int) -> None:
        """S streams' blocks [pending, mask x (block - 1)] at their committed lengths (``b_meta``), then each
        stream's candidates and projected rows as its solo pass leaves them, stream s at rows s * (block - 1)."""

        d, w = self.d, self.d.w
        n, St = self.block, self.streams
        pend, slot, pos = self.b_meta[:S], self.b_meta[St:St + S], self.b_meta[2 * St:2 * St + S]
        self.ids.view(St, n)[:S, 0].copy_(pend)
        rowpos = (pos[:, None] + self.ar[:n][None, :]).reshape(-1)
        idx = (slot[:, None] * self.cap + self._local(rowpos).view(S, n)).reshape(-1)
        R = S * n
        x = torch.empty((R, d.D), dtype=torch.bfloat16, device=self.dev)
        glue.embed(self.ids[:R], w.embed, d.D, 1, x)
        cos, sin = self._rotary(rowpos)
        for i in range(len(d.layers)):
            x = self._layer(i, x, cos, sin, idx, S, pos, slot)
        m, k = n - 1, d.top_k
        h, hs = d._norm(x.view(S, n, d.D)[:, 1:].reshape(S * m, d.D), d.norm)
        logits = _mm(h, w.draft_head if w.draft_head is not None else w.head, hs).float()
        packed = torch.empty((S * m, 2 * k), dtype=torch.float32, device=self.dev)
        for s in range(S):                  # torch.topk and cuBLAS may pick kernels by row count: one stream each
            rows = slice(s * m, (s + 1) * m)
            vals, local = torch.topk(logits[rows], k, dim=-1)
            gids = (local + w.vocab_offset).to(torch.int32)
            packed[rows] = torch.cat([vals, gids.view(torch.float32)], dim=1)
            self.proj[rows].copy_(F.linear(h[rows], d.hproj).float())
        size = d.gathered * S * m * 2 * k
        if d.gathered > 1:
            w.comm.all_gather(packed.view(-1), self.cand[:size])
        else:
            self.cand[:size].copy_(packed.view(-1))

    def _taps_compute(self, T: int) -> None:
        """Context updates of T stacked rows (``t_meta``: each row's position and destination), then each updated
        stream's device end."""

        d, St, Tm = self.d, self.streams, self.t_max
        pos, idx = self.t_meta[:T], self.t_meta[Tm:Tm + T]
        ctx, _ = d._norm(_mm(self.tap_in[:T], d.fc), d.hidden_norm)
        cos, sin = self._rotary(pos)
        for i, L in enumerate(d.layers):
            _, k, v = d._prep(_mm(ctx, L.kv), L, cos, sin, 0)
            self.kc[i].index_copy_(1, idx, k)
            self.vc[i].index_copy_(1, idx, v)
        self.pos.index_copy_(0, self.t_meta[2 * Tm:2 * Tm + St], self.t_meta[2 * Tm + St:2 * Tm + 2 * St])

    # -- graphs -------------------------------------------------------------------------------------------------
    def _trash_tables(self) -> None:
        """Tables that send every row to the trash rows (captures and their warm-ups touch no stream)."""

        St, Tm = self.streams, self.t_max
        self.b_meta.zero_()
        self.b_meta[St:2 * St].fill_(St)                 # slot St: rows St * cap .., the trash (block <= trash)
        self.t_meta.zero_()
        self.t_meta[Tm:2 * Tm].copy_(St * self.cap + torch.arange(Tm, device=self.dev))
        self.t_meta[2 * Tm:2 * Tm + St].fill_(St)       # the trash end

    @torch.no_grad()
    def capture(self) -> None:
        """CUDA graphs for block passes of 1..streams streams and context updates of each row bucket (a multiple of
        the block up to ``t_max``), both ranks together (the block pass all-gathers). Stream contexts are left as
        they were."""

        ends = [c.context_end for c in self.contexts]
        saved = self.pos.clone()
        self.pool = torch.cuda.graph_pool_handle()
        self._trash_tables()
        for T in self.buckets:
            for _ in range(2):
                self._taps_compute(T)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._taps_compute(T)
            self.tap_graphs[T] = g
        for S in range(1, self.streams + 1):
            for _ in range(2):
                self._block_compute(S)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._block_compute(S)
            self.block_graphs[S] = g
        torch.cuda.synchronize()
        self.pos.copy_(saved)
        for c, e in zip(self.contexts, ends):
            c.context_end = e

    # -- the context ---------------------------------------------------------------------------------------------
    def _launch_taps(self, pieces: list[tuple[DraftContext, torch.Tensor, int]]) -> None:
        St, Tm = self.streams, self.t_max
        if self.t_ready is not None:
            self.t_ready.synchronize()                   # the last update's table has left t_host
        h = self.t_host
        h[2 * Tm:2 * Tm + St].fill_(St)
        h[2 * Tm + St:].zero_()
        off = 0
        for j, (c, part, p0) in enumerate(pieces):
            r = part.shape[0]
            self.tap_in[off:off + r].copy_(part)
            p = torch.arange(p0, p0 + r, dtype=torch.int64)
            h[off:off + r] = p
            h[Tm + off:Tm + off + r] = c.slot * self.cap + (p % self.d.ring if self.d.ring else p)
            h[2 * Tm + j], h[2 * Tm + St + j] = c.slot, p0 + r
            off += r
        T = off
        g = None
        if self.tap_graphs:
            T = next(b for b in self.buckets if b >= off)
            g = self.tap_graphs[T]
        if T > off:                                      # padded rows: the trash
            h[off:T] = 0
            h[Tm + off:Tm + T] = St * self.cap + torch.arange(off, T, dtype=torch.int64)
        self.t_meta.copy_(h, non_blocking=True)
        if self.t_ready is not None:
            self.t_ready.record()
        if g is not None:
            g.replay()
        else:
            self._taps_compute(T)
        for c, part, p0 in pieces:
            c.context_end = p0 + part.shape[0]

    @torch.no_grad()
    def commit(self, items: Sequence[tuple[DraftContext, torch.Tensor]]) -> None:
        """Each stream's committed rows' taps [n, 5 * D] (bf16) at its positions context_end, context_end + 1, ...:
        every stream's rows stacked into launches of up to ``t_max`` rows."""

        seen: set[int] = set()
        for c, taps in items:
            if c.owner is not self or c.slot in seen:
                raise ValueError("commit takes each of this drafter's streams at most once")
            seen.add(c.slot)
            if c.context_end + taps.shape[0] > c.capacity:
                raise ValueError("drafter context past its capacity")
        pieces: list[tuple[DraftContext, torch.Tensor, int]] = []
        used = 0
        for c, taps in items:
            start, end = 0, c.context_end
            while start < taps.shape[0]:
                take = min(taps.shape[0] - start, self.t_max - used)
                pieces.append((c, taps[start:start + take], end))
                start, end, used = start + take, end + take, used + take
                if used == self.t_max:
                    self._launch_taps(pieces)
                    pieces, used = [], 0
        if pieces:
            self._launch_taps(pieces)

    # -- drafts -------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def candidates(self, reqs: Sequence[tuple[DraftContext, int, int]]) -> list[tuple[np.ndarray, ...]]:
        """(stream, pending, depth) -> each stream's ``Drafter.candidates(pending, depth)``: one batched pass."""

        S, St = len(reqs), self.streams
        if not 1 <= S <= St:
            raise ValueError(f"a block pass takes 1..{St} streams, got {S}")
        if len({c.slot for c, _, _ in reqs}) != S or any(c.owner is not self for c, _, _ in reqs):
            raise ValueError("a block pass takes each of this drafter's streams at most once")
        h = self.b_host
        for s, (c, pending, _) in enumerate(reqs):
            h[s], h[St + s], h[2 * St + s] = int(pending), c.slot, c.context_end
        self.b_meta.copy_(h, non_blocking=True)
        g = self.block_graphs.get(S)
        if g is not None:
            g.replay()
        else:
            self._block_compute(S)
        d = self.d
        m, k = self.block - 1, d.top_k
        vocab = d.succ.shape[0]
        # one pinned copy of every stream's candidates and projected rows; ids outside the vocabulary (Mia's 0092: a
        # read-back that raced the pass) are copied again once, after a full device sync, before they fail the round
        for attempt in (0, 1):
            if self.cand_ready is not None:
                self.cand_host.copy_(self.cand, non_blocking=True)
                self.cand_ready.record()
                self.cand_ready.synchronize()
            else:
                self.cand_host.copy_(self.cand)
            g_all = self.cand_host[:d.gathered * S * m * 2 * k].view(d.gathered, S * m, 2 * k)
            p_all = self.cand_host[self.cand_n:].view(St * m, -1)
            try:
                return [merge_candidates(g_all[:, s * m:s * m + depth], p_all[s * m:s * m + depth], k, vocab)
                        for s, (_, _, depth) in enumerate(reqs)]
            except BadCandidates as e:
                if attempt:
                    raise
                print(f"[tensorfold] drafter: {e} ({S} streams); copying the rows again once the device is idle",
                      file=sys.stderr, flush=True)
                torch.cuda.synchronize()
        raise AssertionError("unreachable")

    def propose(self, reqs: Sequence[DraftRequest]) -> list[list[int]]:
        """Each stream's ``Drafter.propose``: streams with a depth and a context share one block pass; the others
        get [] (as a solo drafter runs none for them)."""

        out: list[list[int]] = [[] for _ in reqs]
        live = []
        for i, r in enumerate(reqs):
            depth = min(r.depth, self.block - 1)
            if depth >= 1 and r.ctx.context_end > 0:
                live.append((i, depth))
        if not live:
            return out
        got = self.candidates([(reqs[i].ctx, reqs[i].pending, depth) for i, depth in live])
        for (i, _), (tokens, values, proj) in zip(live, got):
            r = reqs[i]
            first = r.ctx.context_end + 1
            if r.beta > 0:
                out[i] = self.d.chain(tokens, values, proj, r.pending, first, r.sampling, r.confidence, beta=r.beta,
                                      round_ms=r.round_ms)
            else:
                out[i] = self.d.chain(tokens, values, proj, r.pending, first, r.sampling, r.confidence)
        return out

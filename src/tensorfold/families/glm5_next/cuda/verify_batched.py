"""``BatchedVerify``: ONE forward over every stream's rows of a round, back to back (Mia's patch 0030, ``verify.py``).

The row-independent parts (embeddings, hyper-connection glue, projections, MLP / MoE, head) run on the whole window;
KDA layers through the segmented chain (``kda.chain_segments``: each segment from its slot's state, into its slot's
other parity); DSA layers through the segmented latent / sparse attention over the streams' extents
(``forward.dsa_segments``, ``segments.SegRows``); the commit replays each segment's kept rows
(``kda.replay_layers_segments``) and shifts its conv window (``forward.conv_shift_segments``). Every segment's rows
get the bits of that stream's window alone: the dense matmuls (Q4, B16, EXL3 X3 / Stack) keep rows apart at any row
count up to 128, the expert kernels likewise (checked against ``verify.SerialVerify`` in tests/cuda/test_glm_multi.py).
"""

from __future__ import annotations

from typing import Sequence

import torch

from .verify import Segment, Verified


class BatchedVerify:
    """One forward over every segment's rows (``rows`` at most, ``segments.seg_count`` streams), on buffers of its
    own. ``taps``: the drafter's tap layers (DFlash2's inputs), or ()."""

    def __init__(self, e, *, taps: Sequence[int] = (), rows: int = 32) -> None:
        from . import kda as kda_mod
        from . import latent
        from .forward import Buffers
        from .segments import SegRows, SelectScratch, seg_count

        if not latent.ENABLED:
            raise ValueError("batched verify windows need the latent cache (TF_GLM_LATENT=1)")
        w = e.w
        c = w.cfg
        self.e, self.w, self.rows_max = e, w, rows
        dev = w.device
        self.b = Buffers(w, rows, e.caches.rows)
        if taps:
            self.b.set_taps(tuple(taps), c.hidden)
        self.taps = bool(taps)
        self.segs = seg_count(e.slots.count)       # 4 up to four stream slots, one a slot past that (Mia's 0069)
        self.seg_rows = SegRows(rows, dev, max_segs=self.segs, ring=e.caches.ring if e.caches.rings else None)
        self.sel = SelectScratch(rows, e.caches.rows, dev) if e.caches.index is not None else None
        kda_layers = [l for l in w.layers if l.kind == "kda"]
        n = len(kda_layers)
        LL = c.lin_heads // w.world
        width = kda_layers[0].kda.proj.n if kda_layers else 0
        self.kda_index = {l.index: i for i, l in enumerate(kda_layers)}
        self.dsa_index = {l.index: i for i, l in enumerate(l for l in w.layers if l.kind == "dsa")}
        self.proj = torch.zeros((n, rows, width), dtype=torch.bfloat16, device=dev)
        self.scratch = kda_mod.KDAScratchSet(n, rows, LL, dev) if n else None
        self.tmp = (torch.empty((rows, LL, kda_mod.DK), dtype=torch.float32, device=dev),
                    torch.empty((rows, LL, kda_mod.DV), dtype=torch.bfloat16, device=dev)) if n else None
        self.kseg = torch.zeros((self.segs, kda_mod.SEG_COLS), dtype=torch.int32, device=dev)
        self.kseg_host = torch.zeros((self.segs, kda_mod.SEG_COLS), dtype=torch.int32,
                                     pin_memory=torch.cuda.is_available())
        self.staged = torch.cuda.Event() if torch.cuda.is_available() else None
        self.segments: list[Segment] = []
        self.R = 0
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.replays = {"graph": 0, "eager": 0}

    @torch.no_grad()
    def capture(self, rows: Sequence[int] | None = None) -> None:
        """CUDA graphs of the window for each row count (default 1 .. its rows), both ranks together: positions,
        extents, slots and parities live in the device tables, so one graph a row count covers every mix. Captured on
        a window of slot 0 at the pool's first rows (the warm-up runs write there; slot 0's states are cleared after,
        so capture before any stream is admitted). With the indexer the graphs always run the token selection (its
        dense rows ignore it: the same bits as a window that skips it)."""

        from .forward import compute
        from .sparse import SPARSE_FROM

        e, w, b = self.e, self.w, self.b
        index = e.caches.index is not None
        pool = torch.cuda.graph_pool_handle()
        for R in rows or range(1, self.rows_max + 1):
            self._stage([0] * R)
            self.seg_rows.set([self._span(e.home, 0, R)], sparse_from=SPARSE_FROM if index else None)
            self._write_table([(R, 0, 0, 0, R)])
            run = lambda: compute(w, None, b, R, mixer=lambda layer: self._mixer(layer, R, index))  # noqa: E731
            for _ in range(2):
                run()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                run()
            self.graphs[R] = g
        torch.cuda.synchronize()
        e.slots.rec[0].zero_()
        e.slots.conv[0].zero_()

    @torch.no_grad()
    def time_rows(self, reps: int = 5, most: int | None = None) -> list[float]:
        """ms of the captured window of each size (1 .. its rows, at most ``most``): the fastest of ``reps`` replays,
        on the capture's own tables (slot 0 at the pool's first rows, its ring rows with index rings; its states
        cleared after), before any stream is admitted. The rows are ``verify.timing_tokens`` (Mia's 0035 with 0040,
        0041 and 0048)."""

        from .sparse import SPARSE_FROM
        from .verify import timing_tokens

        e = self.e
        index = e.caches.index is not None
        out = []
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        tokens = timing_tokens(self.w.cfg.vocab, self.rows_max)
        for R in range(1, min(most or self.rows_max, self.rows_max) + 1):
            g = self.graphs[R]
            self._stage(tokens[:R])
            self.seg_rows.set([self._span(e.home, 0, R)], sparse_from=SPARSE_FROM if index else None)
            self._write_table([(R, 0, 0, 0, R)])
            best = float("inf")
            for _ in range(reps + 1):
                start.record()
                g.replay()
                stop.record()
                stop.synchronize()
                best = min(best, start.elapsed_time(stop))
            out.append(best)
        e.slots.rec[0].zero_()
        e.slots.conv[0].zero_()
        return out

    # -- the window's tables -------------------------------------------------------------------------------------------
    def _tables(self, segments: Sequence[Segment], keeps: Sequence[int] | None = None) -> None:
        """The KDA segment table (row0, rows, slot, parity, conv slot, keep) into the device copy, in place (a captured
        graph reads the same tensor); unused segments have 0 rows."""

        self._write_table([(len(s.tokens), s.st.slot, s.st.cur[0] if s.st.cur else 0, s.st.slot,
                            len(s.tokens) if keeps is None else keeps[k]) for k, s in enumerate(segments)])

    def _span(self, st, pos: int, n: int) -> tuple:
        """SegRows' segment of ``n`` rows of ``st`` from ``pos``: its extent's base, and with index rings its slot's
        ring base."""
        caches = self.e.caches
        span = (st.base, pos, n)
        return span + (caches.ring_base(st.slot),) if caches.rings else span

    def _write_table(self, rows) -> None:
        from . import kda as kda_mod

        table = kda_mod.segment_table(rows, "cpu", out=torch.zeros_like(self.kseg_host))
        if self.staged is not None:
            self.staged.synchronize()
        self.kseg_host.copy_(table)
        self.kseg.copy_(self.kseg_host, non_blocking=True)
        if self.staged is not None:
            self.staged.record()

    def _stage(self, tokens: list[int]) -> None:
        b = self.b
        R = len(tokens)
        if R > b.rows:
            raise ValueError(f"a window of {R} rows, the batched buffers hold {b.rows}")
        b.staged.synchronize()
        b.ids_host[:R].numpy()[:] = tokens
        b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
        b.staged.record()

    # -- the seam --------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, segments: Sequence[Segment]) -> Verified:
        from .forward import check_room, compute
        from .sparse import SPARSE_FROM

        w, b = self.w, self.b
        index = self.e.caches.index is not None
        spans = []
        for seg in segments:
            check_room(w, seg.st, len(seg.tokens))
            spans.append(self._span(seg.st, seg.st.pos, len(seg.tokens)))
        tokens = [t for seg in segments for t in seg.tokens]
        R = len(tokens)
        self._stage(tokens)
        self.seg_rows.set(spans, sparse_from=SPARSE_FROM if index else None)
        self._tables(segments)
        self.segments, self.R = list(segments), R
        g = self.graphs.get(R)
        if g is not None:
            self.replays["graph"] += 1
            g.replay()
            logits = b.logits[:R]
        else:
            self.replays["eager"] += 1
            select = index and self.seg_rows.any_sparse()
            logits = compute(w, None, b, R, mixer=lambda layer: self._mixer(layer, R, select))
        out, taps, at = [], [], 0
        for seg in segments:
            n = len(seg.tokens)
            out.append(logits[at:at + n])
            taps.append(torch.cat([t[at:at + n] for t in b.taps], dim=1) if self.taps else None)
            at += n
        return Verified(out, taps)

    def _mixer(self, layer, R: int, select: bool):
        """A layer's KDA or DSA block over the whole window, each segment on its own stream's state."""

        from . import prof
        from .forward import dsa_segments, kda_segments

        e = self.e
        if layer.kind == "kda":
            with prof.timed("kda"):
                return kda_segments(layer, self.w, self.b, R, self.kseg, e.slots, self.proj, self.scratch,
                                    self.kda_index[layer.index], self.tmp)
        caches = e.caches
        di = self.dsa_index[layer.index]
        planes = caches.arena.planes
        lc = planes[caches.kc[di]].tensor
        index = None if caches.index is None else caches.index_tensors(di)
        with prof.timed("dsa (total)"):
            return dsa_segments(layer, self.w, lc, self.b, R, self.seg_rows, self.sel, index, select=select)

    @torch.no_grad()
    def commit(self, segments: Sequence[Segment], keeps: Sequence[int]) -> None:
        """Each segment keeps its first ``keeps[k]`` rows: its KDA states replayed to them (when it keeps fewer than
        its rows), its conv window shifted, its parity flipped, its position advanced."""

        from . import kda as kda_mod
        from .forward import conv_shift_segments

        if [id(s) for s in segments] != [id(s) for s in self.segments]:
            raise ValueError("commit takes the segments of the last forward")
        for seg, keep in zip(segments, keeps):
            if not 1 <= keep <= len(seg.tokens):
                raise ValueError("keep must be in 1..rows")
        e = self.e
        if self.scratch is not None:
            self._tables(segments, keeps)
            if any(k < len(s.tokens) for s, k in zip(segments, keeps)):
                kda_mod.replay_layers_segments(self.kseg, e.slots.rec, self.scratch)
            conv_shift_segments(e.slots.conv, self.proj, self.kseg)
        for seg, keep in zip(segments, keeps):
            st = seg.st
            if st.cur:
                st.cur = [1 - st.cur[0]] * len(st.cur)
            st.set_pos(st.pos + keep)
        self.segments = []

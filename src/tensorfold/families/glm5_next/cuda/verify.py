"""The verify seam of GLM-5.3-Flash's concurrent streams (``multi.MultiDecoder``): a round's windows, one per stream
(its state: extent and slot views; its tokens: the pending token, then its drafts), in, each window's logits and
DFlash2 taps out, then each stream's commit of the rows it keeps. Two implementations of the same two calls:

- ``SerialVerify``: each window as its own forward on its own state (the solo kernels: exact by construction; a CUDA
  graph where the state is the one they were captured on);
- ``BatchedVerify``: ONE forward over every stream's rows back to back (``verify_batched.py``).

Ported from Mia's patch 0030 (``verify.py``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass
class Segment:
    """One stream's rows of a round's verify window: its state (extent and slot views) and tokens (the pending
    token, then its drafts)."""

    st: Any
    tokens: list[int]


@dataclass
class Verified:
    logits: list[torch.Tensor]              # per segment [rows, V / world] (this rank's vocabulary columns)
    taps: list[torch.Tensor | None]         # per segment [rows, taps * D] (DFlash2's inputs), or None


class SerialVerify:
    """The reference: each segment's window as its own forward on its own state. Logits and taps are copied out
    before the next segment's forward reuses the buffers; each slot keeps its own KDA window scratch and projections,
    so the commits can come after every forward."""

    def __init__(self, e, *, taps: bool) -> None:
        self.e, self.taps = e, taps

    @torch.no_grad()
    def forward(self, segments: Sequence[Segment]) -> Verified:
        e = self.e
        logits, taps = [], []
        for seg in segments:
            prev = e.use(seg.st)
            try:
                R = len(seg.tokens)
                logits.append(e.forward(seg.tokens)[:R].clone())
                taps.append(e.tap_rows(R).clone() if self.taps else None)
            finally:
                e.use(prev)
        return Verified(logits, taps)

    @torch.no_grad()
    def commit(self, segments: Sequence[Segment], keeps: Sequence[int]) -> None:
        from .forward import commit

        e = self.e
        for seg, keep in zip(segments, keeps):
            commit(e.w, seg.st, e.buf, len(seg.tokens), keep)


def timing_tokens(vocab: int, n: int) -> list[int]:
    """Fixed, distinct token ids for startup timings of verify windows and prompt chunks (Mia's 0048: rows of real
    text route to many experts; one token repeated routes every row to the same few)."""

    import numpy as np

    pool = min(vocab, 150000)
    ids = [int(t) for t in np.random.default_rng(1234).choice(pool, size=min(n, pool), replace=False)]
    return (ids * -(-n // len(ids)))[:n] if n > pool else ids      # a small (test) vocabulary: the ids again

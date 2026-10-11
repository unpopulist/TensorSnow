"""Mia's multi-prompt chunks (her 0049; port-prefill-8s), TF_GLM_MULTI_PREFILL=1 under --parallel: one prompt chunk
holds the next rows of several streams' prompts. Contract: each stream's rows get the bits of a chunk of their own
(``decode.prefill_chunk`` on its state): KDA states, conv windows, latent and indexer caches, DFlash2 taps, head
logits; and through the scheduler every reply equals its solo ``draft=False`` reply.

Checked on the synthetic MLX and EXL3 checkpoints (dense and past the dense limit), with the hyper-connection split
between two ranks (``glm_two_ranks``), and on two real RED-SNOW layers of a rank (X3 / Stack projections, universal
EXL3 experts) when the checkpoint is there."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_engine import _checkpoint, _TwoCopies  # noqa: E402

MODEL = Path(os.environ.get("TENSORFOLD_REDSNOW_MODEL", "/var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw"))


def _states(e, cap: int, n: int):
    from tensorfold.families.glm5_next.cuda.forward import State

    return [State(e.w, cap, e.rows, caches=e.caches, base=k * cap, slots=e.slots, slot=k) for k in range(n)]


def _view(st) -> list[torch.Tensor]:
    out = [st.rec[st.cur[0]].clone(), st.conv.clone()]
    out += [x[:st.pos].clone() for x in st.kc + st.vc if x is not None]
    if st.index is not None:
        for ik, ig, pk in st.index:
            out += [ik[:st.pos].clone(), ig[:st.pos].clone(), pk[:st.pos // 4].clone()]
    return out


class _Taps:
    """A stand-in DFlash2 context: the tap rows it was given, in order."""

    def __init__(self) -> None:
        self.rows: list[torch.Tensor] = []

    def add_taps(self, rows: torch.Tensor) -> None:
        self.rows.append(rows.clone())


def _schedule(prompts, rows: int):
    """Chunks per round: each stream's next ``rows`` (at most), every round all unfinished streams together."""

    pos = [0] * len(prompts)
    rounds = []
    while any(p < len(q) for p, q in zip(pos, prompts)):
        r = []
        for k, q in enumerate(prompts):
            if pos[k] < len(q):
                stop = min(len(q), pos[k] + rows)
                r.append((k, pos[k], stop))
                pos[k] = stop
        rounds.append(r)
    return rounds


def _solo_and_grouped(w, prompts, *, cap: int, rows: int, long_context: bool, split=None):
    """(solo, grouped) results per stream: heads, tap rows, final state."""

    from tensorfold.families.glm5_next.cuda import multi_prefill
    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill_chunk

    n = len(prompts)
    e = Engine(w, capacity=cap, max_rows=8, prefill_rows=rows * n, long_context=long_context, streams=n,
               pool_rows=n * cap, taps=(0, 1), split=split)
    sts = _states(e, cap, n)
    plan = _schedule(prompts, rows)
    solo = []
    for k, q in enumerate(prompts):
        st = sts[k]
        st.reset()
        prev = e.use(st)
        taps = _Taps()
        heads = []
        for rnd in plan:
            for kk, a, z in rnd:
                if kk == k:
                    out = prefill_chunk(e, q, a, z - a, drafter=taps, head=z == len(q))
                    if out is not None:
                        heads.append(out[:1].clone())
        e.use(prev)
        solo.append((heads, taps.rows, _view(st)))
    for st in sts:
        st.reset()
    taps = [_Taps() for _ in prompts]
    heads = [[] for _ in prompts]
    for rnd in plan:
        pieces = [multi_prefill.Piece(sts[k], list(prompts[k][a:z]), drafter=taps[k], head=z == len(prompts[k]))
                  for k, a, z in rnd]
        for (k, _, _), h in zip(rnd, multi_prefill.prefill_pieces(e, pieces)):
            if h is not None:
                heads[k].append(h)
    grouped = [(heads[k], taps[k].rows, _view(sts[k])) for k in range(n)]
    torch.cuda.synchronize()
    return solo, grouped


def _same(solo, grouped):
    for k, (a, b) in enumerate(zip(solo, grouped)):
        for part, (x, y) in enumerate(zip(a, b)):
            assert len(x) == len(y), (k, part)
            for i, (u, v) in enumerate(zip(x, y)):
                assert torch.equal(u, v), (k, part, i)


@pytest.fixture(scope="module", params=["mlx", "exl3"])
def weights_one(request, tmp_path_factory):
    from tensorfold.families.glm5_next.cuda import weights

    path = tmp_path_factory.mktemp("glm_mp_" + request.param)
    _checkpoint(path, exl3=request.param == "exl3", mtp=False)
    w = weights.load(path, rank=0, device="cuda", mtp=False, world=2)
    w.comm = _TwoCopies()
    return w, path


def _prompts(seed, sizes):
    rng = np.random.default_rng(seed)
    return [[int(t) for t in rng.integers(0, 1000, size=n)] for n in sizes]


def test_pieces_equal_their_own_chunks(weights_one):
    w, _ = weights_one
    solo, grouped = _solo_and_grouped(w, _prompts(1, (300, 77, 190)), cap=2560, rows=128, long_context=False)
    _same(solo, grouped)


def test_pieces_past_the_dense_limit(weights_one):
    w, _ = weights_one
    solo, grouped = _solo_and_grouped(w, _prompts(2, (2300, 150)), cap=2600, rows=1024, long_context=True)
    _same(solo, grouped)


def test_pieces_with_the_split_between_two_ranks(weights_one):
    from glm_two_ranks import both, load_pair

    from tensorfold.families.glm5_next.cuda.hcsplit import SplitSettings

    _, path = weights_one
    ws, hub = load_pair(path)
    prompts = _prompts(3, (301, 120, 64))
    split = SplitSettings(True, True, 2, fronts=True)
    res = both(hub, *[lambda r=r: _solo_and_grouped(ws[r], prompts, cap=2560, rows=128, long_context=False,
                                                    split=split) for r in (0, 1)])
    for solo, grouped in res:
        _same(solo, grouped)


@pytest.mark.skipif(not (MODEL / "model.safetensors.index.json").exists(), reason="no RED-SNOW checkpoint")
def test_redsnow_layers_pieces_equal_their_own_chunks():
    from tensorfold.families.glm5_next.cuda import weights

    w = weights.load(MODEL, rank=1, device="cuda", mtp=False, world=2, layers=[3, 4])
    w.comm = _TwoCopies()
    w.comm.rank = 1
    try:
        rng = np.random.default_rng(4)
        prompts = [[int(t) for t in rng.integers(0, 150000, size=n)] for n in (700, 260, 1100)]
        solo, grouped = _solo_and_grouped(w, prompts, cap=2560, rows=512, long_context=False)
        _same(solo, grouped)
    finally:
        del w
        torch.cuda.empty_cache()


def test_settings():
    from tensorfold.families.glm5_next.cuda import multi_prefill

    assert multi_prefill.enabled("") is False and multi_prefill.enabled("1") is True
    assert multi_prefill.wait_ms("") == 0 and multi_prefill.wait_ms("250") == 250
    with pytest.raises(ValueError):
        multi_prefill.enabled("2")
    with pytest.raises(ValueError):
        multi_prefill.wait_ms("5000")

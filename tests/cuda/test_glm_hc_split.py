"""Mia's hyper-connection split (her 0010, 0017, 0022, 0033; port-prefill-8s): prompt chunks' glue split by rows
between the two ranks, its exchanges overlapped on a second stream (TF_GLM_PREFILL_OVERLAP=1) and the next block's
front run during them (=2). Contract: every bit of the unsplit two-rank prefill: first token, head logits, KDA
states and conv windows, latent and indexer caches, DFlash2 taps, on both ranks.

Both ranks run in one process (``glm_two_ranks``: two threads, their collectives paired), so the partials summed are
the real two halves. The synthetic checkpoints of test_glm_engine (MLX 4-bit, EXL3 experts), and two real RED-SNOW
layers per rank (X3 / Stack dense projections in every front: KDA qkv, DSA q_a/kv_a and q_b, the dense MLP's
gate/up, the shared expert), when the checkpoint is there."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from glm_two_ranks import both, load_pair  # noqa: E402
from test_glm_engine import _checkpoint  # noqa: E402

from tensorfold.families.glm5_next.cuda.hcsplit import SplitSettings  # noqa: E402

MODEL = Path(os.environ.get("TENSORFOLD_REDSNOW_MODEL", "/var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw"))

SETTINGS = {
    "split": SplitSettings(True),
    "overlap1": SplitSettings(True, True, 2),
    "overlap1x3": SplitSettings(True, True, 3),
    "overlap2": SplitSettings(True, True, 2, fronts=True),
    "gather": SplitSettings(True, exchange="gather"),
    "gather-overlap2": SplitSettings(True, True, 2, exchange="gather", fronts=True),
}


def _snapshot(e, last: int) -> list[torch.Tensor]:
    """What the prompt leaves; the taps of the last chunk's ``last`` rows (rows past them, the split's pad row
    among them, are scratch nothing reads)."""
    st, b = e.st, e.pbuf
    out = [st.rec[st.cur[0]].clone(), st.conv.clone()]
    out += [x[:st.pos].clone() for x in st.kc + st.vc if x is not None]
    if st.index is not None:
        for ik, ig, pk in st.index:
            out += [ik[:st.pos].clone(), ig[:st.pos].clone(), pk[:st.pos // 4].clone()]
    out += [t[:last].clone() for t in b.taps]
    out.append(b.logits[:1].clone())
    return out


def _run(ws, hub, prompt, settings, *, rows, capacity, long_context=False, taps=()):
    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill

    def rank(r):
        def go():
            start = ws[r].comm.calls
            e = Engine(ws[r], capacity=capacity, max_rows=8, prefill_rows=rows, long_context=long_context,
                       taps=taps, split=settings)
            first = prefill(e, prompt, None, mtp=False)
            torch.cuda.synchronize()
            return first, _snapshot(e, len(prompt) - (len(prompt) - 1) // rows * rows), ws[r].comm.calls - start
        return go

    return both(hub, rank(0), rank(1))


def _same(a, b, label):
    (fa, sa, _), (fb, sb, _) = a, b
    assert fa == fb, label
    assert len(sa) == len(sb)
    for i, (x, y) in enumerate(zip(sa, sb)):
        assert torch.equal(x, y), (label, i)


@pytest.fixture(scope="module", params=["mlx", "exl3"])
def pair(request, tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_split_" + request.param)
    _checkpoint(path, exl3=request.param == "exl3", mtp=False)
    ws, hub = load_pair(path)
    yield ws, hub
    del ws
    torch.cuda.empty_cache()


@pytest.mark.parametrize("name", list(SETTINGS))
def test_split_prefill_equals_unsplit(pair, name):
    ws, hub = pair
    prompt = [int(t) for t in np.random.default_rng(10).integers(0, 1000, size=301)]
    # 128-row chunks (two split, then a 45-row tail below min_rows), and an odd 301-row chunk (the pad row)
    for rows in (128, 512):
        want = _run(ws, hub, prompt, None, rows=rows, capacity=2560, taps=(0, 1))
        got = _run(ws, hub, prompt, SETTINGS[name], rows=rows, capacity=2560, taps=(0, 1))
        for r in (0, 1):
            _same(want[r], got[r], (name, rows, r))
        assert want[0][0] == want[1][0]
        assert got[0][2] != want[0][2]                    # the split path really ran (other collectives)


def test_split_past_the_dense_limit(pair):
    """Sparse rows (the indexer caches, one-pass sparse attention) under the split, overlap level 2."""

    ws, hub = pair
    prompt = [int(t) for t in np.random.default_rng(11).integers(0, 1000, size=2400)]
    want = _run(ws, hub, prompt, None, rows=1024, capacity=2600, long_context=True)
    got = _run(ws, hub, prompt, SETTINGS["overlap2"], rows=1024, capacity=2600, long_context=True)
    for r in (0, 1):
        _same(want[r], got[r], ("long", r))


def test_settings_from_env():
    assert SplitSettings.from_env({}) == SplitSettings()
    s = SplitSettings.from_env({"TF_GLM_HC_SPLIT": "1", "TF_GLM_PREFILL_OVERLAP": "2"})
    assert s.split and s.overlap and s.fronts and s.code() == [1, 2, 2, 64, 0]
    assert SplitSettings.from_env({"TF_GLM_HC_SPLIT": "1"}).code() == [1, 0, 1, 64, 0]
    with pytest.raises(ValueError):
        SplitSettings.from_env({"TF_GLM_PREFILL_OVERLAP": "1"})
    with pytest.raises(ValueError):
        SplitSettings.from_env({"TF_GLM_HC_SPLIT": "1", "TF_GLM_PREFILL_OVERLAP": "3"})
    with pytest.raises(ValueError):
        SplitSettings(True, False, fronts=True)


@pytest.mark.skipif(not (MODEL / "model.safetensors.index.json").exists(), reason="no RED-SNOW checkpoint")
def test_redsnow_layers_split_equals_unsplit():
    """RED-SNOW layers 0 (KDA, dense MLP) and 3 (DSA, MoE) of both ranks: the X3 / Stack prompt GEMMs in every front
    run on row ranges under the split and keep the unsplit bits."""

    ws, hub = load_pair(MODEL, layers=[0, 3])
    try:
        prompt = [int(t) for t in np.random.default_rng(12).integers(0, 150000, size=1500)]
        want = _run(ws, hub, prompt, None, rows=1024, capacity=2560)
        for name in ("split", "overlap2"):
            got = _run(ws, hub, prompt, SETTINGS[name], rows=1024, capacity=2560)
            for r in (0, 1):
                _same(want[r], got[r], (name, r))
    finally:
        del ws
        torch.cuda.empty_cache()

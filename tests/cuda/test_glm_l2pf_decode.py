"""TF_GLM_L2PF (``glm5_next/cuda/l2pf``, Mia's 0046) wired into the decode path: speed only, so every reply and every
decode window's logits keep their bits with the prefetch on.

- decode windows of 1 .. 16 rows, eager and as CUDA graphs, each mechanism (bulk / lines / touch): the logits of the
  window with the prefetch equal those without, bit for bit (tiny synthetic GLM of ``test_glm_engine``; and two real
  RED-SNOW layers, X3 / Stack dense weights, when the checkpoint is there);
- drafted == serial with TF_GLM_L2PF=1, and the replies equal an engine's without it (MLX 4-bit and EXL3 checkpoints);
- --parallel 3 batched verify windows (segmented KDA / DSA, graphs) with TF_GLM_L2PF=1: every stream's reply its solo
  serial one from an engine without the prefetch;
- the ranks' startup settings vector carries TF_GLM_L2PF: two ranks started differently refuse to start."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_engine import _checkpoint, _drafter, _generate, _TwoCopies  # noqa: E402
from test_glm_multi import _concurrent, _prompts, _solo, _with_env, SAMPLINGS  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import l2pf  # noqa: E402

MODEL = Path(os.environ.get("TENSORFOLD_REDSNOW_MODEL", "/var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw"))
ON = {"TF_GLM_L2PF": "1"}


def _engine(path, **kw):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies(), **kw)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_l2pf")
    _checkpoint(path / "model")
    _checkpoint(path / "exl3", exl3=True)
    _drafter(path / "dflash2")
    return path


def _windows(e, prompt, rows=range(1, 17)):
    """Each window's logits from the state the prompt leaves (a window not committed rewrites the same rows)."""

    from tensorfold.families.glm5_next.cuda.decode import prefill

    e.reset()
    prefill(e, prompt, None, mtp=False)
    out = []
    for R in rows:
        out.append(e.forward(prompt[-R:]).clone())
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("kind", ["model", "exl3"])
def test_windows_keep_their_bits(models, kind):
    """1 .. 16-row windows (eager past 4 rows, graphs for 1 .. 4) with the prefetch in each mode equal the windows
    without it."""

    from tensorfold.families.glm5_next.cuda import weights
    from tensorfold.families.glm5_next.cuda.decode import Engine

    w = weights.load(models / kind, rank=0, device="cuda", mtp=False, world=2)
    w.comm = _TwoCopies()
    prompt = [int(t) for t in np.random.default_rng(21).integers(0, 1000, size=40)]
    ref = Engine(w, capacity=512, max_rows=16, prefill_rows=32, graphs=True)
    want = _windows(ref, prompt)
    del ref
    for mode in ("bulk", "lines", "touch"):
        w.meta["l2pf"] = l2pf.Prefetch(w, l2pf.Settings.from_env({"TF_GLM_L2PF": mode}))
        pf = w.meta["l2pf"]
        assert set(n for _, n in pf.sites) == set("afo")
        e = Engine(w, capacity=512, max_rows=16, prefill_rows=32, graphs=True)
        assert pf.launches > 0                              # the captures took the sites
        before = pf.launches
        got = _windows(e, prompt)
        assert pf.launches > before                         # eager windows (5 .. 16 rows) prefetched too
        assert e.replays["main"] > 0 and e.replays["eager"] > 0
        for R, (a, b) in enumerate(zip(want, got), 1):
            assert torch.equal(a, b), (mode, R)
        assert l2pf.ACTIVE is None
        del e
    w.meta.pop("l2pf")


@pytest.mark.parametrize("kind", ["model", "exl3"])
def test_drafted_equals_serial_with_prefetch(models, kind):
    """The engine with TF_GLM_L2PF=1 (graphed verify windows, MTP head, DFlash2): every policy's reply equals the
    serial one, and both equal the replies of an engine without the prefetch."""

    prompt = list(np.random.default_rng(22).integers(0, 1000, size=41))
    samplings = [Sampling(1234, 1.0, 20, 0.95), None]
    off = _engine(models / kind, drafter=models / "dflash2")
    want = [_generate(off, prompt, s, draft=False, tokens=28)[0] for s in samplings]
    del off
    on = _with_env(ON, lambda: _engine(models / kind, drafter=models / "dflash2"))
    pf = on.w.meta.get("l2pf")
    assert pf is not None and pf.launches > 0
    for s, ref in zip(samplings, want):
        serial, _ = _generate(on, prompt, s, draft=False, tokens=28)
        assert serial == ref
        for policy in (None, "2", "f3", "c3:0.35"):
            drafted, _ = _generate(on, prompt, s, policy=policy, tokens=28)
            assert drafted == ref, policy
    assert on.e.replays["main"] > 0


@pytest.mark.parametrize("kind", ["model", "exl3"])
def test_batched_streams_equal_solo_with_prefetch(models, kind):
    """--parallel 3, batched verify windows as graphs with TF_GLM_L2PF=1: every stream's reply is its solo serial
    one from an engine without the prefetch."""

    from tensorfold.families.glm5_next.cuda.verify_batched import BatchedVerify

    jobs = [(p, s, 20) for p, s in zip(_prompts(41, (37, 66, 23)), SAMPLINGS)]
    solo = _engine(models / kind, drafter=models / "dflash2")
    want = [_solo(solo, p, s, n) for p, s, n in jobs]
    del solo
    m = _with_env({**ON, "TF_GLM_MULTI_VERIFY": "batched", "TF_GLM_MULTI_GRAPHS": "1"},
                  lambda: _engine(models / kind, drafter=models / "dflash2", parallel=3,
                                  multi_options={"draft_graphs": False}))
    try:
        v = m.multi.verify
        assert isinstance(v, BatchedVerify) and v.graphs
        pf = m.w.meta["l2pf"]
        assert v.rows_max <= pf.s.rows                       # every batched window takes the sites
        got, _ = _concurrent(m, jobs)
        assert got == want
        for policy in ("0", "f3"):
            got, _ = _concurrent(m, jobs, policy=policy)
            assert got == want, policy
        assert v.replays["graph"] > 0
    finally:
        m.scheduler.close()


def test_ranks_must_agree_on_the_prefetch(models, monkeypatch):
    """TF_GLM_L2PF and its knobs are in the startup settings vector: two ranks started differently refuse."""

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    class Stop(Exception):
        pass

    seen: list[list[int]] = []

    inner = GlmEngine._gather_ints

    def grab(self, values):
        if len(values) < 20:                  # capacity admission's rows: as the one-GPU stand-in gives them
            return inner(self, values)
        seen.append(list(values))             # the settings vector
        raise Stop

    monkeypatch.setattr(GlmEngine, "_gather_ints", grab)
    for env in ({"TF_GLM_L2PF": "0"}, ON, {"TF_GLM_L2PF": "1", "TF_GLM_L2PF_MB": "4"}, ON):
        with pytest.raises(Stop):
            _with_env(env, lambda: _engine(models / "model"))
    assert seen[0] != seen[1] and seen[1] != seen[2] and seen[1] == seen[3]
    assert [a for a, b in zip(seen[0], seen[1]) if a != b]
    monkeypatch.setattr(GlmEngine, "_gather_ints",
                        lambda self, values: inner(self, values) if len(values) < 20 else [list(values), seen[0]])
    with pytest.raises(RuntimeError, match="different settings"):
        _with_env(ON, lambda: _engine(models / "model"))


def test_settings_ints():
    assert l2pf.Settings.from_env({}).ints() == [0] * 7
    a = l2pf.Settings.from_env({"TF_GLM_L2PF": "1"}).ints()
    assert a == [1, 8192, 7, 64, 0, 128, 32]
    assert l2pf.Settings.from_env({"TF_GLM_L2PF": "touch", "TF_GLM_L2PF_SITES": "fa"}).ints()[:3] == [3, 8192, 3]


@pytest.mark.skipif(not (MODEL / "model.safetensors.index.json").exists(), reason="no RED-SNOW checkpoint")
def test_redsnow_layers_windows_keep_their_bits():
    """RED-SNOW layers 0 (KDA, dense MLP) and 3 (DSA, MoE), rank 0 of two, X3 / Stack dense weights and the 8-bit
    head: 1 .. 16-row windows (graphs for 1, 2, 4, 8) with the prefetch equal those without; the plan covers the
    DSA ``o`` site's value half of kv_b."""

    from tensorfold.families.glm5_next.cuda import weights
    from tensorfold.families.glm5_next.cuda.decode import Engine

    w = weights.load(MODEL, rank=0, device="cuda", mtp=False, world=2, layers=[0, 3])
    w.comm = _TwoCopies()
    try:
        prompt = [int(t) for t in np.random.default_rng(23).integers(0, 150000, size=60)]
        kw = dict(capacity=512, max_rows=16, prefill_rows=64, graphs=True, graph_rows=(1, 2, 4, 8))
        ref = Engine(w, **kw)
        want = _windows(ref, prompt)
        del ref
        pf = l2pf.Prefetch(w, l2pf.Settings.from_env(ON))
        dsa = next(l for l in w.layers if l.dsa is not None)
        first, count, _ = pf.sites[(dsa.index, "o")]
        rows = pf.table[first:first + count].tolist()
        value = [t for t in l2pf.tensors(pf._out(dsa)[:-1])]
        assert value and any(t.data_ptr() <= a < t.data_ptr() + t.numel() * t.element_size()
                             for a, _ in rows for t in value)            # kv_b's value half, not only o_proj
        w.meta["l2pf"] = pf
        e = Engine(w, **kw)
        got = _windows(e, prompt)
        for R, (a, b) in enumerate(zip(want, got), 1):
            assert torch.equal(a, b), R
        assert e.replays["main"] > 0 and pf.launches > 0
    finally:
        w.meta.pop("l2pf", None)
        del w
        torch.cuda.empty_cache()

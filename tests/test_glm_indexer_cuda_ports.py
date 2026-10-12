"""Mia's 0105 / 0107 (BadAd84): the indexer's CUDA scoring extensions' start-up and fallback, on the CPU.

Contracts:
  - the extensions are used under Triton 3.7.x only (sparse.TRITON_PTX compares major.minor: the served image's 3.7.1
    matches), and a failed build or another release falls back to Triton without raising, logged once;
  - decode.Engine's constructor loads both before any request with FP8 caches and DSA layers, not with bf16.
The kernels' bits are GPU checks (tests/cuda/test_glm_indexer_cuda_ports.py)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tensorfold.families.glm5_next.cuda import decode, forward, sparse


@pytest.fixture
def fresh(monkeypatch):
    for name in ("_SCORES_EXT", "_SCORES_FAILED", "_SEG_SCORES_EXT", "_SEG_SCORES_FAILED"):
        monkeypatch.setattr(sparse, name, None)


def test_triton_release_gate_and_build_failure_fall_back(monkeypatch, fresh, capsys):
    from tensorfold.cuda import build

    monkeypatch.setattr(sparse.triton, "__version__", "3.7.1")
    assert sparse.triton_matches_ptx()
    loads = []

    def failing(**kw):
        loads.append(kw["name"])
        raise RuntimeError("Error building extension: nvcc not found")

    monkeypatch.setattr(build, "load", failing)
    assert sparse._scores_ext() is None and sparse._seg_scores_ext() is None
    assert sparse._scores_ext() is None                                     # not retried
    assert loads == ["tensorfold_glm_scores_v1", "tensorfold_glm_segscores_v1"]
    assert not sparse.prompt_scores_cuda() and not sparse.seg_scores_cuda()
    assert capsys.readouterr().out.count("is not used") == 2

    for name in ("_SCORES_EXT", "_SCORES_FAILED"):
        monkeypatch.setattr(sparse, name, None)
    monkeypatch.setattr(sparse.triton, "__version__", "3.8.0")
    assert not sparse.triton_matches_ptx()
    assert sparse._scores_ext() is None and len(loads) == 2                 # the loader never called


def _construct(monkeypatch, kv):
    calls = []
    monkeypatch.setattr(sparse, "_scores_ext", lambda: calls.append("scores"))
    monkeypatch.setattr(sparse, "_seg_scores_ext", lambda: calls.append("seg"))
    stub = lambda *a, **k: SimpleNamespace(rows=1)                          # noqa: E731
    for mod, name in ((decode, "Buffers"), (decode, "State"), (forward, "Caches"), (forward, "Slots")):
        monkeypatch.setattr(mod, name, stub)
    monkeypatch.setattr(forward, "index_ring", lambda *a, **k: 0)
    monkeypatch.setenv("TF_GLM_L2PF", "0")
    monkeypatch.delenv("TF_GLM_SCORES_CUDA", raising=False)
    monkeypatch.delenv("TF_GLM_SEG_SCORES_CUDA", raising=False)
    s = SimpleNamespace
    w = s(meta={}, layers=[s(kind="dsa")], world=2, rank=0, mtp=None, head=s(n=1), cfg=s(hidden=4096), device="cpu")
    decode.Engine(w, kv=kv)
    return calls


def test_engine_loads_the_extensions_at_start_for_fp8_only(monkeypatch, fresh):
    assert _construct(monkeypatch, "fp8") == ["scores", "seg"]
    assert _construct(monkeypatch, "bf16") == []

"""GLM-5.3-Flash's engine with TF_GLM_KV=fp8 (kv8.py) on the tiny synthetic checkpoint of test_glm_engine.py, past
the dense limit: the latent caches and the indexer's pooled keys are FP8 rows, and TensorFold's exactness rule still
holds: drafted replies equal serial ones, sparse CUDA graphs equal eager steps, and a resumed prompt (the same
conversation, or one whose saved rows come back after another took the caches) equals a fresh prefill."""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

from test_glm_engine import _checkpoint, _forget, _generate, _TwoCopies  # noqa: E402  (pytest puts tests/cuda on sys.path)

PROMPT = 2100              # past the dense limit: the first reply token is already a sparse row
CONTEXT = 2600


@pytest.fixture(scope="module")
def engine_fp8(tmp_path_factory):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    path = tmp_path_factory.mktemp("glm_kv8")
    _checkpoint(path)
    return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies(), context=CONTEXT, prefill_rows=256, kv="fp8")


def _prompt(seed=11, n=PROMPT):
    return list(np.random.default_rng(seed).integers(0, 1000, size=n))


def test_caches_are_fp8_rows(engine_fp8):
    from tensorfold.families.glm5_next.cuda import kv8

    st = engine_fp8.e.st
    assert engine_fp8.kv == "fp8" and st.kv == "fp8" and engine_fp8.capacity_plan["kv_dtype"] == "fp8"
    lw = engine_fp8.w.cfg.kv_lora
    for kc in st.kc + ([st.mtp_kc] if hasattr(st, "mtp_kc") else []):
        assert kc.dtype == torch.uint8 and kc.shape[1] == lw + kv8.PAD
    assert st.index is not None
    for ik, ig, pk in st.index:
        assert ik.dtype == ig.dtype == torch.bfloat16                      # per-token keys and gates stay bf16
        assert pk.dtype == torch.uint8 and kv8.width(pk) == ik.shape[1]


@pytest.mark.parametrize("n", [300, PROMPT], ids=["dense", "sparse"])
@pytest.mark.parametrize("sampling", [Sampling(99, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_fp8_drafted_equals_serial(engine_fp8, sampling, n):
    prompt = _prompt(n=n)
    serial, stats = _generate(engine_fp8, prompt, sampling, draft=False, tokens=32)
    assert len(serial) == 32 and stats["drafts"] is False
    for policy in (None, "2", "c3:0.35"):
        drafted, _ = _generate(engine_fp8, prompt, sampling, policy=policy, tokens=32)
        assert drafted == serial, policy


def test_fp8_sparse_graphs_equal_eager_steps(engine_fp8):
    sampling = Sampling(5, 1.0, 20, 0.95)
    prompt = _prompt(seed=12)
    before = dict(engine_fp8.e.replays)
    with_graphs, _ = _generate(engine_fp8, prompt, sampling, tokens=32)
    assert engine_fp8.e.replays["sparse"] > before["sparse"]
    graphs, engine_fp8.e.graphs = engine_fp8.e.graphs, None
    try:
        eager, _ = _generate(engine_fp8, prompt, sampling, tokens=32)
    finally:
        engine_fp8.e.graphs = graphs
    assert with_graphs == eager


def test_fp8_switching_conversations_resumes_like_a_fresh_prefill(engine_fp8):
    """A, then B (which takes the live caches: A's FP8 rows are saved), then A again from its saved rows: the same
    reply as a fresh prefill."""
    sampling = Sampling(31, 1.0, 20, 0.95)
    a = _prompt(seed=40, n=2080)
    b = _prompt(seed=41, n=2097)
    reply_a, _ = _generate(engine_fp8, a, sampling, tokens=12)
    _generate(engine_fp8, b, sampling, tokens=12)
    next_a = a + reply_a + _prompt(seed=42, n=9)
    warm_a, stats = _generate(engine_fp8, next_a, sampling, tokens=12)
    assert stats["cached"] == len(a), stats
    _forget(engine_fp8)
    cold_a, stats = _generate(engine_fp8, next_a, sampling, tokens=12)
    assert stats["cached"] == 0 and warm_a == cold_a

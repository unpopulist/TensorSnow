"""TF_GLM_KV=fp8 under --parallel (the merge's requirement A, end to end): a batched / serial multi-stream engine
with an fp8 DSA cache replies exactly as a --parallel 1 engine with the same fp8 cache and draft=False. The batched
verify runs the segmented fp8 kernels (latent.seg_attention / sparse.seg_select_tokens); the serial reference runs
the single-stream fp8 kernels (test_glm_kv8 validates those), so the match proves the fp8 cache is read as e4m3 under
--parallel. One GPU plays rank 0 of two, as in test_glm_multi."""

from __future__ import annotations

import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_multi import SAMPLINGS, _concurrent, _engine, _prompts, _solo, _with_env  # noqa: E402
from test_glm_engine import _checkpoint, _drafter  # noqa: E402

os.environ.setdefault("TF_GLM_CACHE_GIB", "0.05")


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_multi_kv8")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def solo_fp8(models):
    return _engine(models / "model", drafter=models / "dflash2", kv="fp8")


@pytest.fixture(scope="module", params=["serial", "batched"])
def multi_fp8(models, request):
    e = _with_env({"TF_GLM_MULTI_VERIFY": request.param},
                  lambda: _engine(models / "model", drafter=models / "dflash2", parallel=3, kv="fp8",
                                  multi_options={"draft_graphs": False}))
    yield e
    e.scheduler.close()


def test_fp8_streams_equal_their_solo_fp8_replies(solo_fp8, multi_fp8):
    prompts = _prompts(41, (37, 80, 23))
    jobs = [(p, s, 24) for p, s in zip(prompts, SAMPLINGS)]
    want = [_solo(solo_fp8, p, s, n) for p, s, n in jobs]
    got, stats = _concurrent(multi_fp8, jobs)
    assert got == want
    assert all(st["parallel"] == 3 for st in stats)


def test_the_parallel_pool_caches_are_fp8_uint8(multi_fp8):
    """The --parallel pool's latent and pooled-key planes are kv8 uint8 rows (keys/gates stay bf16)."""
    caches = multi_fp8.e.caches
    assert caches.kv == "fp8"
    planes = caches.arena.planes
    assert planes[caches.kc[0]].tensor.dtype == torch.uint8
    if caches.index is not None:
        ik, ig, pk = caches.index_tensors(0)
        assert ik.dtype == torch.bfloat16 and ig.dtype == torch.bfloat16
        assert pk.dtype == torch.uint8

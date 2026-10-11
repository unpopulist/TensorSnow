"""Copy drafts (Mia's 0007, TF_GLM_COPY_DRAFTS=1) on the tiny synthetic GLM: they only propose, so a drafted reply
equals the serial (draft=False) reply token for token, alone (MTP / DFlash2 decode) and under --parallel 3 beside
streams that copy less. The tiny random model never repeats 2 tokens, so the tests then match 1 token (set on the
engine; the env allows 2 and up): proposals really fire, and verification decides."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_engine import _checkpoint, _drafter  # noqa: E402
from test_glm_multi import _concurrent, _engine, _prompts, _solo, _with_env  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

COPY = {"TF_GLM_COPY_DRAFTS": "1", "TF_GLM_COPY_MATCH": "2"}
SAMPLINGS = [None, Sampling(1234, 1.0, 20, 0.95), Sampling(77, 0.8, 0, 0.9)]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_copy")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def plain(models):
    return _engine(models / "model", drafter=models / "dflash2")


def _match_one(engine):
    from tensorfold.families.glm5_next.cuda.copy_drafts import CopySettings

    engine.copy = CopySettings(1, engine.copy.most)
    return engine


@pytest.fixture(scope="module")
def copying(models):
    e = _with_env(COPY, lambda: _engine(models / "model", drafter=models / "dflash2"))
    assert e.copy is not None and e.copy.match == 2
    return _match_one(e)


def test_copy_settings_reach_the_engine(copying, plain):
    assert copying.copy is not None and plain.copy is None


@pytest.mark.parametrize("k", range(3))
def test_drafted_reply_with_copies_equals_serial(copying, plain, k):
    prompt = _prompts(5 + k, (40,))[0]
    want = _solo(plain, prompt, SAMPLINGS[k], 64)
    out: list[int] = []
    copying.request.policy = None
    copying.request.stop_eos = False
    stats = copying.generate(list(prompt), 64, SAMPLINGS[k], lambda new: out.extend(new))
    assert out == want
    assert stats.get("copy_rounds", 0) > 0, stats


@pytest.mark.parametrize("verify", ["serial", "batched"])
def test_parallel_streams_with_copies_equal_their_solo_replies(models, plain, verify):
    multi = _with_env({**COPY, "TF_GLM_MULTI_VERIFY": verify},
                      lambda: _engine(models / "model", drafter=models / "dflash2", parallel=3,
                                      multi_options={"draft_graphs": False}))
    _match_one(multi)
    try:
        prompts = _prompts(41, (37, 80, 23))
        jobs = [(p, s, 64) for p, s in zip(prompts, SAMPLINGS)]
        want = [_solo(plain, p, s, n) for p, s, n in jobs]
        got, stats = _concurrent(multi, jobs)
        assert got == want
        assert all(st.get("copy_rounds", 0) > 0 for st in stats), stats
    finally:
        multi.scheduler.close()

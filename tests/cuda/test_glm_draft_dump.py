"""TF_GLM_DRAFT_DUMP (Mia's 0011) on the tiny synthetic GLM: a DFlash2 request decodes serially (its reply equals
draft=False) and leaves one .npz the simulator reads; --parallel engines ignore the setting."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_engine import _checkpoint, _drafter  # noqa: E402
from test_glm_multi import _engine, _prompts, _solo, _with_env  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402


def test_dump_records_a_serial_reply(tmp_path):
    from tensorfold.families.glm5_next.cuda import draft_sim

    _checkpoint(tmp_path / "model")
    _drafter(tmp_path / "dflash2")
    dump = tmp_path / "dumps"
    e = _with_env({"TF_GLM_DRAFT_DUMP": str(dump)}, lambda: _engine(tmp_path / "model", drafter=tmp_path / "dflash2"))
    assert e.dump is not None
    plain = _engine(tmp_path / "model", drafter=tmp_path / "dflash2")
    assert plain.dump is None
    prompt = _prompts(3, (50,))[0]
    sampling = Sampling(9, 1.0, 20, 0.95)
    want = _solo(plain, prompt, sampling, 40)
    out: list[int] = []
    e.request.policy = "f3"
    e.request.stop_eos = False
    stats = e.generate(list(prompt), 40, sampling, lambda new: out.extend(new))
    assert out == want
    files = sorted(dump.glob("*.npz"))
    assert len(files) == 1 and stats.get("draft_dump") == str(files[0])
    d = draft_sim.load_dump(files[0])
    assert [int(t) for t in d.tokens] == out


def test_parallel_ignores_the_dump(tmp_path):
    _checkpoint(tmp_path / "model")
    _drafter(tmp_path / "dflash2")
    e = _with_env({"TF_GLM_DRAFT_DUMP": str(tmp_path / "d")},
                  lambda: _engine(tmp_path / "model", drafter=tmp_path / "dflash2", parallel=2,
                                  multi_options={"draft_graphs": False}))
    try:
        assert e.dump is None
    finally:
        e.scheduler.close()

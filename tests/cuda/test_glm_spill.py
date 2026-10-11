"""The spill tier wired into GLM-5.3-Flash's engines (Mia's 0088, ``--spill-gib``; cuda/spill.py), with TF_GLM_KV=fp8
on the tiny synthetic checkpoint of test_glm_engine.py, one GPU playing rank 0 of two (``_TwoCopies``):

- one request at a time: a kept prompt state evicted from memory is written to disk, read back for a request that
  extends it, and the reply equals a fresh prefill's;
- a second engine on the same snapshot directory (a restart) resumes from the stored state;
- --parallel 2: a stored state read back while another stream decodes; both replies equal their solo ones.

Spill off is the rest of the suite (no engine there is given ``spill``)."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

from test_glm_engine import _checkpoint, _drafter, _forget, _generate  # noqa: E402  (tests/cuda on sys.path)
from test_glm_multi import _concurrent, _engine, _solo, _with_env  # noqa: E402

os.environ.setdefault("TF_GLM_CACHE_GIB", "0.05")
os.environ.setdefault("TF_SPILL_DIRECT", "0")          # tmp folders may refuse O_DIRECT (spill.py falls back too)


def _cfg(root):
    from tensorfold.cuda.spill import SpillConfig

    return SpillConfig(root=str(root), gib=1.0, highwater=1.0, min_tokens=64, min_free_gib=0.0, writers=1,
                       direct=False, flush_s=10.0)


def _prompt(seed, n):
    return [int(t) for t in np.random.default_rng(seed).integers(0, 1000, size=n)]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_spill")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    return path


SAMPLING = Sampling(31, 1.0, 20, 0.95)
A, MORE, B = _prompt(40, 300), _prompt(42, 9), _prompt(41, 280)
LONG = 400                      # the reply the other stream decodes while a stored state is read (--parallel)


def test_an_evicted_state_comes_back_from_disk_like_a_fresh_prefill_and_after_a_restart(models, tmp_path):
    root = tmp_path / "spill"
    root.mkdir()
    e = _engine(models / "model", drafter=models / "dflash2", kv="fp8", spill=_cfg(root))
    assert e.spill is not None and e.kv == "fp8"
    cold, stats = _generate(e, A + MORE, SAMPLING, tokens=12)       # the fresh prefill (its own state never dropped)
    assert stats["cached"] == 0
    _forget(e)
    e.cache_entries = 1                                             # B's state pushes A's out of memory
    _generate(e, A, SAMPLING, tokens=12)
    _generate(e, B, SAMPLING, tokens=12)
    assert e.spill.drain(60) and e.spill.has(A) and all(c.ids != A for c in e.cache)
    warm, stats = _generate(e, A + MORE, SAMPLING, tokens=12)
    assert stats["cached"] == len(A) and e.spill.stats.loaded == 1
    assert warm == cold
    del e
    torch.cuda.empty_cache()

    again = _engine(models / "model", drafter=models / "dflash2", kv="fp8", spill=_cfg(root))   # a restart
    assert again.spill.has(A) and not again.cache
    warm, stats = _generate(again, A + MORE, SAMPLING, tokens=12)
    assert stats["cached"] == len(A) and again.spill.stats.loaded == 1
    assert warm == cold


def test_parallel_reads_a_stored_state_beside_a_decoding_stream(models, tmp_path):
    solo = _engine(models / "model", drafter=models / "dflash2", kv="fp8")
    other = _prompt(43, 50)
    want = [_solo(solo, A + MORE, SAMPLING, 16), _solo(solo, other, None, LONG)]
    del solo
    torch.cuda.empty_cache()
    root = tmp_path / "spill"
    root.mkdir()
    e = _with_env({"TF_GLM_MULTI_VERIFY": "batched"},
                  lambda: _engine(models / "model", drafter=models / "dflash2", parallel=2, kv="fp8", spill=_cfg(root),
                                  multi_options={"draft_graphs": False}))
    try:
        m = e.multi
        assert m.disk is not None and e.spill is None
        e.cache_entries = 1                                         # B's kept state pushes A's out of the pool
        _concurrent(e, [(A, SAMPLING, 8)])
        _concurrent(e, [(B, SAMPLING, 8)])
        assert m.disk.drain(60) and m.disk.has(A) and all(c.ids != A for c in m.kept)
        reads, prestage = [], m.prestage

        def spy(s):                                                 # streams decoding when a read starts
            h = prestage(s)
            if h is not None:
                reads.append(sum(lane.decoding for lane in m.lanes.values()))
            return h

        m.prestage = spy
        # the other stream decodes; A + MORE arrives meanwhile and waits for its read beside it
        got, stats = _concurrent(e, [(A + MORE, SAMPLING, 16), (other, None, LONG)], delays=[0.1, 0.0])
        assert reads == [1] and stats[0]["cached"] == len(A) and m.disk.stats.loaded == 1
        assert got == want
        assert not m.lanes and not e.scheduler.loading and not m.loads
    finally:
        e.scheduler.close()

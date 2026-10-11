"""The indexer's key/gate ring (Mia's 0038 index ring). With TF_GLM_KV=fp8,
TF_GLM_INDEX_RING=auto puts each indexed layer's per-token keys and gates in a ring of
``index_ring(pool, max(max_rows, prefill_rows))`` rows per stream slot. Full-length planes are no longer kept. A
kept prompt carries the rows of its pools still filling (``decode.Snapshot.tail``). These tests check four things:

1. fp8 with the ring wrapping (prefill_rows=256: a 320-row ring under 2,000-token prompts). A kept prompt resumed
   after another conversation overwrote the ring gives the same state and reply as a fresh prefill. The kept point is
   the prefill keep-point (``decode.prefill``), and it uses the MTP head. The ring engine also gives the same tokens
   and pooled keys as an fp8 engine with full planes (TF_GLM_INDEX_RING=0).
2. bf16 is unchanged: no ring, full planes, and the saved rows still hold the keys and gates. Its index kernels write
   the same bytes as the merge base's (194c74c). A bf16 ring also writes the same pooled keys as the planes do.
3. The geometry reports the larger pool: the exact byte saving on RED-SNOW's config, and about 1.8x the tokens in the
   same memory.
4. --parallel 2 on fp8 with the ring: each stream equals its solo run, including a kept prompt that is taken over
   into another slot (``multi.put_ring_tail``)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REDSNOW = Path(os.environ.get("TENSORFOLD_REDSNOW_MODEL", "/var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw"))
BASE = os.environ.get("TF_TEST_SPARSE_BASE", "194c74c")   # a commit with sparse.py before the ring (dev history)


# -- 3. geometry (CPU) ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not (REDSNOW / "config.json").exists(), reason="needs the RED-SNOW checkpoint's config")
@pytest.mark.parametrize("parallel", [1, 2])
def test_geometry_ring_reports_the_larger_pool(parallel):
    from tensorfold.cuda import capacity
    from tensorfold.cuda.geometry import PREFILL_ROWS, layer_counts, mla_index_ring, mla_ring_bytes
    from tensorfold.families.glm5_next.cuda.engine import MULTI_WINDOW, multi_geometry
    from tensorfold.families.glm5_next.cuda.pool import align_up

    t = capacity.config(REDSNOW)
    count = layer_counts(t)[1]                               # 11 DSA layers (MTP off, as --parallel and auto run)
    assert count == 11
    g = {(kv, ring): multi_geometry(t, 2, parallel, streams=parallel, window=MULTI_WINDOW, latent=True, mtp=False,
                                    kv=kv, ring=ring) for kv in ("bf16", "fp8") for ring in (False, True)}
    cap = 262144
    pool = parallel * align_up(cap) if parallel > 1 else cap
    ring_rows = mla_index_ring(cap, PREFILL_ROWS)
    assert ring_rows == 2112
    # the ring takes out the full-length key and gate planes (11 x 512 B a pool token) for one ring a slot
    saved = count * pool * 2 * 128 * 2 - count * 2 * ring_rows * 128 * 2 - (parallel - 1) * mla_ring_bytes(t, PREFILL_ROWS)
    assert g["fp8", False].bytes_at(cap) - g["fp8", True].bytes_at(cap) == saved
    assert count * 2 * 128 * 2 == 5632
    # the same spare memory holds ~1.84x the tokens (marginal 12,348 -> 6,716 B a token with the selection scratch)
    budget = g["fp8", False].bytes_at(cap)
    lo, hi = cap, 4 * cap
    while hi - lo > 64:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if g["fp8", True].bytes_at(mid) <= budget else (lo, mid)
    assert lo >= 1.75 * cap, lo


# -- 2. bf16 unchanged (and the ring exact at the kernel) ------------------------------------------------------------
def _head_sparse():
    """sparse.py as of the merge base, as a module of the same package (its relative imports resolve)."""
    root = Path(__file__).resolve().parents[2]
    got = subprocess.run(["git", "-C", str(root), "show", f"{BASE}:src/tensorfold/families/glm5_next/cuda/sparse.py"],
                         capture_output=True, text=True)
    if got.returncode != 0:
        pytest.skip(f"commit {BASE} is not in this checkout (set TF_TEST_SPARSE_BASE)")
    src = got.stdout
    path = Path(os.environ.get("TMPDIR", "/tmp")) / f"_tf_sparse_head_{os.getpid()}.py"
    path.write_text(src)
    name = "tensorfold.families.glm5_next.cuda._sparse_head"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_bf16_index_kernels_unchanged_and_ring_exact():
    """bf16 planes: index_update writes the merge base's bytes (keys, gates, pooled keys) over chunked windows past
    several ring lengths. A 320-row ring then writes the same pooled keys as the planes do."""
    from tensorfold.families.glm5_next.cuda import kv8, sparse
    from tensorfold.families.glm5_next.cuda.forward import index_ring

    head = _head_sparse()
    gen = torch.Generator().manual_seed(3)
    cap, ID, N = 2600, 128, 2213
    kr = torch.randn((N, 2 * ID), generator=gen).to(torch.bfloat16).cuda()
    gr = torch.randn((N, ID), generator=gen).cuda()
    lnw = (1 + 0.1 * torch.randn(ID, generator=gen)).to(torch.bfloat16).cuda()
    lnb = (0.1 * torch.randn(ID, generator=gen)).to(torch.bfloat16).cuda()
    ape = (0.5 * torch.randn((4, ID), generator=gen)).cuda()
    chunks = [256, 3, 1, 256, 61, 256, 8, 256, 256, 255, 2, 256, 147, 200]
    assert sum(chunks) == N
    ring = index_ring(cap, 256)
    assert ring == 320
    out = {}
    for kind, mod, rows in (("head", head, cap), ("planes", sparse, cap), ("ring", sparse, ring)):
        ik = torch.zeros((rows, ID), dtype=torch.bfloat16, device="cuda")
        ig = torch.zeros_like(ik)
        pk = kv8.zeros(cap // 4 + 2, ID, "bf16", "cuda")
        a = 0
        for n in chunks:
            mod.index_update(kr[a:a + n, :ID], gr[a:a + n], lnw, lnb, ape, ik, ig, pk,
                             torch.tensor([a], dtype=torch.int32, device="cuda"))
            a += n
        out[kind] = (ik, ig, pk)
    torch.cuda.synchronize()
    for x, y in zip(out["head"], out["planes"]):
        assert torch.equal(x, y)                                  # bf16: the merge base's bytes
    assert torch.equal(out["ring"][2], out["planes"][2])         # the ring's pools == the planes'
    slots = torch.arange(N - ring, N, device="cuda")
    assert torch.equal(out["ring"][0][slots % ring], out["planes"][0][slots])   # the last ring of rows, in place


# -- engines on the tiny checkpoint (test_glm_engine's) --------------------------------------------------------------
if torch.cuda.is_available():
    from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

    from test_glm_engine import _checkpoint, _forget, _generate, _TwoCopies  # noqa: E402


def _engine(path, **kw):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies(), context=2600, prefill_rows=256, **kw)


def _env(env: dict, make):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return make()
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def _prompt(seed, n):
    return [int(t) for t in np.random.default_rng(seed).integers(0, 1000, size=n)]


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("CUDA only")
    path = tmp_path_factory.mktemp("glm_ring")
    _checkpoint(path)
    return path


def _pools(st) -> list[torch.Tensor]:
    """Every indexed layer's pooled keys up to its position (the MTP layer's to mtp_len), cloned."""
    idx = st.index
    main = [pk[:st.pos // 4].clone() for _, _, pk in idx[:len(st.kc)]]
    return main + ([idx[-1][2][:st.mtp_len // 4].clone()] if len(idx) > len(st.kc) else [])


def _resume_run(e):
    """A (2,080 tokens), then B (takes the live caches and overwrites the whole ring), then A's next turn: warm (from
    A's kept prompt end, len(A) = 2080 (a multiple of 4: no main ring tail, 3 MTP rows of a pool still filling)) and cold."""
    sampling = Sampling(31, 1.0, 20, 0.95)
    a, b = _prompt(40, 2080), _prompt(41, 2097)
    reply_a, _ = _generate(e, a, sampling, tokens=12)
    _generate(e, b, sampling, tokens=12)
    next_a = a + reply_a + _prompt(42, 9)
    warm, stats = _generate(e, next_a, sampling, tokens=12)
    assert stats["cached"] == len(a), stats
    warm_pools, warm_mtp = _pools(e.e.st), e.e.st.mtp_len
    _forget(e)
    cold, stats = _generate(e, next_a, sampling, tokens=12)
    assert stats["cached"] == 0
    return warm, warm_pools, warm_mtp, cold, _pools(e.e.st), e.e.st.mtp_len


# -- 1. fp8 with the ring wrapping: resume == fresh (the prefill keep-point) == full planes --------------------------
def test_fp8_ring_resume_equals_fresh_and_full_planes(model):
    ring_e = _env({"TF_GLM_INDEX_RING": "auto"}, lambda: _engine(model, kv="fp8"))
    caches = ring_e.e.caches
    assert ring_e.index_ring and caches.ring == 320 and caches.rings is not None
    assert ring_e.capacity_plan["index_ring"] == 320
    assert ring_e.e.st.index[0][0].shape == (320, 128) and ring_e.e.w.mtp is not None
    assert ring_e.limit > 2 * caches.ring                        # prompts wrap the ring six times
    warm, wp, wm, cold, cp, cm = _resume_run(ring_e)
    assert warm == cold and wm == cm
    assert all(torch.equal(x, y) for x, y in zip(wp, cp))        # main and MTP pools: the tail came back exactly
    del ring_e
    torch.cuda.empty_cache()
    planes_e = _env({"TF_GLM_INDEX_RING": "0"}, lambda: _engine(model, kv="fp8"))
    assert planes_e.e.caches.rings is None and not planes_e.index_ring
    pwarm, pwp, _, pcold, _, _ = _resume_run(planes_e)
    assert pwarm == warm and pcold == cold                         # the ring changes no token
    assert all(torch.equal(x, y) for x, y in zip(pwp, wp))        # ... nor any pooled key
    del planes_e
    torch.cuda.empty_cache()


def test_bf16_engine_keeps_full_planes(model):
    """bf16 (TF_GLM_INDEX_RING=auto): no ring. The keys and gates are pool planes as before, and a snapshot saves
    them with its rows and has no tail."""
    from tensorfold.families.glm5_next.cuda import decode

    e = _env({"TF_GLM_INDEX_RING": "auto"}, lambda: _engine(model))
    try:
        assert e.kv == "bf16" and not e.index_ring and e.e.caches.rings is None and e.e.caches.ring is None
        st = e.e.st
        assert st.index[0][0].shape[0] == st.capacity                  # a full-length plane over the extent
        _generate(e, _prompt(7, 600), None, tokens=4)
        snap = decode.take_snapshot(e.e, list(range(st.pos)), None, mtp=False)
        assert snap.tail == []
        n = st.pos
        views = decode._row_views(st, n, 0)
        planes = {t.data_ptr() for ik, ig, _ in st.index[:len(st.kc)] for t in (ik, ig)}
        assert sum(v.shape[0] == n and v.data_ptr() in planes for v in views) == 2 * len(st.kc)   # keys, gates saved
    finally:
        del e
        torch.cuda.empty_cache()


# -- 4. --parallel 2, fp8 with the ring: each stream == its solo run; a kept prompt taken over in another slot -------
def test_parallel2_fp8_ring_streams_equal_solo(model):
    from test_glm_multi import _concurrent, _solo

    sampling = [Sampling(1234, 1.0, 20, 0.95), None]
    prompts = [_prompt(51, 2100), _prompt(52, 700)]
    first_after = None
    solo = _engine(model, kv="fp8")
    want = [_solo(solo, p, s, 20) for p, s in zip(prompts, sampling)]
    m = _env({"TF_GLM_MULTI_VERIFY": "batched", "TF_GLM_CACHE_GIB": "0.2"},
             lambda: _engine(model, kv="fp8", parallel=2))
    try:
        assert m.e.caches.ring == 320 and m.e.caches.rings[0][0].shape[0] == 2 * 320
        got, _ = _concurrent(m, [(p, s, 20) for p, s in zip(prompts, sampling)])
        assert got == want
        # the 2,100-token prompt was kept at its end, 2,100 (a multiple of 4: no main ring tail). Its next turn starts after another
        # stream holds slot 0, so it resumes in slot 1 from the tail of the ring it was kept in (slot 0's or 1's)
        first_after = prompts[0] + got[0] + [7, 8, 9]
        other = _prompt(53, 300)
        got2, stats = _concurrent(m, [(other, None, 12), (first_after, sampling[0], 12)], delays=[0.0, 0.3])
        assert stats[1]["cached"] == len(prompts[0]), stats
        assert got2[0] == _solo(solo, other, None, 12)
        assert got2[1] == _solo(solo, first_after, sampling[0], 12)
        # tokens barely depend on one pool's key (512 of ~530 pools are selected), so compare the pools: the resumed
        # stream's (kept again at len - 1 in its extent) equal the solo run's, including the pool its tail completed
        c = max((c for c in m.multi.kept if c.ids == first_after[:len(c.ids)]), key=lambda c: len(c.ids))
        assert len(c.ids) > len(prompts[0])
        K, b4 = len(c.ids) // 4, c.extent.base // 4
        caches, st = m.e.caches, solo.e.st
        for i in range(len(caches.index)):
            assert torch.equal(caches.index_tensors(i)[2][b4:b4 + K], st.index[i][2][:K]), i
    finally:
        m.scheduler.close()

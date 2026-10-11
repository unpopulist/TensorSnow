"""Mia's G3 patches on the GPU (0013 verify windows up to 16 rows, TF_GLM_KVB; 0016/0019 decode kernels; 0018
noise-aware DFlash2 stops; 0032 wide graphs and copy refinements; 0092 checked candidates), on the tiny synthetic GLM of
``test_glm_engine``: drafted replies equal the serial (draft=False) reply token for token with windows of 9..16 rows
(captured 16-row graphs and eager widths) and with TF_GLM_MAX_ROWS=24; every --parallel stream equals its solo reply
with 16-row copied windows; TF_GLM_KVB=fp8/q4 stay exact against themselves. Kernel switches that must keep the bits
(TF_GLM_KDA_WIDE_ROWS, TF_GLM_LATENT_STAGES, TF_GLM_HC_DEC) are compared bit for bit."""

from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_engine import _checkpoint, _drafter, _generate  # noqa: E402
from test_glm_multi import _concurrent, _engine, _solo, _with_env  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import engine as E  # noqa: E402

SAMPLINGS = [None, Sampling(1234, 1.0, 20, 0.95)]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_rounds")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    _checkpoint(path / "exl3", exl3=True)
    return path


@pytest.fixture(scope="module")
def plain(models):
    return _engine(models / "model", drafter=models / "dflash2")


def _copying(make, *, max_rows=16, wide=(16,), copy_max=15, **kw):
    """An engine built with TF_GLM_MAX_ROWS / TF_GLM_WIDE_GRAPHS (engine module constants, read at construction) and
    copy drafts of up to ``copy_max``; the tiny random model never repeats 2 tokens, so its copies match 1 token."""

    saved = E.MAX_ROWS, E.WIDE_GRAPHS
    E.MAX_ROWS, E.WIDE_GRAPHS = max_rows, tuple(wide)
    try:
        e = _with_env({"TF_GLM_COPY_DRAFTS": "1", "TF_GLM_COPY_MAX": str(copy_max)}, lambda: make(**kw))
    finally:
        E.MAX_ROWS, E.WIDE_GRAPHS = saved
    assert e.copy is not None and e.copy.most == copy_max and e.copy.pad == tuple(r for r in wide if r <= copy_max + 1)
    e.copy = dataclasses.replace(e.copy, match=1)
    return e


def _recorded(engine):
    """Every verify window's width, from now on (Engine.forward wrapped)."""

    widths: list[int] = []
    fwd = engine.e.forward

    def forward(tokens):
        widths.append(len(tokens))
        return fwd(tokens)

    engine.e.forward = forward
    return widths


def _prompt(seed, n=600):
    """Most of the vocabulary, so nearly every pending token occurred before: copies of up to COPY_MAX each round."""

    return [int(t) for t in np.random.default_rng(seed).permutation(1000)[:n]]


@pytest.fixture(scope="module")
def wide16(models):
    return _copying(lambda **kw: _engine(models / "model", drafter=models / "dflash2"))


@pytest.fixture(scope="module")
def eager16(models):
    return _copying(lambda **kw: _engine(models / "model", drafter=models / "dflash2"), wide=())


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "sampled"])
@pytest.mark.parametrize("policy", [None, "fnc7:0.3", "f3", "c3:0.3"], ids=["auto", "fnc7", "f3", "mtp"])
def test_16_row_windows_equal_serial(plain, wide16, eager16, policy, sampling):
    prompt = _prompt(11)
    want = _solo(plain, prompt, sampling, 48)
    for e, graphed in ((wide16, True), (eager16, False)):
        widths = _recorded(e)
        before = dict(e.e.replays)
        try:
            got, stats = _generate(e, prompt, sampling, policy=policy, tokens=48)
        finally:
            del e.e.forward
        assert got == want, (policy, graphed)
        assert stats.get("copy_rounds", 0) > 0 and max(widths) == 16, (policy, graphed, widths)
        if graphed:          # 9..15-row windows padded to the captured 16-row graph (room allowing)
            assert widths.count(16) > sum(8 < w < 16 for w in widths), widths
            assert (16, 0) in e.e.graphs.main and e.e.replays["main"] > before["main"]
        else:
            assert (16, 0) not in e.e.graphs.main and e.e.replays["eager"] > before["eager"]


def test_wide_graph_capture_and_kda_reserve(wide16):
    from tensorfold.families.glm5_next.cuda import kda

    assert sorted({r for r, _ in wide16.e.graphs.main}) == list(E.GRAPH_ROWS) + [16]
    assert max(wide16.e.graphs.mtp) <= E.MTP_ROWS and wide16.e.mbuf.rows == E.MTP_ROWS
    assert wide16.e.rows == 16 and wide16.e.buf.rows == 16
    assert wide16.drafter.tap_rows == 16 and max(wide16.drafter.tap_graphs) == 16
    heads = wide16.w.cfg.lin_heads // wide16.w.world
    q, _ = kda._tmp[(heads, torch.device("cuda", torch.cuda.current_device()))]
    assert q.shape[0] >= wide16.e.prefill_rows and not kda._outgrown        # sized before any capture, never grown


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "sampled"])
def test_24_row_windows_equal_serial(models, plain, sampling):
    """TF_GLM_MAX_ROWS=24, TF_GLM_WIDE_GRAPHS=16: windows of 9..16 rows padded to the 16-row graph, 17..24 eager."""

    e = _copying(lambda **kw: _engine(models / "model", drafter=models / "dflash2"), max_rows=24, wide=(16,),
                 copy_max=23)
    prompt = _prompt(12)
    want = _solo(plain, prompt, sampling, 48)
    for policy in (None, "fnc7:0.3"):
        widths = _recorded(e)
        got, _ = _generate(e, prompt, sampling, policy=policy, tokens=48)
        del e.e.forward
        assert got == want, policy
        assert max(widths) > 16 and 16 in widths, widths


@pytest.mark.parametrize("verify", ["serial", "batched"])
def test_parallel_streams_with_16_row_copies_equal_solo(models, plain, verify):
    multi = _copying(lambda **kw: _with_env({"TF_GLM_MULTI_VERIFY": verify}, lambda: _engine(
        models / "model", drafter=models / "dflash2", parallel=3, multi_options={"draft_graphs": False})))
    try:
        assert multi.multi.rows_max == 16
        prompts = [_prompt(41), _prompt(42, 300), [int(t) for t in np.random.default_rng(43).integers(0, 1000, 23)]]
        samplings = [Sampling(1234, 1.0, 20, 0.95), None, Sampling(77, 0.8, 0, 0.9)]
        jobs = [(p, s, 48) for p, s in zip(prompts, samplings)]
        want = [_solo(plain, p, s, n) for p, s, n in jobs]
        for policy in (None, "fnc7:0.3"):
            got, stats = _concurrent(multi, jobs, policy=policy)
            assert got == want, policy
            assert sum(st.get("copy_rounds", 0) for st in stats) > 0, stats
    finally:
        multi.scheduler.close()


@pytest.mark.parametrize("kind", ["fp8", "q4"])
def test_kvb_engine_drafted_equals_serial(models, kind):
    from tensorfold.families.glm5_next.cuda import latent

    e = _with_env({"TF_GLM_KVB": kind}, lambda: _engine(models / "exl3", drafter=models / "dflash2"))
    a = e.w.layers[1].dsa
    assert e.kvb == kind and a.kv_k is None and a.kv_v is None
    assert isinstance(a.absorb, latent.AbsorbF8 if kind == "fp8" else latent.AbsorbQ4)
    total = e.w.nbytes()
    absorb, a.absorb = a.absorb, None
    try:
        assert total - e.w.nbytes() == absorb.nbytes()                  # Weights.nbytes counts the absorb
    finally:
        a.absorb = absorb
    prompt = list(np.random.default_rng(8).integers(0, 1000, size=45))
    for sampling in SAMPLINGS:
        serial, _ = _generate(e, prompt, sampling, draft=False, tokens=32)
        for policy in (None, "fnc7:0.3", "2"):
            drafted, _ = _generate(e, prompt, sampling, policy=policy, tokens=32)
            assert drafted == serial, (kind, policy)


def test_kvb_bf16_holds_kv_b_once(models):
    from tensorfold.families.glm5_next.cuda import latent

    e = _engine(models / "exl3", drafter=models / "dflash2")
    a = e.w.layers[1].dsa
    assert e.kvb == "bf16" and a.kv_k is None and isinstance(a.absorb, latent.AbsorbW)


def test_bad_candidates_are_copied_again(plain, monkeypatch, capsys):
    """0092: a read-back with ids outside the vocabulary is copied again once; the reply is the serial one."""

    from tensorfold.families.glm5_next.cuda import dflash2

    prompt = list(np.random.default_rng(21).integers(0, 1000, size=40))
    sampling = Sampling(5, 1.0, 20, 0.95)
    want = _solo(plain, prompt, sampling, 24)
    real, calls = dflash2.merge_candidates, []

    def flaky(g, proj, k, vocab=None):
        calls.append(vocab)
        if len(calls) == 1:
            raise dflash2.BadCandidates("1 of 32 candidate ids are outside the vocabulary (test)")
        return real(g, proj, k, vocab)

    monkeypatch.setattr(dflash2, "merge_candidates", flaky)
    got, _ = _generate(plain, prompt, sampling, policy="f3", tokens=24)
    assert got == want and len(calls) > 2 and calls[0] == plain.drafter.succ.shape[0]
    assert "copying the rows again" in capsys.readouterr().err


# -- kernels whose switches must keep the bits ----------------------------------------------------------------------
@pytest.mark.parametrize("heads", [2, 32])
def test_kda_wide_chain_bits_for_decode_windows(heads):
    """TF_GLM_KDA_WIDE_ROWS=1 (the default) runs decode windows of 1..16 rows on the three-kernel chain: the fused
    chain's outputs, replay rows and state."""

    from tensorfold.families.glm5_next.cuda import kda

    assert kda.WIDE_ROWS == int(os.environ.get("TF_GLM_KDA_WIDE_ROWS", "1"))
    for rows in range(1, 17):
        g = torch.Generator().manual_seed(rows * 7 + heads)
        C = 3 * heads * kda.DK
        b_off = C + 256
        p = (torch.randn((rows, b_off + heads + 32), generator=g) * 0.5).to(torch.bfloat16).cuda()
        a = torch.randn((rows, heads * kda.DK), generator=g).to(torch.bfloat16).cuda()
        gate = torch.randn((rows, heads * kda.DV), generator=g).to(torch.bfloat16).cuda()
        cs = (torch.randn((3, C), generator=g) * 0.5).to(torch.bfloat16).cuda()
        cw = (torch.randn((C, 4), generator=g) * 0.5).to(torch.bfloat16).cuda()
        state = (torch.randn((heads, kda.DV, kda.DK), generator=g) * 0.1).cuda()
        a_log = (torch.rand(heads, generator=g) * 2 - 1).cuda()
        dt_bias = (torch.randn(heads * kda.DK, generator=g) * 0.1).cuda()
        norm_w = (torch.rand(kda.DV, generator=g) + 0.5).to(torch.bfloat16).cuda()
        got = []
        for wide in (False, True):
            sc = kda.KDAScratch(rows, heads, "cuda")
            out_state = torch.empty_like(state)
            out = kda.chain(p, b_off, a, gate, cs, cw, state, a_log, dt_bias, norm_w, 1e-5, -5.0, rows, sc,
                            out_state, wide=wide).clone()
            got.append((out, out_state, sc.k[:rows], sc.v[:rows], sc.g[:rows], sc.b[:rows]))
        for name, x, y in zip(("out", "state", "k", "v", "g", "beta"), *got):
            assert torch.equal(x, y), (rows, name)


@pytest.mark.parametrize("kv", ["bf16", "fp8"])
def test_latent_chunk_stages_keep_the_bits(kv, monkeypatch):
    """TF_GLM_LATENT_STAGES (3 by default, 1 before 0019): dense and sparse latent attention, the same bits."""

    from tensorfold.families.glm5_next.cuda import kv8, latent

    H, L = 32, 512
    gen = torch.Generator().manual_seed(5)
    P, n = 1500, 6000
    lat = (torch.randn((n, L), generator=gen) * 0.5).to(torch.bfloat16).cuda()
    cache = kv8.quantize_rows(lat) if kv == "fp8" else lat
    tokens = torch.full((16, 2051), -1, dtype=torch.int32)
    counts = []
    for r in range(16):
        sel = torch.sort(torch.randperm(n - 10, generator=gen)[:2048 - r % 3]).values
        tokens[r, :sel.numel()] = sel.to(torch.int32)
        counts.append(sel.numel())
    tokens, counts = tokens.cuda(), torch.tensor(counts, dtype=torch.int32, device="cuda")
    for R in (1, 5, 8, 16):
        qa = (torch.randn((R, H, L), generator=gen) * 0.3).to(torch.bfloat16).cuda()
        outs = []
        for stages in ("1", "3"):
            monkeypatch.setenv("TF_GLM_LATENT_STAGES", stages)
            nch = latent.chunks_for(P + R)
            s = latent.LatentScratch(R, H, nch, "cuda")
            dense = torch.empty((R, H, L), dtype=torch.bfloat16, device="cuda")
            latent.attention(qa, cache, torch.tensor([P], dtype=torch.int32, device="cuda"), s, scale=0.06,
                             nch=nch, out=dense)
            sparse = torch.zeros((R, H, L), dtype=torch.bfloat16, device="cuda")
            latent.sparse_attention(qa, cache, tokens[:R].contiguous(), counts[:R].contiguous(), sparse, 0.06)
            outs.append((dense, sparse))
        assert torch.equal(outs[0][0], outs[1][0]), ("dense", R)
        assert torch.equal(outs[0][1], outs[1][1]), ("sparse", R)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_hc_dec_gives_hc_partials_bits(dtype):
    """TF_GLM_HC_DEC: hc.cu's decode mixing dots equal _hc_partial's for bf16 mixing weights (Mia's checkpoint) and
    fp32 ones (RED-SNOW's; full fp32 values here, whose products with bf16 inputs round)."""

    from tensorfold.families.glm5_next.cuda import glue

    gen = torch.Generator().manual_seed(9)
    fn = (torch.randn((24, 16384), generator=gen) * 0.01).to(dtype).cuda()
    for rows in range(1, 17):
        x = torch.randn((rows, 16384), generator=gen).to(torch.bfloat16).cuda()
        ref = torch.zeros((rows, glue.HC_BLOCKS, 32), device="cuda")
        glue._hc_partial[(rows, glue.HC_BLOCKS)](x, fn, ref, WIDE=16384, NB=glue.HC_BLOCKS, SUB=128, num_warps=4)
        got = torch.zeros_like(ref)
        glue._hc_ext().hc_partial(x, fn, got, rows)
        assert torch.equal(got[..., :25], ref[..., :25]), rows


def test_hc_mma_split_fp32_weights_row_bits_and_accuracy():
    """Prompt rows with fp32 mixing weights: _hc_partial_mma's three-term split gives a row the same bits alone and
    inside a chunk (other rows, a partial row block), and stays at fp32 accuracy (a bf16 copy of the weights does not)."""

    import triton

    from tensorfold.families.glm5_next.cuda import glue

    gen = torch.Generator().manual_seed(5)
    W, NB, R = 16384, glue.HC_BLOCKS, 150
    fn = (torch.randn((24, W), generator=gen) * 0.01).cuda()
    x = torch.randn((R, W), generator=gen).to(torch.bfloat16).cuda()

    def mma(rows, f, split=True):
        n = rows.shape[0]
        part = torch.zeros((n, NB, 32), device="cuda")
        glue._hc_partial_mma[(triton.cdiv(n, 64), NB)](rows.contiguous(), f, part, n, WIDE=W, NB=NB, BM=64, BK=64,
                                                         SPLIT=split, num_warps=4, num_stages=3)
        return part

    whole = mma(x, fn)
    for lo, hi in ((0, 1), (37, 38), (149, 150), (64, 150), (10, 75)):
        assert torch.equal(mma(x[lo:hi], fn)[..., :25], whole[lo:hi, :, :25]), (lo, hi)
    ref = x.double() @ fn.double().T
    err = ((whole[..., :24].sum(1).double() - ref).abs() / ref.abs().clamp_min(1e-3)).median()
    copy = mma(x, fn.to(torch.bfloat16).contiguous(), split=False)
    err_copy = ((copy[..., :24].sum(1).double() - ref).abs() / ref.abs().clamp_min(1e-3)).median()
    assert err < 2e-5 and err_copy > 20 * err, (float(err), float(err_copy))


def test_slot_bytes_track_the_slots(models):
    """``engine.slot_bytes`` (the startup estimate of a stream slot, which grows with MAX_ROWS) equals what
    ``forward.Slots`` allocates a slot at 8 and 16 rows."""

    from tensorfold.cuda.capacity import config as read_config
    from tensorfold.families.glm5_next.cuda.forward import Slots

    e = _engine(models / "model")
    t = read_config(models / "model")
    sizes = {}
    for rows in (8, 16):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        s = Slots(e.w, 3, rows)
        torch.cuda.synchronize()
        sizes[rows] = (torch.cuda.memory_allocated() - before) / 3
        del s
    for rows, got in sizes.items():
        want = E.slot_bytes(t, 2, rows=rows)
        assert abs(got - want) <= 0.02 * want + 4096, (rows, got, want)
    print(f"tiny model slot bytes: 8 rows {sizes[8]:.0f}, 16 rows {sizes[16]:.0f}", file=sys.stderr)

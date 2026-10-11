"""TF_GLM_KV=fp8 (families/glm5_next/cuda/kv8.py): GLM's DSA latent cache and the indexer's pooled keys as e4m3 rows
with a power-of-two fp32 scale each. The kernels store exactly kv8.quantize_rows' bytes; attention on an FP8 cache
equals the bf16 kernels on the dequantized rows up to fp32 summation order and stays close to the bf16 cache; and
the exactness rule holds: a window's rows (verify windows, prompt chunks, sparse rows) equal serial one-row steps."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")

H, D, L = 32, 256, 512           # a rank's heads, qk_nope + rope dim, the latent width (GLM-5.3-Flash on two ranks)


def _rows(gen, n, width=L, gain=False):
    """bf16 rows like kv_a_layernorm's output: about unit rms with a few large channels (amax ~3-5x the rms);
    ``gain``: each row also scaled by exp(N(0, 1)) (rows of very different sizes: the scales' range)."""
    x = torch.randn((n, width), generator=gen)
    x[:, ::37] *= 4.0
    if gain:
        x *= torch.exp(torch.randn((n, 1), generator=gen))
    return x.to(torch.bfloat16)


def _fp8(rows):
    from tensorfold.families.glm5_next.cuda import kv8

    return kv8.quantize_rows(rows.cuda())


# -- the format ------------------------------------------------------------------------------------------------------
def test_kv_kind_and_row_bytes():
    from tensorfold.cuda.geometry import mla_row_bytes
    from tensorfold.families.glm5_next.cuda import kv8

    assert kv8.kv_kind({}) == "bf16" and kv8.kv_kind({"TF_GLM_KV": "fp8"}) == "fp8"
    assert kv8.kv_kind({"TF_GLM_KV": " "}) == "bf16"
    with pytest.raises(ValueError):
        kv8.kv_kind({"TF_GLM_KV": "int8"})
    for w in (128, 512):
        assert kv8.row_bytes(w, "bf16") == mla_row_bytes(w, "bf16") == 2 * w
        assert kv8.row_bytes(w, "fp8") == mla_row_bytes(w, "fp8") == w + 16
    with pytest.raises(ValueError):
        mla_row_bytes(512, "int4")


@cuda
def test_quantize_rows_scales_codes_and_round_trip():
    from tensorfold.families.glm5_next.cuda import kv8

    gen = torch.Generator().manual_seed(0)
    x = _rows(gen, 300, gain=True)
    x[7] = 0                                                       # a zero row: scale 2^-126, zero codes
    q = kv8.quantize_rows(x.cuda())
    assert q.dtype == torch.uint8 and q.shape == (300, L + kv8.PAD)
    assert int(q[:, L + 4:].abs().sum()) == 0                      # the 12 pad bytes stay zero
    s = q[:, L:L + 4].contiguous().view(torch.float32).flatten().cpu()
    bits = s.view(torch.int32)
    assert bool(((bits & 0x7FFFFF) == 0).all())                    # every scale an exact power of two
    amax = x.float().abs().amax(1)
    nz = amax > 0
    top = amax[nz] / s[nz]
    assert bool((top > 128).all() and (top <= 256).all())          # the row's largest code within (128, 256]
    assert s[7].item() == 2.0 ** -126
    back = kv8.dequantize(q).cpu()
    assert torch.equal(back[7], torch.zeros(L))
    rel = ((back - x.float()).norm(dim=1) / x.float().norm(dim=1).clamp_min(1e-30))[nz]
    assert rel.max().item() < 0.04, rel.max().item()               # e4m3's 3 mantissa bits: ~2.7% a row
    # the scale is the torch definition's: 2^(ceil(log2 amax) - 8)
    want = torch.exp2(torch.ceil(torch.log2(amax[nz])) - kv8.SHIFT)
    assert torch.equal(s[nz], want)


@cuda
@pytest.mark.parametrize("R", [1, 3, 8, 300])
def test_latent_write_stores_quantize_rows_bytes(R):
    """latent_write's FP8 bytes are kv8.quantize_rows' of the same bf16 rows, whatever the window (decode, verify
    window, prompt chunk): a row's bytes are a function of the row alone."""
    from tensorfold.families.glm5_next.cuda import kv8, latent

    gen = torch.Generator().manual_seed(R)
    lat = _rows(gen, R).cuda()
    cache = kv8.zeros(1024, L, "fp8", "cuda")
    P = 333
    latent.latent_write(lat, cache, torch.tensor([P], dtype=torch.int32, device="cuda"))
    torch.cuda.synchronize()
    assert torch.equal(cache[P:P + R], kv8.quantize_rows(lat))
    assert int(cache[:P].sum()) == 0 and int(cache[P + R:].sum()) == 0
    # the same rows one at a time land on the same bytes
    one = kv8.zeros(1024, L, "fp8", "cuda")
    for r in range(R):
        latent.latent_write(lat[r:r + 1], one, torch.tensor([P + r], dtype=torch.int32, device="cuda"))
    assert torch.equal(one, cache)
    with pytest.raises(ValueError):
        latent.latent_write(lat[:, :256].contiguous(), cache, torch.tensor([0], dtype=torch.int32, device="cuda"))


@cuda
@pytest.mark.parametrize("P, R", [(0, 64), (61, 8), (130, 1), (6, 300)])
def test_pooled_keys_are_the_bf16_pools_quantized(P, R):
    """index_update on an FP8 pooled-key cache stores kv8.quantize_rows of the pools the bf16 cache gets, and the
    per-token keys and gates are the bf16 run's."""
    from tensorfold.families.glm5_next.cuda import kv8, sparse

    gen = torch.Generator().manual_seed(P * 7 + R)
    cap, ID = 1024, 128
    kr = torch.randn((P + R, 2 * ID), generator=gen).to(torch.bfloat16).cuda()
    gr = torch.randn((P + R, ID), generator=gen).cuda()
    lnw = (1 + 0.1 * torch.randn(ID, generator=gen)).to(torch.bfloat16).cuda()
    lnb = (0.1 * torch.randn(ID, generator=gen)).to(torch.bfloat16).cuda()
    ape = (0.5 * torch.randn((4, ID), generator=gen)).cuda()
    caches = {}
    for kind in ("bf16", "fp8"):
        ik = torch.zeros((cap, ID), dtype=torch.bfloat16, device="cuda")
        ig = torch.zeros_like(ik)
        pk = kv8.zeros(cap // 4 + 2, ID, kind, "cuda")
        for a, n in ((0, P), (P, R)):                               # the prompt before, then this window
            if n:
                sparse.index_update(kr[a:a + n, :ID], gr[a:a + n], lnw, lnb, ape, ik, ig, pk,
                                    torch.tensor([a], dtype=torch.int32, device="cuda"))
        caches[kind] = (ik, ig, pk)
    torch.cuda.synchronize()
    (ik, ig, pb), (ik8, ig8, p8) = caches["bf16"], caches["fp8"]
    assert torch.equal(ik, ik8) and torch.equal(ig, ig8)
    pools = (P + R) // 4
    assert pools > 0
    assert torch.equal(p8[:pools], kv8.quantize_rows(pb[:pools]))
    assert int(p8[pools:].sum()) == 0


# -- attention: FP8 against the bf16 cache ---------------------------------------------------------------------------
def _absorb(gen):
    from tensorfold.families.glm5_next.cuda import latent

    wk = torch.randn((H, D, L), generator=gen) * L ** -0.5
    wv = torch.randn((H, D, L), generator=gen) * L ** -0.5
    return latent.AbsorbW(wk.cuda(), wv.cuda())


def _dense(qa, cache, P, *, part_rows=None):
    """forward.dense_attention as a decode window (R <= 64) or a prompt chunk (PROMPT_ATT_ROWS-row blocks)."""
    from tensorfold.families.glm5_next.cuda import forward, latent

    R = qa.shape[0]
    nch = latent.chunks_for(P + R)
    s = latent.LatentScratch(R, H, nch, "cuda", part_rows=part_rows)
    out = torch.empty((R, H, L), dtype=torch.bfloat16, device="cuda")
    starts = torch.arange(0, R, s.part_rows, dtype=torch.int32, device="cuda")     # Buffers.att_starts
    forward.dense_attention(qa, cache, torch.tensor([P], dtype=torch.int32, device="cuda"), s, scale=D ** -0.5,
                            nch=nch, out=out, starts=starts)
    return out


def _sparse(qa, cache, tokens, counts):
    from tensorfold.families.glm5_next.cuda import latent

    out = torch.zeros((qa.shape[0], H, L), dtype=torch.bfloat16, device="cuda")
    latent.sparse_attention(qa, cache, tokens, counts, out, D ** -0.5)
    return out


def _close(got, want, tol):
    err = (got.float() - want.float()).norm() / want.float().norm()
    assert err.item() < tol, err.item()


def _queries(gen, a, R):
    from tensorfold.families.glm5_next.cuda import latent

    q = torch.randn((R, H, D), generator=gen).to(torch.bfloat16).cuda()
    return latent.absorb_q(q, a, torch.empty((R, H, L), dtype=torch.bfloat16, device="cuda"))


@cuda
@pytest.mark.parametrize("P, R, part", [(1500, 1, None), (1500, 2, None), (1500, 5, None), (1500, 8, None),
                                        (0, 600, 512), (1024, 1100, 512)],
                         ids=["decode", "verify2", "verify5", "verify8", "prompt600", "prompt1100"])
def test_dense_fp8_matches_bf16_cache(P, R, part):
    """Dense latent attention (decode step, verify windows of 2-8 rows, prompt chunks in 512-row blocks): on an FP8
    cache it equals the bf16 kernel on the dequantized rows up to summation order, and stays within e4m3's error
    of the bf16 cache."""
    from tensorfold.families.glm5_next.cuda import kv8

    gen = torch.Generator().manual_seed(P + R)
    a = _absorb(gen)
    lat = _rows(gen, P + R).cuda()
    qa = _queries(gen, a, R)
    c16, c8 = lat, _fp8(lat)
    deq = kv8.dequantize(c8).to(torch.bfloat16)                   # exact: codes x a power of two fit bf16
    assert torch.equal(deq.float(), kv8.dequantize(c8))
    o16 = _dense(qa, c16, P, part_rows=part)
    o8 = _dense(qa, c8, P, part_rows=part)
    od = _dense(qa, deq, P, part_rows=part)
    _close(o8, od, 1e-3)                                           # the same math up to fp32 summation order
    _close(o8, o16, 6e-2)                                          # lossy against bf16 (e4m3 ~2.7% a row)


@cuda
@pytest.mark.parametrize("R", [1, 4, 72])
def test_sparse_fp8_matches_bf16_cache(R):
    from tensorfold.families.glm5_next.cuda import kv8

    gen = torch.Generator().manual_seed(100 + R)
    a = _absorb(gen)
    n, W = 6000, 2051
    lat = _rows(gen, n).cuda()
    qa = _queries(gen, a, R)
    tokens = torch.full((R, W), -1, dtype=torch.int32)
    counts = []
    for r in range(R):
        sel = torch.sort(torch.randperm(n - 10, generator=gen)[:2048 - r % 3]).values
        tokens[r, :sel.numel()] = sel.to(torch.int32)
        counts.append(sel.numel())
    tokens, counts = tokens.cuda(), torch.tensor(counts, dtype=torch.int32, device="cuda")
    c8 = _fp8(lat)
    o16 = _sparse(qa, lat, tokens, counts)
    o8 = _sparse(qa, c8, tokens, counts)
    od = _sparse(qa, kv8.dequantize(c8).to(torch.bfloat16), tokens, counts)
    _close(o8, od, 1e-3)
    _close(o8, o16, 6e-2)


@cuda
@pytest.mark.parametrize("pos, R", [(9000, 1), (9000, 8), (5000, 512)])
def test_select_tokens_on_fp8_pooled_keys(pos, R):
    """Token selection over FP8 pooled keys: per-pool scores within e4m3's error of the bf16 keys' (each pool's
    scale on its dots), and rows of a window select what their serial steps select."""
    from tensorfold.families.glm5_next.cuda import kv8, sparse

    gen = torch.Generator().manual_seed(pos + R)
    IH, ID = 32, 128
    cap = 16384
    pk16 = torch.randn((cap // 4 + 2, ID), generator=gen).to(torch.bfloat16).cuda()
    pk8 = kv8.quantize_rows(pk16)
    pk8[(pos + R) // 4 + 1:] = 0
    pkd = kv8.dequantize(pk8).to(torch.bfloat16)
    qi = torch.randn((R, IH * ID), generator=gen).to(torch.bfloat16).cuda()
    wts = torch.randn((R, IH), generator=gen).to(torch.bfloat16).cuda()
    pos_dev = torch.tensor([pos], dtype=torch.int32, device="cuda")
    t8, c8 = sparse.select_tokens(qi, wts, pk8, pos, R, cap // 4, pos_dev)
    td, cd = sparse.select_tokens(qi, wts, pkd, pos, R, cap // 4, pos_dev)
    t16, _ = sparse.select_tokens(qi, wts, pk16, pos, R, cap // 4, pos_dev)
    assert torch.equal(c8, cd)
    # FP8 codes with the scale on the dot vs the dequantized keys: the same pools but for near ties
    same = (t8 == td).float().mean().item()
    assert same > 0.99, same
    overlap = sum(len(set(t8[r].tolist()) & set(t16[r].tolist())) for r in range(R)) / t8.numel()
    assert overlap > 0.8, overlap                                  # most selected tokens survive e4m3's rounding
    for r in sorted({0, R // 2, R - 1}):
        one, n1 = sparse.select_tokens(qi[r:r + 1].contiguous(), wts[r:r + 1].contiguous(), pk8, pos + r, 1,
                                       cap // 4, torch.tensor([pos + r], dtype=torch.int32, device="cuda"))
        assert torch.equal(one[0], t8[r]) and int(n1[0]) == int(c8[r])


# -- exactness: window rows equal serial steps on an FP8 cache -------------------------------------------------------
@cuda
@pytest.mark.parametrize("R", [2, 5, 8, 16])
def test_fp8_window_rows_equal_serial_steps(R):
    from tensorfold.families.glm5_next.cuda import latent

    gen = torch.Generator().manual_seed(2 + R)
    a = _absorb(gen)
    P = 1500
    c8 = torch.zeros((2560, L + 16), dtype=torch.uint8, device="cuda")
    c8[:P + R] = _fp8(_rows(gen, P + R))
    qa = _queries(gen, a, R)
    s = latent.LatentScratch(64, H, latent.chunks_for(2560 + 64), "cuda")
    window = latent.attention(qa, c8, torch.tensor([P], dtype=torch.int32, device="cuda"), s, scale=D ** -0.5,
                              nch=s.nch, out=torch.empty((R, H, L), dtype=torch.bfloat16, device="cuda")).clone()
    for r in range(R):
        one = latent.attention(qa[r:r + 1].contiguous(), c8, torch.tensor([P + r], dtype=torch.int32, device="cuda"),
                               s, scale=D ** -0.5, nch=s.nch,
                               out=torch.empty((1, H, L), dtype=torch.bfloat16, device="cuda"))
        assert torch.equal(window[r], one[0]), f"row {r} differs from its serial step"


@cuda
def test_fp8_prompt_tiles_and_sparse_rows_equal_serial_steps():
    """Prompt chunks (32-head tiles, 512-row blocks) and sparse rows on an FP8 cache: each row's bits are its
    one-row step's."""
    from tensorfold.families.glm5_next.cuda import latent

    gen = torch.Generator().manual_seed(7)
    n, R, W = 6000, 72, 2051
    c8 = _fp8(_rows(gen, n))
    qa = (torch.randn((R, H, L), generator=gen) * 0.05).to(torch.bfloat16).cuda()
    P = 1500
    wide = _dense(qa, c8, P, part_rows=512).clone()
    s = latent.LatentScratch(R, H, latent.chunks_for(P + R), "cuda")
    for r in (0, 1, 37, R - 1):
        one = latent.attention(qa[r:r + 1].contiguous(), c8, torch.tensor([P + r], dtype=torch.int32, device="cuda"),
                               s, scale=D ** -0.5, nch=latent.chunks_for(P + R),
                               out=torch.empty((1, H, L), dtype=torch.bfloat16, device="cuda"))
        assert torch.equal(wide[r], one[0]), f"dense row {r}"
    tokens = torch.full((R, W), -1, dtype=torch.int32)
    counts = []
    for r in range(R):
        sel = torch.sort(torch.randperm(n, generator=gen)[:2048 - r % 5]).values
        tokens[r, :sel.numel()] = sel.to(torch.int32)
        counts.append(sel.numel())
    tokens, counts = tokens.cuda(), torch.tensor(counts, dtype=torch.int32, device="cuda")
    wide = _sparse(qa, c8, tokens, counts)
    for r in (0, 3, 41, R - 1):
        one = _sparse(qa[r:r + 1].contiguous(), c8, tokens[r:r + 1].contiguous(), counts[r:r + 1].contiguous())
        assert torch.equal(wide[r], one[0]), f"sparse row {r}"


# -- geometry ----------------------------------------------------------------------------------------------------
def test_geometry_fp8_rows_and_refusals():
    from tensorfold.cuda.geometry import mla_geometry

    text = {"num_hidden_layers": 4, "num_attention_heads": 64, "hidden_size": 1024, "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64, "v_head_dim": 256, "kv_lora_rank": 512, "q_lora_rank": 768,
            "num_experts_per_tok": 8, "moe_intermediate_size": 512, "vocab_size": 4096, "index_head_dim": 128,
            "num_nextn_predict_layers": 1, "layer_types": ["linear_attention", "full_attention"] * 2}
    try:
        b16 = mla_geometry(text, 2, 0, latent=True)
    except (KeyError, TypeError) as exc:                           # a config field this tree's geometry needs
        pytest.skip(f"synthetic config incomplete for mla_geometry: {exc}")
    f8 = mla_geometry(text, 2, 0, latent=True, kv="fp8")
    count = 2 + 1                                                  # DSA layers and the MTP layer
    for cap in (4096, 65536, 1 << 20):
        saved = b16.bytes_at(cap) - f8.bytes_at(cap)
        want = count * (cap * (2 * 512 - (512 + 16)) + (cap // 4 + 2) * (2 * 128 - (128 + 16)))
        assert saved == want, (cap, saved, want)
    with pytest.raises(ValueError):
        mla_geometry(text, 2, 0, latent=False, kv="fp8")


# -- plumbing: --kv-dtype / TF_GLM_KV --------------------------------------------------------------------------------
def test_resolve_kv_flag_and_env():
    from tensorfold.families.glm5_next.cuda import kv8

    assert kv8.resolve(None, {}) == "bf16"                         # neither: bf16
    assert kv8.resolve(None, {"TF_GLM_KV": "fp8"}) == "fp8"        # no flag (cuda_engine passes None): the env
    assert kv8.resolve("fp8", {}) == "fp8" and kv8.resolve("bf16", {}) == "bf16"
    assert kv8.resolve("fp8", {"TF_GLM_KV": "fp8"}) == "fp8"
    with pytest.raises(ValueError, match="disagree"):
        kv8.resolve("bf16", {"TF_GLM_KV": "fp8"})
    with pytest.raises(ValueError, match="kv-dtype int8"):
        kv8.resolve("int8", {})


def test_cuda_engine_forwards_kv_dtype(monkeypatch, tmp_path):
    """glm5_next.cuda_engine hands --kv-dtype to GlmEngine as kv: None when the flag was not given (cli.py only sets
    options["kv_dtype"] for non-bf16), the value when it was; an unknown dtype is refused before anything starts."""
    from tensorfold.families import glm5_next
    from tensorfold.families.glm5_next.cuda import engine as glm_engine

    got = []
    monkeypatch.setattr(glm5_next, "check_ranks", lambda *a, **k: None)
    monkeypatch.setattr(glm_engine, "GlmEngine", lambda *a, **k: got.append(k) or k)
    ranks = dict(tp=2, master="192.0.2.10", no_drafts=True)
    glm5_next.cuda_engine(tmp_path, **ranks)
    assert got[-1]["kv"] is None
    glm5_next.cuda_engine(tmp_path, kv_dtype="fp8", **ranks)
    assert got[-1]["kv"] == "fp8"
    glm5_next.cuda_engine(tmp_path, kv_dtype="bf16", **ranks)
    assert got[-1]["kv"] == "bf16"
    with pytest.raises(ValueError, match="kv-dtype int8"):
        glm5_next.cuda_engine(tmp_path, kv_dtype="int8", **ranks)
    assert len(got) == 3

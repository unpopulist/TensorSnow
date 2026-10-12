"""Mia's 0101 / 0105 / 0107 (BadAd84) on the GPU, one case a contract (her tools' select_floor_check.py,
prompt_scores_check.py and decode_indexer_check.py are the sweeps):
  - prompt_pools (one pass over the scores) gives top_pools' pools, ties included, on its floor path and on rows
    that fall back to _select_rows;
  - prompt_scores.cu stores _scores' scores bit for bit (FP8, 32 x 128 indexer heads: RED-SNOW's, replicated);
  - a segmented window's indexer (seg_scores.cu for segments of 4+ rows, _seg_select_floor for 8+ rows) gives the
    old path's scores, tokens and counts bit for bit."""

from __future__ import annotations

import pytest
import torch
import triton

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")
H, D = 32, 128


def _ext_or_skip(get):
    from tensorfold.families.glm5_next.cuda import sparse

    if not sparse.triton_matches_ptx():
        pytest.skip(f"Triton {triton.__version__}: the CUDA ports serve under {sparse.TRITON_PTX} only")
    ext = get()
    assert ext is not None, "the extension failed to build or load under the Triton release it reproduces"
    return ext


@cuda
def test_prompt_pools_equal_top_pools():
    from tensorfold.families.glm5_next.cuda import sparse

    g = torch.Generator(device="cuda").manual_seed(1)
    R, NP = 8, 40_960
    s = torch.randn((R, NP), device="cuda", generator=g)
    s[1] = torch.randint(0, 4, (NP,), device="cuda", generator=g).float()        # ties: past CAP, _select_rows
    s[2] = 0.0                                                                      # all equal
    s[3, 30_000:] = float("-inf")                                                   # -inf past the visible pools
    s[4] = s[4].bfloat16().float()                                                  # bf16-rounded: near ties
    assert torch.equal(sparse.prompt_pools(s, sparse.TOPK_POOLS), sparse.top_pools(s, sparse.TOPK_POOLS))


@cuda
def test_prompt_scores_cuda_equals_triton_bits():
    from tensorfold.families.glm5_next.cuda import kv8, sparse

    ext = _ext_or_skip(sparse._scores_ext)
    torch.manual_seed(0)
    pos, R = 113_000, 72                                                            # a CTA tail (72 = 9 x 8)
    NP = sparse.pool_count(pos, R, 1 << 20)
    pk = kv8.quantize_rows(torch.randn(NP + 2, D, device="cuda"))
    qi = torch.randn(R, H * D, device="cuda").bfloat16()
    ikr = torch.randn(R, D + H, device="cuda").bfloat16()                         # weights a strided view, as forward's
    wts = ikr[:, D:]
    at = torch.tensor([pos], dtype=torch.int32, device="cuda")
    pkv, pks, rs, fp8 = kv8.parts(pk)
    wscale = 1.0 / 5.656854249492381
    want = torch.full((R, NP), float("nan"), device="cuda")
    got = want.clone()
    sparse._scores[(R, triton.cdiv(triton.cdiv(NP, 64), sparse.SCORE_LOOP))](
        qi, wts, wts.stride(0), pkv, pks, want, at, R, NP, D ** -0.5, wscale, H=H, HP=H, D=D, BP=64,
        L=sparse.SCORE_LOOP, RS=rs, FP8=fp8, num_warps=4)
    ext.scores(qi, wts, wts.stride(0), pk, rs, got, at, R, NP, D ** -0.5, wscale)
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))


@cuda
def test_segmented_indexer_equals_old_path(monkeypatch):
    from tensorfold.families.glm5_next.cuda import kv8, segments, sparse

    _ext_or_skip(sparse._seg_scores_ext)
    contexts, rows_each = [300_000, 290_000], [8, 3]           # 8 rows: seg_scores.cu; 3: _seg_scores; R=11: floor
    R = sum(rows_each)
    rows = segments.SegRows(R, "cuda", max_segs=segments.SEGS_DEFAULT)
    segs, base = [], 0
    for ctx, n in zip(contexts, rows_each):
        segs.append((base, ctx - n, n))
        base += -(-ctx // segments.EXTENT) * segments.EXTENT
    torch.manual_seed(0)
    pk = kv8.quantize_rows(torch.randn(base // sparse.POOL + 2, D, device="cuda"))
    rows.set(segs)
    qi = torch.randn(R, H * D, device="cuda").bfloat16()
    wts = torch.randn(R, H, device="cuda").bfloat16()
    scratch = segments.SelectScratch(R, 1_048_576, "cuda")
    assert scratch.split.numel() >= R * 2 * sparse.SEG_FLOOR[0]                      # the floor path is taken

    def run():
        scratch.scores.fill_(float("nan"))
        scratch.tokens.fill_(-7)
        scratch.counts.fill_(-7)
        sparse.seg_select_tokens(qi, wts, pk, rows, scratch)
        torch.cuda.synchronize()
        return scratch.scores[:R].clone().view(torch.int32), scratch.tokens[:R].clone(), scratch.counts[:R].clone()

    new = run()
    monkeypatch.setenv("TF_GLM_SEG_SCORES_CUDA", "0")
    monkeypatch.setenv("TF_GLM_SEG_SELECT_FLOOR", "0")
    old = run()
    assert torch.equal(new[2], old[2]) and bool((old[2] > 0).all())
    assert torch.equal(new[1], old[1]) and torch.equal(new[0], old[0])

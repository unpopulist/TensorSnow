"""Mia's single-stream prefill kernels (her 0009, ported in port-prefill-8s): the prompt token selection
(``sparse._select_prompt``), one-pass sparse attention for prompt rows (``latent.sparse_onepass``), the shared expert
read in place by the MoE combine (``glue.combine_shared``) and the dense pass trimmed to the rows before the sparse
limit (``forward.dense_rows``).

Contracts: _select_prompt gives every counted row select_tokens' tokens and count; sparse_onepass gives a row the
same bits in any chunk (rows alone, any subset) and lands within bf16 rounding of sparse_attention (other bits:
one online-softmax chain instead of merged 512-token chunks, so prompt states move within that tolerance);
combine_shared gives combine's bits."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)


def _select_inputs(pos: int, R: int, H: int = 32, D: int = 128, seed: int = 0):
    gen = torch.Generator().manual_seed(seed + pos + R)
    npool_max = (pos + R) // 4 + 2
    qi = torch.randn((R, H * D), generator=gen).to(torch.bfloat16).cuda()
    wts = torch.randn((R, H), generator=gen).to(torch.bfloat16).cuda()
    pk = torch.randn((npool_max, D), generator=gen).to(torch.bfloat16).cuda()
    pos_dev = torch.tensor([pos], dtype=torch.int32, device="cuda")
    return qi, wts, pk, pos_dev, npool_max - 2


@pytest.mark.parametrize("pos, R", [(2047, 64), (2000, 300), (5000, 512), (9000, 1100), (130000, 2048),
                                    (262144 - 2048, 2048)])
def test_prompt_selection_gives_every_counted_row_the_window_lists(monkeypatch, pos, R):
    from tensorfold.families.glm5_next.cuda import sparse

    qi, wts, pk, pos_dev, np_max = _select_inputs(pos, R)
    got_t, got_c = sparse.select_tokens(qi, wts, pk, pos, R, np_max, pos_dev)          # R >= 64: _select_prompt
    monkeypatch.setattr(sparse, "PROMPT_ROWS", 1 << 30)                                  # the window selection
    want_t, want_c = sparse.select_tokens(qi, wts, pk, pos, R, np_max, pos_dev)
    assert torch.equal(got_c, want_c)
    counted = want_c > 0
    assert torch.equal(got_t[counted], want_t[counted])


@pytest.mark.parametrize("loop", [1, 4])
def test_prompt_selection_score_loop_never_changes_the_lists(monkeypatch, loop):
    """Mia's 0086: _scores' pool blocks a program (SCORE_LOOP) are speed only."""
    from tensorfold.families.glm5_next.cuda import sparse

    qi, wts, pk, pos_dev, np_max = _select_inputs(7000, 700)
    want = sparse.select_tokens(qi, wts, pk, 7000, 700, np_max, pos_dev)
    monkeypatch.setattr(sparse, "SCORE_LOOP", loop)
    got = sparse.select_tokens(qi, wts, pk, 7000, 700, np_max, pos_dev)
    assert torch.equal(got[1], want[1]) and torch.equal(got[0][want[1] > 0], want[0][want[1] > 0])


def _latent_case(H: int, R: int, pos: int, seed: int = 1):
    from tensorfold.families.glm5_next.cuda import sparse

    gen = torch.Generator().manual_seed(seed)
    cap = pos + R + 8
    lc = torch.randn((cap, 512), generator=gen).to(torch.bfloat16).cuda()
    qa = (torch.randn((R, H, 512), generator=gen) * 0.2).to(torch.bfloat16).cuda()
    qi, wts, pk, pos_dev, np_max = _select_inputs(pos, R, H=2, seed=seed)
    tokens, counts = sparse.select_tokens(qi, wts, pk, pos, R, np_max, pos_dev)
    return lc, qa, tokens, counts


@pytest.mark.parametrize("H", [2, 32])
def test_onepass_rows_keep_their_bits_in_any_chunk_and_stay_close(H):
    from tensorfold.families.glm5_next.cuda import latent

    R, pos = 300, 2000                         # rows 0..50 stay dense (count 0), the rest are sparse
    lc, qa, tokens, counts = _latent_case(H, R, pos)
    assert 0 < int((counts == 0).sum()) < R
    scale = 576 ** -0.5
    sentinel = torch.full((R, H, 512), 7.0, dtype=torch.bfloat16, device="cuda")
    one = sentinel.clone()
    latent.sparse_onepass(qa, lc, tokens, counts, one, scale)
    dense = counts == 0
    assert bool((one[dense] == 7.0).all())                     # count-0 rows are left alone
    for a, b in ((0, 1), (51, 52), (100, 164), (R - 1, R), (17, 299)):
        part = sentinel[a:b].clone()
        latent.sparse_onepass(qa[a:b].contiguous(), lc, tokens[a:b].contiguous(), counts[a:b].contiguous(), part,
                              scale)
        assert torch.equal(part, one[a:b]), (a, b)
    ref = sentinel.clone()
    latent.sparse_attention(qa, lc, tokens, counts, ref, scale)
    s = ~dense
    err = (one[s].float() - ref[s].float()).abs().max()
    assert float(err) <= 2e-2 * float(ref[s].float().abs().max()), float(err)
    # the double-precision definition for one row
    r = R - 1
    sel = tokens[r, :int(counts[r])].long()
    kv = lc[sel].double()
    p = torch.softmax(torch.einsum("hd,td->ht", qa[r].double(), kv) * scale, dim=-1)
    want = p @ kv
    assert float((one[r].double() - want).abs().max()) <= 2e-2 * float(want.abs().max())


def test_combine_shared_gives_combines_bits():
    from tensorfold.families.glm5_next.cuda import glue

    gen = torch.Generator().manual_seed(4)
    R, slots, D = 37, 9, 1024
    y = torch.randn((R, slots, D), generator=gen).cuda()
    sy = torch.randn((R, D), generator=gen).cuda()
    wts = torch.rand((R, slots), generator=gen).cuda()
    got = torch.empty((R, D), device="cuda")
    glue.combine_shared(y, sy, wts, got)
    y2 = y.clone()
    y2[:, slots - 1] = sy
    want = torch.empty((R, D), device="cuda")
    glue.combine(y2, wts, want)
    assert torch.equal(got, want)


def test_dense_rows():
    from types import SimpleNamespace

    from tensorfold.families.glm5_next.cuda import sparse
    from tensorfold.families.glm5_next.cuda.forward import dense_rows

    c = SimpleNamespace(dense_limit=sparse.SPARSE_FROM)
    assert sparse.SPARSE_FROM == 2051
    assert dense_rows(c, 2048, 0) == 2048 and dense_rows(c, 2048, 2048) == 3 and dense_rows(c, 64, 2051) == 0
    assert dense_rows(c, 64, None) == 64 and dense_rows(SimpleNamespace(dense_limit=4096), 64, 2048) == 64

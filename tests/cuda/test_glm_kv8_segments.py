"""TF_GLM_KV=fp8 under --parallel: the multi-stream *segmented* kernels (latent.seg_attention, sparse.seg_select_tokens)
read a quantized cache as e4m3, exactly as the single-stream kernels do, NOT as bf16.

The segmented kernels (Mia's 0009/0029) were ported before FP8 KV existed; the exl3<-multistream merge reconciled them
to the post-0038 FP8 forms. These tests are the check: for one stream's window
(a segment at base 0) on an fp8 cache,

  seg kernel(fp8)  ==  single-stream kernel(fp8)        (the serial fp8 path: validated by test_glm_kv8.py)
  seg kernel(fp8)  ~=  single-stream kernel(dequantized bf16)
  seg kernel(fp8)  !=  the result of reading the uint8 rows as bf16  (proves the cache is read as fp8, not bf16).

One GPU, one stream segment; the arena row is the token position (base 0), so a segment behaves as a solo window."""

from __future__ import annotations

import pytest
import torch

from test_glm_kv8 import H, D, L, _absorb, _close, _fp8, _queries, _rows  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")


@cuda
@pytest.mark.parametrize("P, R", [(1500, 1), (1500, 5), (1500, 8), (1500, 16)],
                         ids=["decode", "window5", "window8", "window16"])
def test_seg_dense_reads_fp8_not_bf16(P, R):
    from tensorfold.families.glm5_next.cuda import latent
    from tensorfold.families.glm5_next.cuda.segments import SegRows

    gen = torch.Generator().manual_seed(P + R)
    a = _absorb(gen)
    lat = _rows(gen, P + R).cuda()
    qa = _queries(gen, a, R)
    c8 = _fp8(lat)                                                   # [P+R, L+pad] uint8
    deq = latent_kv8_dequant(c8)

    # single-stream fp8 path (the serial reference the parallel path must equal)
    ss = latent.LatentScratch(R, H, latent.chunks_for(P + R), "cuda")
    solo = latent.attention(qa, c8, _pos([P]), ss, scale=D ** -0.5, nch=ss.nch,
                            out=torch.empty((R, H, L), dtype=torch.bfloat16, device="cuda"), hb=16).clone()

    # segmented path: one dense segment (base 0, position P, R rows)
    rows = SegRows(R, "cuda", max_segs=1)
    rows.set([(0, P, R)], sparse_from=None)
    seg_scratch = latent.LatentScratch(R, H, latent.seg_chunks(), "cuda")
    seg = latent.seg_attention(qa, c8, rows, None, None, seg_scratch, scale=D ** -0.5,
                               out=torch.empty((R, H, L), dtype=torch.bfloat16, device="cuda"), hb=16).clone()

    # seg fp8 == single-stream fp8, bit for bit. The single-stream fp8 path is validated by test_glm_kv8.py, so this
    # equality is the proof the segmented kernel reads the uint8 rows as e4m3 (a kernel that read them as bf16 could
    # not reproduce the fp8 path nor land near the dequantized rows below).
    assert torch.equal(seg, solo)
    _close(seg, _dense_bf16(qa, deq, P), 1e-3)                       # and ~= the kernel on the dequantized rows
    # the dequantized (correct) rows differ materially from the raw code bytes: fp8 is lossy, not a bf16 relabelling
    assert not torch.allclose(deq.float(), c8[:, :L].to(torch.float32), atol=1.0)


@cuda
@pytest.mark.parametrize("pos, R", [(9000, 1), (9000, 8), (5000, 16)])
def test_seg_select_reads_fp8_pooled_keys_not_bf16(pos, R):
    from tensorfold.families.glm5_next.cuda import kv8, sparse
    from tensorfold.families.glm5_next.cuda.segments import SegRows, SelectScratch

    gen = torch.Generator().manual_seed(pos + R)
    IH, ID = 32, 128
    cap = 16384
    pk16 = torch.randn((cap // 4 + 2, ID), generator=gen).to(torch.bfloat16).cuda()
    pk8 = kv8.quantize_rows(pk16)
    pk8[(pos + R) // 4 + 1:] = 0
    qi = torch.randn((R, IH * ID), generator=gen).to(torch.bfloat16).cuda()
    wts = torch.randn((R, IH), generator=gen).to(torch.bfloat16).cuda()

    # single-stream fp8 selection (the reference)
    t_solo, c_solo = sparse.select_tokens(qi, wts, pk8, pos, R, cap // 4, _pos([pos]))

    # segmented fp8 selection: one sparse segment at base 0, positions pos..pos+R-1
    rows = SegRows(R, "cuda", max_segs=1)
    rows.set([(0, pos, R)], sparse_from=0)
    scratch = SelectScratch(R, cap, "cuda")
    t_seg, c_seg = sparse.seg_select_tokens(qi, wts, pk8, rows, scratch)

    assert torch.equal(c_seg[:R], c_solo)
    same = (t_seg[:R] == t_solo).float().mean().item()
    assert same > 0.999, same                                        # seg fp8 selects the single-stream fp8 tokens

    # reading the pooled keys as bf16 selects materially different tokens
    pk_wrong = _bf16_view128(pk8)
    t_wrong, _ = sparse.select_tokens(qi, wts, pk_wrong, pos, R, cap // 4, _pos([pos]))
    overlap = sum(len(set(t_seg[r].tolist()) & set(t_wrong[r].tolist())) for r in range(R)) / max(1, t_seg[:R].numel())
    assert overlap < 0.9, overlap                                    # the bf16 misreading diverges


# -- helpers ---------------------------------------------------------------------------------------------------------
def _pos(xs):
    return torch.tensor(xs, dtype=torch.int32, device="cuda")


def latent_kv8_dequant(c8):
    from tensorfold.families.glm5_next.cuda import kv8

    return kv8.dequantize(c8).to(torch.bfloat16)


def _dense_bf16(qa, cache, P):
    from tensorfold.families.glm5_next.cuda import latent

    R = qa.shape[0]
    s = latent.LatentScratch(R, H, latent.chunks_for(P + R), "cuda")
    return latent.attention(qa, cache, _pos([P]), s, scale=D ** -0.5, nch=s.nch,
                            out=torch.empty((R, H, L), dtype=torch.bfloat16, device="cuda"), hb=16).clone()


def _bf16_view128(pk8):
    return pk8[:, : 2 * 128].contiguous().view(torch.bfloat16)

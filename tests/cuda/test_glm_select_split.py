"""The indexer's long-context token selection (Mia's 0029 serial part, 0031, 0043, 0085, 0086; port of her
tools/select_split_check.py and scores_loop_check.py, smaller).

Contracts:
  - decode windows past SPLIT_FROM scores take the split selection, which gives the one-program selection's tokens and
    counts bit for bit, and both equal a torch oracle (each row's 512 best complete pools, ties to the lower pool,
    ascending, then its incomplete pool's visible tokens; 0 for rows within the dense limit);
  - SPLIT_BLK and SCORE_LOOP are speed only;
  - a decode window's rows (_scores_rows) get the prompt selection's (_scores) lists;
  - segmented windows (seg_select_tokens, through select_split's SEG path) equal the oracle on their own scores and on
    tie-heavy, all-equal and special-value scores; bf16 and FP8 pooled keys."""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")
H, D = 32, 128


def _order_key(s: torch.Tensor) -> torch.Tensor:
    """sparse._order_key in int64: the float32 scores' order as unsigned 32-bit keys (-0 counted as +0)."""
    b = (s.float() + 0.0).view(torch.int32).to(torch.int64)
    return (b & 0xFFFFFFFF) ^ torch.where(b < 0, torch.full_like(b, 0xFFFFFFFF), torch.full_like(b, 0x80000000))


def _oracle(scores: torch.Tensor, pos: list[int], sparse_rows: list[bool]):
    from tensorfold.families.glm5_next.cuda import sparse

    K, PL, W = sparse.TOPK_POOLS, sparse.POOL, sparse.TOKENS
    R = scores.shape[0]
    toks = torch.full((R, W), -1, dtype=torch.int32)
    cnt = torch.zeros(R, dtype=torch.int32)
    for r in range(R):
        q = pos[r]
        npool = (q + 1) // PL
        if not sparse_rows[r] or npool <= K:
            continue
        best = torch.sort(-_order_key(scores[r, :npool]), stable=True).indices[:K]
        sel = torch.sort(best).values
        toks[r, :K * PL] = (sel[:, None] * PL + torch.arange(PL)).reshape(-1).to(torch.int32)
        n = K * PL
        for j in range(PL - 1):
            t = npool * PL + j
            toks[r, K * PL + j] = t if t <= q else -1
            n += int(t <= q)
        cnt[r] = n
    return toks, cnt


def _inputs(pos: int, R: int, fp8: bool, seed: int = 0):
    from tensorfold.families.glm5_next.cuda import kv8

    gen = torch.Generator().manual_seed(seed + pos + R)
    npool_max = (pos + R) // 4 + 2
    qi = torch.randn((R, H * D), generator=gen).to(torch.bfloat16).cuda()
    wts = torch.randn((R, H), generator=gen).to(torch.bfloat16).cuda()
    raw = torch.randn((npool_max, D), generator=gen).cuda()
    pk = kv8.quantize_rows(raw) if fp8 else raw.to(torch.bfloat16)
    pos_dev = torch.tensor([pos], dtype=torch.int32, device="cuda")
    return qi, wts, pk, pos_dev, npool_max - 2


def _same_counted(a, b):
    (ta, ca), (tb, cb) = a, b
    counted = cb > 0
    return torch.equal(ca.cpu(), cb.cpu()) and torch.equal(ta[counted].cpu(), tb[counted].cpu())


@cuda
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("pos, R", [(40_000, 1), (40_000, 16), (130_000, 5), (32_760, 16), (300_000, 8)])
def test_split_decode_selection_equals_one_program_and_oracle(monkeypatch, fp8, pos, R):
    from tensorfold.families.glm5_next.cuda import sparse

    qi, wts, pk, pos_dev, np_max = _inputs(pos, R, fp8)
    assert sparse.pool_bucket(pos, R, np_max) >= sparse.SPLIT_FROM          # the split path is taken
    captured = []
    real = sparse.select_split

    def capture(scores, k, **kw):
        captured.append(scores.clone())
        return real(scores, k, **kw)

    monkeypatch.setattr(sparse, "select_split", capture)
    split = sparse.select_tokens(qi, wts, pk, pos, R, np_max, pos_dev)
    monkeypatch.setattr(sparse, "select_split", real)
    assert len(captured) == 1
    monkeypatch.setattr(sparse, "SPLIT_FROM", 1 << 30)                       # one program a row
    one = sparse.select_tokens(qi, wts, pk, pos, R, np_max, pos_dev)
    assert torch.equal(split[1], one[1]) and torch.equal(split[0], one[0])
    rows = [pos + r for r in range(R)]
    want = _oracle(captured[0].cpu(), rows, [True] * R)
    assert _same_counted(split, want)


@cuda
@pytest.mark.parametrize("blk", [1, 4, 32])
def test_split_blk_is_speed_only(monkeypatch, blk):
    from tensorfold.families.glm5_next.cuda import sparse

    qi, wts, pk, pos_dev, np_max = _inputs(590_000, 16, True)
    want = sparse.select_tokens(qi, wts, pk, 590_000, 16, np_max, pos_dev)
    monkeypatch.setattr(sparse, "SPLIT_BLK", blk)
    got = sparse.select_tokens(qi, wts, pk, 590_000, 16, np_max, pos_dev)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


@cuda
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("pos, R", [(2_045, 8), (20_000, 16), (40_000, 3)])
def test_decode_rows_get_the_prompt_selection_lists(monkeypatch, fp8, pos, R):
    """_scores_rows (decode windows) and _scores (prompt chunks) score alike: the same lists for counted rows."""
    from tensorfold.families.glm5_next.cuda import sparse

    qi, wts, pk, pos_dev, np_max = _inputs(pos, R, fp8)
    window = sparse.select_tokens(qi, wts, pk, pos, R, np_max, pos_dev)
    monkeypatch.setattr(sparse, "PROMPT_ROWS", 1)                             # _select_prompt
    prompt = sparse.select_tokens(qi, wts, pk, pos, R, np_max, pos_dev)
    assert _same_counted(window, prompt)


def _special(shape, kind, npool_max):
    g = torch.Generator(device="cuda").manual_seed(5)
    s = torch.zeros(shape, device="cuda")
    if kind == "ties":
        s[:, :npool_max] = torch.randint(0, 8, (shape[0], npool_max), device="cuda", generator=g).float()
    elif kind == "special":
        v = torch.randn(shape[0], npool_max, device="cuda", generator=g)
        pick = torch.randint(0, 8, v.shape, device="cuda", generator=g)
        v = torch.where(pick == 0, torch.full_like(v, -0.0), v)
        v = torch.where(pick == 1, torch.full_like(v, float("-inf")), v)
        v = torch.where(pick == 2, torch.full_like(v, 3.0e38), v)
        v = torch.where(pick == 3, torch.full_like(v, -3.0e38), v)
        s[:, :npool_max] = v
    return s                                      # "equal": all zeros


@cuda
@pytest.mark.parametrize("contexts, rows_each", [([2_060], [16]), ([49_160], [16]), ([226_000], [16]),
                                                 ([1_000, 50_000, 300_000], [3, 8, 5])],
                         ids=["dense-edge", "chunk-edge", "226k", "3-streams"])
def test_segmented_selection_equals_oracle(monkeypatch, contexts, rows_each):
    from tensorfold.families.glm5_next.cuda import kv8, segments, sparse

    R = sum(rows_each)
    rows = segments.SegRows(R, "cuda", max_segs=max(segments.SEGS_DEFAULT, len(contexts)))
    segs, base = [], 0
    for ctx, n in zip(contexts, rows_each):
        segs.append((base, ctx - n, n))
        base += -(-ctx // segments.EXTENT) * segments.EXTENT
    torch.manual_seed(0)
    pk = kv8.quantize_rows(torch.randn(base // sparse.POOL + 2, D, device="cuda"))
    rows.set(segs)
    torch.cuda.synchronize()
    qi = torch.randn(R, H * D, device="cuda", dtype=torch.bfloat16)
    wts = torch.randn(R, H, device="cuda", dtype=torch.bfloat16)
    pos, spr = rows.pos[:R].tolist(), [bool(x) for x in rows.sparse[:R].tolist()]
    npool_max = (max(pos) + 1) // sparse.POOL
    scratch = segments.SelectScratch(R, 65_536 if base <= 65_536 else 1_048_592, "cuda")

    def run(fn):
        scratch.tokens.fill_(-1)
        scratch.counts.fill_(-7)
        fn()
        torch.cuda.synchronize()
        return scratch.tokens[:R].cpu(), scratch.counts[:R].cpu()

    got = run(lambda: sparse.seg_select_tokens(qi, wts, pk, rows, scratch))
    assert _same_counted(got, _oracle(scratch.scores[:R].cpu(), pos, spr))
    monkeypatch.setattr(sparse, "SPLIT_BLK", 1)
    assert torch.equal(run(lambda: sparse.seg_select_tokens(qi, wts, pk, rows, scratch))[0], got[0])
    monkeypatch.setattr(sparse, "SPLIT_BLK", 16)
    for kind in ("ties", "equal", "special"):
        s = _special(scratch.scores[:R].shape, kind, npool_max)
        out = run(lambda: sparse.select_split(s, sparse.TOPK_POOLS, scratch=scratch.split, rows=rows,
                                              tokens=scratch.tokens[:R], counts=scratch.counts[:R]))
        assert _same_counted(out, _oracle(s.cpu(), pos, spr)), kind

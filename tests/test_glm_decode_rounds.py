"""Mia's G3 patches on the host side (0013 verify windows up to 16 rows and TF_GLM_KVB, 0018 noise-aware DFlash2 stop
rules, 0021 TF_GLM_DFLASH_POLICY, 0032 TF_GLM_MAX_ROWS / TF_GLM_WIDE_GRAPHS and the copy-draft refinements, 0092
checked drafter candidates): settings parsing, policy codes, copy proposals, the KVB weight estimate and the per-slot
DFlash2 contexts' own buffers. No GPU."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorfold.families.glm5_next.cuda import engine as E
from tensorfold.families.glm5_next.cuda.copy_drafts import CopyDrafts, CopySettings


def test_window_constants():
    assert E.GRAPH_ROWS == (1, 2, 3, 4, 5, 6, 7, 8)
    assert E.MAX_ROWS == 16 and E.MTP_ROWS == 8
    assert E.multi_decode_rows(1, 32) == 64 and E.multi_decode_rows(4, 32, 16) == 64
    assert E.multi_decode_rows(8, 64, 16) == 80


def test_max_rows_and_wide_graphs_parse():
    assert E.max_rows("") == 16
    for ok in ("16", "24", "64"):
        assert E.max_rows(ok) == int(ok)
    for bad in ("8", "20", "65", "72", "x"):
        with pytest.raises(ValueError):
            E.max_rows(bad)
    assert E.wide_graphs("", 16) == () and E.wide_graphs("0", 16) == ()
    assert E.wide_graphs("16", 16) == (16,) and E.wide_graphs("24, 16", 24) == (16, 24)
    for bad in ("8", "17", "16,x"):
        with pytest.raises(ValueError):
            E.wide_graphs(bad, 16)


def test_noise_aware_policy_codes():
    assert E.encode_policy("fnc7:0.3") == [16, 7, 300000, 2000000]
    assert E.encode_policy("fnc5:0.25:1.5") == [16, 5, 250000, 1500000]
    assert E.encode_policy("fcost7:noisy") == [17, 7, 2000000, 0]
    for bad in ("fnc8:0.3", "fnc0:0.3", "fnc7:1.5", "fcost7:x", "fnc7"):
        with pytest.raises(ValueError):
            E.encode_policy(bad)
    p = E.decode_policy(E.encode_policy("fnc7:0.3"))
    assert p.most == 7 and p.confidence == 0.3 and p.chain_rule == {"beta": 2.0, "round_ms": None}
    assert E.decode_policy(E.encode_policy("fc5:0.3")).chain_rule == {}
    with pytest.raises(ValueError):
        E.decode_policy(E.encode_policy("fcost7:noisy"))            # needs the cost table
    costs = {"verify": [1.0 + r for r in range(8)], "block": 0.5, "taps_row": 0.1}
    assert len(E.round_costs(costs)) == 8
    p = E.decode_policy(E.encode_policy("fcost7:noisy"), costs)
    assert p.chain_rule["round_ms"] == E.round_costs(costs)
    # --parallel passes DFlash2 policies (kind 16 / 17) through, remaps auto to the DFlash2 default
    from tensorfold.families.glm5_next.cuda.multi import multi_code

    assert multi_code([16, 7, 300000, 2000000], True, [13, 5, 300000, 0]) == [16, 7, 300000, 2000000]
    assert multi_code([4, 2, 8, 30000], True, [16, 7, 300000, 2000000]) == [16, 7, 300000, 2000000]


def test_noisy_confidence_and_best_depth():
    from tensorfold.families.glm5_next.cuda import dflash2

    score = np.array([2.0, 1.0, 0.5])
    c = dflash2.noisy_confidence(score, None, 0, 2.0)
    e = np.exp(score / 2.0)
    assert c == pytest.approx(e[0] / e.sum())
    # cheap rounds: every draft pays; a costly wide window stops the chain early
    assert dflash2.best_depth([0.9, 0.9, 0.9], [1.0, 1.0, 1.0, 1.0]) == 3
    assert dflash2.best_depth([0.9, 0.2, 0.2], [1.0, 1.1, 3.0, 4.0]) == 1


def test_recipe_copy_settings():
    """Mia's recipe: COPY 1, COPY_MAX 15, COPY_CODE 1 (TF_GLM_WIDE_GRAPHS=16, TF_GLM_COPY_REPLY_MATCH=16)."""

    env = {"TF_GLM_COPY_DRAFTS": "1", "TF_GLM_COPY_MAX": "15", "TF_GLM_COPY_REPLY_MATCH": "16"}
    c = CopySettings.from_env(E.MAX_ROWS - 1, env, pad=E.wide_graphs("16", 16))
    assert (c.match, c.most, c.reply_match, c.pad, c.miss_most) == (8, 15, 16, (16,), 0)
    assert c.code() == [8, 15, 16, 0]
    with pytest.raises(ValueError):
        CopySettings.from_env(15, {**env, "TF_GLM_COPY_MAX": "16"})
    with pytest.raises(ValueError):
        CopySettings.from_env(15, {**env, "TF_GLM_COPY_REPLY_MATCH": "8"})      # not past the match
    assert CopySettings.from_env(15, {}) is None


def test_copy_padding_reply_match_and_miss_cap():
    text = list(range(100, 140))
    # padding: a 10-row window (9 drafts) is filled to the 16-row graph by repeating its last draft
    d = CopyDrafts(text + text[:3], match=3, most=15, prompt=len(text), pad=(16,))
    got = d.propose()
    assert got[:12] == text[3:15] and len(got) == 15
    d = CopyDrafts(text + text[30:33], match=3, most=15, prompt=len(text), pad=(16,))
    got = d.propose()
    # 10 copied (the copy runs on into the suffix itself), padded to 15 by repeating the last
    assert got == text[33:40] + text[30:33] + [text[32]] * 5
    assert d.proposed == 10
    assert d.padded([1] * 5) == [1] * 5                                  # within the 8-row graphs: as it is
    assert d.padded(list(range(9)), room=12) == list(range(9))           # no room for the 16-row width
    # reply match: an occurrence inside the reply needs the longer match
    prompt = [1, 2, 3, 4, 5, 6]
    reply = [7, 8, 9, 50, 51, 9, 50]
    d = CopyDrafts(prompt + reply, match=2, most=4, prompt=len(prompt), reply_match=3)
    assert d.propose() == []                                             # "9 50" occurs in the reply: 2 tokens only
    d = CopyDrafts(prompt + reply, match=2, most=4, prompt=len(prompt))
    assert d.propose() == [51, 9, 50]
    # the miss cap: after a copied round that kept fewer than all its drafts, at most miss_most drafts
    d = CopyDrafts(text + text[:3], match=3, most=10, miss_most=2)
    assert len(d.propose()) == 10
    d.extend([text[3], 999])                                             # one draft kept of 10: a miss
    assert d.missed
    d.extend(text[:3])
    assert len(d.propose()) == 2


def test_kvb_estimate_and_kind(monkeypatch):
    from tensorfold.families.glm5_next.cuda import qmm

    monkeypatch.delenv("TF_GLM_KVB", raising=False)
    assert qmm.kvb_kind() == "bf16"
    assert qmm.kvb_kind({"TF_GLM_KVB": "fp8"}) == "fp8"
    with pytest.raises(ValueError):
        qmm.kvb_kind({"TF_GLM_KVB": "int8"})
    base = lambda name, info: (int(np.prod(info["shape"])) * 2, 0)      # noqa: E731
    name = "model.language_model.layers.3.self_attn.kv_b_proj.weight"
    for dtype in ("F16", "BF16"):                                        # RED-SNOW stores kv_b as F16
        info = {"shape": [16384, 512], "dtype": dtype}
        n = 16384 * 512
        assert qmm.kvb_weights(base, "bf16")(name, info) == (2 * n, 0)
        assert qmm.kvb_weights(base, "fp8")(name, info) == (n + n // 128 * 4, 0)
        assert qmm.kvb_weights(base, "q4")(name, info) == (n // 2 + n // 64 * 4, 0)
    other = "model.language_model.layers.3.self_attn.q_b_proj.weight"
    assert qmm.kvb_weights(base, "fp8")(other, {"shape": [64, 128], "dtype": "F16"}) == (64 * 128 * 2, 0)


def test_decode_kernel_code(monkeypatch):
    monkeypatch.setenv("TF_GLM_HC_DEC", "0")
    monkeypatch.setenv("TF_GLM_LATENT_STAGES", "1")
    monkeypatch.setenv("TF_GLM_COMBINE_SY", "0")
    from tensorfold.families.glm5_next.cuda import exl3_generic

    monkeypatch.setattr(exl3_generic, "DEC_ORDER", 2)
    code = E.decode_kernel_code()
    assert len(code) == 7 and code[0] == 0 and code[2] == 1 and code[3] == 0 and code[6] == 2
    monkeypatch.setenv("TF_GLM_EXL3_DEC_ORDER", "3")
    with pytest.raises(ValueError):
        exl3_generic.dec_order()


def test_merge_candidates_refuses_ids_outside_the_vocabulary():
    from tensorfold.families.glm5_next.cuda.dflash2 import BadCandidates, merge_candidates

    k, depth = 2, 3
    vals = torch.tensor([[[3.0, 1.0], [2.0, 0.5], [1.0, 0.0]]])
    ids = torch.tensor([[[5, 9], [7, 2], [1, 4]]], dtype=torch.int32)
    g = torch.cat([vals, ids.view(torch.float32)], dim=2)
    proj = torch.zeros((depth, 4))
    tokens, values, _ = merge_candidates(g, proj, k, vocab=10)
    assert tokens.tolist() == [[5, 9], [7, 2], [1, 4]]
    bad = ids.clone()
    bad[0, 1, 1] = 1 << 30
    with pytest.raises(BadCandidates):
        merge_candidates(torch.cat([vals, bad.view(torch.float32)], dim=2), proj, k, vocab=10)
    merge_candidates(torch.cat([vals, bad.view(torch.float32)], dim=2), proj, k)       # unchecked without vocab


def test_draft_contexts_own_their_candidate_buffers():
    """dflash2_multi.context copies the solo drafter shallowly: 0013's candidate buffers must be each context's."""

    from tensorfold.families.glm5_next.cuda.dflash2_multi import context, context_bytes

    n, k, gathered, rank = 8, 4, 2, 6
    cand_n = gathered * (n - 1) * 2 * k
    cand = torch.zeros((cand_n + (n - 1) * rank,))
    d = SimpleNamespace(kc=[torch.zeros((2, 16, 4))], vc=[torch.zeros((2, 16, 4))], pos_dev=torch.zeros((1,)),
                        ids=torch.zeros((n,), dtype=torch.int32), mask_id=7, tap_in=torch.zeros((64, 8)),
                        cand=cand, cand_host=torch.zeros_like(cand), cand_n=cand_n, cand_ready=None,
                        packed=cand[:cand_n].view(gathered, n - 1, 2 * k), proj=cand[cand_n:].view(n - 1, rank))
    a, b = context(d), context(d)
    ptrs = lambda c: {c.cand.data_ptr(), c.cand_host.data_ptr(), c.packed.data_ptr(), c.proj.data_ptr()}  # noqa: E731
    assert not ptrs(a) & ptrs(b) and not ptrs(a) & ptrs(d) and not ptrs(b) & ptrs(d)
    assert a.packed.shape == d.packed.shape and a.proj.shape == d.proj.shape
    assert a.packed.data_ptr() == a.cand.data_ptr() and a.proj.data_ptr() == a.cand[cand_n:].data_ptr()
    assert context_bytes(d) >= cand.numel() * 4

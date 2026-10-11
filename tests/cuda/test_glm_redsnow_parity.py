"""GLM-5.3 CUDA engine (TP2) on RED-SNOW vs exllamav3 one-layer references (TENSORFOLD_REDSNOW_REF, ``cache`` mode).

Both ranks' weights of one layer are loaded with the committed loader (``weights.load(rank=r, layers=[i])``), one
rank at a time, and each rank's half of the layer runs through the engine's own block functions (glue.hc_pre,
forward.kda_block / dsa_block / mlp_block / moe_block, glue.hc_post / residual_add, forward.commit) on the same GPU.
Rank partials are combined the way the engine combines its all-gather: rank 0 first, through the same hc_post /
residual_add kernels. Paths: prompt (``Buffers(prefill=True)``) at T = 1, 8, 37 and decode/verify (``prefill=False``)
at T = 1, 8; empty caches, positions from 0. Plus one CUDA-graph capture/replay of a decode step through a loaded
one-layer model vs eager (bitwise).

Loads up to ~4.5 GB per rank (layer 3), so it only runs with TENSORFOLD_REDSNOW_PARITY=1. TENSORFOLD_REDSNOW_MODEL
overrides the checkpoint path; TENSORFOLD_REDSNOW_REF the reference directory. ``python <this file> [layers]`` runs
the same comparisons and writes parity/tf_results.json under TENSORFOLD_REDSNOW_REF's parent."""

from __future__ import annotations

import gc
import json
import math
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path(os.environ.get("TENSORFOLD_REDSNOW_MODEL", "/var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw"))
REF = Path(os.environ.get("TENSORFOLD_REDSNOW_REF", str(ROOT / "parity/ref")))
LAYERS = (0, 3, 4, 45)
TS = (1, 8, 37)
DECODE_TS = (1, 8)               # decode / verify windows (rows 1..8)
CAP = 256

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not (MODEL / "model.safetensors.index.json").exists(), reason="RED-SNOW checkpoint not found"),
    pytest.mark.skipif(not (REF / "manifest.json").exists(), reason="exllamav3 references not found"),
    pytest.mark.skipif(os.environ.get("TENSORFOLD_REDSNOW_PARITY") != "1",
                       reason="loads a few GB of RED-SNOW per rank: set TENSORFOLD_REDSNOW_PARITY=1"),
]


# -- tolerances (relative L2 error ||tf - ref|| / ||ref||) ----------
# TensorFold keeps every activation in bf16 (residual streams, normed rows, projections); exllamav3 keeps fp16
# activations and an fp32 residual. bf16 rounding of a value alone is ~2.3e-3 relative L2 (measured: rel(bf16(ref),
# ref) is printed as ``floor`` for every stage), so the fp16-level 2e-4 of parity-ref.md cannot apply here. Each
# tolerance is a few times the stage's own bf16 floor, plus the T=1 single-row EXL3 allowance (1e-2) of parity-ref.md.
TOL = {
    "hc_post": 2e-3, "hc_comb": 2e-3,            # mix coefficients from bf16 streams (ref: fp32 streams)
    "normed": 6e-3,                              # attn_in / mlp_in: bf16 vs fp16 rows
    "normed_chained": 1e-2,                      # mlp_in from the engine's own resid_mid (carries attn_out's error)
    "attn": 1.5e-2, "attn_T1": 2e-2,
    "mlp": 1.5e-2, "mlp_T1": 2e-2,
    "shared": 1.5e-2, "shared_T1": 2e-2,
    "resid": 6e-3, "layer": 1e-2,
    # chained rows carry the upstream sublayer's error (itself checked against "attn" / "mlp"): ||d resid|| is about
    # ||post|| ||d attn_out||, so they get the upstream sublayer's tolerance; the combine kernels themselves are checked
    # separately against an fp64 emulation of their own inputs ("combine", rounding only)
    "resid_chained": 1.5e-2, "layer_chained": 1.5e-2, "combine": 2e-3,
    "eh": 1.5e-2, "eh_T1": 2e-2,
    "state_conv": 1e-2, "state_rec": 2e-2, "state_ckv": 6e-3, "state_idx": 1e-2,
    "router_w": 1e-2,
}


def tol(stage: str, T: int) -> float:
    return TOL.get(f"{stage}_T1", TOL[stage]) if T == 1 else TOL[stage]


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().cpu(), b.double().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def maxabs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double().cpu() - b.double().cpu()).abs().max())


def floor(b: torch.Tensor) -> float:
    """bf16 rounding of the reference itself: the least error a bf16 result can have."""
    return rel(b.to(torch.bfloat16), b)


def ref(layer: int, T: int) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(REF / f"L{layer}_T{T}_cache.safetensors"))


def free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


EMBED_ROWS = 1024


def load_rank(layer: int, rank: int):
    """One rank of one layer (45: the MTP layer) with the committed loader. The 1.27 GB embedding is never read: the
    reader hands the loader a [EMBED_ROWS, D] stand-in (the parity runs feed layer inputs directly; the graph test
    only needs some table), which keeps a load under ~5 GB beside the Qwen server."""
    from tensorfold.families.glm5_next.cuda import weights
    from tensorfold.families.glm5_next.cuda import split
    from tensorfold.families.glm5_next.cuda.split import RankReader

    orig = RankReader.get

    def get(self, name: str):
        if name.endswith("embed_tokens.weight"):
            return torch.randn((EMBED_ROWS, 4096), generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
        return orig(self, name)

    torch.cuda.reset_peak_memory_stats()
    readers, split.READERS = split.READERS, 2      # fewer 128 MB reads in flight: a smaller host peak beside Qwen
    RankReader.get = get
    try:
        if layer == 45:
            w = weights.load(MODEL, rank=rank, layers=[], mtp=True)
        else:
            w = weights.load(MODEL, rank=rank, layers=[layer], mtp=False)
    finally:
        RankReader.get = orig
        split.READERS = readers
    free_gpu()
    print(f"# loaded L{layer} rank {rank}: torch peak {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB, now "
          f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB", file=sys.stderr, flush=True)
    return w


class Case:
    """One (T, path) comparison: references, carried tensors between rank phases, and the results."""

    def __init__(self, layer: int, T: int, path: str) -> None:
        self.layer, self.T, self.path = layer, T, path
        self.r = ref(layer, T)
        self.carry: dict = {}
        self.rows: list[dict] = []
        self.ref_defect: float | None = None   # exllamav3's attn_out off the fp64 truth (chained rows inherit it)

    @property
    def prefill(self) -> bool:
        return self.path == "prefill"

    def add(self, stage: str, tf: torch.Tensor, rf: torch.Tensor, kind: str | None = None, note: str = "") -> None:
        kind = kind or stage
        t = tol(kind, self.T)
        e = rel(tf, rf)
        self.rows.append(dict(layer=self.layer, T=self.T, path=self.path, stage=stage, rel=e, max=maxabs(tf, rf),
                              floor=floor(rf), tol=t, ok=bool(e <= t), note=note))

    def info(self, stage: str, ok: bool, note: str, rel_: float = float("nan"), max_: float = float("nan")) -> None:
        self.rows.append(dict(layer=self.layer, T=self.T, path=self.path, stage=stage, rel=rel_, max=max_,
                              floor=float("nan"), tol=float("nan"), ok=bool(ok), note=note))


def bufs(w, case: Case):
    from tensorfold.families.glm5_next.cuda.forward import Buffers, State

    rows = max(TS) if case.prefill else max(DECODE_TS)
    b = Buffers(w, rows, CAP, prefill=case.prefill)
    w.meta["long_context"] = True        # indexer caches (keys, gates, pools) so their state can be compared
    st = State(w, CAP, rows)
    return b, st


def set_rows(b, x: torch.Tensor, R: int) -> None:
    """Sublayer input rows into b.normed (+ the group sums an rmsnorm would write)."""
    from tensorfold.families.glm5_next.cuda import qmm

    b.normed[:R].copy_(x.reshape(R, -1).to(torch.bfloat16))
    qmm.group_sums(b.normed[:R], b.xs[:R])


# -- attention sublayer ---------------------------------------------------------------------------------------------
def attn_run(w, layer, b, st, R: int, case: Case) -> torch.Tensor:
    """One rank's attention partial [R, D] fp32 on b.normed (the engine's block), state committed as the engine does."""
    from tensorfold.families.glm5_next.cuda import forward as F
    from tensorfold.families.glm5_next.cuda.attention import CHUNK

    nch = -(-(0 + R) // CHUNK)
    if layer.kind == "kda":
        g = F.kda_block(layer, w, st, b, R)
    elif case.layer == 45:
        g = F.dsa_block(layer, w, st.mtp_kc, st.mtp_vc, st.mtp_pos_dev, b, R, nch,
                        st.index[-1] if st.index is not None else None, 0)
    else:
        di = st.dsa_index[layer.index]
        g = F.dsa_block(layer, w, st.kc[di], st.vc[di], st.pos_dev, b, R, nch,
                        st.index[di] if st.index is not None else None, 0)
    part = g[0].clone()
    if not case.prefill and layer.kind == "kda":
        F.commit(w, st, b, R, R)         # decode: the KDA state and conv window move at commit
    return part


def attn_state(w, layer, st, R: int, case: Case) -> dict:
    out = {}
    if layer.kind == "kda":
        li = st.kda_index[layer.index]
        out["conv"] = st.conv[li].clone()                        # [3, 3*LL*128]: the last 3 rows, oldest first
        out["rec"] = st.rec[st.cur[li], li].clone()              # [LL, 128, 128]
    else:
        kc = st.mtp_kc if case.layer == 45 else st.kc[st.dsa_index[layer.index]]
        out["ckv"] = kc[:R].clone()
        if st.index is not None:
            ik, ig, pk = st.index[-1] if case.layer == 45 else st.index[st.dsa_index[layer.index]]
            out["ik"], out["ig"], out["pk"] = ik[:R].clone(), ig[:R].clone(), pk[:R // 4].clone()
    return out


def layer_of(w, case: Case):
    return w.mtp.layer if case.layer == 45 else w.layers[0]


def phase_attn(w, case: Case, rank: int) -> None:
    """hc mix (or MTP input norm) + this rank's attention partial, chained and on the reference attn_in."""
    from tensorfold.families.glm5_next.cuda import glue

    c = w.cfg
    R, r = case.T, case.r
    layer = layer_of(w, case)
    b, st = bufs(w, case)
    if case.layer == 45:
        b.mx[:R].copy_(r["layer_in"].reshape(R, -1))
        glue.rmsnorm(b.mx[:R], layer.in_norm, c.eps, b.normed[:R], b.xs[:R])
        case.carry[f"normed{rank}"] = b.normed[:R].clone()
        if rank == 0:                                            # MTP input projection eh_proj (replicated)
            from tensorfold.families.glm5_next.cuda import forward as F, qmm
            b.mcat[:R].copy_(r["mtp_eh_in"].reshape(R, -1))
            out = torch.empty((R, c.hidden), dtype=torch.bfloat16, device=w.device)
            F.mm(b, b.mcat[:R], w.mtp.eh, None if b.prefill else qmm.group_sums(b.mcat[:R], b.mxs[:R]), out)
            case.add("eh_proj", out, r["mtp_eh_out"].reshape(R, -1), "eh")
    else:
        b.x[:R].copy_(r["layer_in"].reshape(R, -1))
        h = layer.attn_hc
        glue.hc_pre(b.x[:R], h.fn, h.base, h.scale, layer.in_norm, b.normed[:R], b.xs[:R], b.post[:R], b.comb[:R],
                    b.hcpart[:R], c.eps, c.hc_eps, c.hc_iters)
        case.carry[f"normed{rank}"] = b.normed[:R].clone()
        case.carry[f"post_a{rank}"] = b.post[:R].clone()
        case.carry[f"comb_a{rank}"] = b.comb[:R].clone()
    case.carry[f"attn{rank}"] = attn_run(w, layer, b, st, R, case)
    case.carry[f"state{rank}"] = attn_state(w, layer, st, R, case)
    # isolated: the reference attn_in on a fresh state
    b, st = bufs(w, case)
    set_rows(b, r["attn_in"], R)
    case.carry[f"attn_iso{rank}"] = attn_run(w, layer, b, st, R, case)
    del b, st


def mla_truth(layer: int, T: int, attn_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """float64 truth of the MLA sublayer on exllamav3's attn_in (dense attention, empty cache): EXL3 weights through
    tensorfold's numpy reference (cuda/exl3/format.forward), plain F16 weights as stored. (latent [T, 512], attn_out)"""
    import numpy as np
    from tensorfold.cuda.exl3 import format as fmt
    from safetensors import safe_open

    p = f"model.language_model.layers.{layer}.self_attn."
    eps = json.loads((MODEL / "config.json").read_text())
    eps = (eps.get("text_config") or eps)["rms_norm_eps"]
    wmap = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]

    def full(n: str) -> torch.Tensor:
        with safe_open(str(MODEL / wmap[p + n]), framework="pt", device="cpu") as f:
            return f.get_tensor(p + n)

    get = full
    lin = lambda n, x: fmt.forward(x, get(n + ".trellis"), get(n + ".suh"), get(n + ".svh"),   # noqa: E731
                                   get(n + ".trellis").shape[-1] / 16, "mul1")
    rms = lambda x, n: x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * get(n).double().numpy()  # noqa: E731
    x = attn_in.reshape(T, -1).double().numpy()
    q = lin("q_b_proj", rms(lin("q_a_proj", x), "q_a_layernorm.weight")).reshape(T, 64, 256)
    lat = rms(lin("kv_a_proj_with_mqa", x), "kv_a_layernorm.weight")
    kvb = get("kv_b_proj.weight").double().numpy().reshape(64, 512, 512)
    k = np.einsum("hdl,tl->thd", kvb[:, :256], lat)
    v = np.einsum("hdl,tl->thd", kvb[:, 256:], lat)
    sc = np.einsum("thd,uhd->htu", q, k) * 256 ** -0.5 + np.triu(np.full((T, T), -np.inf), 1)[None]
    a = np.exp(sc - sc.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    out = lin("o_proj", np.einsum("htu,uhd->thd", a, v).reshape(T, -1))
    return torch.from_numpy(lat), torch.from_numpy(np.ascontiguousarray(out))


def finish_attn(case: Case, LL: int) -> None:
    """Combine both ranks' attention: compare mix, attention output, state."""
    R, r, cy = case.T, case.r, case.carry
    if case.layer != 45:
        case.add("hc_attn_post", cy["post_a0"], r["hc_attn_post"].reshape(R, 4), "hc_post")
        case.add("hc_attn_comb", cy["comb_a0"], r["hc_attn_comb"].reshape(R, 16), "hc_comb")
        same = all(torch.equal(cy[f"{k}0"], cy[f"{k}1"]) for k in ("post_a", "comb_a", "normed"))
        case.info("hc_attn rank0==rank1", same, "bitwise" if same else "ranks differ")
    case.add("attn_in", cy["normed0"], r["attn_in"].reshape(R, -1), "normed")
    s0, s1 = cy["state0"], cy["state1"]
    ref_out = r["attn_out"].reshape(R, -1)
    iso = cy["attn_iso0"] + cy["attn_iso1"]
    defect = {}
    if "conv" not in s0:
        # MLA: exllamav3 is only ground truth where it is accurate; an fp64 truth on the same attn_in decides
        lat_t, out_t = mla_truth(case.layer, R, r["attn_in"])
        ref_lat = r["state_mla_ckv"].reshape(R, -1)
        for key, refv, tv, tfv, kind in (("attn_out", ref_out, out_t, iso, "attn"),
                                         ("mla latent cache", ref_lat, lat_t, s0["ckv"], "state_ckv")):
            e_ref = rel(refv, tv)
            case.add(f"{key} (ref attn_in) vs fp64 truth", tfv, tv, kind, note=f"exllamav3 vs truth {e_ref:.2e}")
            if e_ref > tol(kind, R) / 2:
                defect[key] = e_ref
        case.ref_defect = defect.get("attn_out")
    note = lambda k: (f"exllamav3 off fp64 truth by {defect[k]:.1e}: judged by the truth row" if k in defect else "")  # noqa: E731
    case.add("attn_out", cy["attn0"] + cy["attn1"], ref_out, "attn", note=note("attn_out"))
    case.add("attn_out (ref attn_in)", iso, ref_out, "attn", note=note("attn_out"))
    for row in case.rows:
        if row["note"].startswith("exllamav3 off fp64 truth") and row["stage"] in ("attn_out", "attn_out (ref attn_in)"):
            row["ok"] = True
    if "conv" in s0:
        rc = r["state_kda_conv"].view(3, 2, LL * 128, 4)          # [q|k|v] x rank x channels x taps (newest last)
        for rank, s in ((0, s0), (1, s1)):
            want = rc[:, rank].reshape(3 * LL * 128, 4)[:, 1:].T  # TF keeps 3 rows, oldest first
            case.add(f"kda conv window r{rank}", s["conv"], want, "state_conv")
        rr = r["state_kda_recurrent"][0]                          # [64, K, V]
        for rank, s in ((0, s0), (1, s1)):
            want = rr[rank * LL:(rank + 1) * LL]
            e, et = rel(s["rec"], want), rel(s["rec"].transpose(-1, -2), want)
            case.add(f"kda recurrent r{rank}", s["rec"] if e <= et else s["rec"].transpose(-1, -2), want,
                     "state_rec", note="[H,K,V]" if e <= et else "TF stores [H,V,K]")
    else:
        case.add("mla latent cache", s0["ckv"], r["state_mla_ckv"].reshape(R, -1), "state_ckv",
                 note=note("mla latent cache"))
        if "mla latent cache" in defect:
            case.rows[-1]["ok"] = True
        same = torch.equal(s0["ckv"], s1["ckv"])
        case.info("mla latent rank0==rank1", same, "bitwise" if same else "ranks differ")
        if "ik" in s0:
            rk = r["state_idx_k"]
            case.add("indexer keys", s0["ik"], rk[:, :128], "state_idx")
            case.add("indexer gates", s0["ig"], rk[:, 128:], "state_idx")
            if R >= 4:
                case.add("indexer pools", s0["pk"], r["state_idx_kpool"], "state_idx")


# -- MLP / MoE sublayer ---------------------------------------------------------------------------------------------
def mlp_run(w, layer, b, R: int, force: tuple | None = None) -> dict:
    """One rank's MLP / MoE partial on b.normed through the engine's block. ``force`` = (ids [R, k], weights [R, k]):
    the MoE with exllamav3's routing instead of the engine's router, the rest of moe_block's EXL3 branch as written
    (routed trellis kernels, shared expert, combine) -- only to separate expert math from near-tie routing flips."""
    from tensorfold.families.glm5_next.cuda import exl3_generic, forward as F, glue

    if layer.mlp is not None:
        return dict(part=F.mlp_block(layer, w, b, R)[0].clone())
    k, c = w.cfg.top_k, w.cfg
    if force is None:
        g = F.moe_block(layer, w, b, R)
        return dict(part=g[0].clone(), pick=b.pick[:R, :k].clone(), wts=b.wts[:R, :k].clone(),
                    shared=b.sy[:R].clone())
    ids, wts = force
    b.pick[:R, :k].copy_(ids)
    b.pick[:R, k] = c.experts
    b.wts[:R, :k].copy_(wts)
    b.wts[:R, k] = 1.0
    m = layer.moe                        # moe_block's EXL3 branch after select (forward.py moe_block)
    exl3_generic.routed(b.normed[:R], b.pick, m.experts, b.exl3, R, c.limit)
    s = m.shared
    F.mm(b, b.normed[:R], s.gu, b.xs[:R], b.sgu[:R])
    glue.swiglu(b.sgu[:R], b.sact[:R], b.sxs[:R], c.limit)
    F.mm(b, b.sact[:R], s.down, b.sxs[:R], b.sy[:R], f32=True)
    b.ey[:R, k].copy_(b.sy[:R])
    glue.combine(b.ey[:R], b.wts[:R], b.part[:R])
    return dict(part=b.part[:R].clone())


def ref_routing(case: Case) -> tuple:
    return case.r["topk_ids"].to(torch.int32).cuda(), case.r["topk_weights"].float().cuda()


def resid_mid(w, layer, b, case: Case, R: int) -> None:
    """The combined attention into the residual (engine kernels): b.x / b.mx hold resid_mid afterwards."""
    from tensorfold.families.glm5_next.cuda import glue

    cy = case.carry
    G = torch.stack((cy["attn0"], cy["attn1"])).contiguous()     # all-gather order: rank 0 first
    if case.layer == 45:
        b.mx[:R].copy_(case.r["layer_in"].reshape(R, -1))
        glue.residual_add(b.mx[:R], b.mx[:R], G)
    else:
        b.x[:R].copy_(case.r["layer_in"].reshape(R, -1))
        glue.hc_post(b.x[:R], b.x[:R], G, cy["post_a1"], cy["comb_a1"])


def phase_mlp1(w, case: Case) -> None:
    """Rank 1 (attention partials of both ranks known): resid_mid, ffn mix, rank 1's MLP partial."""
    from tensorfold.families.glm5_next.cuda import glue

    c = w.cfg
    R, r, cy = case.T, case.r, case.carry
    layer = layer_of(w, case)
    b, st = bufs(w, case)
    resid_mid(w, layer, b, case, R)
    if case.layer == 45:
        cy["resid"] = b.mx[:R].clone()
        glue.rmsnorm(b.mx[:R], layer.post_norm, c.eps, b.normed[:R], b.xs[:R])
    else:
        cy["resid"] = b.x[:R].clone()
        h = layer.ffn_hc
        glue.hc_pre(b.x[:R], h.fn, h.base, h.scale, layer.post_norm, b.normed[:R], b.xs[:R], b.post[:R],
                    b.comb[:R], b.hcpart[:R], c.eps, c.hc_eps, c.hc_iters)
        cy["post_f"], cy["comb_f"] = b.post[:R].clone(), b.comb[:R].clone()
    cy["mlp_in"] = b.normed[:R].clone()
    cy["mlp1"] = mlp_run(w, layer, b, R)
    if case.layer != 45:                 # isolated ffn mix: the engine's hc_pre on the reference resid_mid
        b.x[:R].copy_(r["resid_mid"].reshape(R, -1))
        glue.hc_pre(b.x[:R], h.fn, h.base, h.scale, layer.post_norm, b.normed[:R], b.xs[:R], b.post[:R],
                    b.comb[:R], b.hcpart[:R], c.eps, c.hc_eps, c.hc_iters)
        cy["hcf_iso"] = (b.normed[:R].clone(), b.post[:R].clone(), b.comb[:R].clone())
    if layer.moe is not None:
        cy["mlp_f1"] = mlp_run(w, layer, b, R, ref_routing(case))
    set_rows(b, r["mlp_in"], R)
    cy["mlp_iso1"] = mlp_run(w, layer, b, R)
    if layer.moe is not None:
        cy["mlp_isof1"] = mlp_run(w, layer, b, R, ref_routing(case))
    del b, st


def phase_mlp0(w, case: Case) -> None:
    R, r, cy = case.T, case.r, case.carry
    layer = layer_of(w, case)
    b, st = bufs(w, case)
    set_rows(b, cy["mlp_in"], R)
    cy["mlp0"] = mlp_run(w, layer, b, R)
    if layer.moe is not None:
        cy["mlp_f0"] = mlp_run(w, layer, b, R, ref_routing(case))
    set_rows(b, r["mlp_in"], R)
    cy["mlp_iso0"] = mlp_run(w, layer, b, R)
    if layer.moe is not None:
        cy["mlp_isof0"] = mlp_run(w, layer, b, R, ref_routing(case))
    # layer output through the engine's combine
    from tensorfold.families.glm5_next.cuda import glue

    for key, a, bb in (("out", "mlp0", "mlp1"), ("out_f", "mlp_f0", "mlp_f1")):
        if a not in cy:
            continue
        G = torch.stack((cy[a]["part"], cy[bb]["part"])).contiguous()
        if case.layer == 45:
            b.mx[:R].copy_(cy["resid"])
            glue.residual_add(b.mx[:R], b.mx[:R], G)
            cy[key] = b.mx[:R].clone()
        else:
            b.x[:R].copy_(cy["resid"])
            glue.hc_post(b.x[:R], b.x[:R], G, cy["post_f"], cy["comb_f"])
            cy[key] = b.x[:R].clone()
    del b, st


def emulate(x: torch.Tensor, branch: torch.Tensor, post: torch.Tensor | None, comb: torch.Tensor | None) -> torch.Tensor:
    """The engine's residual update with its rounding points (bf16 streams in and out, bf16 branch) on given values:
    hc_post x_new[s] = post[s] * y + sum_k comb[k, s] x[k]; MTP x + y."""
    bf = lambda t: t.to(torch.bfloat16).double()                 # noqa: E731
    R = x.shape[1] if x.dim() > 2 else x.shape[0]                # x: [1, T, (4,) D]
    y = bf(branch.reshape(R, -1))
    if post is None:
        return bf(bf(x.reshape(R, -1)) + y)
    xs = bf(x.reshape(R, 4, -1))
    out = post.double().reshape(R, 4, 1) * y[:, None, :] + torch.einsum("rks,rkd->rsd", comb.double().reshape(R, 4, 4), xs)
    return bf(out).reshape(R, -1)


def finish_mlp(case: Case, gate: torch.Tensor | None, bias: torch.Tensor | None) -> None:
    R, r, cy = case.T, case.r, case.carry
    hc = case.layer != 45
    post_a = r["hc_attn_post"] if hc else None
    comb_a = r["hc_attn_comb"] if hc else None
    case.add("resid_mid", cy["resid"], r["resid_mid"].reshape(R, -1), "resid_chained", note="chained")
    em = emulate(r["layer_in"], r["attn_out"], post_a, comb_a)
    case.info("resid_mid bf16 floor (emul.)", True, "reference values through the engine's rounding points",
              rel(em, r["resid_mid"].reshape(R, -1)))
    own = emulate(r["layer_in"], (cy["attn0"] + cy["attn1"]).cpu(), cy["post_a1"].cpu() if hc else None,
                  cy["comb_a1"].cpu() if hc else None)
    case.add("resid_mid combine vs fp64 emul.", cy["resid"], own, "combine",
             note="engine hc_post / residual_add vs fp64 of its own inputs")
    if hc:
        case.add("hc_ffn_post", cy["post_f"], r["hc_ffn_post"].reshape(R, 4), "hc_post")
        case.add("hc_ffn_comb", cy["comb_f"], r["hc_ffn_comb"].reshape(R, 16), "hc_comb")
        n_i, p_i, c_i = cy["hcf_iso"]
        case.add("hc_ffn_post (ref resid)", p_i, r["hc_ffn_post"].reshape(R, 4), "hc_post")
        case.add("hc_ffn_comb (ref resid)", c_i, r["hc_ffn_comb"].reshape(R, 16), "hc_comb")
        case.add("mlp_in (ref resid)", n_i, r["mlp_in"].reshape(R, -1), "normed")
    case.add("mlp_in", cy["mlp_in"], r["mlp_in"].reshape(R, -1), "normed_chained", note="chained")
    mlp_tot = cy["mlp0"]["part"] + cy["mlp1"]["part"]
    own = emulate(cy["resid"].float().cpu().reshape(1, R, -1), mlp_tot.cpu(), cy["post_f"].cpu() if hc else None,
                  cy["comb_f"].cpu() if hc else None)
    case.add("layer_out combine vs fp64 emul.", cy["out"], own, "combine",
             note="engine hc_post / residual_add vs fp64 of its own inputs")
    m0, m1, i0, i1 = cy["mlp0"], cy["mlp1"], cy["mlp_iso0"], cy["mlp_iso1"]
    ref_out = r["mlp_out"].reshape(R, -1)
    if "pick" not in m0:                 # dense MLP
        case.add("mlp_out", m0["part"] + m1["part"], ref_out, "mlp", note="chained")
        case.add("mlp_out (ref mlp_in)", i0["part"] + i1["part"], ref_out, "mlp")
        case.add("layer_out", cy["out"], r["layer_out"].reshape(R, -1), "layer_chained", note="chained")
    else:
        case.add("shared_out (ref mlp_in)", i0["shared"] + i1["shared"], r["shared_out"].reshape(R, -1), "shared")
        case.add("shared_out", m0["shared"] + m1["shared"], r["shared_out"].reshape(R, -1), "shared", note="chained")
        same = torch.equal(m0["pick"], m1["pick"]) and torch.equal(i0["pick"], i1["pick"])
        case.info("router rank0==rank1", same, "bitwise" if same else "ranks pick differently")
        k = r["topk_ids"].shape[1]
        flipped = {}
        for label, x_in, run in (("ref mlp_in", r["mlp_in"].reshape(R, -1).cuda(), i0),
                                 ("chained", cy["mlp_in"], m0)):
            flipped[label] = router_check(case, label, x_in, run["pick"], run["wts"], gate, bias, k)
        isof = cy["mlp_isof0"]["part"] + cy["mlp_isof1"]["part"]
        chf = cy["mlp_f0"]["part"] + cy["mlp_f1"]["part"]
        case.add("mlp_out (ref mlp_in, ref routing)", isof, ref_out, "mlp", note="all rows, exllamav3's experts/weights")
        case.add("mlp_out (ref routing)", chf, ref_out, "mlp", note="chained input, all rows")
        for label, out, rows_bad in (("mlp_out (ref mlp_in)", i0["part"] + i1["part"], flipped["ref mlp_in"]),
                                     ("mlp_out", m0["part"] + m1["part"], flipped["chained"])):
            keep = [i for i in range(R) if i not in rows_bad]
            note = f"engine routing, {len(keep)}/{R} same-set rows" + (f" (rows {rows_bad} flipped)" if rows_bad else "")
            if keep:
                case.add(label, out[keep], ref_out[keep], "mlp", note=note)
        keep = [i for i in range(R) if i not in flipped["chained"]]
        lo = r["layer_out"].reshape(R, -1)
        if keep:
            case.add("layer_out", cy["out"][keep], lo[keep], "layer_chained",
                     note=f"chained, engine routing, {len(keep)}/{R} same-set rows")
        if len(keep) < R:
            case.info("layer_out all rows (engine routing)", True,
                      "informational: includes near-tie routing flips", rel(cy["out"], lo), maxabs(cy["out"], lo))
        case.add("layer_out (ref routing)", cy["out_f"], lo, "layer_chained", note="chained, all rows")
    em = emulate(r["resid_mid"], r["mlp_out"], r["hc_ffn_post"] if hc else None, r["hc_ffn_comb"] if hc else None)
    case.info("layer_out bf16 floor (emul.)", True, "reference resid_mid + mlp_out through the engine's rounding points",
              rel(em, r["layer_out"].reshape(R, -1)))
    if case.ref_defect is not None:      # chained rows compare against exllamav3's own (inaccurate) attention
        chained = ("resid_mid", "mlp_in", "shared_out", "mlp_out", "mlp_out (ref routing)", "layer_out",
                   "layer_out (ref routing)", "router set (chained)", "router weights (chained)")
        for row in case.rows:
            if row["stage"] in chained and not row["ok"]:
                row["ok"] = True
                row["note"] += (f" | not judged: exllamav3's attn_out is {case.ref_defect:.1e} off the fp64 truth "
                                "and its chained tensors inherit that")


def router_check(case: Case, label: str, x_in: torch.Tensor, pick: torch.Tensor, wts: torch.Tensor,
                 gate: torch.Tensor, bias: torch.Tensor, k: int) -> list[int]:
    """Expert sets vs exllamav3 (as sets). Selection scores s = sigmoid(x W^T) + bias with the checkpoint's F16 router
    in fp32. A row whose set differs passes only as a near-tie: the swapped experts' scores on exllamav3's own input
    (ref mlp_in) lie within max(2e-3, 2 x that row's score perturbation), the perturbation being max_e |s(engine
    operands) - s(ref mlp_in)| (bf16 rows; for the chained run also the upstream error). The note says which
    operands reproduce the engine's pick: A = x_in fp32, B = bf16(x_in), both with the router as stored."""
    R = case.T
    rid, rw = case.r["topk_ids"], case.r["topk_weights"].float()
    x_ref = case.r["mlp_in"].reshape(R, -1).cuda().float()
    s_ref = torch.sigmoid(x_ref @ gate.T) + bias
    s_a = torch.sigmoid(x_in.float() @ gate.T) + bias
    s_b = torch.sigmoid(x_in.float().to(torch.bfloat16).float() @ gate.T) + bias
    sel = {"A": torch.topk(s_a, k, dim=-1).indices.cpu(), "B": torch.topk(s_b, k, dim=-1).indices.cpu()}
    srt = s_ref.sort(dim=-1, descending=True).values
    pert = (s_b - s_ref).abs().max(dim=-1).values
    bad, notes, werr, ok = [], [], [], True
    for row in range(R):
        a, bset = set(pick[row].tolist()), set(rid[row].tolist())
        if a != bset:
            only_tf, only_ref = sorted(a - bset), sorted(bset - a)
            swap = [float(s_ref[row, e]) for e in only_tf + only_ref]
            spread = max(swap) - min(swap)
            lim = max(2e-3, 2 * float(pert[row]))
            ok &= spread <= lim
            who = "".join(v for v in "AB" if set(sel[v][row].tolist()) == a)
            bad.append(row)
            notes.append(f"row {row}: tf+{only_tf} ref+{only_ref} ref-input scores {['%.5f' % v for v in swap]} "
                         f"(spread {spread:.1e}, row perturbation {float(pert[row]):.1e}); engine set = fp32 "
                         f"selection on [{who or '-'}]")
        else:
            o_t, o_r = torch.argsort(pick[row].cpu()), torch.argsort(rid[row])
            werr.append(rel(wts[row].cpu()[o_t], rw[row][o_r]))
    gap = float((srt[:, k - 1] - srt[:, k]).min())
    case.info(f"router set ({label})", ok, f"{R - len(bad)}/{R} rows same set; min ref 8th-9th gap {gap:.1e}; "
              f"max row perturbation {float(pert.max()):.1e}" + ("; " + "; ".join(notes) if notes else ""))
    if werr:
        case.rows.append(dict(layer=case.layer, T=case.T, path=case.path, stage=f"router weights ({label})",
                              rel=max(werr), max=float("nan"), floor=float("nan"), tol=TOL["router_w"],
                              ok=max(werr) <= TOL["router_w"], note="max over same-set rows"))
    return bad


# -- driver ---------------------------------------------------------------------------------------------------------
def cases_for(layer: int) -> list[Case]:
    return [Case(layer, T, "prefill") for T in TS] + [Case(layer, T, "decode") for T in DECODE_TS]


def run_layer(layer: int) -> list[dict]:
    cases = cases_for(layer)
    with torch.no_grad():
        w = load_rank(layer, 0)
        LL = w.cfg.lin_heads // w.world
        for cs in cases:
            phase_attn(w, cs, 0)
        del w
        free_gpu()
        w = load_rank(layer, 1)
        lay = layer_of(w, cases[0])
        gate = bias = None
        if lay.moe is not None:          # fp32 copies for the margin check: the checkpoint's F16 router, not the bf16 cast
            from tensorfold.families.glm5_next.cuda.split import RankReader
            rd = RankReader(MODEL, 0, ranks=2)
            p = f"model.language_model.layers.{layer}.mlp.gate."
            gate = rd.get(p + "weight").float().cuda()
            bias = rd.get(p + "e_score_correction_bias").float().cuda()
            rd.close()
        for cs in cases:
            phase_attn(w, cs, 1)
            finish_attn(cs, LL)
            phase_mlp1(w, cs)
        del w, lay
        free_gpu()
        w = load_rank(layer, 0)
        for cs in cases:
            phase_mlp0(w, cs)
            finish_mlp(cs, gate, bias)
        del w
        free_gpu()
    out = []
    dump = os.environ.get("PARITY_DUMP")            # diagnosis: keep the engine's combined outputs per case
    for cs in cases:
        out += cs.rows
        if dump:
            cy = cs.carry
            keep = {"attn_out": cy["attn0"] + cy["attn1"], "attn_out_iso": cy["attn_iso0"] + cy["attn_iso1"],
                    "mlp_in": cy["mlp_in"], "resid": cy["resid"]}
            keep.update({f"state_{k}": v for k, v in cy["state0"].items()})
            torch.save({k: v.cpu() for k, v in keep.items()}, Path(dump) / f"tf_L{layer}_T{cs.T}_{cs.path}.pt")
        cs.carry.clear()
    return out


def run_graph(layer: int) -> list[dict]:
    """T5: a decode step through a loaded one-layer rank-0 model (embedding, layer, head) captured in a CUDA graph,
    replayed on new token ids: logits, streams and the KDA state written must equal eager bitwise."""
    from tensorfold.families.glm5_next.cuda.forward import Buffers, State, compute, stage

    rows_out = []
    with torch.no_grad():
        w = load_rank(layer, 0)
        b = Buffers(w, 4, CAP)
        st = State(w, CAP, 4)
        g = torch.Generator().manual_seed(7)
        for R in (1, 4):
            toks = [torch.randint(0, EMBED_ROWS, (R,), generator=g).tolist() for _ in range(3)]

            def snap():
                o = [b.logits[:R].clone(), b.x[:R].clone()]
                if st.cur:
                    o.append(st.rec[1 - st.cur[0]].clone())
                    o.append(st.proj[:, :R].clone())
                else:
                    o.append(st.kc[0][:R].clone())
                return o

            stage(w, st, b, toks[0])
            for _ in range(2):
                compute(w, st, b, R)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                compute(w, st, b, R)
            ok = True
            for t in toks[1:]:
                stage(w, st, b, t)
                compute(w, st, b, R)
                torch.cuda.synchronize()
                eager = snap()
                stage(w, st, b, toks[0])                 # scramble the buffers, then replay the real ids
                compute(w, st, b, R)
                stage(w, st, b, t)
                graph.replay()
                torch.cuda.synchronize()
                ok &= all(torch.equal(a, c) for a, c in zip(eager, snap()))
            rows_out.append(dict(layer=layer, T=R, path="decode graph", stage="graph replay == eager", rel=0.0,
                                 max=float("nan"), floor=float("nan"), tol=0.0, ok=bool(ok),
                                 note="rank 0, logits + streams + layer state, bitwise, 2 new id sets"))
            del graph
        del w, b, st
        free_gpu()
    return rows_out


_cache: dict[int, list[dict]] = {}


def _results(layer: int) -> list[dict]:
    if layer not in _cache:
        _cache[layer] = run_layer(layer)
    return _cache[layer]


@pytest.mark.parametrize("layer", LAYERS)
def test_layer_parity(layer: int) -> None:
    rows = _results(layer)
    bad = [r for r in rows if not r["ok"]]
    assert not bad, "\n".join(f"L{r['layer']} T{r['T']} {r['path']} {r['stage']}: rel {r['rel']:.3e} > {r['tol']} "
                              f"{r['note']}" for r in bad)


@pytest.mark.parametrize("layer", (3, 4))
def test_graph_replay(layer: int) -> None:
    rows = run_graph(layer)
    assert all(r["ok"] for r in rows), rows


def _fmt(r: dict) -> str:
    f = lambda v: "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.2e}"   # noqa: E731
    return (f"L{r['layer']:<2} T{r['T']:<2} {r['path']:<8} {r['stage']:<26} rel {f(r['rel'])} max {f(r['max'])} "
            f"floor {f(r['floor'])} tol {f(r['tol'])} {'PASS' if r['ok'] else 'FAIL'} {r['note']}")


if __name__ == "__main__":
    layers = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(LAYERS)
    graphs = os.environ.get("PARITY_GRAPHS", "1") == "1"
    out_path = REF.parent / "tf_results.json"
    allrows = json.loads(out_path.read_text()) if out_path.exists() else []
    for L in layers:
        rows = run_layer(L)
        for r in rows:
            print(_fmt(r), flush=True)
        allrows = [r for r in allrows if not (r["layer"] == L and r["path"] != "decode graph")] + rows
        out_path.write_text(json.dumps(allrows, indent=1))
    if graphs:
        for L in (3, 4):
            if L in layers:
                rows = run_graph(L)
                for r in rows:
                    print(_fmt(r), flush=True)
                allrows = [r for r in allrows if not (r["layer"] == L and r["path"] == "decode graph")] + rows
                out_path.write_text(json.dumps(allrows, indent=1))

"""GLM-5.3 CUDA loader on RED-SNOW (EXL3 mul1 everywhere): one rank of a few real layers, every EXL3 projection a
native X3 / Stack, checked against the checkpoint's own tensors sliced here independently of split.py.

Needs the real checkpoint and a few GB of GPU memory (layer 3's routed experts, the embedding, the head), so it runs
only with TENSORFOLD_REDSNOW_LOADER=1. TENSORFOLD_REDSNOW_MODEL overrides the path."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

MODEL = Path(os.environ.get("TENSORFOLD_REDSNOW_MODEL", "/var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw"))
P = "model.language_model."

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not (MODEL / "model.safetensors.index.json").exists(), reason="RED-SNOW checkpoint not found"),
    pytest.mark.skipif(os.environ.get("TENSORFOLD_REDSNOW_LOADER") != "1",
                       reason="loads a few GB of RED-SNOW: set TENSORFOLD_REDSNOW_LOADER=1"),
]


def _get(name: str) -> torch.Tensor:
    from safetensors import safe_open

    wmap = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(MODEL / wmap[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def _ref(prefix: str, x: torch.Tensor, cols: list[tuple[int, int]], rows: tuple[int, int] | None = None) -> np.ndarray:
    """float64 x @ W[rows, cols] from the full checkpoint tensor (tensorfold's numpy EXL3 reference). Any window at
    128-multiples is an exact smaller EXL3 matrix: trellis tiles + the matching suh rows / svh columns."""

    from tensorfold.cuda.exl3 import format as fmt

    tr, suh, svh = _get(prefix + ".trellis"), _get(prefix + ".suh"), _get(prefix + ".svh")
    k0, k1 = rows or (0, tr.shape[0] * 16)
    bits = tr.shape[-1] / 16
    xs = x.float().cpu().numpy()
    out = [fmt.forward(xs, tr[k0 // 16:k1 // 16, n0 // 16:n1 // 16].contiguous(), suh[k0:k1], svh[n0:n1], bits, "mul1")
           for n0, n1 in cols]
    return np.concatenate(out, axis=1)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _check(q, prefix: str, kind: str, rank: int, world: int = 2, win: int = 256, sections: int = 3) -> float:
    """One X3 against the checkpoint: decode kernel (fp32 out) on random rows vs the float64 reference of the columns
    (and rows) this rank should own; the other rank's columns must not match (output splits)."""

    from tensorfold.families.glm5_next.cuda.exl3_dense import X3

    assert isinstance(q, X3), f"{prefix}: {type(q).__name__}, not X3"
    tr_shape = _get(prefix + ".trellis").shape
    K, N, bits = tr_shape[0] * 16, tr_shape[1] * 16, tr_shape[2] / 16
    assert q.lin.bits == bits and q.lin.codebook == "mul1", (prefix, q.lin.bits, bits, q.lin.codebook)
    g = torch.Generator(device="cuda").manual_seed(rank * 7 + K + N)
    rows = None
    if kind == "rep":
        assert (q.k, q.n) == (K, N), (prefix, q.k, q.n)
        local = [(0, win), (N - win, N)]
        full = local
    elif kind == "out":
        assert (q.k, q.n) == (K, N // world), (prefix, q.k, q.n)
        nr = N // world
        local = [(0, win), (nr - win, nr)]
        full = [(rank * nr + a, rank * nr + b) for a, b in local]
        other = [((1 - rank) * nr + a, (1 - rank) * nr + b) for a, b in local]
    elif kind == "in":
        assert (q.k, q.n) == (K // world, N), (prefix, q.k, q.n)
        kr = K // world
        rows = (rank * kr, (rank + 1) * kr)
        local = [(0, win), (N - win, N)]
        full = local
    elif kind == "fused3":                       # [q | k | v] sections, rank r takes its half of each
        sec, sr = N // sections, N // sections // world
        assert (q.k, q.n) == (K, sections * sr), (prefix, q.k, q.n)
        local = [(s * sr + a, s * sr + a + 128) for s in range(sections) for a in (0, sr - 128)]
        full = [(s * sec + rank * sr + a, s * sec + rank * sr + a + 128) for s in range(sections) for a in (0, sr - 128)]
        other = [(s * sec + (1 - rank) * sr + a, s * sec + (1 - rank) * sr + a + 128)
                 for s in range(sections) for a in (0, sr - 128)]
    else:
        raise ValueError(kind)
    x = (torch.randn((3, q.k), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    y = q(x, f32=True).cpu().numpy()
    got = np.concatenate([y[:, a:b] for a, b in local], axis=1)
    ref = _ref(prefix, x, full, rows)
    rel = _rel(got, ref)
    assert rel < 3e-3, f"{prefix} rank {rank}: relative error {rel:.2e}"
    if kind in ("out", "fused3"):
        assert _rel(got, _ref(prefix, x, other, rows)) > 0.5, f"{prefix}: rank {rank} matches the other rank's columns"
    return rel


def _load(rank: int, layers: list[int], mtp: bool):
    from tensorfold.families.glm5_next.cuda import weights

    return weights.load(MODEL, rank=rank, device="cuda", mtp=mtp, world=2, layers=layers)


def _free():
    gc.collect()
    torch.cuda.empty_cache()


def _check_dsa(a, i: int, rank: int, errs: dict) -> None:
    from tensorfold.families.glm5_next.cuda.exl3_dense import Stack
    from tensorfold.families.glm5_next.cuda.qmm import B16

    s = f"{P}layers.{i}.self_attn."
    assert isinstance(a.proj, Stack) and len(a.proj.parts) == 2 and a.proj.n == 1536 + 512
    errs[f"{i}.q_a"] = _check(a.proj.parts[0], s + "q_a_proj", "rep", rank)
    errs[f"{i}.kv_a"] = _check(a.proj.parts[1], s + "kv_a_proj_with_mqa", "rep", rank)
    errs[f"{i}.q_b"] = _check(a.q_b, s + "q_b_proj", "out", rank)
    errs[f"{i}.o"] = _check(a.o, s + "o_proj", "in", rank)
    if a.kv_k is None:      # the latent cache (default): kv_b held once, split per head for absorb/expand (Mia's 0013)
        assert a.absorb.wk.shape == a.absorb.wv.shape == (32, 256, 512)
    else:                   # TF_GLM_LATENT=0: kv_b as plain key and value rows
        assert isinstance(a.kv_k, B16) and isinstance(a.kv_v, B16) and a.kv_k.weight.shape == (32 * 256, 512)
    assert isinstance(a.index.kw, B16) and isinstance(a.index.qb, B16)
    # Stack[q_a, kv_a] == the two parts side by side (one rows count with the tmp+copy path)
    x = torch.randn((4, 4096), device="cuda").to(torch.bfloat16)
    y = a.proj(x)
    assert torch.equal(y[:, :1536], a.proj.parts[0](x)) and torch.equal(y[:, 1536:], a.proj.parts[1](x))


def _check_shared(m, i: int, rank: int, errs: dict) -> None:
    from tensorfold.cuda.exl3.experts import Exl3RoutedExperts
    from tensorfold.cuda.exl3.linear import CODEBOOK_IDS
    from tensorfold.families.glm5_next.cuda.exl3_dense import Stack

    s = f"{P}layers.{i}.mlp.shared_experts."
    assert isinstance(m.shared.gu, Stack) and m.shared.width == 1024 and m.shared.gu.n == 2048
    errs[f"{i}.sh_gate"] = _check(m.shared.gu.parts[0], s + "gate_proj", "out", rank)
    errs[f"{i}.sh_up"] = _check(m.shared.gu.parts[1], s + "up_proj", "out", rank)
    errs[f"{i}.sh_down"] = _check(m.shared.down, s + "down_proj", "in", rank)
    ex = m.experts
    assert isinstance(ex, Exl3RoutedExperts) and ex.count == 288 and ex.width == 1024 and ex.dims == 4096
    assert ex.cb == CODEBOOK_IDS["mul1"], ex.cb
    # widths per expert from the trellis shapes (expert 0 gate, last expert down)
    for e, proj, k2 in ((0, "gate_proj", ex.gate_k2), (287, "down_proj", ex.down_k2)):
        tr = _get(f"{P}layers.{i}.mlp.experts.{e}.{proj}.trellis")
        assert int(k2[e]) == tr.shape[-1] // 8, (i, e, proj)


@pytest.mark.parametrize("rank", [0, 1])
def test_layers_0_3_4_and_head(rank):
    from tensorfold.families.glm5_next.cuda import qmm
    from tensorfold.families.glm5_next.cuda.exl3_dense import SCRATCH, Stack, X3
    from tensorfold.families.glm5_next.cuda.qmm import B16

    w = _load(rank, [0, 3, 4], mtp=False)
    errs: dict[str, float] = {}
    try:
        assert [l.index for l in w.layers] == [0, 3, 4] and w.mtp is None
        assert w.cfg.quant == "exl3" and w.cfg.codebook == "mul1"
        # KDA layers 0 and 4: fused qkv + plain tail, fused conv, o input split
        for i in (0, 4):
            k = w.layers[[0, 3, 4].index(i)].kda
            s = f"{P}layers.{i}.self_attn."
            assert isinstance(k.proj, Stack) and len(k.proj.parts) == 2 and k.proj.n == 12288 + 288
            qkv, tail = k.proj.parts
            errs[f"{i}.qkv"] = _check(qkv, s + "qkv_proj", "fused3", rank)
            assert isinstance(tail, B16) and tail.weight.shape == (288, 4096)
            b_rows = _get(s + "b_proj.weight")[rank * 32:(rank + 1) * 32]
            want = torch.cat([_get(s + "f_a_proj.weight"), _get(s + "g_a_proj.weight"), b_rows]).to(torch.bfloat16)
            assert torch.equal(tail.weight.cpu(), want)
            assert (k.fa_off, k.ga_off, k.b_off) == (12288, 12416, 12544)
            conv = _get(s + "conv1d.weight").view(3, 2, 4096, 4)[:, rank].reshape(12288, 4)
            assert k.conv.shape == (12288, 4) and torch.equal(k.conv.cpu(), conv)
            assert isinstance(k.fb, B16) and k.fb.weight.shape == (4096, 128)
            errs[f"{i}.o"] = _check(k.o, s + "o_proj", "in", rank)
            # the Stack output = [qkv | tail] side by side
            x = torch.randn((3, 4096), device="cuda").to(torch.bfloat16)
            y = k.proj(x)
            assert torch.equal(y[:, :12288], qkv(x))
            assert torch.equal(y[:, 12288:], qmm.matmul(x, tail))
        # dense MLP layer 0
        m = w.layers[0].mlp
        s = f"{P}layers.0.mlp."
        assert isinstance(m.gu, Stack) and m.gu.n == 12288 and m.width == 6144
        errs["0.gate"] = _check(m.gu.parts[0], s + "gate_proj", "out", rank)
        errs["0.up"] = _check(m.gu.parts[1], s + "up_proj", "out", rank)
        errs["0.down"] = _check(m.down, s + "down_proj", "in", rank)
        # DSA layer 3 + its shared expert and routed experts; MoE layer 4 too
        _check_dsa(w.layers[1].dsa, 3, rank, errs)
        _check_shared(w.layers[1].moe, 3, rank, errs)
        _check_shared(w.layers[2].moe, 4, rank, errs)
        # head: this rank's vocabulary half, no draft copy; MTP / DFlash2 read it through qmm.matmul
        assert isinstance(w.head, X3) and (w.head.k, w.head.n) == (4096, 77440) and not w.head.gemm
        assert w.vocab_offset == rank * 77440 and w.draft_head is None
        errs["head"] = _check(w.head, "lm_head", "out", rank, win=512)
        x = torch.randn((2, 4096), device="cuda").to(torch.bfloat16)
        dec = w.head(x)
        assert torch.equal(qmm.matmul(x, w.head, None), dec)              # dflash2._mm / mtp head (decode)
        assert torch.equal(w.head.prompt(x[1:]), dec[1:])               # prefill head: 1 row on decode pieces
        # memory accounting and the reserved scratch
        users = __import__("tensorfold.families.glm5_next.cuda.weights", fromlist=["x"]).exl3_users(w)
        assert sum(isinstance(u, X3) for u in users) and sum(isinstance(u, Stack) for u in users)
        assert SCRATCH.buffers["tmp:torch.bfloat16"].numel() >= 128 * 77440
        x3_bytes = sum(u.nbytes() for u in users)
        total = w.nbytes()
        assert total > x3_bytes + SCRATCH.nbytes() + w.embed.numel() * 2
        print(f"\nrank {rank}: nbytes {total / 2**30:.2f} GiB (EXL3 dense {x3_bytes / 2**20:.0f} MiB, scratch "
              f"{SCRATCH.nbytes() / 2**20:.0f} MiB); max rel err {max(errs.values()):.2e}")
        print(" ".join(f"{k}={v:.1e}" for k, v in errs.items()))
    finally:
        del w
        _free()


@pytest.mark.parametrize("rank", [0, 1])
def test_mtp_layer(rank):
    from tensorfold.families.glm5_next.cuda.exl3_dense import X3

    w = _load(rank, [], mtp=True)
    errs: dict[str, float] = {}
    try:
        assert w.layers == [] and w.mtp is not None
        assert isinstance(w.mtp.eh, X3)
        errs["45.eh"] = _check(w.mtp.eh, f"{P}layers.45.eh_proj", "rep", rank)
        _check_dsa(w.mtp.layer.dsa, 45, rank, errs)
        _check_shared(w.mtp.layer.moe, 45, rank, errs)
        print(f"\nrank {rank} MTP: " + " ".join(f"{k}={v:.1e}" for k, v in errs.items()))
    finally:
        del w
        _free()

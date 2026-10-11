"""RED-SNOW memory budget: the prompt buffer's BF16 split-K partials (b.sk), the EXL3
prompt GEMM workspace and the decode scratch as mla_geometry counts them, against the sizes the loader's objects ask
for; and the --tp 4 refusal. CPU only: the matrices are stand-ins of the real per-rank shapes (no weight data)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda import geometry

MODEL = Path(os.environ.get("TENSORFOLD_REDSNOW_MODEL", "/var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw"))
P = "model.language_model."
real = pytest.mark.skipif(not (MODEL / "model.safetensors.index.json").exists(), reason="RED-SNOW checkpoint not found")


def test_copies_match_the_originals():
    from tensorfold.families.glm5_next.cuda import exl3_dense, qmm

    assert geometry.EXL3_DENSE_ROWS == exl3_dense.ROWS
    assert geometry.EXL3_GEMM_MAX == exl3_dense.GEMM_MAX
    for n in (32, 64, 160, 288, 1024, 4096, 8192, 16384, 77440):
        for k in (64, 128, 512, 1536, 4096, 8192):
            assert geometry.b16_split_k(n, k) == qmm.b16_split_k(n, k), (n, k)


def test_dense_exl3_detection():
    q = lambda **kw: {"_quantization": {"quant_method": "exl3", **kw}}   # noqa: E731
    assert geometry.mla_dense_exl3(q(codebook="mul1"))
    assert geometry.mla_dense_exl3(q(codebook="MUL1"))
    assert not geometry.mla_dense_exl3(q(codebook="mcg"))
    assert not geometry.mla_dense_exl3(q())
    assert not geometry.mla_dense_exl3({"_quantization": {"bits": 4, "group_size": 64}})


# -- stand-ins built from the checkpoint's headers through split.py's rules (independent of geometry's config math)
def _headers() -> dict:
    from tensorfold.cuda.capacity import headers

    return headers(MODEL)


def _rank_shape(h: dict, name: str, world: int = 2) -> list[int]:
    from tensorfold.families.glm5_next.cuda.split import GEOMETRY_KIND, rule

    shape = list(h[name]["shape"])
    kind = GEOMETRY_KIND.get(rule(name), rule(name))
    if kind == "row":
        shape[0] //= world
    elif kind == "dim1":
        shape[1] //= world
    elif kind == "col":
        shape[-1] //= world
    return shape


def _x3(h: dict, prefix: str):
    from tensorfold.cuda.exl3.linear import Exl3Linear
    from tensorfold.families.glm5_next.cuda.exl3_dense import X3

    s = _rank_shape(h, prefix + ".trellis")
    e = torch.empty(0)
    return X3(Exl3Linear(e, e, e, None, s[2] / 16, "mul1", s[0] * 16, s[1] * 16))


def _b16(h: dict, *names: str):
    from tensorfold.families.glm5_next.cuda.qmm import B16

    shapes = [_rank_shape(h, n + ".weight") for n in names]
    return B16(torch.empty(0), sum(s[0] for s in shapes), shapes[0][1])


def _standin(h: dict, mtp: bool = True):
    """Rank 0 of TP2 as weights.load builds it: KDA layer 0 (dense MLP), DSA layer 3 (MoE), the MTP layer, the head."""

    from tensorfold.families.glm5_next.cuda.exl3_dense import Stack
    from tensorfold.families.glm5_next.cuda.qmm import B16
    from tensorfold.families.glm5_next.cuda.weights import DSAW, KDAW, IndexW, LayerW, MLPW, MoEW, MTPW

    e = torch.empty(0)

    def dsa(i: int) -> DSAW:
        p = f"{P}layers.{i}.self_attn."
        kv = _rank_shape(h, p + "kv_b_proj.weight")          # [HL * 512, 512]: key rows, value rows
        half = B16(e, kv[0] // 2, kv[1])
        ix = IndexW(_b16(h, p + "indexer.wk", p + "indexer.weights_proj"), _b16(h, p + "indexer.wq_b"), e, e, e, e)
        return DSAW(Stack([_x3(h, p + "q_a_proj"), _x3(h, p + "kv_a_proj_with_mqa")]), e, e, _x3(h, p + "q_b_proj"),
                    half, half, _x3(h, p + "o_proj"), 0, ix)

    def mlp(p: str) -> MLPW:
        return MLPW(Stack([_x3(h, p + "gate_proj"), _x3(h, p + "up_proj")]), _x3(h, p + "down_proj"), 0)

    p0 = f"{P}layers.0.self_attn."
    kda = KDAW(Stack([_x3(h, p0 + "qkv_proj"), _b16(h, p0 + "f_a_proj", p0 + "g_a_proj", p0 + "b_proj")]),
               _b16(h, p0 + "f_b_proj"), _b16(h, p0 + "g_b_proj"), e, e, e, e, _x3(h, p0 + "o_proj"), 0)
    l0 = LayerW(0, "kda", None, None, e, e, kda=kda, mlp=mlp(f"{P}layers.0.mlp."))
    l3 = LayerW(3, "dsa", None, None, e, e, dsa=dsa(3), moe=MoEW(e, e, None, mlp(f"{P}layers.3.mlp.shared_experts.")))
    m = None
    if mtp:
        lm = LayerW(45, "dsa", None, None, e, e, dsa=dsa(45), moe=MoEW(e, e, None, mlp(f"{P}layers.45.mlp.shared_experts.")))
        m = MTPW(e, e, _x3(h, f"{P}layers.45.eh_proj"), e, lm)
    return SimpleNamespace(layers=[l0, l3], mtp=m, head=_x3(h, "lm_head"), draft_head=None)


@real
@pytest.mark.parametrize("mtp", [True, False])
def test_geometry_equals_the_loaders_objects(mtp):
    from tensorfold.cuda.capacity import config
    from tensorfold.families.glm5_next.cuda import exl3_dense
    from tensorfold.families.glm5_next.cuda.forward import b16_partials
    from tensorfold.families.glm5_next.cuda.weights import exl3_users

    h = _headers()
    w = _standin(h, mtp)
    users = exl3_users(w)
    t = config(MODEL)
    got = geometry.mla_exl3_dense_scratch(t, 2, mtp=mtp)
    rows = geometry.PREFILL_ROWS
    assert got["partials"] == 4 * rows * b16_partials(w)
    pw, pxh = exl3_dense.prompt_sizes(users, rows)
    assert got["prompt"] == 2 * (pw + pxh) + geometry.HADAMARD_BYTES
    xh, z, tmp = exl3_dense.decode_sizes(users)
    assert got["decode"] == 2 * xh + 4 * z + 6 * tmp
    # the exact RED-SNOW figures (computed from the checkpoint headers)
    assert b16_partials(w) == 16384                       # indexer wq_b (4096 x 1536, 4 K slices) and kv_b halves
    assert got["partials"] == 128 * 2**20                 # was 8 * 2048 * 16384 * 4 = 1 GiB
    assert (pw, pxh) == (4096 * 12288, 2048 * 8192)       # qkv's W_q; DSA o_proj's K (the head is excluded)
    assert got["prompt"] == 134_250_496
    assert (xh, z, tmp) == (128 * 8192, 8 * 128 * 4096, 128 * 77440)
    assert got["decode"] == 78_348_288
    assert not w.head.gemm and w.head.n == 77440


@real
def test_mla_geometry_budgets_the_dense_workspaces_not_the_old_gib():
    from tensorfold.cuda.capacity import config

    t = config(MODEL)
    new = geometry.mla_geometry(t, 2, 8, latent=True, mtp=True).bytes_at(4096)
    mcg = dict(t, _quantization={**t["_quantization"], "codebook": "mcg"})
    old = geometry.mla_geometry(mcg, 2, 8, latent=True, mtp=True).bytes_at(4096)
    extra = sum(geometry.mla_exl3_dense_scratch(t, 2, mtp=True).values())
    assert old - new == 8 * geometry.PREFILL_ROWS * 16384 * 4 - extra
    assert extra == 128 * 2**20 + 134_250_496 + 78_348_288


@pytest.mark.parametrize("rows, slots, d, width, experts", [
    (geometry.PREFILL_ROWS, 9, 4096, 1024, 288),   # RED-SNOW's prompt chunk on a rank (the prompt kernel fits)
    (64, 9, 4096, 1024, 288),                      # its decode window
    (3, 9, 4096, 1024, 288),                       # a window below PROMPT_ROWS: z for every pair
    (256, 9, 4096, 1088, 288),                     # a width the prompt kernel does not take: z for every pair
    (40, 3, 512, 256, 16),                         # fewer experts than pairs
])
def test_exl3_expert_scratch_equals_the_universal_kernels_scratch(rows, slots, d, width, experts):
    """geometry.exl3_expert_scratch == the bytes of the Scratch forward.Buffers builds (Mia's 0028, re-derived for
    tensorfold/cuda/exl3/experts.py)."""
    from tensorfold.cuda.exl3.experts import Scratch

    s = Scratch(SimpleNamespace(dims=d, width=width, count=experts), rows, slots, device="cpu")
    got = sum(v.numel() * v.element_size() for v in vars(s).values() if isinstance(v, torch.Tensor))
    assert geometry.exl3_expert_scratch(rows, slots, d, width, experts) == got


def test_onepass_drops_exactly_the_sparse_partials():
    """mla_geometry(onepass=True) (TF_GLM_SPARSE_ONEPASS) budgets a prompt chunk's selection only: smaller by the
    sparse partials ((topk + 515) // 512 chunks x PREFILL_ROWS x heads x (kv_lora + 2) fp32), latent cache only."""
    t = {"num_attention_heads": 64, "kv_lora_rank": 512, "index_topk": 2048, "qk_nope_head_dim": 192,
         "qk_rope_head_dim": 64}
    partials = ((2048 + 515) // 512) * geometry.PREFILL_ROWS * 32 * 514 * 4
    for cap in (4096, 262144):
        off = geometry.mla_chunk_scratch(t, 2, cap, latent=True)
        assert off - geometry.mla_chunk_scratch(t, 2, cap, latent=True, onepass=True) == partials
        assert geometry.mla_chunk_scratch(t, 2, cap, latent=False, onepass=True) == geometry.mla_chunk_scratch(
            t, 2, cap, latent=False)


@real
def test_redsnow_geometry_onepass_and_expert_scratch():
    """RED-SNOW's startup geometry: onepass lowers bytes_at by exactly the sparse partials at every capacity."""
    from tensorfold.cuda.capacity import config

    t = config(MODEL)
    for mtp in (False, True):
        off = geometry.mla_geometry(t, 2, 8, latent=True, mtp=mtp, kv="fp8")
        on = geometry.mla_geometry(t, 2, 8, latent=True, mtp=mtp, kv="fp8", onepass=True)
        for cap in (4096, 65536, 262144):
            assert off.bytes_at(cap) - on.bytes_at(cap) == 5 * geometry.PREFILL_ROWS * 32 * 514 * 4


def _fake_ckpt(tmp_path: Path, head_cols: int) -> Path:
    """A one-shard EXL3 checkpoint header with an lm_head of ``head_cols`` columns (zero-filled data)."""

    import numpy as np
    from safetensors.numpy import save_file

    d = tmp_path / f"m{head_cols}"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "glm5_next", "quantization_config": {
        "quant_method": "exl3", "codebook": "mul1", "bits": 5}}))
    tensors = {"lm_head.trellis": np.zeros((8, head_cols // 16, 128), np.int16),
               "lm_head.suh": np.zeros((128,), np.float16), "lm_head.svh": np.zeros((head_cols,), np.float16),
               "lm_head.mul1": np.zeros((), np.int32),
               P + "layers.0.self_attn.o_proj.trellis": np.zeros((32, 8, 64), np.int16)}
    save_file(tensors, str(d / "model-00001-of-00001.safetensors"))
    (d / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}))
    return d


def test_tp4_refused_for_an_unaligned_head(tmp_path):
    from tensorfold.families.glm5_next import check_ranks

    bad = _fake_ckpt(tmp_path, 1280)          # 1280 / 4 = 320 columns a rank: not a multiple of 128
    check_ranks(bad, 2)                        # 640 a rank: allowed
    with pytest.raises(ValueError, match=r"--tp 4: .*lm_head.*--tp 2"):
        check_ranks(bad, 4)
    check_ranks(_fake_ckpt(tmp_path, 2048), 4)  # 512 a rank: allowed


def test_tp4_refused_before_the_engine_starts(tmp_path, monkeypatch):
    import tensorfold.families.glm5_next as fam

    bad = _fake_ckpt(tmp_path, 1280)
    with pytest.raises(ValueError, match="--tp 4"):
        fam.cuda_engine(bad, tp=4, rank=0, master="127.0.0.1")   # raises before importing / building GlmEngine


def test_tp_check_skips_rank_folders_and_mlx(tmp_path):
    from tensorfold.families.glm5_next import check_ranks

    bad = _fake_ckpt(tmp_path, 1280)
    (bad / "model-00001-of-00001.rank0.safetensors").write_bytes(b"")
    check_ranks(bad, 4)                        # split.py's pieces: already cut, nothing to check
    mlx = tmp_path / "mlx"
    mlx.mkdir()
    (mlx / "config.json").write_text(json.dumps({"model_type": "glm5_next", "quantization": {"bits": 4, "group_size": 64}}))
    check_ranks(mlx, 4)


@real
def test_redsnow_tp4_refused_tp2_allowed():
    from tensorfold.families.glm5_next import check_ranks

    check_ranks(MODEL, 2)
    with pytest.raises(ValueError, match=r"--tp 4: 2 EXL3 tensor\(s\) .*lm_head.*lm_head.svh row \[154880\]"):
        check_ranks(MODEL, 4)

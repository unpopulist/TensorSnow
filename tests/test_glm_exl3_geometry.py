"""CPU tests: the GLM startup weight estimate (``geometry.split_weights``) on RED-SNOW's mul1 EXL3 split."""

import os
from pathlib import Path

import numpy as np
import pytest

from tensorfold.cuda import geometry
from tensorfold.families.glm5_next.cuda import split

L = "model.language_model.layers."
MODEL = Path(os.environ.get("TENSORFOLD_REDSNOW_MODEL", "/var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw"))


def info(dtype, shape, pre_split=False):
    return {"dtype": dtype, "shape": list(shape), "split": pre_split}


def pieces(name, i):
    """Rank 0 and rank 1 bytes of split.py's actual pieces (stride-0 stand-in bytes, no checkpoint read)."""

    kind = split.rule(name)
    if kind == "drop":
        return [0, 0]
    shape, it = i["shape"], split.DTYPE_BYTES[i["dtype"]]
    n = int(np.prod(shape)) * it
    raw = np.lib.stride_tricks.as_strided(np.zeros(1, np.uint8), (n,), (0,))
    return [split.split_bytes(raw, shape, it, kind, r, 2)[0].size for r in (0, 1)]


# RED-SNOW header shapes (KDA layer 4, DSA layer 3, dense layer 0, expert, head)
CASES = {
    L + "4.self_attn.qkv_proj.trellis": info("I16", [256, 1536, 96]),
    L + "4.self_attn.qkv_proj.svh": info("F16", [24576]),
    L + "4.self_attn.qkv_proj.suh": info("F16", [4096]),
    L + "4.self_attn.qkv_proj.mul1": info("I32", []),
    L + "4.self_attn.conv1d.weight": info("BF16", [24576, 1, 4]),
    L + "4.self_attn.o_proj.trellis": info("I16", [512, 256, 96]),
    L + "3.self_attn.q_b_proj.trellis": info("I16", [128, 1024, 96]),
    L + "3.self_attn.q_a_proj.trellis": info("I16", [256, 128, 96]),
    L + "0.mlp.down_proj.suh": info("F16", [12288]),
    L + "4.mlp.experts.17.gate_proj.trellis": info("I16", [256, 128, 80]),
    L + "4.mlp.experts.17.gate_proj.mul1": info("I32", []),
    "lm_head.trellis": info("I16", [256, 9680, 128]),
    "lm_head.svh": info("F16", [154880]),
    "lm_head.suh": info("F16", [4096]),
    "lm_head.mul1": info("I32", []),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_exl3_estimate_equals_split_piece(name):
    """geometry_rule (what the engine passes) and rule both give exactly the bytes of each rank's split piece."""

    i = CASES[name]
    want = pieces(name, i)
    assert want[0] == want[1]
    for rule in (split.geometry_rule, split.rule):
        assert geometry.split_weights(rule)(name, i) == (want[0], 0)


def test_fused_kinds_and_scalar_head():
    t = geometry.split_weights(split.geometry_rule)
    assert t(L + "4.self_attn.qkv_proj.trellis", CASES[L + "4.self_attn.qkv_proj.trellis"])[0] == 256 * 768 * 96 * 2
    assert t(L + "4.self_attn.conv1d.weight", CASES[L + "4.self_attn.conv1d.weight"])[0] == 12288 * 4 * 2
    assert t("lm_head.mul1", CASES["lm_head.mul1"]) == (4, 0)                    # 0-D: no IndexError
    assert t("lm_head.trellis", CASES["lm_head.trellis"])[0] == 256 * 4840 * 128 * 2   # vocab half only, not a quarter
    assert t("lm_head.svh", CASES["lm_head.svh"])[0] == 77440 * 2
    assert t("lm_head.suh", CASES["lm_head.suh"])[0] == 4096 * 2                  # copied whole on each rank


def test_pre_split_exl3_head_is_not_halved_again():
    t = geometry.split_weights(split.geometry_rule)
    assert t("lm_head.trellis", info("I16", [256, 4840, 128], True))[0] == 256 * 4840 * 128 * 2
    assert t("lm_head.svh", info("F16", [77440], True))[0] == 77440 * 2
    assert t("lm_head.mul1", info("I32", [], True)) == (4, 0)


def test_mlx_and_bf16_heads_unchanged():
    t = geometry.split_weights(split.geometry_rule)
    # MLX 4-bit head: copied by the rule, the rank's vocabulary rows kept at load
    assert t("lm_head.weight", info("U32", [154880, 768])) == (77440 * 768 * 4, 0)
    assert t("lm_head.scales", info("BF16", [154880, 96])) == (77440 * 96 * 2, 0)
    assert t("lm_head.weight", info("U32", [154880, 768], True)) == (77440 * 768 * 4, 0)
    # BF16 head: half the vocabulary plus the 4-bit draft head
    assert t("lm_head.weight", info("BF16", [154880, 6144])) == (77440 * 6144 * 2 + 77440 * 6144 * 9 // 16, 0)


def test_scalar_with_a_split_kind_raises_cleanly():
    with pytest.raises(ValueError, match="no split axis"):
        geometry.split_weights(lambda name: "row")("x.mul1", info("I32", []))
    with pytest.raises(ValueError, match="no split axis"):
        geometry.split_weights(lambda name: "diagonal")("x.weight", info("BF16", [4, 4]))


@pytest.mark.skipif(not (MODEL / "model.safetensors.index.json").exists(), reason="RED-SNOW checkpoint not present")
def test_whole_redsnow_checkpoint_estimate_matches_split_pieces():
    """Every tensor header: the engine's per-rank estimate equals the sum of each rank's split.py pieces."""

    from tensorfold.cuda.capacity import estimate_weights, headers

    est = geometry.split_weights(split.geometry_rule, 2)
    total = [0, 0]
    for name, i in headers(MODEL).items():
        got = est(name, i)[0]
        want = pieces(name, i)
        assert got == want[0] == want[1], name
        total = [total[0] + want[0], total[1] + want[1]]
    assert total[0] == total[1] == estimate_weights(MODEL, est).resident == 100_261_183_480

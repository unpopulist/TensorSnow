"""RED-SNOW (GLM-5.3, EXL3 mul1) split over two ranks, checked on the real checkpoint's bytes on the host (no GPU).

Each rank's piece of a KDA layer (fused qkv_proj / conv1d), a DSA layer, a dense-MLP layer, the shared and one routed
expert, the MTP eh_proj and lm_head must be the expected slice of the full tensor, byte for byte, with every split
boundary on a 128-element (Hadamard block) multiple, and the two pieces must put the full tensor back together.
Reads only safetensors headers and the tensors named here; skipped where the checkpoint is not on this machine."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.glm5_next.cuda import split  # noqa: E402

CKPT = Path(os.environ.get("TF_REDSNOW_CKPT", "/var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw"))
pytestmark = pytest.mark.skipif(not (CKPT / "model.safetensors.index.json").exists(),
                                reason=f"needs the RED-SNOW EXL3 checkpoint at {CKPT}")
L = "model.language_model.layers."
NP = {"I16": np.int16, "I32": np.int32, "F16": np.float16, "BF16": np.uint16, "F32": np.float32}
LIN = (".trellis", ".suh", ".svh", ".mul1")


def _names() -> list[str]:
    def lin(p):
        return [p + s for s in LIN]

    kda = lambda i: [n for x in ("qkv_proj", "o_proj") for n in lin(f"{L}{i}.self_attn.{x}")] + [  # noqa: E731
        f"{L}{i}.self_attn.{x}" for x in ("conv1d.weight", "A_log", "dt_bias", "b_proj.weight", "f_a_proj.weight",
                                          "f_b_proj.weight", "g_a_proj.weight", "g_b_proj.weight")]
    dsa = [n for x in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj") for n in lin(f"{L}3.self_attn.{x}")]
    dsa += [f"{L}3.self_attn.kv_b_proj.weight"]
    dense = [n for x in ("gate_proj", "up_proj", "down_proj") for n in lin(f"{L}0.mlp.{x}")]
    shared = [n for x in ("gate_proj", "up_proj", "down_proj") for n in lin(f"{L}4.mlp.shared_experts.{x}")]
    routed = [n for x in ("gate_proj", "up_proj", "down_proj") for n in lin(f"{L}4.mlp.experts.17.{x}")]
    return kda(4) + kda(0) + dsa + dense + shared + routed + lin(f"{L}45.eh_proj") + lin("lm_head")


NAMES = _names()


def expected(name: str):
    """The spec, written out independently of split.rule: (axis, fused3) or None for a copy on both ranks."""

    if name.endswith(".mul1") or re.search(r"\.(q_a_proj|kv_a_proj_with_mqa|eh_proj)\.", name):
        return None
    part = name.rsplit(".", 1)[1]
    if ".qkv_proj." in name:
        return {"trellis": (1, True), "svh": (0, True), "suh": None}[part]
    if name.endswith(".conv1d.weight"):
        return (0, True)
    if re.search(r"(gate|up)_proj\.|q_b_proj\.|^lm_head\.", name) and part in ("trellis", "suh", "svh"):
        return {"trellis": (1, False), "svh": (0, False), "suh": None}[part]
    if re.search(r"(down|o)_proj\.", name) and part in ("trellis", "suh", "svh"):
        return {"trellis": (0, False), "suh": (0, False), "svh": None}[part]
    if re.search(r"\.(A_log|dt_bias|b_proj\.weight|f_b_proj\.weight|g_b_proj\.weight|kv_b_proj\.weight)$", name):
        return (0, False)
    return None


@pytest.fixture(scope="module")
def index() -> dict:
    return json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]


@pytest.fixture(scope="module")
def headers(index) -> dict:
    out = {}
    for file in sorted(set(index.values())):
        header, base = split.read_header(CKPT / file)
        header.pop("__metadata__", None)
        out.update({k: (file, base, v) for k, v in header.items()})
    return out


def _full(headers, name) -> np.ndarray:
    file, base, info = headers[name]
    a, b = info["data_offsets"]
    mm = np.memmap(CKPT / file, dtype=np.uint8, mode="r", offset=base + a, shape=(b - a,))
    return mm.view(NP[info["dtype"]]).reshape(info["shape"])


def _piece(full: np.ndarray, spec, rank: int, ranks: int = 2) -> np.ndarray:
    if spec is None:
        return full
    axis, fused = spec
    n = full.shape[axis]
    if not fused:
        c = n // ranks
        return np.take(full, range(rank * c, (rank + 1) * c), axis=axis)
    c = n // (3 * ranks)
    idx = [t * n // 3 + rank * c + j for t in range(3) for j in range(c)]
    return np.take(full, idx, axis=axis)


def _bounds(n: int, spec, ranks: int = 2) -> list[int]:
    """Every element index where one rank's (or one of q/k/v's) piece meets another's on the split axis."""

    axis, fused = spec
    parts = 3 * ranks if fused else ranks
    return [i * n // parts for i in range(1, parts)]


def _bytes(t) -> np.ndarray:
    return t.contiguous().reshape(-1).view(torch.uint8).numpy()


def test_every_index_name_has_a_rule_and_aligned_boundaries(index, headers):
    kinds = {}
    for name in index:
        kind = split.rule(name)                                 # raises on an unmatched or ambiguous name
        kinds[name] = kind
        if kind != "drop":
            split.check_exl3(name, headers[name][2]["shape"], kind, 2)
    assert not [n for n, k in kinds.items() if k == "drop" and not n.startswith("model.visual.")]
    assert {k for n, k in kinds.items() if n.endswith(".mul1")} == {"rep"}
    assert {kinds[n] for n in index if ".qkv_proj." in n} == {"fused3_dim1", "fused3_row", "rep"}
    for name in NAMES:                                          # the rules agree with the spec written out above
        spec = expected(name)
        want = "rep" if spec is None else {(0, False): "row", (1, False): "dim1", (0, True): "fused3_row",
                                           (1, True): "fused3_dim1"}[spec]
        assert kinds[name] == want, name


def test_rank_pieces_are_exact_aligned_slices_that_reassemble(headers):
    readers = [split.RankReader(CKPT, r) for r in (0, 1)]
    try:
        for name in NAMES:
            full = _full(headers, name)
            spec = expected(name)
            pieces = [readers[r].get(name) for r in (0, 1)]
            for r, got in enumerate(pieces):
                want = _piece(full, spec, r)
                assert list(got.shape) == list(want.shape), name
                assert np.array_equal(_bytes(got), np.ascontiguousarray(want).view(np.uint8).reshape(-1)), (name, r)
            if spec is None:
                continue
            axis, fused = spec
            tile = 16 if name.endswith(".trellis") else 1         # trellis axes count 16-element tiles
            if name.endswith(LIN) or name.endswith(".conv1d.weight"):  # EXL3 (and the conv fused like qkv)
                assert all(b * tile % 128 == 0 for b in _bounds(full.shape[axis], spec)), name
            a, b = (_bytes(p).view(full.dtype).reshape(p.shape) for p in pieces)
            if fused:
                k = full.shape[axis] // 6
                thirds = [np.take(x, range(t * k, (t + 1) * k), axis=axis) for t in range(3) for x in (a, b)]
                whole = np.concatenate(thirds, axis=axis)
            else:
                whole = np.concatenate([a, b], axis=axis)
            assert np.array_equal(whole.view(np.uint8), np.ascontiguousarray(full).view(np.uint8)), name
    finally:
        for rd in readers:
            rd.close()


def test_rank_piece_shapes(headers):
    rd = split.RankReader(CKPT, 1)
    try:
        q = L + "4.self_attn.qkv_proj."
        w = headers[q + "trellis"][2]["shape"][2]
        assert list(rd.get(q + "trellis").shape) == [256, 768, w]           # K 4096 x N' 12288 = [q_r | k_r | v_r]
        assert list(rd.get(q + "svh").shape) == [12288] and list(rd.get(q + "suh").shape) == [4096]
        assert list(rd.get(L + "4.self_attn.conv1d.weight").shape) == [12288, 1, 4]
        assert list(rd.get(L + "4.self_attn.o_proj.trellis").shape)[:2] == [256, 256]   # K' 4096 of 8192
        assert list(rd.get(L + "3.self_attn.o_proj.trellis").shape)[:2] == [512, 256]   # K' 8192 of 16384
        assert list(rd.get(L + "3.self_attn.q_b_proj.trellis").shape)[:2] == [96, 512]
        assert list(rd.get(L + "3.self_attn.q_a_proj.trellis").shape)[:2] == [256, 96]  # copied
        assert list(rd.get(L + "0.mlp.gate_proj.trellis").shape)[:2] == [256, 384]
        assert list(rd.get(L + "0.mlp.down_proj.suh").shape) == [6144]
        assert list(rd.get("lm_head.trellis").shape)[:2] == [256, 4840]                 # vocab 77440 a rank
        assert list(rd.get("lm_head.svh").shape) == [77440] and list(rd.get("lm_head.mul1").shape) == []
    finally:
        rd.close()


def test_device_cut_matches_host_cut_for_fused_kinds(headers):
    """split_device (the prefetch path, here on host torch tensors) gives the same bytes as split_bytes."""

    for name in (L + "4.self_attn.qkv_proj.trellis", L + "4.self_attn.qkv_proj.svh", L + "4.self_attn.conv1d.weight",
                 L + "0.mlp.down_proj.trellis", L + "4.mlp.shared_experts.up_proj.trellis"):
        info = headers[name][2]
        raw = np.ascontiguousarray(_full(headers, name)).view(np.uint8).reshape(-1)
        kind, itemsize = split.rule(name), split.DTYPE_BYTES[info["dtype"]]
        for r in (0, 1):
            host, hs = split.split_bytes(raw, info["shape"], itemsize, kind, r)
            dev, ds = split.split_device(torch.from_numpy(raw.copy()), info["shape"], itemsize, kind, r)
            assert hs == ds and np.array_equal(host, dev.numpy()), (name, r)


def test_rank_folders_hold_the_same_pieces(headers, tmp_path):
    """A small checkpoint of real tensors (lm_head left out for size) split into rank folders by split.main reads back
    the same pieces as the full checkpoint read in place."""

    names = [n for n in NAMES if not n.startswith("lm_head.")]
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    file = "model-00001-of-00001.safetensors"
    tensors = []
    for n in names:
        info = headers[n][2]
        tensors.append((n, info["dtype"], info["shape"], np.ascontiguousarray(_full(headers, n)).view(np.uint8).reshape(-1)))
    split.write(str(ckpt / file), tensors, None)
    (ckpt / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(names, file)}))
    for r in (0, 1):
        out = tmp_path / f"rank{r}"
        split.main([str(ckpt), "--rank", str(r), str(out)])
        full, folder = split.RankReader(ckpt, r), split.RankReader(out, r)
        assert folder.split and not full.split
        folder.prefetch(names)
        for n in names:
            a, b = full.get(n), folder.get(n)
            assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(_bytes(a), _bytes(b)), (n, r)
        full.close()
        folder.close()

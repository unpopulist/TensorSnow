"""GLM-5.3's EXL3 dense matrices (``glm5_next/cuda/exl3_dense.py``) on real RED-SNOW tensors: the X3 wrapper against
``Exl3Linear`` itself, decode row invariance, prompt chunk invariance, the GEMM_MAX gate, Stacks of EXL3 and plain
parts against concatenated outputs, ``forward.mm`` / ``qmm.matmul`` dispatch, and CUDA-graph replay.

Loads single tensors of layers 0 and 3 (a few hundred MB of GPU memory). Model folder: TENSORFOLD_REDSNOW_MODEL."""

from __future__ import annotations

import json
import os
import types
from pathlib import Path

import pytest
import torch

MODEL = Path(os.environ.get("TENSORFOLD_REDSNOW_MODEL", "/var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw"))
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
if not (MODEL / "model.safetensors.index.json").exists():
    pytest.skip(f"no RED-SNOW checkpoint at {MODEL}", allow_module_level=True)

from safetensors import safe_open  # noqa: E402

from tensorfold.cuda.exl3.linear import Exl3Linear  # noqa: E402
from tensorfold.families.glm5_next.cuda import exl3_dense, forward, qmm  # noqa: E402

P = "model.language_model.layers."
ROWS = [1, 2, 3, 4, 5, 6, 7, 8, 16, 128]
_MAP = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]


def tensor(name: str) -> torch.Tensor:
    with safe_open(str(MODEL / _MAP[name]), framework="pt") as f:
        return f.get_tensor(name)


def x3_of(prefix: str, cols: list[tuple[int, int]] | None = None) -> exl3_dense.X3:
    """An X3 for one EXL3 group; ``cols`` keeps these output-column ranges (multiples of 128), as a TP rank piece."""

    tr, suh, svh = tensor(prefix + ".trellis"), tensor(prefix + ".suh"), tensor(prefix + ".svh")
    assert tensor(prefix + ".mul1").numel() == 1
    if cols:
        tr = torch.cat([tr[:, a // 16:b // 16] for a, b in cols], dim=1)
        svh = torch.cat([svh[a:b] for a, b in cols])
    return exl3_dense.x3(tr, suh, svh, "mul1")


@pytest.fixture(scope="module")
def layers() -> dict:
    a = P + "0.self_attn."
    got = {
        # rank 0's [q_r | k_r | v_r] of the fused qkv (q, k, v each 8192 wide): 4096 x 12288, 8-bit, plan (1, 8)
        "qkv": x3_of(a + "qkv_proj", [(0, 4096), (8192, 12288), (16384, 20480)]),
        "o": x3_of(a + "o_proj"),                                         # 8192 x 4096, 6-bit, split-K
        "q_a": x3_of(P + "3.self_attn.q_a_proj"),                         # 4096 x 1536, 7-bit
        "kv_a": x3_of(P + "3.self_attn.kv_a_proj_with_mqa"),              # 4096 x 512, 8-bit
        "f_a": qmm.make_b16(tensor(a + "f_a_proj.weight").cuda()),
        "g_a": qmm.make_b16(tensor(a + "g_a_proj.weight").cuda()),
        "b": qmm.make_b16(tensor(a + "b_proj.weight")[:32].cuda()),       # rank 0's b rows
    }
    assert (got["qkv"].k, got["qkv"].n) == (4096, 12288)
    return got


def inputs(rows: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((rows, k), generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)


@pytest.mark.parametrize("name", ["qkv", "o", "q_a", "kv_a"])
def test_wrapper_matches_exl3linear(layers, name):
    q = layers[name]
    x = inputs(128, q.k, 1)
    for r in ROWS:
        assert torch.equal(q(x[:r]), Exl3Linear.__call__(q.lin, x[:r])), r
        assert torch.equal(q(x[:r], f32=True), Exl3Linear.__call__(q.lin, x[:r], out_dtype=torch.float32)), r


@pytest.mark.parametrize("name", ["qkv", "o", "q_a", "kv_a"])
@pytest.mark.parametrize("f32", [False, True])
def test_decode_row_invariant(layers, name, f32):
    q = layers[name]
    x = inputs(200, q.k, 2)
    full = q(x[:128], f32=f32)
    for r in ROWS:
        assert torch.equal(q(x[:r], f32=f32), full[:r]), r
    for r in range(8):
        assert torch.equal(q(x[r:r + 1], f32=f32), full[r:r + 1]), r
    y200 = q(x, f32=f32)                                   # two decode pieces (128 + 72)
    assert torch.equal(y200[:128], full)
    assert torch.equal(y200[128:], q(x[128:], f32=f32))


@pytest.mark.parametrize("name", ["qkv", "o", "q_a", "kv_a"])
def test_prompt_chunk_invariant(layers, name):
    q = layers[name]
    assert q.gemm
    x = inputs(300, q.k, 3)
    full = q.prompt(x)
    assert torch.equal(q.prompt(x[:37]), full[:37])
    assert torch.equal(q.prompt(x[:128]), full[:128])
    assert torch.equal(q.prompt(x[128:]), full[128:])
    assert torch.equal(q.prompt(x[:5]), full[:5])
    dec = q(x).float()
    rel = ((full.float() - dec).norm() / dec.norm()).item()
    assert rel < 2e-3, rel                                 # different arithmetic from decode, same matrix


def test_big_matrix_takes_decode_pieces(layers, monkeypatch):
    q = layers["qkv"]
    x = inputs(200, q.k, 4)
    want = q(x)
    monkeypatch.setattr(exl3_dense, "GEMM_MAX", q.k * q.n - 1)   # qkv now counts as "the vocabulary head"

    def boom(*a, **k):
        raise AssertionError("prompt GEMM used above GEMM_MAX")

    monkeypatch.setattr(exl3_dense.x3prefill, "matmul", boom)
    assert not q.gemm
    assert torch.equal(q.prompt(x), want)
    out = torch.empty((200, q.n + 256), dtype=torch.bfloat16, device="cuda")
    q.prompt(x, out[:, 128:128 + q.n])                     # row-strided out through tmp pieces
    assert torch.equal(out[:, 128:128 + q.n], want)


def _concat(parts, x, prompt):
    outs = []
    for p in parts:
        if isinstance(p, exl3_dense.X3):
            outs.append(p.prompt(x) if prompt else p(x))
        else:
            outs.append(qmm.matmul(x, p))
    return torch.cat(outs, dim=1)


@pytest.mark.parametrize("names", [["qkv", "f_a", "g_a", "b"], ["q_a", "kv_a"]])
def test_stack_equals_concat(layers, names):
    parts = [layers[n] for n in names]
    s = exl3_dense.Stack(parts)
    assert s.n == sum(p.n for p in parts)
    x = inputs(300, s.k, 5)
    for r in ROWS + [200]:
        assert torch.equal(s(x[:r]), _concat(parts, x[:r], False)), r
    for r in (1, 5, 37, 300):
        assert torch.equal(s.prompt(x[:r]), _concat(parts, x[:r], True)), r
    # into a column slice of a wider buffer (row stride != n): decode and prompt
    wide = torch.zeros((8, s.n + 384), dtype=torch.bfloat16, device="cuda")
    s(x[:8], wide[:, 256:256 + s.n])
    assert torch.equal(wide[:, 256:256 + s.n], _concat(parts, x[:8], False))
    assert not wide[:, :256].any() and not wide[:, 256 + s.n:].any()
    wide = torch.zeros((37, s.n + 384), dtype=torch.bfloat16, device="cuda")
    s.prompt(x[:37], wide[:, 128:128 + s.n])
    assert torch.equal(wide[:, 128:128 + s.n], _concat(parts, x[:37], True))


def test_dispatch(layers, monkeypatch):
    calls = []
    real = exl3_dense.x3prefill.matmul

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(exl3_dense.x3prefill, "matmul", spy)
    sk = torch.empty((8 * 300 * 1536,), dtype=torch.float32, device="cuda")
    stack = exl3_dense.Stack([layers["qkv"], layers["f_a"], layers["g_a"], layers["b"]])
    for q in (layers["qkv"], stack):
        x = inputs(37, q.k, 6)
        out = torch.empty((37, q.n), dtype=torch.bfloat16, device="cuda")
        calls.clear()
        forward.mm(types.SimpleNamespace(prefill=True, sk=sk), x, q, None, out)
        assert calls and torch.equal(out, q.prompt(x))
        calls.clear()
        for r in (1, 4, 8):
            out = torch.empty((r, q.n), dtype=torch.bfloat16, device="cuda")
            forward.mm(types.SimpleNamespace(prefill=False, sk=sk), x[:r], q, None, out)
            assert torch.equal(out, q(x[:r]))
            assert torch.equal(qmm.matmul(x[:r], q), q(x[:r]))          # dflash2's direct call, no buffers
        assert not calls
    # fp32 partials (out_proj) on both paths
    o = layers["o"]
    x = inputs(37, o.k, 7)
    part = torch.empty((37, o.n), dtype=torch.float32, device="cuda")
    forward.mm(types.SimpleNamespace(prefill=True, sk=sk), x, o, None, part, f32=True)
    assert torch.equal(part, o.prompt(x, f32=True))
    forward.mm(types.SimpleNamespace(prefill=False, sk=sk), x[:4], o, None, part[:4], f32=True)
    assert torch.equal(part[:4], Exl3Linear.__call__(o.lin, x[:4], out_dtype=torch.float32))


def test_graph_replay(layers):
    s = exl3_dense.Stack([layers["qkv"], layers["f_a"], layers["g_a"], layers["b"]])
    o = layers["o"]
    exl3_dense.SCRATCH.reserve([s, o], "cuda")
    sk = torch.empty((8 * 8 * 4096,), dtype=torch.float32, device="cuda")
    x = inputs(4, s.k, 8)
    xo = inputs(4, o.k, 9)
    y = torch.empty((4, s.n), dtype=torch.bfloat16, device="cuda")
    yo = torch.empty((4, o.n), dtype=torch.float32, device="cuda")

    def step():
        qmm.matmul(x, s, out=y, part=sk)
        qmm.matmul(xo, o, out=yo, f32=True)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    for seed in range(3):
        x.copy_(inputs(4, s.k, 100 + seed))
        xo.copy_(inputs(4, o.k, 200 + seed))
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(y, _concat(s.parts, x, False))
        assert torch.equal(yo, o(xo, f32=True))

"""RED-SNOW buffers as forward.Buffers allocates them: b.sk sized from the plain matrices,
the EXL3 prompt GEMM's workspace reserved at startup and not regrown by a prompt chunk.

Loads rank 0 of layer 0 (KDA + dense MLP), the embedding and the head of the real checkpoint (about 2 GB), so it runs
only with TENSORFOLD_REDSNOW_LOADER=1 (TENSORFOLD_REDSNOW_MODEL overrides the path)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

MODEL = Path(os.environ.get("TENSORFOLD_REDSNOW_MODEL", "/var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw"))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not (MODEL / "model.safetensors.index.json").exists(), reason="RED-SNOW checkpoint not found"),
    pytest.mark.skipif(os.environ.get("TENSORFOLD_REDSNOW_LOADER") != "1",
                       reason="loads about 2 GB of RED-SNOW: set TENSORFOLD_REDSNOW_LOADER=1"),
]


def test_buffers_sizes_and_prompt_workspace():
    from tensorfold.families.glm5_next.cuda import exl3_dense, forward, qmm
    from tensorfold.families.glm5_next.cuda.weights import exl3_users, load

    exl3_dense.SCRATCH.__init__()                 # a fresh shared scratch: sizes below are this layer's alone
    w = load(MODEL, rank=0, mtp=False, world=2, layers=[0])
    users = exl3_users(w)
    assert users
    # layer 0's plain matrices: the KDA tail [f_a | g_a | b_r] (288 x 4096, 8 K slices) splits K; f_b / g_b do not
    assert forward.b16_partials(w) == 8 * 288
    sc = exl3_dense.SCRATCH
    dec = forward.Buffers(w, 8)
    assert dec.sk.numel() == 8 * 8 * 288
    assert sc.prefill.w is None                    # decode buffers reserve no prompt workspace
    rows = 512
    pre = forward.Buffers(w, rows, prefill=True)
    assert pre.sk.numel() == rows * 8 * 288        # was 8 * rows * 16384
    # the routed experts' universal-kernel scratch exactly as geometry.exl3_expert_scratch counts it (Mia's 0028)
    from tensorfold.cuda.geometry import exl3_expert_scratch

    c = w.cfg
    from tensorfold.families.glm5_next.cuda.exl3_generic import DECODE_PROMPT_ROWS

    for b, pr in ((dec, DECODE_PROMPT_ROWS), (pre, None)):
        got = sum(v.numel() * v.element_size() for v in vars(b.exl3).values() if isinstance(v, torch.Tensor))
        assert got == exl3_expert_scratch(b.rows, c.top_k + 1, c.hidden, c.moe_width // 2, c.experts, pr), b.rows
    assert pre.att_starts.tolist() == list(range(0, rows, forward.PROMPT_ATT_ROWS))
    assert dec.att_starts is None
    pw, pxh = exl3_dense.prompt_sizes(users, rows)
    assert (pw, pxh) == (4096 * 12288, rows * 6144)   # qkv's W_q; the dense down_proj's K (head excluded)
    assert sc.prefill.w.numel() == pw and sc.prefill.xh.numel() == pxh
    ptr = (sc.prefill.w.data_ptr(), sc.prefill.xh.data_ptr())
    xh, z, tmp = exl3_dense.decode_sizes(users)
    assert sc.buffers["xh:torch.float16"].numel() == xh and sc.buffers["z:torch.float32"].numel() == z
    assert sc.buffers["tmp:torch.bfloat16"].numel() == tmp == 128 * 77440
    # a full prompt chunk through every EXL3 / plain projection of the layer: nothing grows, nothing regrows
    layer = w.layers[0]
    x = torch.randn((rows, w.cfg.hidden), dtype=torch.bfloat16, device="cuda") * 0.1
    before = sc.nbytes()
    forward.mm(pre, x, layer.kda.proj, None, pre.kproj[0, :rows])
    forward.mm(pre, x, layer.mlp.gu, None, pre.gu[:rows])
    forward.mm(pre, pre.act[:rows].normal_(), layer.mlp.down, None, pre.part[:rows], f32=True)
    fa = pre.kproj[0, :rows, layer.kda.fa_off:layer.kda.fa_off + 128]
    forward.mm(pre, fa, layer.kda.fb, None, pre.ka[:rows])
    torch.cuda.synchronize()
    assert sc.nbytes() == before
    assert (sc.prefill.w.data_ptr(), sc.prefill.xh.data_ptr()) == ptr
    assert isinstance(layer.kda.proj.parts[1], qmm.B16)

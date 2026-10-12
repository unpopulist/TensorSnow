"""RED-SNOW's routed experts (rank 0's share of two real layers: 6-bit gate/up with 4/5/6-bit down, and mostly 5-bit
gate/up with 4-bit down) through the decode kernel's 16-byte loads give the bits of its 32-bit loads and of the prompt
kernel, at 1, 2, 8 and 16 rows, and every row alone gives the bits it gives inside its window. Loads about 4 GB of
RED-SNOW: only with TENSORFOLD_REDSNOW_LOADER=1 (TENSORFOLD_REDSNOW_MODEL overrides the path)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

MODEL = Path(os.environ.get("TENSORFOLD_REDSNOW_MODEL", "/var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw"))
pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only"),
              pytest.mark.skipif(not (MODEL / "model.safetensors.index.json").exists(), reason="no RED-SNOW checkpoint"),
              pytest.mark.skipif(os.environ.get("TENSORFOLD_REDSNOW_LOADER") != "1",
                                 reason="loads about 4 GB of RED-SNOW: set TENSORFOLD_REDSNOW_LOADER=1")]

LAYERS = (4, 7)
E, TOPK, SLOTS, D = 288, 8, 9, 4096


def _layer(rd, i):
    from tensorfold.cuda.exl3 import experts

    p = f"model.language_model.layers.{i}.mlp.experts."
    mats = [[tuple(rd.get(p + f"{e}.{proj}.{x}").cuda() for x in ("trellis", "suh", "svh")) for e in range(E)]
            for proj in ("gate_proj", "up_proj", "down_proj")]
    full = p + "0.gate_proj"
    cb = "mcg" if full + ".mcg" in rd.index else "mul1"        # RED-SNOW's config.json: mul1
    return experts.prepare(*mats, cb)


def _picks(R, rng):
    """R rows of 8 distinct experts drawn from a pool of about 4 R + 4 (windows share experts); slot 8 the shared one."""

    pool = rng.choice(E, size=min(E, 4 * R + 4), replace=False)
    out = np.full((R, SLOTS), E, dtype=np.int32)
    for r in range(R):
        out[r, :TOPK] = rng.choice(pool, size=TOPK, replace=False)
    return torch.from_numpy(out).cuda()


@pytest.fixture(scope="module")
def layers():
    from tensorfold.families.glm5_next.cuda.split import RankReader

    rd = RankReader(MODEL, 0, ranks=2)
    out = {i: _layer(rd, i) for i in LAYERS}
    assert out[4].k2_gu == (12, 12) and out[7].k2_gu[0] < out[7].k2_gu[1]   # mixed widths in layer 7
    assert all(ex.aligned16 for ex in out.values())
    return out


def test_sixteen_byte_loads_keep_the_bits(layers, monkeypatch):
    from tensorfold.cuda.exl3 import experts
    from tensorfold.families.glm5_next.cuda.exl3_generic import dec_order

    order = dec_order()                                         # TF_GLM_EXL3_DEC_ORDER (0090): the same bits at each
    rng = np.random.default_rng(7)
    g = torch.Generator(device="cuda").manual_seed(3)
    for i, ex in layers.items():
        dec = experts.Scratch(ex, 16, SLOTS, prompt_rows=17)        # 1-16 rows: the decode kernel
        pro = experts.Scratch(ex, 16, SLOTS, prompt_rows=1)         # every window: the prompt kernel
        for R in (1, 2, 8, 16):
            x = (torch.randn((R, D), device="cuda", generator=g) * 0.5).to(torch.bfloat16)
            pick = _picks(R, rng)
            live = (pick < E).view(-1)

            def run(loads, s, rows=slice(None)):
                monkeypatch.setattr(experts, "LOADS", loads)
                xr, pr = x[rows].contiguous(), pick[rows].contiguous()
                y = experts.routed(xr, pr, None, ex, s, None, xr.shape[0], 10.0, act_mode=experts.ACT_BF16,
                                   order=order)
                return y[:xr.shape[0] * SLOTS].clone()

            new = run(1, dec)[live]
            assert torch.isfinite(new).all()
            assert torch.equal(new.view(torch.int32), run(0, dec)[live].view(torch.int32)), (i, R, "32-bit loads")
            assert torch.equal(new.view(torch.int32), run(1, pro)[live].view(torch.int32)), (i, R, "prompt kernel")
            win = new.view(R, TOPK, D)
            for r in range(R):                                      # a row alone == the row inside its window
                alone = run(1, dec, slice(r, r + 1))[:TOPK]
                assert torch.equal(alone.view(torch.int32), win[r].view(torch.int32)), (i, R, r)

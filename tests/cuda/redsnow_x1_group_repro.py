"""Isolated single-thread repro: does X1's exl3.experts.group/routed work on REAL RED-SNOW layer-3 experts at the
R a prompt chunk drives (up to PREFILL_ROWS=2048)? No merge-modified engine code, no two-thread harness."""
import os
from pathlib import Path

import torch

from tensorfold.families.glm5_next.cuda import weights, exl3_generic
from tensorfold.families.glm5_next.cuda.forward import Buffers
from tensorfold.families.glm5_next.cuda.split import RankReader
from tensorfold.families.glm5_next.cuda import split as splitmod

MODEL = Path(os.environ.get("TENSORFOLD_REDSNOW_MODEL", "/var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw"))
EMBED_ROWS = 1024

orig = RankReader.get
def get(self, name):
    if name.endswith("embed_tokens.weight"):
        return torch.randn((EMBED_ROWS, 4096), generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
    return orig(self, name)

splitmod.READERS = 2
RankReader.get = get
try:
    w = weights.load(MODEL, rank=0, layers=[3], mtp=False)
finally:
    RankReader.get = orig

w.meta["long_context"] = True
layer = w.layers[0]
assert layer.moe is not None, "layer 3 should be MoE"
m = layer.moe
c = w.cfg
k = c.top_k
print(f"# top_k={k} experts={c.experts} limit={c.limit}", flush=True)

b = Buffers(w, 2048, 2560, prefill=True)
gen = torch.Generator().manual_seed(1)
for R in (37, 512, 1024, 2048):
    b.normed[:R].copy_((torch.randn((R, 4096), generator=gen) * 0.1).to(torch.bfloat16).cuda())
    ids = torch.stack([torch.randperm(c.experts, generator=gen)[:k] for _ in range(R)]).to(torch.int32).cuda()
    b.pick[:R, :k].copy_(ids)
    b.pick[:R, k] = c.experts
    try:
        exl3_generic.routed(b.normed[:R], b.pick, m.experts, b.exl3, R, c.limit)
        torch.cuda.synchronize()
        print(f"R={R}: OK", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"R={R}: FAIL {type(exc).__name__}: {str(exc).splitlines()[0]}", flush=True)

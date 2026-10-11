"""GLM's L2 prefetch kernel (``glm5_next/cuda/l2pf.cu``) on the GPU: the extension builds and one prefetch over an
EXL3 X3's ranges (scales + warp-stream heads) and a BF16 tensor runs in each mode (bulk / lines / touch) on a side
stream, and leaves the weights and an X3 call's result as they were. Small synthetic tensors, no checkpoint."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from tensorfold.families.glm5_next.cuda import exl3_dense, l2pf  # noqa: E402


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_prefetch_runs(mode):
    g = torch.Generator().manual_seed(1)
    trellis = torch.randint(-2**15, 2**15, (1024 // 16, 512 // 16, 64), dtype=torch.int16, generator=g)
    x = exl3_dense.x3(trellis, torch.randn(1024, generator=g).half(), torch.randn(512, generator=g).half(),
                      "mul1", device="cuda", sc=exl3_dense.Scratch())
    plain = torch.randn(256, 1024, device="cuda").bfloat16()
    inp = torch.randn(1, 1024, device="cuda").bfloat16()
    want = x(inp).clone()
    words = x.lin.words.clone()

    spans = l2pf.ranges([x, plain], 256 << 10)
    assert len(spans) > 4
    pieces = l2pf.pieces(spans, 32 << 10)
    table = torch.tensor(pieces, dtype=torch.int64, device="cuda").view(-1, 2).contiguous()
    sink = torch.zeros((1,), dtype=torch.int32, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        l2pf._ext().prefetch(table, 0, len(pieces), mode, 4, 128, sink)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    assert torch.equal(x.lin.words, words)
    assert torch.equal(x(inp), want)

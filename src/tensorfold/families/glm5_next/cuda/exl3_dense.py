"""GLM-5.3's EXL3 dense matrices: trellis projections (``X3``), side-by-side stacks of separately quantized parts
(``Stack``) and their shared scratch.

Dispatch:
- decode / verify / MTP rows (``__call__``): the row-invariant EXL3 decode kernel in pieces of at most 128 rows, so a
  row's bits never depend on the window it is in.
- prompt chunks (``prompt``, chosen by ``forward.mm`` on ``b.prefill``): the prompt GEMM (W_q decoded to fp16 once a
  call) for matrices of at most GEMM_MAX weights, chunk-invariant but not bit-equal to decode; bigger matrices (the
  vocabulary head) take 128-row decode pieces, so no K x N fp16 copy of them is ever made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from tensorfold.cuda.exl3 import prefill as x3prefill
from tensorfold.cuda.exl3.linear import Exl3Linear

ROWS = 128                  # most rows one call of the EXL3 decode kernel takes
GEMM_MAX = 96 * 2**20       # most weights (K * N) a matrix may have to take the prompt GEMM (its fp16 W_q workspace)


class Scratch:
    """Grow-only named buffers shared by every EXL3 matrix (calls run one after another on one stream).

    A grown buffer's predecessor is kept alive in ``retired``: a captured CUDA graph may still point at it."""

    def __init__(self) -> None:
        self.buffers: dict[str, torch.Tensor] = {}
        self.retired: list[torch.Tensor] = []
        self.prefill = x3prefill.Workspace()

    def get(self, name: str, numel: int, dtype: torch.dtype, device) -> torch.Tensor:
        key = f"{name}:{dtype}"
        t = self.buffers.get(key)
        if t is None or t.numel() < numel or t.device != torch.device(device):
            if t is not None:
                self.retired.append(t)
            t = torch.empty((max(numel, 1),), dtype=dtype, device=device)
            self.buffers[key] = t
        return t[:numel]

    def reserve(self, users: list[Any], device, rows: int = ROWS) -> None:
        """Preallocate the decode scratch for ``users`` (X3 / Stack) so no call allocates (graph capture)."""

        xh, z, tmp = decode_sizes(users, rows)
        self.get("xh", xh, torch.float16, device)
        self.get("z", z, torch.float32, device)
        self.get("tmp", tmp, torch.bfloat16, device)
        self.get("tmp", tmp, torch.float32, device)

    def reserve_prompt(self, users: list[Any], rows: int, device) -> None:
        """Preallocate the prompt GEMM's workspace (decoded W_q, rotated inputs, Hadamard block) for prompt chunks of
        ``rows`` rows, so its size is fixed at startup rather than on the first prompt."""

        w, xh = prompt_sizes(users, rows)
        if w:
            self.prefill._grow("w", w, device)
            self.prefill._grow("xh", xh, device)
            self.prefill.hadamard(device)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.buffers.values()) + self.prefill.nbytes()


SCRATCH = Scratch()


def _parts(users: list[Any]) -> list[Any]:
    return [p for u in users for p in (u.parts if isinstance(u, Stack) else [u])]


def decode_sizes(users: list[Any], rows: int = ROWS) -> tuple[int, int, int]:
    """Elements of the decode scratch ``Scratch.reserve`` holds for ``users``: (xh fp16, z fp32, tmp), tmp twice
    (bf16 and fp32). Bytes: 2 * xh + 4 * z + 6 * tmp."""

    xh = z = tmp = 1
    for p in _parts(users):
        tmp = max(tmp, rows * p.n)
        if isinstance(p, X3):
            xh = max(xh, rows * p.k)
            if p.lin.split[0] > 1:
                z = max(z, p.lin.split[0] * rows * p.n)
    return xh, z, tmp


def prompt_sizes(users: list[Any], rows: int) -> tuple[int, int]:
    """fp16 elements of the prompt GEMM's workspace for chunks of ``rows`` rows: (decoded W_q, the largest K x N of
    a matrix taking the GEMM; rotated inputs, rows x its largest K). (0, 0) when no matrix takes the GEMM.
    The 128 x 128 bf16 Hadamard block (32 KiB) comes on top."""

    w = xh = 0
    for p in _parts(users):
        if isinstance(p, X3) and p.gemm:
            w, xh = max(w, p.k * p.n), max(xh, rows * p.k)
    return w, xh


def _direct(out: torch.Tensor) -> bool:
    """The decode kernel / BF16 matmul can write ``out`` itself: contiguous (a 1-row column slice is) and 16-byte aligned."""

    return out.is_contiguous() and out.data_ptr() % 16 == 0


def _into(sc: Scratch, write: Callable[[torch.Tensor, torch.Tensor], Any], x: torch.Tensor, out: torch.Tensor,
          n: int, any_rows: bool = False) -> torch.Tensor:
    """``write(x_piece, dst)`` for pieces of at most ROWS rows; dst is ``out`` itself when it can be, else a scratch
    tmp copied into the (column-sliced) ``out``. Rows are independent, so the pieces keep every row's bits.
    ``any_rows``: ``write`` takes any row count (the BF16 matmul), so a direct ``out`` is written in one call."""

    m = x.shape[0]
    if (m <= ROWS or any_rows) and _direct(out):
        write(x, out)
        return out
    for r0 in range(0, m, ROWS):
        r1 = min(m, r0 + ROWS)
        piece = out[r0:r1]
        if _direct(piece):
            write(x[r0:r1], piece)
        else:
            tmp = sc.get("tmp", (r1 - r0) * n, out.dtype, out.device).view(r1 - r0, n)
            write(x[r0:r1], tmp)
            piece.copy_(tmp)
    return out


def _out(x: torch.Tensor, n: int, out: torch.Tensor | None, f32: bool) -> torch.Tensor:
    if out is None:
        return torch.empty((x.shape[0], n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    if out.shape != (x.shape[0], n) or out.stride(-1) != 1:
        raise ValueError(f"EXL3 matmul: out {tuple(out.shape)} must be ({x.shape[0]}, {n}) with unit column stride")
    if f32 and out.dtype != torch.float32:
        raise ValueError("EXL3 matmul: f32 needs an fp32 out")
    return out


@dataclass
class X3:
    """One EXL3 projection [K -> N] (``Exl3Linear``): decode kernel in 128-row pieces, or the prompt GEMM."""

    lin: Exl3Linear
    sc: Scratch = field(default=SCRATCH, repr=False)

    @property
    def n(self) -> int:
        return self.lin.n

    @property
    def k(self) -> int:
        return self.lin.k

    @property
    def gemm(self) -> bool:
        """Prompt chunks take the prompt GEMM (False: the vocabulary head, 128-row decode pieces)."""

        return self.lin.k * self.lin.n <= GEMM_MAX

    def nbytes(self) -> int:
        lin = self.lin
        parts = [lin.words, lin.suh, lin.svh, lin.counters] + ([lin.bias] if lin.bias is not None else [])
        return sum(t.numel() * t.element_size() for t in parts)

    def _decode(self, x: torch.Tensor, dst: torch.Tensor) -> None:
        lin, sk, m = self.lin, self.lin.split[0], x.shape[0]
        if x.shape[1] != lin.k:
            raise ValueError(f"EXL3 matmul: x {tuple(x.shape)} does not match K={lin.k}")
        if x.data_ptr() % 16 or not x.is_contiguous():    # the kernel reads a contiguous, 16-byte aligned x
            x = x.clone(memory_format=torch.contiguous_format)
        xh = self.sc.get("xh", m * lin.k, torch.float16, x.device).view(m, lin.k)
        z = self.sc.get("z", sk * m * lin.n, torch.float32, x.device) if sk > 1 else None
        lin(x, out=dst, xh=xh, z=z)

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None, f32: bool = False,
                 part: torch.Tensor | None = None) -> torch.Tensor:
        """Decode / verify rows: bf16 out, or fp32 with ``f32`` (an unrounded TP partial). ``part`` is unused."""

        return _into(self.sc, self._decode, x, _out(x, self.n, out, f32), self.n)

    def prompt(self, x: torch.Tensor, out: torch.Tensor | None = None, f32: bool = False,
               part: torch.Tensor | None = None) -> torch.Tensor:
        """Prompt-chunk rows: the prompt GEMM (writes a row-strided ``out`` directly) or, above GEMM_MAX, decode pieces."""

        out = _out(x, self.n, out, f32)
        if not self.gemm:
            return _into(self.sc, self._decode, x, out, self.n)
        return x3prefill.matmul(self.lin, x, out, self.sc.prefill)


@dataclass
class Stack:
    """Matrices reading the same input, written side by side into one output, like the stacked B16 they replace:
    EXL3 parts (each its own widths and scales, so they cannot share one trellis) and plain ``qmm.B16`` parts."""

    parts: list[Any]
    sc: Scratch = field(default=SCRATCH, repr=False)

    @property
    def n(self) -> int:
        return sum(p.n for p in self.parts)

    @property
    def k(self) -> int:
        return self.parts[0].k

    def nbytes(self) -> int:
        return sum(p.nbytes() for p in self.parts)

    def _run(self, x: torch.Tensor, out: torch.Tensor | None, f32: bool, part: torch.Tensor | None,
             prompt: bool) -> torch.Tensor:
        from . import qmm

        out = _out(x, self.n, out, f32)
        at = 0
        for p in self.parts:
            dst = out[:, at:at + p.n]
            if isinstance(p, X3):
                (p.prompt if prompt else p)(x, dst, f32=f32)
            elif isinstance(p, qmm.B16):                # same bits for any rows, so the pieces are free
                _into(self.sc, lambda xs, o, p=p: qmm._matmul_b16(xs, p, out=o, f32=f32, part=part), x, dst, p.n,
                      any_rows=True)
            else:
                raise TypeError(f"Stack part {type(p).__name__} is neither X3 nor B16")
            at += p.n
        return out

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None, f32: bool = False,
                 part: torch.Tensor | None = None) -> torch.Tensor:
        """Decode / verify rows: a 1-row column slice is written directly, more rows through a tmp and a copy."""

        return self._run(x, out, f32, part, False)

    def prompt(self, x: torch.Tensor, out: torch.Tensor | None = None, f32: bool = False,
               part: torch.Tensor | None = None) -> torch.Tensor:
        """Prompt-chunk rows: EXL3 parts' prompt GEMMs write their column slices directly."""

        return self._run(x, out, f32, part, True)


def x3(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, codebook: str = "mul1",
       bias: torch.Tensor | None = None, device: str | torch.device = "cuda", sc: Scratch = SCRATCH) -> X3:
    """An X3 from one EXL3 group's tensors (trellis int16 [K/16, N/16, 16 * bits], suh/svh fp16)."""

    return X3(Exl3Linear.from_tensors(trellis, suh, svh, codebook, bias, device), sc)

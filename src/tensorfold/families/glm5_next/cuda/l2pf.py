"""L2 prefetch of the next kernels' weights in decode windows (TF_GLM_L2PF, off by default; the same bits).

A decode window's layer alternates DRAM-bound matmuls (projections, routed and shared experts, the head) with phases
that leave DRAM idle: the two all-gathers a layer (the RoCE wait), the hyper-connection glue, the router and top-k,
the KDA recurrence and the latent attention core. At a few sites a layer, a small kernel on a side stream asks for
the weights the main stream reads next to be brought into L2 (``l2pf.cu``) while it is in such a phase; the matmul
that follows then finds part of its bytes in L2 instead of DRAM. The prefetch only reads, and every model kernel keeps
its inputs, outputs and stream order, so replies keep their bits.

Sites (``Prefetch.site``, keyed by (layer index, name)):

- ``a``, at the attention block's all-gather: the FFN's hyper-connection weights and norm, the router, and the shared
  expert's gate/up and down (a dense MLP's gate/up and down); with the prefetcher on, the MoE runs its shared expert
  before the routed experts (disjoint buffers, the same kernels: the same bits), so the routed experts' ~100+ MB
  stream does not evict it first.
- ``f``, at the FFN's all-gather: the next layer's hyper-connection weights and input norm, then its input
  projections (KDA: proj, f/g low-rank; DSA: proj, q_b, the indexer's keys); after the last layer the final norm and
  the head.
- ``o``, after the attention block's input projections, while the recurrence / attention core runs: its output
  projection.

Each site takes up to TF_GLM_L2PF_MB (default 8) MiB, small tensors first (scales, biases, norms), then the large
matrices' leading bytes. TF_GLM_L2PF_SITES (default ``afo``) picks the sites. TF_GLM_L2PF=1|bulk (one
cp.async.bulk.prefetch.L2 per piece), ``lines`` (prefetch.global.L2::evict_last per line) or ``touch`` (ld.global.cg
per line). Windows over TF_GLM_L2PF_ROWS (default 64) rows and prompt chunks skip every site.

Adapted from jayleaton/glm53-tensorfold-spark patches/0460 (l2pf.py / l2pf.cu; Apache-2.0, Copyright 2026 Jay
Leaton): the sites at the two all-gathers and after the attention projections, the shared expert first, a budget a
site, bulk / lines / touch mechanisms, a 4-bit matrix's chunk heads for one-wave launches (``q4_heads``), the side
stream forked at a site and joined before a forward returns. Changes: written onto this engine's forward (hooks in
``forward.compute`` / ``partials`` / the KDA and DSA blocks and their multi-stream ``*_segments`` forms, so batched
verify windows prefetch too); one device table of (address, bytes) pieces for every site with each site's grid sized
at load; DSA's ``o`` site fires after absorb_q and takes kv_b's value half (this engine's latent layout) before the
output projection; the next layer's KDA / DSA small weights (f/g low-rank, conv, norms, the indexer's) join site
``f``; no expert (``e``) or MTP sites; ``touch`` loads inside the same kernel; default 8 MiB a site.

RED-SNOW (native EXL3 dense weights, ``exl3_dense``): an ``X3`` contributes its ``suh`` / ``svh`` / ``bias`` and then
its trellis words' heads in the order the EXL3 decode kernel's warps start reading them (``x3_heads``, from the
matrix's own ``lin.split`` / ``lin.strides``); a ``Stack``'s parts are taken one after another, each a group of its
own. Never collected: the shared EXL3 scratch (``X3.sc`` / ``Stack.sc``: ``exl3_dense.Scratch`` and its prompt
``Workspace``) and ``Exl3Linear``'s ``counters`` (semaphores the kernel writes). Routed experts take no site."""

from __future__ import annotations

import dataclasses
import os
from functools import lru_cache
from typing import Any, Iterable

import torch

MODES = {"1": 0, "bulk": 0, "lines": 1, "touch": 2}
PIECE = 32 << 10           # bytes a table piece covers at most (a bulk prefetch's size, a warp's walk); TF_GLM_L2PF_KB
ALIGN = 16                 # cp.async.bulk wants 16-byte addresses and sizes


@dataclasses.dataclass
class Settings:
    mode: int = -1          # -1: off
    mb: float = 8.0
    sites: str = "afo"
    rows: int = 64
    blocks: int = 0         # 0: enough for the site's pieces (a thread a piece in bulk mode, a warp a piece else)
    threads: int = 128
    piece: int = PIECE

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        env = os.environ if env is None else env
        s = cls()
        v = (env.get("TF_GLM_L2PF", "") or "0").strip().lower()
        if v not in ("0", "off", *MODES):
            raise ValueError(f"TF_GLM_L2PF: 0, 1 (bulk), bulk, lines or touch, not {v!r}")
        s.mode = MODES.get(v, -1)
        try:
            s.mb = float(env.get("TF_GLM_L2PF_MB", "") or 8.0)
        except ValueError:
            s.mb = -1.0
        if not 0 < s.mb <= 64:
            raise ValueError("TF_GLM_L2PF_MB: above 0 and at most 64 (MiB a site)")
        s.sites = (env.get("TF_GLM_L2PF_SITES", "") or "afo").strip().lower()
        if not s.sites or set(s.sites) - set("afo"):
            raise ValueError(f"TF_GLM_L2PF_SITES: letters of 'afo', not {s.sites!r}")
        s.rows = int(env.get("TF_GLM_L2PF_ROWS", "") or 64)
        s.blocks = int(env.get("TF_GLM_L2PF_BLOCKS", "") or 0)
        s.threads = int(env.get("TF_GLM_L2PF_THREADS", "") or 128)
        s.piece = int(env.get("TF_GLM_L2PF_KB", "") or PIECE >> 10) << 10
        if s.blocks < 0 or s.threads < 32 or s.threads % 32 or s.threads > 1024:
            raise ValueError("TF_GLM_L2PF_BLOCKS / _THREADS: 0 (auto) or more blocks of 32..1024 threads (whole warps)")
        if not 1 << 10 <= s.piece <= 1 << 20:
            raise ValueError("TF_GLM_L2PF_KB: 1 .. 1024 KiB a piece")
        return s

    @property
    def on(self) -> bool:
        return self.mode >= 0

    def code(self) -> str:
        return "off" if not self.on else f"{self.mode}:{self.mb}:{self.sites}:{self.rows}:{self.blocks}:{self.threads}:{self.piece}"

    def ints(self) -> list[int]:
        """The settings as ints for the ranks' startup agreement (``GlmEngine``'s settings vector): every rank must
        capture the same side-stream fork / join in its graphs. All zeros when off."""

        if not self.on:
            return [0] * 7
        return [self.mode + 1, round(self.mb * 1024), sum(1 << "afo".index(ch) for ch in set(self.sites)), self.rows,
                self.blocks, self.threads, self.piece >> 10]


@lru_cache(maxsize=1)
def _ext():
    from pathlib import Path

    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_glm_l2pf_v2", sources=[str(here / "l2pf.cpp"), str(here / "l2pf.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


@lru_cache(maxsize=1)
def _exl3_types() -> tuple[type, type, type, tuple[type, ...]]:
    """(X3, Stack, Exl3Linear, scratch types never collected), imported late (no import cycle with the forward)."""

    from tensorfold.cuda.exl3.linear import Exl3Linear
    from tensorfold.cuda.exl3.prefill import Workspace

    from . import exl3_dense

    return exl3_dense.X3, exl3_dense.Stack, Exl3Linear, (exl3_dense.Scratch, Workspace)


def _lin_tensors(lin) -> list[torch.Tensor]:
    """An ``Exl3Linear``'s weight tensors in the order a call reads them (``rot_in``: suh; the kernel: words, svh,
    bias). Not ``counters`` (``_counters``): semaphores the kernel writes, not weights."""

    return [t for t in (lin.suh, lin.words, lin.svh, lin.bias) if t is not None]


def tensors(obj: Any) -> list[torch.Tensor]:
    """The CUDA tensors an object's weights hold (dataclass / plain-object fields in order, lists, tuples), each once.
    EXL3: an ``X3`` / ``Stack`` gives its ``lin`` / ``parts`` only (not the shared scratch ``sc``), an ``Exl3Linear``
    its words and scales (not its counters); a scratch object met anywhere is skipped."""

    X3, Stack, Exl3Linear, scratch = _exl3_types()
    out: list[torch.Tensor] = []
    seen: set[int] = set()

    def walk(t: Any, depth: int = 0) -> None:
        if depth > 6 or t is None or isinstance(t, scratch):
            return
        if isinstance(t, X3):
            walk(t.lin, depth + 1)
        elif isinstance(t, Stack):
            walk(list(t.parts), depth + 1)
        elif isinstance(t, Exl3Linear):
            for v in _lin_tensors(t):
                walk(v, depth + 1)
        elif isinstance(t, torch.Tensor):
            if t.numel() and t.data_ptr() not in seen:
                seen.add(t.data_ptr())
                out.append(t)
        elif isinstance(t, (list, tuple)):
            for v in t:
                walk(v, depth + 1)
        elif isinstance(t, dict):
            for v in t.values():
                walk(v, depth + 1)
        elif hasattr(t, "__dict__") and not isinstance(t, (int, float, str, bool, torch.device)):
            for v in vars(t).values():
                walk(v, depth + 1)

    walk(obj)
    return out


ONE_WAVE = 192             # qmm programs resident at once on GB10 (4 CTAs an SM x 48 SMs)


def q4_heads(q, left: int) -> list[tuple[int, int]]:
    """A 4-bit matrix (``qmm.Q4``: words [N/64, K/64, 8, 32, 2], one contiguous chunk of K * 32 / SK bytes a
    (64-column tile, K slice), what one decode program streams): the first ``left / chunks`` bytes of EVERY chunk, so
    each program of a one-wave launch starts on L2 hits and the launch's slowest program is shortened too (a
    tensor's first bytes would only help its first programs). [] for a launch of more than one wave or heads under
    512 bytes. (jayleaton/glm53-tensorfold-spark patches/0460 ``_q4_heads``, Apache-2.0, Copyright 2026 Jay Leaton;
    changes: our qmm's split_k, (address, bytes) pairs.)"""

    from . import qmm

    w = q.weight
    if not isinstance(w, torch.Tensor) or not w.is_contiguous():
        return []
    chunks = -(-int(q.n) // qmm.BN) * qmm.split_k(int(q.n), int(q.k))
    total = w.numel() * w.element_size()
    if chunks <= 0 or chunks > ONE_WAVE or total % chunks:
        return []
    size = total // chunks
    per = min(size, left // chunks) // ALIGN * ALIGN
    if per < 512 or w.data_ptr() % ALIGN or size % ALIGN:
        return []
    return [(w.data_ptr() + c * size, per) for c in range(chunks)]


def x3_heads(lin, left: int) -> list[tuple[int, int]]:
    """An EXL3 matrix (``Exl3Linear``, "strips" layout: words [N/128, K/16, 8, TW]): the first ``left / streams``
    bytes of every warp's stream in the decode kernel's first wave, so each warp of the launch starts on L2 hits.

    Derived from the matrix itself, not a fixed tiling: ``linear.cu``'s grid is (N/128 column blocks, SK splits) of WK
    warps (``lin.split``, fixed by ``plan(k, n)`` at load), and warp w of block (nb, split) walks per_warp = K/16 / SK
    / WK k tiles from kt0 = split * per_warp * WK + w * per_warp, each ``stride_k`` words on from the last (one
    contiguous run of the strips layout). Programs go x first, so a launch of more than ONE_WAVE programs takes the
    first ONE_WAVE (plan() sizes launches for 192 blocks; ONE_WAVE is not tuned for this kernel's residency).
    [] for the "stored" layout, a non-contiguous or unaligned words tensor, a split that does not tile K, or heads
    under 512 bytes (then the caller takes the words' leading bytes)."""

    w = lin.words
    if getattr(lin, "layout", "strips") != "strips" or not isinstance(w, torch.Tensor) or not w.is_contiguous():
        return []
    sk, wk = lin.split
    k, n = int(lin.k), int(lin.n)
    nb, kt = n // 128, k // 16
    if nb <= 0 or sk <= 0 or wk <= 0 or kt % (sk * wk):
        return []
    per_warp = kt // sk // wk
    stride_k, stride_nb = (4 * s for s in lin.strides)          # words -> bytes
    total = w.numel() * w.element_size()
    if total != nb * stride_nb or stride_nb != kt * stride_k:
        return []
    run = per_warp * stride_k                                    # bytes one warp streams
    programs = min(nb * sk, ONE_WAVE)
    streams = programs * wk
    per = min(run, left // streams) // ALIGN * ALIGN
    base = w.data_ptr()
    if per < 512 or base % ALIGN or stride_k % ALIGN:
        return []
    out = []
    for p in range(programs):
        col, split = p % nb, p // nb
        for warp in range(wk):
            kt0 = split * per_warp * wk + warp * per_warp
            out.append((base + col * stride_nb + kt0 * stride_k, per))
    return out


def _span(t: torch.Tensor, left: int) -> tuple[int, int] | None:
    a = t.data_ptr()
    n = t.numel() * t.element_size()
    lo = a - a % ALIGN
    hi = min(a + n, lo + left)
    hi += (-hi) % ALIGN
    return (lo, hi - lo) if hi > lo else None


def _units(groups: Iterable[Any]) -> list[Any]:
    """The groups with each ``Stack`` replaced by its parts, in order (the main stream runs them one after another)."""

    _, Stack, _, _ = _exl3_types()
    out: list[Any] = []
    for g in groups:
        if isinstance(g, Stack):
            out += _units(g.parts)
        else:
            out.append(g)
    return out


def ranges(groups: Iterable[Any], budget: int) -> list[tuple[int, int]]:
    """(address, bytes) ranges of ``groups`` (weight objects in the order the main stream reads them) up to
    ``budget`` bytes: within a group its small tensors first (scales, biases, norms), then a 4-bit matrix's chunk
    heads when its decode launch is one wave (``q4_heads``), an EXL3 matrix's warp-stream heads (``x3_heads``), else
    the large tensors' leading bytes; 16-byte aligned. A ``Stack`` is its parts, each a group."""

    X3, _, Exl3Linear, _ = _exl3_types()
    out: list[tuple[int, int]] = []
    left = budget
    for g in _units(groups):
        ts = sorted(tensors(g), key=lambda x: x.numel() * x.element_size())
        heads = []
        lin = g.lin if isinstance(g, X3) else g if isinstance(g, Exl3Linear) else None
        if lin is not None:
            small = sum(t.numel() * t.element_size() for t in ts if t is not lin.words)
            heads = x3_heads(lin, left - small) if left > small else []
            if heads:
                ts = [t for t in ts if t is not lin.words]
        elif all(hasattr(g, a) for a in ("weight", "scales", "biases", "n", "k")) and isinstance(g.weight, torch.Tensor):
            small = sum(t.numel() * t.element_size() for t in ts if t is not g.weight)
            heads = q4_heads(g, left - small) if left > small else []
            if heads:
                ts = [t for t in ts if t is not g.weight]
        for t in ts:
            if left < ALIGN:
                return out
            sp = _span(t, left)
            if sp is not None:
                out.append(sp)
                left -= sp[1]
        if heads and left >= sum(n for _, n in heads):
            out += heads
            left -= sum(n for _, n in heads)
    return out


def pieces(spans: list[tuple[int, int]], piece: int = PIECE) -> list[tuple[int, int]]:
    out = []
    for a, n in spans:
        for o in range(0, n, piece):
            out.append((a + o, min(piece, n - o)))
    return out


class Prefetch:
    """The sites' tables for one set of weights (one device table; (first, count) a site) and the side stream."""

    def __init__(self, w, settings: Settings | None = None) -> None:
        self.s = settings or Settings.from_env()
        self.w = w
        self.sites: dict[tuple[int, str], tuple[int, int, int]] = {}
        self.bytes: dict[tuple[int, str], int] = {}
        budget = int(self.s.mb * (1 << 20))
        rows: list[tuple[int, int]] = []
        layers = list(w.layers)
        for i, layer in enumerate(layers):
            nxt = layers[i + 1] if i + 1 < len(layers) else None
            plan = {"a": self._ffn(layer), "f": self._next(nxt), "o": self._out(layer)}
            for name, groups in plan.items():
                if name not in self.s.sites or not groups:
                    continue
                spans = ranges(groups, budget)
                p = pieces(spans, self.s.piece)
                if p:
                    self.sites[(layer.index, name)] = (len(rows), len(p), self._grid(len(p)))
                    self.bytes[(layer.index, name)] = sum(n for _, n in spans)
                    rows += p
        dev = w.device
        self.table = torch.tensor(rows or [(0, 0)], dtype=torch.int64, device=dev).view(-1, 2).contiguous()
        self.sink = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.side = torch.cuda.Stream(device=dev) if torch.cuda.is_available() else None
        if self.side is not None:
            _ext()                                       # built / loaded now, before any graph capture
        self.forked = False
        self.launches = 0

    def _grid(self, n: int) -> int:
        """Blocks for a site of ``n`` pieces: TF_GLM_L2PF_BLOCKS, or a thread a piece (bulk) / a warp a piece."""

        if self.s.blocks:
            return self.s.blocks
        per = self.s.threads if self.s.mode == 0 else self.s.threads // 32
        return max(1, min(48, -(-n // per)))

    # -- what each site prefetches ---------------------------------------------------------------------------------
    def _ffn(self, layer) -> list:
        g: list = [layer.ffn_hc, layer.post_norm]
        if layer.mlp is not None:
            g += [layer.mlp.gu, layer.mlp.down]
        elif layer.moe is not None:
            g += [layer.moe.router, layer.moe.bias]
            if layer.moe.shared is not None:
                g += [layer.moe.shared.gu, layer.moe.shared.down]
        return g

    def _next(self, nxt) -> list:
        if nxt is None:
            return [self.w.norm, self.w.head]
        g: list = [nxt.attn_hc, nxt.in_norm]
        if nxt.kda is not None:
            k = nxt.kda
            g += [k.fb, k.gb, k.conv, k.a_log, k.dt_bias, k.proj]
        elif nxt.dsa is not None:
            a = nxt.dsa
            g += [a.q_norm, a.kv_norm, a.proj, a.q_b]
            if a.index is not None:
                g += [a.index.kw, a.index.gate]
        return g

    def _out(self, layer) -> list:
        """KDA: its output projection. DSA (fired after absorb_q, before the attention core): the value half of kv_b
        that expand_v reads, then the output projection."""

        if layer.kda is not None:
            return [layer.kda.o]
        if layer.dsa is not None:
            a = layer.dsa.absorb
            value = [getattr(a, n) for n in ("wvs", "wvb", "sv", "wvw", "wv") if getattr(a, n, None) is not None]
            return value + [layer.dsa.o]
        return []

    # -- the forward's calls ------------------------------------------------------------------------------------------
    def site(self, index: int, name: str) -> None:
        """Fork the side stream at this point of the main stream and prefetch the site's ranges there."""

        key = self.sites.get((index, name))
        if key is None or self.side is None:
            return
        first, count, grid = key
        main = torch.cuda.current_stream()
        self.side.wait_stream(main)
        with torch.cuda.stream(self.side):
            _ext().prefetch(self.table, first, count, self.s.mode, grid, self.s.threads, self.sink)
        self.forked = True
        self.launches += 1

    def join(self) -> None:
        """The main stream waits for the side stream (a captured graph's branches rejoin before its end)."""

        if self.forked:
            torch.cuda.current_stream().wait_stream(self.side)
            self.forked = False

    def summary(self) -> str:
        per = {}
        for (_, name), n in self.bytes.items():
            per.setdefault(name, []).append(n)
        parts = [f"{k} {len(v)} sites {sum(v) / len(v) / (1 << 20):.1f} MiB" for k, v in sorted(per.items())]
        mode = {0: "bulk", 1: "lines", 2: "touch"}[self.s.mode]
        return f"L2 prefetch in decode windows ({mode}, up to {self.s.mb:g} MiB a site): " + ", ".join(parts)


ACTIVE: Prefetch | None = None         # set by forward.compute for a decode window while it runs


def site(index: int, name: str) -> None:
    if ACTIVE is not None:
        ACTIVE.site(index, name)

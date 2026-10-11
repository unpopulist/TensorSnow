"""GLM-5.3-Flash on two NCCL ranks; both sample by one keyed rule from the same gathered candidates, so no broadcast."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable

DEFAULT_POLICY = "auto"
# DFlash2 drafts every round: by default up to 5 while their probability product holds 0.3; TF_GLM_DFLASH_POLICY picks
# another DFlash2 policy (e.g. fnc7:0.3, the noise-aware stop rule) for requests that name none
DFLASH_POLICY = os.environ.get("TF_GLM_DFLASH_POLICY", "").strip() or "fc5:0.3"
EXL3_AUTO = DFLASH_POLICY             # what auto runs on an EXL3 checkpoint with the draft model
GRAPH_ROWS = (1, 2, 3, 4, 5, 6, 7, 8)  # verify windows (and MTP steps) captured as CUDA graphs; wider ones run eagerly


def max_rows(value: str | None = None) -> int:
    """TF_GLM_MAX_ROWS: the widest verify window, 16 (default) to 64 in steps of 8 (``geometry.MLA_DECODE_ROWS``,
    which the startup estimate sizes decode buffers for). Wider windows serve copy drafts only (TF_GLM_COPY_MAX up to
    this minus one); past 16 rows a window's all-gathers exceed TF_ROCE_MAX_KB's default 256 KiB (16 KiB a row) and
    go through NCCL unless it is raised (TF_ROCE_MAX_KB=512 for 32 rows)."""

    from tensorfold.cuda.geometry import MLA_DECODE_ROWS

    value = (os.environ.get("TF_GLM_MAX_ROWS", "") if value is None else value).strip()
    if value == "":
        return 16
    if not value.isdecimal() or not 16 <= int(value) <= MLA_DECODE_ROWS or int(value) % 8:
        raise ValueError(f"TF_GLM_MAX_ROWS: 16 to {MLA_DECODE_ROWS} in steps of 8, not {value!r}")
    return int(value)


MAX_ROWS = max_rows()                 # the widest verify window (a pending token and up to 15 drafts: copy drafts)


def wide_graphs(value: str | None = None, most: int | None = None) -> tuple[int, ...]:
    """TF_GLM_WIDE_GRAPHS: verify-window widths past GRAPH_ROWS (9 to TF_GLM_MAX_ROWS) that also replay CUDA graphs,
    comma separated (e.g. ``16`` or ``16,32``; off by default: wider windows run eagerly). Copy drafts pad a window
    that falls short of a captured width up to it (``copy_drafts.CopySettings.pad``). Each width costs a graph a KDA
    parity, and one more a pool bucket past the dense limit, at startup."""

    most = MAX_ROWS if most is None else most
    value = (os.environ.get("TF_GLM_WIDE_GRAPHS", "") if value is None else value).strip()
    if value in ("", "0"):
        return ()
    out = set()
    for part in value.split(","):
        part = part.strip()
        if not part.isdecimal() or not max(GRAPH_ROWS) < int(part) <= most:
            raise ValueError(f"TF_GLM_WIDE_GRAPHS: widths from {max(GRAPH_ROWS) + 1} to {most} (TF_GLM_MAX_ROWS), "
                             f"comma separated, not {value!r}")
        out.add(int(part))
    return tuple(sorted(out))


WIDE_GRAPHS = wide_graphs()


def l2pf_code() -> list[int]:
    """TF_GLM_L2PF and its knobs as ints both ranks compare at startup (``l2pf.Settings.ints``): the decode windows'
    graphs capture the prefetch's side-stream fork / join, so both ranks must run it alike. The same bits either
    way."""

    from .l2pf import Settings

    return Settings.from_env().ints()


def decode_kernel_code() -> list[int]:
    """The kernel switches of Mia's 0004/0016/0019 as ints both ranks compare at startup: TF_GLM_HC_DEC,
    TF_GLM_KDA_WIDE_ROWS, TF_GLM_LATENT_STAGES, TF_GLM_COMBINE_SY, TF_GLM_HC_MMA and TF_GLM_HC_PRE_ROWS. All but
    HC_MMA give the same bits either way (they choose kernels and launches, not arithmetic); HC_MMA changes prompt
    rows' bits."""

    from . import glue, kda, latent

    # TF_GLM_HC_MMA does change prompt rows' bits (tensor-core mixing dots): under TF_GLM_HC_SPLIT the ranks must agree
    return [int(glue.hc_dec()), int(kda.WIDE_ROWS), int(latent.chunk_stages()),
            int(os.environ.get("TF_GLM_COMBINE_SY", "1") != "0"), int(glue.HC_MMA), glue.HC_PRE_ROWS]
MTP_ROWS = 8                          # rows an MTP step takes at once (a kept backlog is absorbed in steps of this)
DENSE_CAPACITY = 2560                 # cache slots while DSA attention stays dense (contexts up to 2,051 tokens)
# TF_GLM_DRAFT_RING=0: DFlash2's context in a flat buffer of the whole window, not a 2,176-row ring (the same drafts)
DRAFT_RING = os.environ.get("TF_GLM_DRAFT_RING", "1").strip() != "0"
# --vision (rank 0): what startup reserves beside the tower's 1.05 GiB of bf16 weights for an encode and its
# prompt's feature rows. Measured on RED-SNOW (8,192 patches a pass): a 1,998-token picture peaks 0.43 GiB above the
# weights; a request's rows add 8 KiB a token (32,768 clip tokens: 0.25 GiB). Mia reserved 1.5 GiB for 16,384-patch
# passes. A picture cap past 4,096 tokens scales it; TENSORFOLD_VISION_WORKSPACE_MIB sets it outright
VISION_WORKSPACE = 2**30


def _trim_host_heap() -> None:
    """glibc's malloc_trim(0): loading leaves GiBs of freed host memory in malloc's heap and per-thread arenas (measured
    on RED-SNOW: 3.1 GiB heap + ~5 GiB in 64 MiB arenas per rank), which a GB10 Spark's GPU then cannot use."""

    import ctypes

    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):         # not glibc
        pass


def vision_workspace() -> int:
    value = os.environ.get("TENSORFOLD_VISION_WORKSPACE_MIB", "")
    if value == "":
        from tensorfold.vision.glm import GlmVisionLimits

        return VISION_WORKSPACE * max(4096, GlmVisionLimits.from_env().image_tokens) // 4096
    if not value.isdecimal() or int(value) > 16384:
        raise ValueError(f"TENSORFOLD_VISION_WORKSPACE_MIB: 0 to 16,384 MiB, not {value!r}")
    return int(value) * 2**20


def vision_admission(weights_estimate, geometry, workspace: int):
    """Rank 0's admission with the tower: its bf16 weights (which the split rule drops) and its encode workspace on
    top of the caches' bytes."""

    import math

    from tensorfold.cuda.capacity import Geometry

    def weights(name, info):
        if name.startswith("model.visual."):
            return math.prod(info["shape"]) * 2, 0
        return weights_estimate(name, info)

    def geo(text):
        g = geometry(text)
        return Geometry(lambda slots: g.bytes_at(slots) + workspace, g.reserve, g.minimum_slots)

    return weights, geo


GRID = 64                             # TF_GLM_KDA_CHUNKED=1: prompt chunks and kept prompt states on this grid


def prompt_grid(prefill_rows: int, chunked: bool | None = None) -> int:
    """The token grid prompt chunks start on and kept prompt states sit on (Mia's 0008, reduced to what her chunked
    KDA prompt kernel needs): 0 (anywhere) unless TF_GLM_KDA_CHUNKED=1, whose rows depend on their 32-row
    sub-chunks at absolute positions; then 64, so a resumed prompt passes through the fresh prefill's sub-chunks."""

    if chunked is None:
        from .kda import CHUNKED as chunked
    if not chunked:
        return 0
    if prefill_rows % GRID:
        raise ValueError(f"TF_GLM_KDA_CHUNKED=1: prompt chunks of {prefill_rows} rows must be a multiple of {GRID}")
    return GRID


def grid_point(n: int, cut: int, grid: int) -> int | None:
    """Where a drafted request of an n-token prompt resumed at ``cut`` keeps its prompt state, or None for nowhere.

    Off (grid 0) and on-grid prompts: at n (the prompt's own end state, kept with its head's logits row so an
    identical resend replays it, ``decode.whole``). Otherwise at B = floor(n / grid) * grid, when B lies past
    ``cut`` (so B > 0); the request then prefills up to B, keeps that state, and resumes from it to n."""

    if not grid or n % grid == 0:
        return n
    point = n // grid * grid
    return point if point > cut else None


SHARED_DEFAULT = 256                  # TF_GLM_SHARED_PREFIX=1: the fewest tokens a shared-prefix state adds
SHARED_MIN = 16
SHARED_MOST = 2                       # shared-prefix states a request keeps at most (the header's two slots)


def shared_prefix(value: str | None = None) -> int:
    """TF_GLM_SHARED_PREFIX: 0 (off, the default), or the fewest tokens past a request's resume point a shared-prefix
    state must add (1 or on: 256; else 16 or more).

    On, a drafted text request also keeps its state where its prompt stops sharing a kept one and at the end of its
    system block (its first user message), each on the prompt grid, so a later prompt sharing only that prefix resumes
    from it; saved states keep DFlash2's context window too."""

    value = os.environ.get("TF_GLM_SHARED_PREFIX", "") if value is None else value
    value = value.strip().lower()
    if value in ("", "0", "off"):
        return 0
    if value in ("1", "on"):
        return SHARED_DEFAULT
    if not value.isdecimal() or int(value) < SHARED_MIN:
        raise ValueError(f"TF_GLM_SHARED_PREFIX: 0 (off), 1 (on: {SHARED_DEFAULT} tokens) or the fewest tokens a "
                         f"shared-prefix state adds past a request's resume point, {SHARED_MIN} or more; not {value!r}")
    return int(value)


def common_prefix(a, b) -> int:
    """How many leading tokens two id sequences (lists or int64 arrays) share."""

    import numpy as np

    n = min(len(a), len(b))
    if n == 0:
        return 0
    x = np.asarray(a[:n], dtype=np.int64)
    y = np.asarray(b[:n], dtype=np.int64)
    diff = np.flatnonzero(x != y)
    return int(diff[0]) if len(diff) else n


def shared_points(prompt: list[int], cut: int, point: int | None, grid: int, least: int, known,
                  opener: int | None = None) -> list[int]:
    """Where a request resumed at ``cut``, keeping its own state at ``point`` (``grid_point``), also keeps shared-prefix
    states: at the longest prefix it shares with any of ``known`` (id sequences: the kept prompts) and at its first
    ``opener`` token (the end of the system block), each rounded down to the grid, kept when it adds at least
    ``least`` tokens past ``cut`` (and past the other point) and lies short of ``point``. Sorted, at most
    SHARED_MOST."""

    if not least or point is None:
        return []
    import numpy as np

    arr = np.asarray(prompt, dtype=np.int64)
    wanted = [max((common_prefix(arr, ids) for ids in known if len(ids) >= cut + least), default=0)]
    if opener is not None:
        at = np.flatnonzero(arr == opener)
        if len(at):
            wanted.append(int(at[0]))
    out: list[int] = []
    for p in sorted(p // grid * grid if grid else p for p in wanted):
        if p - (out[-1] if out else cut) >= least and p < point:
            out.append(p)
    return out[-SHARED_MOST:]


def kept_entries(configured: int, share_pct: float, budget: int, state: int) -> tuple[int, str]:
    """How many kept prompt states ``--parallel`` reserves memory for and keeps (Mia's 0097): (count, a note for the
    log, "" when nothing changed).

    Each kept state owns ``state`` bytes outside the pool (its KDA recurrent state and conv window, the index rings'
    tail and the DFlash2 window copy), reserved at start out of ``budget`` (what TF_GLM_CACHE_GIB leaves the pool
    beyond the window), so the count is a memory reservation, not only a limit. ``share_pct`` (TF_GLM_CACHE_SHARE_PCT,
    0: off) makes the count at least that share of the budget: a pool-sized count, so a pool of millions of free tokens
    is not pushed empty by 32 one-shot conversations. Past three quarters of the budget the count is cut: the states
    would own memory the start-up estimate never counted. Pure arithmetic, the same on every rank."""

    state = max(1, state)
    n = max(1, configured)
    note = ""
    if share_pct > 0:
        n = max(n, int(budget * min(share_pct, 75.0) / 100 // state))
        note = f"{n} (at least {share_pct:g}% of the {budget / 2 ** 30:.1f} GiB the pool shares, ~{state >> 20} MiB each)"
    ceiling = max(1, int(budget * 0.75) // state)
    if n > ceiling:
        n = ceiling
        note = (f"{n}, not the {configured} asked: they would take more than three quarters of the {budget / 2 ** 30:.1f} "
                f"GiB the pool shares ({state >> 20} MiB each reserved at start)")
    return n, note


def user_opener(model_dir: Path) -> int | None:
    """The id of GLM's ``<|user|>`` role token (the first one ends the system block), or None without a tokenizer."""

    path = Path(model_dir) / "tokenizer.json"
    if not path.is_file():
        return None
    try:
        added = json.loads(path.read_text()).get("added_tokens", [])
    except (OSError, ValueError):
        return None
    return next((int(t["id"]) for t in added if t.get("content") == "<|user|>"), None)


def encode_policy(spec: str) -> list[int]:
    """Encode policy kind, maximum drafts, and two parameters in millionths as four integers, with 10 added to kind for DFlash2.

    DFlash2 only: ``fncN:P[:B]`` (kind 16: the fcN:P product of noise-aware confidences, B their temperature,
    default ``dflash2.NOISY_BETA``) and ``fcostN:noisy[:B]`` (kind 17: the depth k <= N maximizing E[tokens] / round
    ms by the startup cost table, E from the same confidences; N up to the captured verify windows' drafts)."""

    spec = str(spec).strip()
    bad = ValueError(f"draft policy {spec!r}: expected auto[:E:EVERY:MARGIN], 0, N, a[:LOW:HIGH], cN:P, or one of "
                     f"these after f (N from 1 to {MAX_ROWS - 1}), fncN:P[:B] or fcostN:noisy[:B] "
                     f"(N up to {len(GRAPH_ROWS) - 1})")
    try:
        if spec == "auto" or spec.startswith("auto:"):
            parts = spec.split(":")
            if len(parts) not in (1, 4):
                raise bad
            explore, every, margin = (int(parts[1]), int(parts[2]), float(parts[3])) if len(parts) == 4 else (2, 8, 0.03)
            if explore < 1 or every < 0 or not 0 <= margin < 1:
                raise bad
            return [4 if len(parts) == 1 else 5, explore, every, int(round(margin * 1e6))]
        if spec.startswith(("fnc", "fcost")):
            return _encode_noisy(spec[1:], bad)
        if spec.startswith("f"):
            code = encode_policy(spec[1:])
            return [code[0] + 10] + code[1:] if code[0] else code
        if spec.startswith("a"):
            parts = spec.split(":")
            if parts[0] != "a" or len(parts) not in (1, 3):
                raise bad
            low, high = (float(parts[1]), float(parts[2])) if len(parts) == 3 else (0.8, 0.9)
            return [2, 3, int(round(low * 1e6)), int(round(high * 1e6))]
        if spec.startswith("c"):
            most_text, conf = spec[1:].split(":")
            most = int(most_text)
            if not 0 < most < MAX_ROWS:
                raise bad
            return [3, most, int(round(float(conf) * 1e6)), 0]
        most = int(spec)
    except ValueError:
        raise bad from None
    if not 0 <= most < MAX_ROWS:
        raise bad
    return [1 if most > 0 else 0, most, 0, 0]


def _encode_noisy(spec: str, bad: ValueError) -> list[int]:
    """``ncN:P[:B]`` -> [16, N, P, B] and ``costN:noisy[:B]`` -> [17, N, B, 0], P and B in millionths (DFlash2 only)."""

    from .dflash2 import NOISY_BETA

    cost = spec.startswith("cost")
    parts = spec[4 if cost else 2:].split(":")
    if len(parts) not in (2, 3) or (cost and parts[1] != "noisy"):
        raise bad
    most = int(parts[0])
    beta = float(parts[2]) if len(parts) == 3 else NOISY_BETA
    if not 0 < most <= len(GRAPH_ROWS) - 1 or not 0 < beta <= 1000 or round(beta * 1e6) < 1:
        raise bad
    if cost:
        return [17, most, int(round(beta * 1e6)), 0]
    conf = float(parts[1])
    if not 0 <= conf <= 1:
        raise bad
    return [16, most, int(round(conf * 1e6)), int(round(beta * 1e6))]


def decode_policy(code: list[int], costs: dict | None = None):
    """Decode a serial, MTP, or automatic policy, retaining exploration, sampling, and margin settings; ``costs``: the
    startup cost table (``GlmEngine._calibrate``) the fcost rule weighs a round's depths by."""

    from .decode import DepthPolicy

    kind, most, a, b = code
    if kind in (4, 5):
        return ("auto", most, a, b / 1e6, kind == 5)
    kind %= 10
    if kind == 2:
        return DepthPolicy(min(most, MAX_ROWS - 1), low=a / 1e6, high=b / 1e6)
    if kind == 3:
        return DepthPolicy(min(most, MAX_ROWS - 1), fixed=True, confidence=a / 1e6)
    if kind == 6:
        return DepthPolicy(min(most, MAX_ROWS - 1), fixed=True, confidence=a / 1e6, beta=b / 1e6)
    if kind == 7:
        if costs is None:
            raise ValueError("the fcost policy needs the engine's cost table")
        return DepthPolicy(min(most, MAX_ROWS - 1), fixed=True, beta=a / 1e6, round_ms=round_costs(costs))
    return DepthPolicy(min(most, MAX_ROWS - 1), fixed=True) if kind == 1 else None


def round_costs(costs: dict) -> tuple[float, ...]:
    """ms of a DFlash2 round of k drafts, k = 0 .. len(verify) - 1: the verify window of k + 1 rows, the block and a
    kept row's context update, as draft_sim's Costs(verify, block, 0.0, taps_row).round_ms(k, 1, True)."""

    return tuple(v + costs["block"] + 0.0 + costs["taps_row"] * 1 for v in costs["verify"])


def _f64_ints(x: float) -> list[int]:
    return list(struct.unpack("<2i", struct.pack("<d", float(x))))


def _ints_f64(lo: int, hi: int) -> float:
    return struct.unpack("<d", struct.pack("<2i", lo, hi))[0]


MTP_DEFAULT = "1"                     # TF_GLM_MTP when unset: the MTP head stays beside DFlash2 (auto, 0: left out)


def mtp_head(drafter: bool, serial_only: bool, layers: int, value: str | None = None) -> bool:
    """TF_GLM_MTP: load the MTP head? auto: not beside DFlash2 or with --no-drafts; 1: whenever it exists; 0: never."""

    value = os.environ.get("TF_GLM_MTP", "") if value is None else value
    value = value.strip().lower() or MTP_DEFAULT
    if value not in ("0", "1", "auto"):
        raise ValueError(f"TF_GLM_MTP: 0, 1 or auto, not {value!r}")
    if value == "auto":
        return bool(layers) and not drafter and not serial_only
    return value == "1" and bool(layers)


PARALLEL_MOST = 8                     # --parallel: concurrent requests at most (Mia's 0069; segments.MAX_SEGS)
MULTI_WINDOW = 32                     # --parallel: rows of every stream's verify windows together (multi.MAX_WINDOW)


def multi_window(value: str | None = None) -> int:
    """TF_GLM_MULTI_WINDOW (Mia's 0069): --parallel's shared verify window, the rows of every stream's windows together
    in a round (the longest tails trimmed past it): 32 (default) or 16 to 64 in steps of 8. The batched window's
    buffers, token selection scratch and CUDA graphs (one a row count) are sized for it, and the startup estimate
    counts them (``multi_decode_rows``)."""

    value = (os.environ.get("TF_GLM_MULTI_WINDOW", "") if value is None else value).strip()
    if value == "":
        return MULTI_WINDOW
    if not value.isdecimal() or not 16 <= int(value) <= 64 or int(value) % 8:
        raise ValueError(f"TF_GLM_MULTI_WINDOW: 16 to 64 rows in steps of 8, not {value!r}")
    return int(value)


def multi_decode_rows(parallel: int, window: int, rows: int = MAX_ROWS) -> int:
    """Decode-window rows the startup estimate sizes buffers for: MLA_DECODE_ROWS (64), or under --parallel the
    engine's own window (``rows``: TF_GLM_MAX_ROWS) and the batched one (``window``) together when they take more
    (TF_GLM_MULTI_WINDOW past 48 at the default 16 rows)."""

    from tensorfold.cuda.geometry import MLA_DECODE_ROWS

    return max(MLA_DECODE_ROWS, int(rows) + int(window)) if parallel > 1 else MLA_DECODE_ROWS


def pool_streams(parallel: int, value: str | None = None) -> int:
    """TF_GLM_POOL_STREAMS: how many streams' full context windows the --parallel cache pool holds at once (default
    every stream, ``parallel``), so startup refuses a --context that N streams cannot all reach; a smaller value
    lets the streams share a smaller pool (the youngest given back to the queue when it is full, as before)."""

    if parallel <= 1:
        return 1
    value = (os.environ.get("TF_GLM_POOL_STREAMS", "") if value is None else value).strip()
    if value == "":
        return parallel
    if not value.isdecimal() or not 1 <= int(value) <= parallel:
        raise ValueError(f"TF_GLM_POOL_STREAMS: 1 to --parallel ({parallel}) streams' windows, not {value!r}")
    return int(value)


def pool_cache_bytes(t: dict, world: int, tokens: int, *, mtp: bool = False, kv: str = "bf16",
                     ring: bool = False) -> int:
    """The per-token caches of ``tokens`` pool tokens on a rank (mla_geometry's latent cache, indexer keys, gates and
    pooled keys; the latent cache, TF_GLM_LATENT's default). ``kv`` (TF_GLM_KV): fp8 shrinks the latent rows and the
    pooled keys to ``mla_row_bytes``; keys and gates stay bf16, per token, or with ``ring`` (TF_GLM_INDEX_RING) in
    per-slot rings outside the pool (``mla_ring_bytes``)."""

    from tensorfold.cuda.geometry import layer_counts, mla_row_bytes

    _, attention = layer_counts(t)
    count = attention + int(mtp)
    lw, index = int(t.get("kv_lora_rank", 512)), int(t.get("index_head_dim", 128))
    return (count * tokens * mla_row_bytes(lw, kv) + (0 if ring else count * 2 * tokens * index * 2)
            + count * (tokens // 4 + 2) * mla_row_bytes(index, kv))


def multi_geometry(t: dict, world: int, parallel: int, *, streams: int, window: int, latent: bool = True,
                   mtp: bool = False, kv: str = "bf16", ring: bool = False):
    """The startup geometry of the GLM engine under ``parallel`` streams (``Geometry.bytes_at(slots)``, slots: one
    stream's context window): ``mla_geometry`` (decode buffers for the engine's and the batched window), every slot
    past the first (``slot_bytes``), the pool's per-token caches for ``streams`` full windows (``pool_streams``) in
    whole 2,048-token extents, and the batched selection's fp32 scores over the pool; with ``ring`` (the index ring,
    TF_GLM_INDEX_RING) every slot past the first also holds its key and gate rings. Without parallel streams: the
    plain ``mla_geometry``."""

    from tensorfold.cuda.capacity import Geometry
    from tensorfold.cuda.geometry import PREFILL_ROWS, mla_geometry, mla_ring_bytes

    from .latent import SPARSE_ONEPASS
    from .pool import align_up

    # SPARSE_ONEPASS: prompt chunks' sparse attention allocates no partials (Mia's 0028)
    g = mla_geometry(t, world, MAX_ROWS, minimum_slots=DENSE_CAPACITY, latent=latent, mtp=mtp,
                     decode_rows=multi_decode_rows(parallel, window), kv=kv, ring=ring, onepass=SPARSE_ONEPASS)
    if parallel <= 1:
        return g
    extra = (parallel - 1) * (slot_bytes(t, world) + (mla_ring_bytes(t, PREFILL_ROWS, mtp=mtp) if ring else 0))

    def bytes_at(slots: int) -> int:
        pool = streams * align_up(slots)
        return (g.bytes_at(slots) + extra + pool_cache_bytes(t, world, pool, mtp=mtp, kv=kv, ring=ring)
                - pool_cache_bytes(t, world, slots, mtp=mtp, kv=kv, ring=ring) + window * pool)

    return Geometry(bytes_at, g.reserve, g.minimum_slots)


def slot_bytes(t: dict, world: int = 2, rows: int = MAX_ROWS) -> int:
    """What one more stream slot (``forward.Slots``) takes on each rank, from the text config: every KDA layer's
    recurrent states (two parities), conv window, a decode window's projections and replay scratch (Mia's 0030)."""

    from tensorfold.cuda.geometry import layer_counts

    linear, _ = layer_counts(t)
    lin = t.get("linear_attn_config") or {}
    lh = int(lin.get("num_heads", t.get("linear_num_heads", 64))) // world
    ld = int(lin.get("head_dim", t.get("linear_head_dim", 128)))
    conv = int(lin.get("short_conv_kernel_size", t.get("linear_conv_kernel_dim", 4)))
    width = 3 * lh * ld + 2 * ld + lh                  # the KDA input projection's rows (q, k, v, f_a, g_a, beta)
    scratch = rows * lh * (ld * 2 + ld * 4 + ld * 2 + ld * 4 + 4)
    return linear * (2 * lh * ld * ld * 4 + (conv - 1) * 3 * lh * ld * 2 + rows * width * 2 + scratch)


def draft_context_bytes(t: dict, world: int = 2, *, ring: bool, capacity: int) -> int:
    """One more DFlash2 context (``dflash2_multi.context``) on each rank: keys and values of every draft layer over
    its ring (or capacity + block rows)."""

    from tensorfold.cuda.geometry import draft_ring_rows

    layers = int(t["num_hidden_layers"])
    heads = int(t["num_key_value_heads"]) // world
    hd = int(t["head_dim"])
    block = int((t.get("dflash_config") or {}).get("block_size", 16))
    window = int(t.get("sliding_window", 0))
    ring_rows = draft_ring_rows(window - 1, block) if ring and window > 0 else 0
    cap = ring_rows if 0 < ring_rows < capacity + block else capacity + block
    return 2 * layers * heads * hd * cap * 2 + (1 << 20)


def without_mtp(transform, layers: int):
    """A startup weight transform that leaves out the MTP layer's tensors (``layers.<num_hidden_layers>.``)."""

    prefix = f"model.language_model.layers.{layers}."
    return lambda name, info: (0, 0) if name.startswith(prefix) else transform(name, info)


class GlmEngine:
    """GLM-5.3-Flash on two ranks (this one ``rank``): weights, MTP and DFlash2 drafting, per-request policies."""

    parallel = 1                # --parallel N: requests decoded together (``multi``); 1 serves one at a time
    concurrent = False          # the server hands every request straight to ``generate`` (the scheduler orders them)
    grid = 0                    # TF_GLM_KDA_CHUNKED: the prompt grid (``prompt_grid``)
    shared = 0                  # TF_GLM_SHARED_PREFIX: the fewest tokens a shared-prefix state adds (``shared_prefix``)
    opener = None               # rank 0 only: GLM's ``<|user|>`` id, the end of a prompt's system block
    multi = None                # --parallel > 1: the ``multi.MultiDecoder`` (both ranks)
    scheduler = None            # ... and rank 0's ``multi.GlmScheduler``
    spill = None                # --spill-gib, one request at a time: the ``spill.SpillStore`` (--parallel: the decoder's)

    def __init__(self, model_dir: Path, *, rank: int, master: str, port: int, policy: str = DEFAULT_POLICY,
                 drafter: Path | None = None, context: int | None = None, context_explicit: bool | None = None, serial_only: bool = False, comm=None,
                 prefill_rows: int | None = None, world: int = 2, parallel: int = 1,
                 multi_options: dict | None = None, split=None, kv: str | None = None, vision: bool = False,
                 vision_urls: bool = False, spill=None) -> None:
        """``comm``: a communicator with ``all_gather`` and ``barrier`` instead of NCCL between two machines (tests);
        ``split``: a ``hcsplit.SplitSettings`` instead of TF_GLM_HC_SPLIT / TF_GLM_PREFILL_OVERLAP (tests);
        ``parallel``: requests decoded together (``multi``: their caches share one pool, each stream up to the
        context window; DFlash2 drafts only); ``multi_options``: keyword arguments for ``multi.MultiDecoder``
        (tests); ``kv``: the DSA caches' format, bf16 or fp8 (``--kv-dtype``; None: TF_GLM_KV, bf16 unless set);
        ``vision``: rank 0 runs the image tower (the other ranks only take its rows); ``spill``: --spill-gib's
        ``tensorfold.cuda.spill.SpillConfig`` (the disk tier for kept prompt states, Mia's 0088), None: off."""

        import torch

        from tensorfold.cuda.comm import open_comm
        from .decode import Engine
        from .weights import Config, load
        from .split import geometry_rule
        from tensorfold.cuda.capacity import admit
        from tensorfold.cuda.geometry import (PREFILL_ROWS, dflash2_geometry, dflash2_weights, mla_geometry,
                                              split_weights)

        encode_policy(policy)                           # a bad default fails here, not in the first request
        if not encode_policy(DFLASH_POLICY)[0] >= 10:
            raise ValueError(f"TF_GLM_DFLASH_POLICY={DFLASH_POLICY!r}: a DFlash2 policy (f...), e.g. fc5:0.3 or "
                             "fnc7:0.3")
        from .copy_drafts import CopySettings

        # TF_GLM_COPY_DRAFTS=1: a reply that repeats earlier text drafts its continuation (prompt lookup) ahead of
        # the MTP head and DFlash2; both ranks must agree, so the settings join the startup comparison below
        self.copy = None if serial_only else CopySettings.from_env(MAX_ROWS - 1, pad=WIDE_GRAPHS)
        from .draft_dump import dump_dir

        # TF_GLM_DRAFT_DUMP=DIR (Mia's 0011): DFlash2 requests decode serially and record the drafter's view, one
        # .npz a request, for tools like draft_sim.py; the single-stream engine only
        self.dump = dump_dir() if drafter is not None and not serial_only and parallel == 1 else None
        torch.cuda.set_device(0)
        self.torch = torch
        self.rank = rank
        parallel = int(parallel)
        if not 1 <= parallel <= PARALLEL_MOST:
            raise ValueError(f"--parallel: 1 to {PARALLEL_MOST} requests at once for GLM-5.3-Flash, not {parallel}")
        self.parallel = parallel
        self.spill_cfg = spill                # --spill-gib: a ``spill.SpillConfig`` (every rank alike), else None
        if spill is not None:                 # its staging buffers pinned now, before the caches are sized
            from tensorfold.cuda.spill import preallocate

            preallocate(spill)
        # TF_GLM_MULTI_WINDOW / TF_GLM_POOL_STREAMS (--parallel): the round's rows and the pool's full windows
        self.multi_window = multi_window()
        self.pool_streams = pool_streams(parallel)
        if parallel > 1 and self.multi_window < parallel:
            raise ValueError(f"TF_GLM_MULTI_WINDOW={self.multi_window} rows cannot hold {parallel} streams' tokens")
        if parallel > 1 and MAX_ROWS > self.multi_window:
            raise ValueError(f"TF_GLM_MAX_ROWS={MAX_ROWS}: a stream's verify window cannot be wider than "
                             f"TF_GLM_MULTI_WINDOW={self.multi_window}, the round's rows under --parallel")
        self.policy = "0" if serial_only else policy
        self.serial_only = serial_only
        world_ranks = getattr(comm, "world", world) if comm is not None else world
        self.comm = comm if comm is not None else open_comm(rank, world_ranks, master, port)
        self.world = world_ranks
        from .hcsplit import SplitSettings

        # TF_GLM_HC_SPLIT=1 / TF_GLM_PREFILL_OVERLAP=1|2: prompt chunks' hyper-connection glue split by rows between
        # the ranks (and its exchanges overlapped); the same bits, both ranks must agree (the comparison below)
        self.split = SplitSettings.from_env() if split is None else split
        if self.split.split and world_ranks != 2:
            raise ValueError(f"TF_GLM_HC_SPLIT splits prompt rows between two ranks, not {world_ranks}")
        self.comm.barrier()
        cfg = Config.read(model_dir)
        # Without --context the window stays dense, attending every key without indexer work.
        explicit = context is not None if context_explicit is None else bool(context_explicit)
        from . import LATENT
        from .kv8 import KINDS as KV_KINDS, resolve as resolve_kv

        # TF_GLM_KV=fp8 / --kv-dtype fp8: the DSA latent cache and the indexer's pooled keys as e4m3 rows (``kv8``)
        kv = resolve_kv(kv)
        if kv != "bf16" and not LATENT:
            raise ValueError("TF_GLM_KV=fp8 holds the latent cache: it needs TF_GLM_LATENT=1")
        self.kv = kv
        # TF_GLM_INDEX_RING (auto: with fp8): the indexer's per-token keys and gates in a ring per stream slot
        from .forward import ring_enabled

        ring = ring_enabled(kv)
        self.index_ring = ring
        # TF_GLM_MTP off: the MTP layer's tensors, caches and buffers are neither loaded nor estimated
        self.mtp_on = mtp_head(drafter is not None, serial_only, cfg.mtp_layers) and parallel == 1   # multi: DFlash2
        weights_estimate = split_weights(geometry_rule, self.world)
        from .qmm import KVB_KINDS, kvb_kind, kvb_weights

        # TF_GLM_KVB=fp8/q4 (an EXL3 checkpoint, latent cache): kv_b as the absorb and expand kernels read it (Mia's
        # 0013; RED-SNOW's kv_b is F16, so it applies); lossy, off by default
        kvb = kvb_kind() if getattr(cfg, "quant", None) == "exl3" and LATENT else "bf16"
        weights_estimate = kvb_weights(weights_estimate, kvb)
        self.kvb = kvb
        if not self.mtp_on:
            weights_estimate = without_mtp(weights_estimate, cfg.layers)
        from tensorfold.cuda.capacity import Geometry

        def geometry(text):
            # --parallel: every stream slot past the first, the pool's full windows (TF_GLM_POOL_STREAMS) and the
            # batched window's buffers (``multi_geometry``); ``kv``: fp8 shrinks the latent cache and pooled keys
            return multi_geometry(text, self.world, parallel, streams=self.pool_streams, window=self.multi_window,
                                  latent=LATENT, mtp=self.mtp_on, kv=kv, ring=ring)

        def draft_geometry(text):
            g = dflash2_geometry(text, self.world, MAX_ROWS, ring=DRAFT_RING)
            if parallel == 1:
                return g
            # --parallel: a DFlash2 context per stream slot beside the solo drafter's
            return Geometry(lambda slots: g.bytes_at(slots) + parallel * draft_context_bytes(
                text, self.world, ring=DRAFT_RING, capacity=slots), g.reserve)

        vision = bool(vision) and rank == 0
        workspace = vision_workspace() if vision else 0
        if vision:                           # rank 0's tower weights and workspace shorten its window (--parallel too)
            weights_estimate, geometry = vision_admission(weights_estimate, geometry, workspace)
        try:
            self.capacity_plan = admit(model_dir, context if explicit else cfg.dense_limit, explicit, torch,
                                       geometry, weights_estimate, rank=rank, world=self.world,
                                       gather=self._gather_ints, draft_dir=drafter,
                                       draft_weights=lambda d: dflash2_weights(d, self.world),
                                       draft_geometry=draft_geometry)
        except ValueError as exc:
            if parallel <= 1:
                raise
            # both ranks refuse alike (admit compares them), before any weight loads
            raise ValueError(
                f"--parallel {parallel}: the cache pool must hold {self.pool_streams} streams' full context windows "
                f"(each up to --context, beside {parallel} stream slots and DFlash2 contexts), and it does not fit: "
                f"{exc} The window named there is a stream's. Lower --context or --parallel, or set "
                f"TF_GLM_POOL_STREAMS below {self.pool_streams} to let the streams share a smaller pool (the youngest "
                f"is then given back to the queue when it is full).") from exc
        self.limit = self.capacity_plan["context_window"]
        self.capacity_plan["kv_dtype"] = kv
        capacity = self.capacity_plan["cache_slots"]
        long_context = self.limit > cfg.dense_limit
        # both ranks must run the same calls: refuse to start when they were given different settings
        prefill_rows = PREFILL_ROWS if prefill_rows is None else int(prefill_rows)
        self.grid = prompt_grid(prefill_rows)
        # TF_GLM_SHARED_PREFIX: a drafted text request also keeps shared-prefix states (its system block and where it
        # stops sharing a kept prompt). Rank 0 alone reads the chat template for the user opener; the points it decides
        # ride in the request header to rank 1, which never reads the template.
        self.shared = shared_prefix()
        self.opener = user_opener(model_dir) if self.shared and rank == 0 else None
        from tensorfold.cuda.sampling import union_cover

        verify_code = 0
        tune_code = [0] * 7
        if parallel > 1:                      # serial and batched verify windows make different collectives
            from .multi import verify_kind
            from .multi_tune import MultiSettings

            verify_code = 1 + ("batched", "serial").index(verify_kind())
            tune_code = MultiSettings.from_env().code()     # Mia's 0035: sampler, depth, async, lone, profile
        draft_code = 0
        if parallel > 1 and drafter is not None:  # one batched DFlash2 pass a round or one a stream: other all-gathers
            from .dflash2_multi import DRAFT_KINDS, draft_kind

            draft_code = 1 + DRAFT_KINDS.index(draft_kind())
        from .sliced_fill import budget_ms

        fill_budget = _f64_ints(budget_ms()) if parallel > 1 else [0, 0]     # both ranks slice prompt chunks alike
        mine = [int(drafter is not None), capacity, int(long_context), int(serial_only), int(LATENT),
                prefill_rows, int(self.mtp_on), int(DRAFT_RING), parallel, verify_code, self.grid,
                self.multi_window if parallel > 1 else 0, self.pool_streams, KV_KINDS.index(kv), int(ring),
                KVB_KINDS.index(kvb), int(self.shared), len(cfg.eos), sum(cfg.eos)] \
            + fill_budget + self.split.code() + (self.copy.code() if self.copy is not None else [0, 0, 0, 0]) \
            + [int(self.dump is not None), MAX_ROWS, len(WIDE_GRAPHS), sum(WIDE_GRAPHS), max(WIDE_GRAPHS, default=0)] \
            + decode_kernel_code() + l2pf_code() + tune_code + [int(union_cover())]
        # other conversations' kept prompts get what the window leaves, at most TF_GLM_CACHE_GIB, the same on both ranks
        plan = self.capacity_plan
        wanted = int(float(os.environ.get("TF_GLM_CACHE_GIB", "3")) * 2 ** 30)
        spare = max(0, min(wanted, plan["budget_bytes"] - plan["total_bytes_estimate"]))
        self.agreed = mine[:1] + mine[2:]     # the spill tier's folders are keyed on these (not the window's slots)
        # TF_GLM_MULTI_DRAFT is compared but keys no folder: it changes no kept state's bits
        both = self._gather_ints(mine + [draft_code, spare >> 20])
        if both[0][:-1] != both[1][:-1]:
            raise RuntimeError("the two ranks were started with different settings (draft model, context, drafts, "
                               "TF_GLM_LATENT, TF_GLM_MTP, TF_GLM_DRAFT_RING, --parallel, TF_GLM_MULTI_VERIFY, "
                               "TF_GLM_MULTI_WINDOW, TF_GLM_POOL_STREAMS, TF_GLM_KV / --kv-dtype, TF_GLM_INDEX_RING, "
                               "TF_GLM_KVB, TF_GLM_SHARED_PREFIX, TF_GLM_ASSISTANT_ENDS, "
                               "TF_GLM_FILL_BUDGET_MS, TF_GLM_KDA_CHUNKED, TF_GLM_HC_SPLIT*, TF_GLM_PREFILL_OVERLAP, "
                               "TF_GLM_OVERLAP_PIECES, TF_GLM_HC_EXCHANGE, TF_GLM_COPY_*, TF_GLM_DRAFT_DUMP, "
                               "TF_GLM_MAX_ROWS, TF_GLM_WIDE_GRAPHS, TF_GLM_HC_DEC, TF_GLM_KDA_WIDE_ROWS, "
                               "TF_GLM_LATENT_STAGES, TF_GLM_COMBINE_SY, TF_GLM_HC_MMA, TF_GLM_L2PF*, "
                               "TF_GLM_MULTI_SAMPLER / _DEPTH / _OVERHEAD_MS / _ASYNC / _LONE / _PROFILE / _DRAFT, "
                               "TENSORFOLD_NUCLEUS_UNION): "
                               f"rank 0 {both[0][:-1]}, rank 1 {both[1][:-1]}; pull the draft model on both machines "
                               "(or pass --drafter none to both) and give both the same flags")
        self.cache_bytes = min(both[0][-1], both[1][-1]) << 20
        sp = spill                            # --spill-gib: every rank takes the tier's decisions alike, on these
        tiers = self._gather_ints([1, int(sp.gib * 1024), sp.min_tokens, int(sp.min_free_gib * 1024)]
                                  if sp is not None else [0] * 4)
        if any(list(r) != list(tiers[0]) for r in tiers[1:]):
            raise RuntimeError("the ranks were started with different spill settings (--spill-gib, --spill-min-tokens, "
                               "--spill-min-free-gib): " + ", ".join(f"rank {r} {list(v)}" for r, v in enumerate(tiers)))
        plan["kept_bytes"] = self.cache_bytes
        for key in ("serving_peak_bytes_estimate", "total_bytes_estimate"):
            plan[key] = plan[key] + self.cache_bytes
        # --parallel: one pool of per-token caches for every stream and kept prompt: the window's slots, plus what
        # TF_GLM_CACHE_GIB gives kept prompts (their rows in the pool, their KDA states and DFlash2 windows beside
        # it), whole 2,048-token extents; each stream holds up to the window (pool.py, multi.py)
        pool_rows = None
        if parallel > 1:
            from tensorfold.cuda.capacity import config as read_config

            from .pool import ALIGN

            text = read_config(model_dir)
            per_token = max(1, -(-pool_cache_bytes(text, self.world, 65536, kv=kv, ring=ring) // 65536)
                            + self.multi_window)
            # a kept prompt's KDA state (one parity) and DFlash2 window live beside the pool
            window = 0
            if drafter is not None:
                window = draft_context_bytes(read_config(drafter), self.world, ring=True, capacity=capacity)
            # one kept state's fixed cost outside the pool: its KDA state (one parity) and its DFlash2 window copy
            state = slot_bytes(text, self.world, rows=0) // 2 + window
            configured = self.cache_entries_setting()
            # TF_GLM_CACHE_SHARE_PCT (Mia's 0097, 0: off): the count follows a share of the pool's budget, cut to
            # three quarters of it. Off, the configured count stands as before (Mia cuts it to the three quarters
            # either way; at RED-SNOW's KEPT_GIB=2 that is 16 of 32, so here only when the share is asked for)
            share = float(os.environ.get("TF_GLM_CACHE_SHARE_PCT", "0") or 0)
            self.entries = configured
            if share > 0:
                self.entries, why = kept_entries(configured, share, self.cache_bytes, state)
                if rank == 0 and why:
                    print(f"[tensorfold] kept prompt states: {why}", flush=True)
            states = self.entries * state
            pool_rows = (self.pool_streams * (-(-capacity // ALIGN) * ALIGN)
                         + max(0, self.cache_bytes - states) // per_token // ALIGN * ALIGN)
            if pool_rows < ALIGN:
                raise ValueError(f"--parallel {parallel}: a pool of {pool_rows} tokens is less than one {ALIGN}-token "
                                 f"extent; free memory or lower --context")
            self.limit = min(self.limit, pool_rows - MAX_ROWS)
            self.cache_bytes = 0                 # kept prompts live in the pool
            plan["pool_tokens"] = pool_rows
        if rank == 0 and self.cache_bytes < wanted and parallel == 1:
            print(f"[tensorfold] other conversations' prompts are kept in {self.cache_bytes / 2 ** 30:.1f} GiB, what "
                  f"the {self.limit}-token window leaves (TF_GLM_CACHE_GIB asks {wanted / 2 ** 30:.1f})", flush=True)
        if not self.mtp_on and drafter is None and not serial_only and parallel == 1:
            raise ValueError(("TF_GLM_MTP=0 leaves" if cfg.mtp_layers else "this checkpoint has") + " no MTP head and "
                             "no DFlash2 draft model was given, so every round would decode one token: pull the draft "
                             "model on both machines (--drafter), or pass --no-drafts to both for the serial reference")
        if self.split.split and self.split.exchange == "p2p" and hasattr(self.comm, "exchange"):
            # both ranks (Mia's 0064): open the split's send/receive connection before the weights and the cache pool
            # take the memory; opened later, NCCL's buffer registration could fail with ENOMEM (ibv_reg_mr, her
            # issue #36). The connection stays open; nothing it sends changes
            a = torch.zeros((1,), dtype=torch.float32, device="cuda")
            self.comm.exchange([a], [torch.empty_like(a)], 1 - rank)
            torch.cuda.synchronize()
        w = load(model_dir, rank=rank, mtp=self.mtp_on, world=self.world)
        w.comm = self.comm
        self.comm.ready("loading")                   # a peer stuck loading is named, not waited on in NCCL
        self.comm.barrier()
        self.w = w
        if rank == 0 and cfg.mtp_layers and not self.mtp_on:
            print("[tensorfold] the checkpoint's MTP head is not loaded (TF_GLM_MTP=" +
                  (os.environ.get("TF_GLM_MTP", "").strip() or MTP_DEFAULT) + "): " +
                  ("DFlash2 drafts every request" if drafter is not None else "--no-drafts"), flush=True)
        self.drafter = None
        self.drafter_dir = drafter
        if drafter is not None:
            from .dflash2 import Drafter

            self.drafter = Drafter(drafter, w, capacity=pool_rows or capacity, tap_rows=MAX_ROWS, ring=DRAFT_RING)
        self.e = Engine(w, capacity=pool_rows or capacity, max_rows=MAX_ROWS, prefill_rows=prefill_rows, graphs=True,
                        graph_rows=GRAPH_ROWS + WIDE_GRAPHS, long_context=long_context,
                        taps=self.drafter.tap_layers if self.drafter is not None else (), streams=parallel,
                        pool_rows=pool_rows, split=self.split, kv=kv, ring=ring, mtp_rows=MTP_ROWS, grid=self.grid)
        self.capacity_plan["index_ring"] = self.e.caches.ring or 0
        if rank == 0 and kv != "bf16":
            print("[tensorfold] TF_GLM_KV=fp8: the DSA latent cache and the indexer's pooled keys as e4m3 rows with a "
                  f"power-of-two scale each (lossy; drafted replies still equal serial ones), a {self.limit}-token "
                  "window", flush=True)
        if rank == 0 and self.e.caches.ring is not None:
            print(f"[tensorfold] TF_GLM_INDEX_RING: the indexer's keys and gates in a {self.e.caches.ring}-row ring per "
                  f"stream slot ({self.e.caches.ring_bytes() / 2 ** 20:.1f} MiB), not {self.limit}-token planes",
                  flush=True)
        if rank == 0 and self.grid:
            print(f"[tensorfold] TF_GLM_KDA_CHUNKED=1: KDA prompt chunks in chunked (WY) form, 32-row sub-chunks "
                  f"(close to the serial kernel, not its bits); prompt states kept on a {self.grid}-token grid",
                  flush=True)
        if self.e.pbuf.split is not None:
            self.e.pbuf.split.warm()                 # both ranks: the send/receive connection (already open: 0064)
            if rank == 0:
                print(f"[tensorfold] {self.split.describe()}", flush=True)
        c = self.copy
        if rank == 0 and (WIDE_GRAPHS or MAX_ROWS != 16 or (c is not None and (c.reply_match or c.miss_most))):
            print(f"[tensorfold] verify windows up to {MAX_ROWS} rows, CUDA graphs for "
                  f"{', '.join(map(str, GRAPH_ROWS + WIDE_GRAPHS))} rows (TF_GLM_MAX_ROWS, TF_GLM_WIDE_GRAPHS)"
                  + (f"; copies from the reply match {c.reply_match} tokens (TF_GLM_COPY_REPLY_MATCH)"
                     if c is not None and c.reply_match else "")
                  + (f"; {c.miss_most} copied drafts after a copied round that missed (TF_GLM_COPY_MISS_MAX)"
                     if c is not None and c.miss_most else ""), flush=True)
        if rank == 0 and type(self.comm).__name__ == "RoceComm":
            from tensorfold.cuda.roce import max_bytes_from_env

            # a window's widest all-gather: 16 KiB a row (Mia's 0032); past TF_ROCE_MAX_KB it goes through NCCL
            if MAX_ROWS * 16 > max_bytes_from_env() >> 10:
                print(f"[tensorfold] TF_GLM_MAX_ROWS={MAX_ROWS}: windows past TF_ROCE_MAX_KB "
                      f"({max_bytes_from_env() >> 10} KiB) go through NCCL; TF_ROCE_MAX_KB={MAX_ROWS * 16} keeps them "
                      "on RoCE", flush=True)
        if rank == 0 and self.kvb != "bf16":
            print(f"[tensorfold] TF_GLM_KVB={self.kvb}: kv_b_proj for the latent absorb and expand as "
                  f"{'FP8 e4m3, a scale per 128 columns' if self.kvb == 'fp8' else 'MSE-searched 4-bit groups of 64'} "
                  "(lossy; drafted replies still equal serial ones)", flush=True)
        if rank == 0 and self.drafter is not None:
            print(f"[tensorfold] DFlash2 policy of requests that name none: {DFLASH_POLICY} (TF_GLM_DFLASH_POLICY)",
                  flush=True)
        if rank == 0 and self.dump is not None:
            print(f"[tensorfold] TF_GLM_DRAFT_DUMP: DFlash2 requests decode serially and record the drafter's view into "
                  f"{self.dump} (one .npz a request; replies equal draft: false)", flush=True)
        if self.drafter is not None:
            self.drafter.capture()
        self.costs = self._calibrate()
        if rank == 0:
            c = self.costs
            print(f"[tensorfold] drafter timings (ms, fastest of 7): {c['timed']}", flush=True)
            mtp = (f"; MTP draft {c['mtp']:.2f} (+{c['mtp_step']:.2f} a chained draft, +{c['mtp_row']:.2f} a row)"
                   if w.mtp is not None else "")
            print("[tensorfold] drafter costs (ms): verify " + " ".join(f"{v:.1f}" for v in c["verify"]) + mtp +
                  f"; DFlash2 block {c['block']:.2f} (+{c['taps_row']:.3f} a tap row)", flush=True)
        self.vision = None                   # rank 0's image tower (``tensorfold.vision.glm.GlmVision``), --vision
        if vision:
            from tensorfold.vision.glm import GlmVision

            self.vision = GlmVision(Path(model_dir), torch.device("cuda", 0), allow_urls=vision_urls)
            self.vision.warm()
            torch.cuda.empty_cache()
            lim = self.vision.limits
            print(f"[tensorfold] vision: image and video input, a {self.vision.weight_bytes / 2**30:.2f} GiB tower "
                  f"with {workspace / 2**30:.2f} GiB of workspace; at most {lim.image_tokens} tokens a picture, "
                  f"{lim.video_tokens} a clip ({lim.video_frames} frames at {lim.fps:g} a second)"
                  f"{'; https URLs allowed' if vision_urls else ''}", flush=True)
        self.eos = tuple(w.cfg.eos)
        self.model_dir = Path(model_dir)
        self.request = threading.local()    # the calling request's policy and stop-at-EOS (``app.GlmApp``)
        # kept conversations (decode.Snapshot, least recently used first) and the live caches' ids; states and saved rows stay within cache_bytes
        self.cache: list = []
        self.live: list[int] = []
        self.cache_entries = getattr(self, "entries", None) or self.cache_entries_setting()   # (--parallel: 0097's)
        # --spill-gib, one request at a time (--parallel: the decoder's own tier, ``multi.MultiDecoder._spill_store``)
        self.spill = self._spill_store() if spill is not None and parallel == 1 else None
        if parallel > 1:
            from .multi import GlmScheduler, MultiDecoder

            self.multi = MultiDecoder(self, parallel, **(multi_options or {}))
            if rank == 0:
                self.scheduler = GlmScheduler(self.multi, max_streams=parallel)
                self.concurrent = True
                print(f"[tensorfold] --parallel {parallel}: up to {parallel} requests decode together, their caches "
                      f"in one pool of {pool_rows} tokens (each up to {self.limit}); "
                      f"{'DFlash2 drafts' if self.drafter is not None else 'one token a round (no draft model)'}, "
                      f"prompt chunks of {self.multi.fill_rows} rows while others decode (TF_GLM_FILL_ROWS, "
                      f"TF_GLM_FILL_SHARE {self.multi.share:g}), {type(self.multi.verify).__name__}"
                      f"{'; waiting prompts share prompt chunks (TF_GLM_MULTI_PREFILL)' if self.multi.group else ''}"
                      f"{f'; prompt layer slices target {self.multi.fill_budget:g} ms (TF_GLM_FILL_BUDGET_MS)' if self.multi.fill_budget else ''}"
                      f"; {self.multi.tune.describe()}", flush=True)
                if self.multi.row_ms is not None:
                    print("[tensorfold] --parallel batched window ms by rows: " +
                          " ".join(f"{i + 1}:{ms:.1f}" for i, ms in enumerate(self.multi.row_ms)), flush=True)
        from tensorfold.cuda.comm import settle

        settle(self.comm)       # startup is over on both ranks: TF_COMM_BACKEND=roce gathers go without a barrier first
        _trim_host_heap()       # the loaders' freed host buffers back to the OS (unified memory: they are the GPU's too)

    @staticmethod
    def cache_entries_setting() -> int:
        return int(os.environ.get("TF_GLM_CACHE_ENTRIES", "8"))

    def _calibrate(self) -> dict:
        """Per-piece ms for ``drafter_choice.DrafterChoice``: fastest of interleaved passes, equal on both ranks."""

        import statistics

        import numpy as np

        from .decode import draft, prefill

        torch = self.torch
        e, st = self.e, self.e.st
        rng = np.random.default_rng(0)
        vocab = self.w.cfg.vocab

        def tokens(n: int) -> list[int]:
            return [int(t) for t in rng.integers(0, vocab, n)]

        prefill(e, tokens(64), None, mtp=True, drafter=self.drafter)
        hidden = e.pbuf.fnormed[:8].clone()                 # rows for timing the draft steps
        one, six = tokens(1), tokens(6)
        start = st.mtp_len

        def rewind() -> None:
            st.set_mtp_len(start)
            st.mtp_drafted = 0

        # the verify windows the drafters' rounds use (MTP chains of 3, DFlash2 blocks of 5: the captured graphs);
        # wider (copied) windows are not the choice's, and their eager launches would bend the fitted line
        widest = max(GRAPH_ROWS)
        pieces: dict[str, tuple] = {f"v{r}": (lambda w=tokens(r): e.forward(w), None) for r in range(1, widest + 1)}
        if self.w.mtp is not None:
            pieces["m1"] = (lambda: draft(e, hidden[:1], one, st.pos + 1, 1, None), rewind)
            pieces["m3"] = (lambda: draft(e, hidden[:1], one, st.pos + 1, 3, None), rewind)
            pieces["m6"] = (lambda: draft(e, hidden[:6], six, st.pos + 1, 1, None), rewind)
        if self.drafter is not None:
            d = self.drafter
            taps = e.tap_rows(8, e.pbuf).clone()
            ctx = d.context_end

            def back() -> None:
                if d.context_end != ctx:
                    d.pos_dev.sub_(d.context_end - ctx)
                    d.context_end = ctx

            pieces["block"] = (lambda: d.propose(one[0], 5, None, 0.0), None)
            pieces["taps8"] = (lambda: d.add_taps(taps), back)
        best = {name: float("inf") for name in pieces}
        for turn in range(9):
            for name, (fn, prep) in pieces.items():
                if prep is not None:
                    prep()
                torch.cuda.synchronize()
                t = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                if turn >= 2:
                    best[name] = min(best[name], (time.perf_counter() - t) * 1e3)
            rewind()
            if self.drafter is not None:
                back()
        names = list(best)
        mine = torch.tensor([best[n] for n in names], dtype=torch.float32, device="cuda")
        world = self.world
        got = torch.empty((world * mine.numel(),), dtype=torch.float32, device="cuda")
        self.comm.all_gather(mine, got)
        per = got.view(world, -1)
        both = {name: per[:, j].max().item() for j, name in enumerate(names)}
        e.reset()
        if self.drafter is not None:
            self.drafter.reset()
        rows = list(range(2, widest + 1))
        ys = [both[f"v{r}"] for r in rows]
        slope = statistics.median((ys[j] - ys[i]) / (rows[j] - rows[i]) for i in range(len(rows))
                                  for j in range(i + 1, len(rows)))
        base = statistics.median(y - slope * r for r, y in zip(rows, ys))
        verify = [both["v1"]] + [base + slope * r for r in rows]
        mtp = both.get("m1", 0.0)
        return {"verify": verify, "mtp": mtp, "mtp_step": max((both.get("m3", 0.0) - mtp) / 2, 0.0),
                "mtp_row": max((both.get("m6", 0.0) - mtp) / 5, 0.0), "block": both.get("block", 0.0),
                "taps_row": max(both.get("taps8", 0.0) / 8, 0.0), "timed": {k: round(v, 2) for k, v in both.items()}}

    def _gather_ints(self, values: list[int]) -> list[list[int]]:
        torch = self.torch
        mine = torch.tensor(values, dtype=torch.int32, device="cuda")
        world = self.world
        got = torch.empty((world * len(values),), dtype=torch.int32, device="cuda")
        self.comm.all_gather(mine, got)
        return [got[i * len(values):(i + 1) * len(values)].tolist() for i in range(world)]

    # the idle doorbell: rank 1 waits for each request on the rendezvous store (no store: no doorbell), not in NCCL
    def _store(self):
        return getattr(self.comm, "store", None)

    def _ring(self) -> None:
        store = self._store()
        if store is not None:
            self._bell = getattr(self, "_bell", 0) + 1
            store.set(f"tf_glm_request_{self._bell}", b"1")

    def _await_bell(self) -> None:
        store = self._store()
        if store is None:
            return
        from datetime import timedelta

        key = f"tf_glm_request_{getattr(self, '_bell', 0) + 1}"
        while True:
            try:
                store.wait([key], timedelta(hours=1))
                break
            except Exception as e:            # an idle hour: wait again (a lost rank 0 is a connection error instead)
                if "timeout" not in str(e).lower():
                    raise
        store.delete_key(key)
        self._bell = getattr(self, "_bell", 0) + 1

    def _share(self, values: list[int] | None) -> list[int]:
        """Rank 0's int list on every rank (a length, then the values, through the all-gather)."""

        torch = self.torch
        n = torch.tensor([len(values) if self.rank == 0 else 0], dtype=torch.int32, device="cuda")
        world = self.world
        got = torch.empty((world,), dtype=torch.int32, device="cuda")
        self.comm.all_gather(n, got)
        count = int(got[0].item())
        buf = (torch.tensor(values, dtype=torch.int32, device="cuda") if self.rank == 0
               else torch.zeros((count,), dtype=torch.int32, device="cuda"))
        allv = torch.empty((world * count,), dtype=torch.int32, device="cuda")
        self.comm.all_gather(buf, allv)
        return [int(v) for v in allv[:count].tolist()]

    def _effective(self, code: list[int]) -> list[int]:
        """Resolve auto and MTP policies to the available heads, using EXL3_AUTO for EXL3 with DFlash2 and DFlash2 when MTP is absent."""

        if code[0] == 4 and self.drafter is not None and self.w.cfg.quant == "exl3":
            return encode_policy(EXL3_AUTO)
        if self.w.mtp is None and code[0] in (1, 2, 3, 4, 5):
            return encode_policy(DFLASH_POLICY) if code[0] in (4, 5) else [code[0] + 10] + code[1:]
        return code

    def _drafters(self, code: list[int]) -> tuple[bool, bool, bool]:
        """(auto, MTP drafts, DFlash2 drafts) for a policy code."""

        auto = code[0] in (4, 5)
        dflash = (auto or code[0] // 10 == 1) and self.drafter is not None
        return auto, auto or not dflash, dflash

    def _resume(self, key: list[int], code: list[int]):
        """The longest snapshot of a strict prefix of ``key`` (a prompt's resume key: its ids, or an image prompt's
        ids with each picture's content key in its rows, Mia's 0054) whose draft caches fit the request's drafters,
        or of the whole prompt when it kept its head's logits row (a replay of an earlier prompt, ``decode.whole``)."""

        _, mtp, dflash = self._drafters(code)
        best = None
        for snap in self.cache:
            fits = (not dflash or snap.drafter_end == len(snap.ids)) and (not mtp or snap.mtp_len >= 0)
            fits = fits and not (self.grid and len(snap.ids) % self.grid)     # prompt chunks start on the grid
            short = len(snap.ids) < len(key) or (len(snap.ids) == len(key) and snap.head is not None)
            if fits and short and snap.key is not None and key[:len(snap.ids)] == snap.key and (
                    best is None or len(snap.ids) > len(best.ids)):
                best = snap
        return best

    def _drop(self, snap) -> None:
        """Forget a kept snapshot and free its saved rows now, even while a caller still holds the object (with the
        spill tier on, written there first)."""
        if self.spill is not None:
            self._spill_snap(snap, leaving=True)
        snap.rows, snap.nbytes, snap.drafter_rows = None, 0, None
        self.cache.remove(snap)

    # -- the spill tier, one request at a time (``--spill-gib``; cuda/spill.py, Mia's 0088). The kept
    # ``decode.Snapshot`` goes through ``spill.encode``; ``_spill_rows`` gives a snapshot whose rows still live in the
    # caches views of them, and the caches wait for those copies (``release``) before they change. Every call runs on
    # every rank in the same order (a save and a read each agree through one all-gather); reads run on this thread.
    _SPILL_SKIP = ("ids", "ids_array", "nbytes", "key", "kid", "extent", "images", "shared")

    def _spill_store(self):
        from tensorfold.cuda.spill import SpillStore, build_id, weights_id

        from .decode import _row_views

        shapes = [(tuple(v.shape[1:]), str(v.dtype)) for v in _row_views(self.e.st, 1, 0)]
        # what the stored bits depend on: build and runtime, weights and drafter, the KV and kv_b formats, the index
        # ring, TP, the cache rows' shapes, and every setting the ranks agreed at start (``agreed``: drafts, grid,
        # verify rows, copy drafts, ...)
        sig = "|".join(["single", build_id(self.model_dir, self.drafter_dir), f"kv={self.kv}", f"kvb={self.kvb}",
                        f"grid={self.grid}", f"ring={self.e.caches.ring}", f"tp{self.world}", repr(shapes),
                        repr(self.agreed)])
        self.spill_jobs: list = []
        return SpillStore(self.spill_cfg, rank=self.rank, world=self.world, device=self.w.device, signature=sig,
                          weights=weights_id(self.model_dir, self.drafter_dir),
                          classes=["tensorfold.families.glm5_next.cuda.decode:Snapshot"],
                          model_id=os.path.basename(str(self.model_dir)), share=self._share,
                          gather=self._gather_ints)

    def _spill_rows(self, snap) -> None:
        """Materialize: a snapshot without saved rows gets views of the live caches it owns (``_saved_views``)."""

        from .decode import _saved_views

        if snap.rows is None:
            snap.rows = _saved_views(self.e, snap, self._kept_drafter())
            snap._viewed = True

    def _spill_wait(self) -> None:
        """Every pending write's copy off the device ends before the live caches change (the next request)."""

        for job in getattr(self, "spill_jobs", []):
            self.spill.release(job)
        self.spill_jobs = []

    def _unwritable(self, snap) -> str | None:
        """Why a written copy of a kept snapshot could not be resumed (None: it could): pictures, rows gone, off the
        prompt grid, or without the DFlash2 window a drafting request needs (``decode.prefill``'s fit)."""

        n = len(snap.ids)
        if getattr(snap, "images", False):          # one stream resumes no state with pictures, from memory either
            return "images"
        if snap.rows is None and self.live[:n] != list(snap.ids):
            return "no_rows"
        if self.grid and n % self.grid:
            return "off_grid"
        if self.drafter is not None:
            ring = getattr(self.drafter, "ring", 0)
            if snap.drafter_end != n or (snap.drafter_rows is None if ring else self._kept_drafter() is None):
                return "no_draft_window"
        return None

    def _superseded(self, snap) -> bool:
        """A longer kept snapshot of the same conversation, stored or writable itself, holds this one's tokens (never
        a shared-prefix state's: other conversations resume from it, not from the longer one)."""

        if getattr(snap, "shared", False):
            return False
        n = len(snap.ids)
        return any(c is not snap and len(c.ids) > n and c.ids[:n] == snap.ids
                   and (self.spill.has(c.ids) or self._unwritable(c) is None) for c in self.cache)

    def _spill_snap(self, snap, leaving: bool = False, keep: frozenset = frozenset()) -> None:
        """Every rank alike: write a kept snapshot whose rows are at hand (saved, or still the live caches').
        ``leaving``: it is being dropped, and a skip counts as unwritten (``unwritten_<reason>``). ``keep``: stored
        entries no room is made from."""

        n = len(snap.ids)
        if n < self.spill.cfg.min_tokens or self.spill.has(snap.ids):
            return
        why = self._unwritable(snap) or ("superseded" if self._superseded(snap) else None)
        if why is not None:
            if leaving:
                st = self.spill.stats
                st.unwritten_evicted += 1
                st.unwritten[why] = st.unwritten.get(why, 0) + 1
            return
        viewed = False
        try:
            job = self.spill.save(snap.ids, snap, (), skip=self._SPILL_SKIP, materialize=self._spill_rows,
                                  head=getattr(snap, "head", None) is not None, keep=keep)
        except Exception as exc:                  # noqa: BLE001  (the prompt is only lost)
            print(f"[tensorfold] spill rank {self.rank}: not written: {exc!r}", flush=True)
            job = None
        finally:
            if getattr(snap, "_viewed", False):   # the views go with the live caches: not kept on the snapshot
                snap.rows = None
                del snap._viewed
                viewed = True
        if job is not None and viewed:            # saved rows stay alive in the job; live caches must wait for it
            self.spill_jobs.append(job)

    def _fit(self, n: int, drafter_end: int, mtp_len: int, code: list[int]) -> bool:
        """``_resume``'s test of a resume point's draft caches against a request's drafters (a stored one's too)."""

        _, mtp, dflash = self._drafters(code)
        return (not dflash or drafter_end == n) and (not mtp or mtp_len >= 0)

    def _spill_load(self, ids: list[int], code: list[int]):
        """Every rank: read the stored prompt ``ids`` into a kept snapshot with saved rows; one all-gather agrees every
        read worked and fits the request's drafters (``_resume``'s test), else every rank prefills from scratch
        (None) and the entry goes."""

        from tensorfold.cuda.spill import ids_key

        key, ok, fit, snap, meta = ids_key(ids), 0, 0, None, None
        try:
            snap, _, meta = self.spill.load(key)
            ok = int(snap is not None and int(meta["n"]) == len(ids))
            if ok:
                fit = int(self._fit(len(ids), snap.drafter_end, snap.mtp_len, code))
        except Exception as exc:                  # noqa: BLE001  (a failed read: every rank prefills)
            print(f"[tensorfold] spill rank {self.rank}: restore failed: {exc!r}", flush=True)
        every = self._gather_ints([ok, fit])
        if not all(r[0] and r[1] for r in every):
            if all(r[0] for r in every):
                self.spill.stats.load_unfit += 1
            else:
                self.spill.stats.load_failed += 1
            self.spill.forget(key)
            return None
        snap.ids = list(ids)
        snap.key, snap.images = list(ids), False      # a text prompt's state: its resume key is its ids (0054)
        snap.nbytes = sum(r.numel() * r.element_size() for r in (snap.rows or []))
        self._remember(snap)
        return snap

    def flush_spill(self, budget: float | None = None) -> None:
        """Rank 0, a clean shutdown between requests: the kept snapshots not on disk, newest first, as many as the
        budget allows; the other ranks are told through a request header with max_tokens 0 and the list, and write
        the same."""

        if self.spill is None:
            return
        budget = self.spill.cfg.flush_s if budget is None else budget
        st = self.spill.stats
        rate = (st.written_bytes / st.write_s if st.write_s > 1 and st.written_bytes else 0.5 * 2 ** 30)
        rate *= self.spill.cfg.writers
        from .decode import row_bytes, snapshot_bytes

        picks, t = [], 0.0
        for i in range(len(self.cache) - 1, -1, -1):
            snap = self.cache[i]
            if (len(snap.ids) < self.spill.cfg.min_tokens or self.spill.has(snap.ids) or self._unwritable(snap)
                    or self._superseded(snap)):
                continue
            nb = snapshot_bytes(snap) + (row_bytes(self.e, snap, self._kept_drafter()) if snap.rows is None else 0)
            t += nb / rate
            if t > 0.8 * budget:
                break
            picks.append(i)
        if self.world > 1:
            self._ring()
            self._share([0] * 17 + [0] * SHARED_MOST)     # max_tokens 0: a flush (a chat's is at least 1)
            self._share(picks)
        self._do_flush(picks, budget)

    def _do_flush(self, picks: list[int], budget: float) -> None:
        t = time.perf_counter()
        from tensorfold.cuda.spill import ids_key

        done: list = []
        for i in picks:                          # newest first: the flush's own earlier (newer) picks stay
            self._spill_snap(self.cache[i], keep=frozenset(ids_key(c.ids) for c in done))
            done.append(self.cache[i])
        ok = self.spill.drain(max(1.0, budget - (time.perf_counter() - t)))
        print(f"[tensorfold] spill rank {self.rank}: flushed {len(picks)} kept prompts in "
              f"{time.perf_counter() - t:.1f}s{'' if ok else ' (not finished)'}", flush=True)

    def _remember(self, snap) -> None:
        # the same ids mean the same prompt, unless pictures are in it (same placeholder ids, different content: 0054)
        for c in [c for c in self.cache if c.ids == snap.ids and c is not snap and not (c.images or snap.images)]:
            self._drop(c)
        self.cache[:] = [c for c in self.cache if c is not snap] + [snap]   # a resumed prompt kept again moves last
        dropped = False
        while len(self.cache) > 1 and (len(self.cache) > self.cache_entries or self._held_bytes() > self.cache_bytes):
            self._drop(self.cache[0])
            dropped = True
        if dropped:
            import torch

            torch.cuda.empty_cache()

    def _take_over(self, keep: list[int]) -> None:
        """Save the rows of every kept snapshot the next prefill overwrites, dropping the oldest entries past the
        memory budget; both ranks decide alike.

        Decided first, copied after (Mia's 0078, issue #421): which snapshots stay within the budget is worked out
        from their sizes alone, the rest are dropped without a copy, and only the survivors are cloned. Cloning each
        in turn and dropping it for the next grew the allocator's pool with every larger copy (reserved memory far
        past the budget)."""
        from .decode import row_bytes, save_rows, snapshot_bytes

        drafter = self._kept_drafter()
        live = self.live
        order = list(self.cache)

        def resumes(c) -> bool:
            return len(c.ids) <= len(keep) and keep[:len(c.ids)] == c.ids

        alive = {id(c) for c in order}
        saved: dict[int, int] = {}                   # a survivor's held bytes once its rows are saved
        drops: list = []
        over_budget = False                          # a drop made for the budget (not a row already gone)

        def held() -> int:
            return sum(saved[id(c)] if id(c) in saved else snapshot_bytes(c) for c in order if id(c) in alive)

        def forget(c) -> None:
            alive.discard(id(c))
            saved.pop(id(c), None)
            drops.append(c)

        for snap in order:                           # the walk of saving one by one, sizes standing in for the copies
            n = len(snap.ids)
            if id(snap) not in alive or snap.rows is not None or resumes(snap):
                continue
            if live[:n] != snap.ids:                  # its rows are already gone: nothing to resume from
                forget(snap)
                continue
            need = row_bytes(self.e, snap, drafter=drafter)
            while held() + need > self.cache_bytes:
                old = next((c for c in order if id(c) in alive and c is not snap and not resumes(c)), None)
                if old is None:
                    break
                forget(old)
                over_budget = True
            if held() + need > self.cache_bytes:
                forget(snap)
                over_budget = True
                continue
            shed = 0 if drafter is not None and snap.drafter_end == n else sum(
                t.numel() * t.element_size() for t in snap.drafter_rows or [])
            saved[id(snap)] = snapshot_bytes(snap) - shed + need
        for snap in drops:
            self._drop(snap)
        for snap in order:
            if id(snap) in saved:
                window, end = snap.drafter_rows, snap.drafter_end
                save_rows(self.e, snap, drafter=drafter)
                if self.spill is not None and end != snap.drafter_end:
                    # the copy kept in memory no longer fits DFlash2 (its window dropped): written now, from the saved
                    # rows, with the window it had (no copy of the live caches to wait for)
                    snap.drafter_rows, snap.drafter_end = window, end
                    try:
                        self._spill_snap(snap)
                    finally:
                        snap.drafter_rows, snap.drafter_end = None, -1
        if self.spill is not None:
            # a snapshot dropped above may be on its way to disk from views of the live caches, which the prefill
            # that follows overwrites: those rows leave the device first
            self._spill_wait()
        if over_budget:
            import torch

            torch.cuda.empty_cache()             # give the freed rows back rather than keep them in torch's pool

    def _kept_drafter(self):
        """The drafter whose context window saved snapshots keep (TF_GLM_SHARED_PREFIX), or None: they drop it."""

        return getattr(self, "drafter", None) if self.shared else None

    def _shared(self, key: list[int], hit) -> list[int]:
        """Rank 0: the shared-prefix points a drafted request keeps (``shared_points``), sent to rank 1. ``key`` is
        the request's resume key (its ids, or an image prompt's keyed ids): shared states are told apart by key too."""

        if not self.shared:
            return []
        cut = len(hit.ids) if hit is not None else 0
        known = [c.key for c in self.cache if c.key is not None]
        return shared_points(key, cut, grid_point(len(key), cut, self.grid), self.grid, self.shared, known,
                             self.opener)

    def _encode_feed(self, vision, cut: int):
        """Rank 0: a ``VisionFeed`` of the tower's rows of the pictures a prompt resumed at ``cut`` still fills (whose
        rows end past it), or None when the kept state already holds them all (Mia's 0054, serial)."""

        first, end = 0, 0
        if cut and getattr(vision, "item_rows", None):
            for rows in vision.item_rows:
                if vision.positions[end + rows - 1] >= cut:
                    break
                end += rows
                first += 1
            if first == len(vision.item_rows):
                return None
            vision = vision.from_item(first)
        positions = [int(p) for p in vision.positions]
        if not positions:
            return None
        t = time.perf_counter()
        rows = self.vision.features(vision)
        if rows.shape[0] != len(positions):
            raise ValueError(f"the tower gave {rows.shape[0]} rows for {len(positions)} image positions")
        feed = VisionFeed(self, positions, rows)
        feed.encode_s = time.perf_counter() - t
        return feed

    def _held_bytes(self) -> int:
        from .decode import snapshot_bytes

        return sum(snapshot_bytes(c) for c in self.cache)

    def _run(self, prompt: list[int], max_tokens: int, sampling, stop_eos: bool, on_tokens: Callable[[list[int]], Any],
             code: list[int], hit, draft: bool, constraint=None, feed=None, shared=(), key=None,
             images: bool = False) -> dict[str, Any]:
        """``shared``: rank 0's shared-prefix points (``_shared``), which both ranks keep states at. ``key``: the
        request's resume key (its ids, or an image prompt's keyed ids, 0054); ``images``: the prompt has pictures."""

        self.e.constraint, self.e.window = constraint, None       # both ranks walk and mask the same rows
        try:
            return self._run_once(prompt, max_tokens, sampling, stop_eos, on_tokens, code, hit, draft, feed, shared,
                                  key, images)
        finally:
            self.e.constraint = self.e.window = self.e.vote = None

    def _run_once(self, prompt: list[int], max_tokens: int, sampling, stop_eos: bool,
                  on_tokens: Callable[[list[int]], Any], code: list[int], hit, draft: bool,
                  feed=None, shared=(), key=None, images: bool = False) -> dict[str, Any]:
        from .decode import (DepthPolicy, StopVote, dflash_decode, mtp_decode, prefill, serial_decode, take_snapshot,
                             whole)
        from .drafter_choice import DrafterChoice, auto_decode

        auto, use_mtp, use_dflash = self._drafters(code)
        drafter = self.drafter if use_dflash else None
        t0 = time.perf_counter()
        # a request writes the caches from its resume point: other conversations' rows are saved first, a saved resume point's restored
        from .decode import load_rows

        cut = len(hit.ids) if hit is not None else 0
        self._take_over(list(hit.ids) if hit is not None else [])
        if hit is not None and hit.rows is not None:
            load_rows(self.e, hit, drafter=self._kept_drafter())
            hit.rows, hit.nbytes = None, 0            # live again
        # a keyed prompt's token (and image, 0054) rows live in the caches and can be resumed; an unkeyed image
        # prompt's rows (key is None) resume nothing and are not kept
        self.live = list(prompt) if key is not None else []
        # a drafted keyed prompt keeps its state at ``point``: its end, or with a grid the last grid multiple in it.
        # Short of the end, the prompt prefills to the point as a prompt of that length would (no first token), keeps
        # that state, and resumes from it as a later request would: any prompt extending prompt[:point] passes through
        # the same state there, fresh or resumed. Both ranks decide alike.
        point = grid_point(len(prompt), cut, self.grid) if draft and key is not None else None
        # TF_GLM_SHARED_PREFIX: rank 0's shared-prefix points (past the resume point, short of ``point``) kept the
        # same way, so fresh and resumed prompts pass through the same states there too
        stops = sorted(p for p in set(shared) if point is not None and cut < p < point)
        stats: dict[str, Any] = {"shared": list(stops)} if stops else {}
        if point is not None and point < len(prompt):
            stops.append(point)
        resume, kept = hit, []
        for stop in stops:
            head = list(prompt[:stop])
            prefill(self.e, head, sampling, mtp=use_mtp, drafter=drafter, resume=resume, sample=False)
            resume = take_snapshot(self.e, head, self.e.last_hidden if use_mtp else None, mtp=use_mtp,
                                   drafter=drafter)
            resume.shared = stop != point                         # a shared-prefix point, not the prompt's own end
            resume.key, resume.images = (key[:stop] if images else head), images   # keyed image states (0054)
            kept.append(resume)
        # a replay of a kept prompt (``decode.whole``: its state and head's logits row) prefills nothing and keeps
        # nothing new; a prompt's own end state keeps the head's row it sampled from, for a later replay
        replay = whole(hit, prompt)
        end = point == len(prompt) and not replay
        first = prefill(self.e, prompt, sampling, mtp=use_mtp, drafter=drafter, resume=resume, feed=feed,
                        keep_head=end)
        prefill_s = time.perf_counter() - t0
        if end:
            snap = take_snapshot(self.e, prompt, self.e.last_hidden if use_mtp else None, mtp=use_mtp,
                                 drafter=drafter)
            snap.head, self.e.head = self.e.head, None
            snap.key, snap.images = (list(key) if images else list(prompt)), images   # keyed image end state (0054)
            kept.append(snap)
        if self.shared and draft and key is not None and hit is not None:     # what it resumed from is used again
            self.cache = [c for c in self.cache if c is not hit] + [hit]
        for snap in kept:
            self._remember(snap)
        if not kept and draft and key is not None and hit is not None:     # the resume point is the grid point: newest
            self.cache = [c for c in self.cache if c is not hit] + [hit]
        stats.update(prefill_s=prefill_s, cached=cut)
        # rank 0's on_tokens may ask to stop (client gone, stop string, a gate's cut); rank 1's never does. Every rank
        # hears it on the next round's verify sample (``decode.StopVote``) and ends after that round (issue #38)
        on_tokens = StopVote(on_tokens)
        on_tokens([first])
        if max_tokens <= 1 or (stop_eos and first in self.eos):
            return stats
        self.e.vote = on_tokens                     # on every rank alike: each verify gather carries one word more
        policy = decode_policy(code, self.costs)
        copies = None
        if policy is not None and self.copy is not None:     # prompt and pending token, alike on both ranks
            copies = self.copy.drafts(list(prompt) + [first])
        if policy is None:
            res = serial_decode(self.e, first, max_tokens, sampling, stop_eos=stop_eos, on_tokens=on_tokens)
        elif self.dump is not None and drafter is not None:      # serial's reply, the drafter's view recorded
            from .draft_dump import dump_decode

            meta = None if self.rank else dict(
                prompt_tokens=len(prompt), cached=cut, policy=list(code), images=feed is not None,
                prompt_sha256=hashlib.sha256(json.dumps(list(prompt)).encode()).hexdigest()[:16])
            res = dump_decode(self.e, drafter, first, max_tokens, sampling, stop_eos=stop_eos, on_tokens=on_tokens,
                              out_dir=self.dump if self.rank == 0 else None, meta=meta)
            if self.rank == 0:
                stats["draft_dump"] = res.dump_path
        elif auto:
            greedy = sampling is None or sampling.temperature <= 0
            m_policy = DepthPolicy(3, fixed=True, confidence=0.35) if greedy else DepthPolicy(3, low=0.6, high=0.85)
            _, explore, every, margin, sampled_too = policy
            choice = None
            if drafter is not None and (greedy or sampled_too):
                choice = DrafterChoice(self.costs, first="f" if greedy else "m", explore=explore, every=every,
                                       margin=margin)
            res = auto_decode(self.e, drafter, first, max_tokens, sampling, choice=choice, m_policy=m_policy,
                              f_policy=DepthPolicy(5, fixed=True, confidence=0.3), stop_eos=stop_eos,
                              on_tokens=on_tokens, copies=copies)
        elif use_dflash:
            res = dflash_decode(self.e, self.drafter, first, max_tokens, sampling, policy=policy, stop_eos=stop_eos,
                                on_tokens=on_tokens, copies=copies)
        else:
            res = mtp_decode(self.e, first, max_tokens, sampling, policy=policy, stop_eos=stop_eos,
                             on_tokens=on_tokens, copies=copies)
        # the caches now hold prompt and reply; only prompts are snapshotted, since a later prompt prefills the reply again
        if feed is None:
            self.live = list(prompt) + res.tokens[:self.e.st.pos - len(prompt)]
        stats.update(decode_s=res.seconds, rounds=res.rounds, min_rows=1 + min(res.depths, default=0),
                     tokens_per_round=round((len(res.tokens) - 1) / max(res.rounds, 1), 3),
                     sha256=hashlib.sha256(json.dumps(res.tokens).encode()).hexdigest()[:16])
        if on_tokens.stop:                          # ended early on a stop rank 0 asked for (``StopVote``)
            stats["stopped"] = True
        if res.arms:
            stats.update(drafters=res.arms, keeps=res.keeps)
        if policy is not None:                   # the drafts' counts, which /health, /metrics and the reply report
            stats.update(drafted=res.drafted, accepted=res.accepted)
        if copies is not None:
            stats.update(copy_rounds=res.copy_rounds, copy_drafted=res.copy_drafted, copy_accepted=res.copy_accepted)
        if res.stages:
            stats["stages_ms"] = {k: round(v * 1e3, 1) for k, v in res.stages.items()}
        return stats

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens, draft: bool = True,
                 constraint=None, *, background: bool = False, vision=None) -> dict[str, Any]:
        """Mirror one rank-0 request on rank 1; draft=False uses serial decoding and fresh prefill as the reference drafted replies must equal; ``background``: under --parallel, after the other requests and yielding to them; ``vision``: an image prompt's prepared pictures (``tensorfold.vision.glm.GlmPrepared``)."""

        if len(prompt) >= self.limit:
            raise ValueError(f"prompt of {len(prompt)} tokens: this engine serves contexts up to {self.limit}")
        max_tokens = max(1, min(int(max_tokens), self.limit - len(prompt)))
        if not draft or self.serial_only:
            spec = "0"
        else:
            spec = getattr(self.request, "policy", None) or self.policy
        code = self._effective(encode_policy(spec))
        stop_eos = bool(getattr(self.request, "stop_eos", True))
        if vision is not None and self.vision is None:
            raise ValueError("image inputs require starting this server with --vision")
        if self.scheduler is not None:          # --parallel: the scheduler's worker runs it with the others
            # an image prompt is encoded on the worker at admission (``multi.MultiDecoder.admit``), which owns the GPU
            stats = self.scheduler.submit(list(prompt), max_tokens, sampling, bool(draft) and not self.serial_only,
                                          on_tokens, stop_eos=stop_eos, vision=vision, constraint=constraint,
                                          background=background, glm={"code": code, "spec": spec},
                                          cancelled=getattr(self.request, "cancelled", None))
            stats.update(policy=spec, drafts=draft)
            return stats
        from .multi import keyed_ids, pack_keys

        positions: list[int] = []
        feed = None
        key: list[int] | None = list(prompt)      # text: the ids themselves; an image prompt: keyed by its pictures
        vrows: list[int] = []
        vkeys: list[int] = []
        vpos: list[int] = []
        if vision is not None:
            pos = [int(p) for p in vision.positions]
            if pos and (pos[-1] >= len(prompt) or pos != sorted(set(pos))):
                raise ValueError("image rows must sit at increasing positions inside the prompt")
            keyed = getattr(vision, "item_keys", None)
            if keyed and sum(vision.item_rows) == len(pos):
                vrows, vkeys, vpos = [int(r) for r in vision.item_rows], [int(k) for k in keyed], list(pos)
                key = keyed_ids(prompt, vpos, vrows, vkeys)   # resumable and kept, told apart by content (0054)
            else:
                key = None                    # no per-item keys: this image prompt neither resumes nor is kept
        hit = self._resume(key, code) if draft and key is not None else None
        # --spill-gib: a longer stored prefix on disk, one whose stored fields fit the request's drafters (every rank
        # reads it). Text prompts only: the tier never writes a state with pictures one stream at a time
        disk = None
        if self.spill is not None and draft and vision is None:
            disk = self.spill.find(key, grid=self.grid, usable=lambda e: self._fit(
                e.n, e.fields.get("drafter_end", -1), e.fields.get("mtp_len", -1), code))
            if disk is not None and hit is not None and len(hit.ids) >= disk.n:
                disk = None
        if disk is not None:
            import types

            hit = types.SimpleNamespace(ids=list(prompt[:disk.n]), key=list(prompt[:disk.n]))
        cut = len(hit.ids) if hit is not None else 0
        if vision is not None:               # encoded before the other ranks are told anything: a failure stops all
            feed = self._encode_feed(vision, cut)
        positions = feed.positions if feed is not None else []
        shared = self._shared(key, hit) if draft and key is not None else []
        seed = (sampling.seed if sampling else 0) & 0xFFFFFFFFFFFFFFFF
        # cached < 0: the resume point is a prompt of -cached tokens stored on disk, which every rank reads
        cached = -disk.n if disk is not None else len(hit.ids) if hit is not None else 0
        header = [max_tokens, int(stop_eos), int(draft), cached,
                  seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF, seed >> 62,
                  *_f64_ints(sampling.temperature if sampling else 0.0), int(sampling.top_k) if sampling else 0,
                  *_f64_ints(sampling.top_p if sampling else 1.0), *_f64_ints(sampling.min_p if sampling else 0.0),
                  int(constraint is not None), len(positions), int(vision is not None),
                  *(shared + [0] * SHARED_MOST)[:SHARED_MOST]] + code
        from tensorfold.engine.grammar import pack

        self._ring()                                   # wakes rank 1, which idles on the store, not in the all-gather
        self._share(header)
        self._share(list(prompt))
        if constraint is not None:                     # the request's grammar: rank 1 compiles the same
            self._share(pack(constraint))
        if positions:
            self._share(positions)
        if vision is not None:                         # the pictures' keys/positions: rank 1 rebuilds the resume key
            self._share(pack_keys(vrows, vkeys, vpos))
        if disk is not None:
            hit = self._spill_load(list(prompt[:disk.n]), code)
        if self.spill is not None:                     # pending copies of the live caches end before they change
            self._spill_wait()
        stats = self._run(list(prompt), max_tokens, sampling, stop_eos, on_tokens, code, hit, draft, constraint, feed,
                          shared, key, vision is not None)
        stats.update(policy=spec, drafts=draft)
        if feed is not None:
            stats.update(image_rows=len(positions), encode_s=round(feed.encode_s, 3))
        from tensorfold.cuda.comm import check

        check(self.comm)        # a transport failure recorded inside a graph (a RoCE wait that timed out) surfaces here
        return stats

    def score_labels(self, prompt_ids: list[int], label_ids: list[int]) -> tuple[list[float], float]:
        """Both ranks prefill the prompt and return its label logits plus the full-vocabulary logsumexp."""

        prompt = [int(token) for token in prompt_ids]
        labels = [int(token) for token in label_ids]
        if not prompt:
            raise ValueError("empty prompt")
        if not labels:
            raise ValueError("empty labels")
        if len(prompt) >= self.limit:
            raise ValueError(f"prompt of {len(prompt)} tokens: this engine serves contexts up to {self.limit}")
        if self.multi is not None:      # the scheduler's worker owns both ranks' collectives
            raise ValueError("scoring (/v1/decisions) is not served under --parallel; start with --parallel 1")
        self._ring()                    # rank 1 waits on the store before this header, as a chat request does
        self._share([0, len(labels)])   # max_tokens on a chat header is at least 1, so 0 is a score
        self._share(prompt)
        self._share(labels)
        return self._score_local(prompt, labels)

    def _score_local(self, prompt: list[int], labels: list[int]) -> tuple[list[float], float]:
        from tensorfold.families.glm5_next.cuda.score import prompt_logits
        from tensorfold.server.decisions import reduce_vocab_shards

        # The score prefill writes attention rows from position 0. Save or drop every snapshot those rows
        # still belong to, then stop naming them: the next chat must not resume the decision as that conversation.
        self._take_over([])
        self.live = []
        if self.drafter is not None:
            self.drafter.reset()
        local = prompt_logits(self.e, prompt)
        rows = [local] if self.comm is None else self._gather_floats(local)
        if self.rank != 0:
            return [], 0.0
        return reduce_vocab_shards(rows, labels, len(rows[0]))

    def _gather_floats(self, values: list[float]) -> list[list[float]]:
        torch = self.torch
        mine = torch.tensor(values, dtype=torch.float32, device="cuda")
        world = self.world
        got = torch.empty((world * len(values),), dtype=torch.float32, device="cuda")
        self.comm.all_gather(mine, got)
        width = len(values)
        return [[float(item) for item in got[i * width:(i + 1) * width].tolist()] for i in range(world)]

    def follow(self) -> None:
        """Rank 1: mirror every request rank 0 serves, forever."""

        from tensorfold.engine.exact_sampling import Sampling

        if self.multi is not None:
            self.multi.follow()
            return

        while True:
            self._await_bell()
            header = self._share(None)
            if len(header) == 2 and header[0] == 0:     # a decision: both ranks prefill, neither samples
                prompt = self._share(None)
                labels = self._share(None)
                self._score_local(prompt, labels)
                continue
            (max_tokens, stop_eos, draft, cached, s_lo, s_hi, s_top, t_lo, t_hi, top_k, p_lo, p_hi, m_lo, m_hi, shaped,
             npos, has_images, *rest) = header
            if max_tokens == 0 and self.spill is not None:     # rank 0's clean shutdown: the spill tier's flush
                self._do_flush(self._share(None), self.spill.cfg.flush_s)
                continue
            shared, code = [p for p in rest[:SHARED_MOST] if p], rest[SHARED_MOST:]
            prompt = self._share(None)
            packed = self._share(None) if shaped else []
            constraint = None
            if packed:                                  # compiled here as on rank 0
                from tensorfold.engine import grammar

                constraint = grammar.compiler(self, self.model_dir, self.eos).follow(packed)
            feed = VisionFeed(self, self._share(None), None) if npos else None
            key: list[int] | None = list(prompt)        # the resume key rank 1 matches and keeps by (0054)
            if has_images:                              # the pictures' keys/positions: rank 1 rebuilds the key
                from .multi import keyed_ids, unpack_keys

                vrows, vkeys, vpos, _ = unpack_keys(self._share(None), 0)
                key = keyed_ids(prompt, vpos, vrows, vkeys) if vrows else None
            temperature = _ints_f64(t_lo, t_hi)
            seed = (s_top << 62) | (s_hi << 31) | s_lo
            sampling = (Sampling(seed, temperature, top_k, _ints_f64(p_lo, p_hi), _ints_f64(m_lo, m_hi))
                        if temperature > 0 else None)
            hit = None
            if cached < 0:                              # rank 0 resumes from a prefix stored on disk: read ours too
                hit = self._spill_load(list(prompt[:-cached]), code)
            elif cached:
                hit = next((c for c in self.cache if len(c.ids) == cached and c.key is not None
                            and key[:cached] == c.key), None)
                if hit is None:
                    raise RuntimeError(f"rank {self.rank} has no snapshot of the {cached} tokens rank 0 resumes from")
            if self.spill is not None:
                self._spill_wait()
            self._run(prompt, max_tokens, sampling, bool(stop_eos), lambda new: None, code, hit, bool(draft),
                      constraint, feed, shared, key, bool(has_images))
            from tensorfold.cuda.comm import check

            check(self.comm)


class VisionFeed:
    """An image prompt's feature rows for each prefill chunk: rank 0 holds them all and hands a chunk's rows to the
    other ranks through the all-gather as the chunk comes (they send zeros), so no rank holds a second copy."""

    def __init__(self, engine: GlmEngine, positions: list[int], rows) -> None:
        self.engine, self.positions, self.rows = engine, positions, rows
        self.width = engine.w.cfg.hidden
        self.encode_s = 0.0

    def chunk(self, start: int, count: int):
        """(row indices within the chunk, their features) for prompt rows start .. start + count, or None. Every
        rank calls it for the same chunks (positions are shared), so the gathers pair up."""

        import bisect

        torch = self.engine.torch
        lo = bisect.bisect_left(self.positions, start)
        hi = bisect.bisect_left(self.positions, start + count)
        if lo == hi:
            return None
        n = hi - lo
        device = getattr(self.engine, "device", "cuda")
        mine = (self.rows[lo:hi].contiguous() if self.rows is not None else
                torch.zeros((n, self.width), dtype=torch.bfloat16, device=device))
        world = getattr(self.engine, "world", 2)
        every = torch.empty((world * n, self.width), dtype=torch.bfloat16, device=device)
        self.engine.comm.all_gather(mine, every)
        index = torch.tensor([p - start for p in self.positions[lo:hi]], dtype=torch.int64, device=device)
        return index, every[:n]

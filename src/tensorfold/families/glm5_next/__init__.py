"""GLM-5.3-Flash (model_type ``glm5_next``): an MLX engine on a 256 GB Mac, a CUDA engine over two DGX Sparks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("glm5_next",)
TITLE = "GLM-5.3-Flash"
LANES = True
# 4-bit weights in groups of 64 with the MTP layer kept; the EXL3 checkpoints are the CUDA engine's alone (Mia's
# 4-bit mcg routed experts with BF16 elsewhere; RED-SNOW's mul1 at 4 to 8 bits a tensor, dense layers and head too)
MODELS = ("TensorFold/GLM-5.3-Flash-MLX-4bit-MTP", "Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw",
          "vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw")
DRAFTER = "incoai/GLM-5.3-Flash-DFlash2"   # the CUDA engine's optional draft model; the Mac engine drafts with MTP
KERNEL_PACKAGE = "tensorfold.kernels.glm.flash.v1"
# the prompt experts' sorted gather (Flash Next's prompt matmuls), hashed into snapshot keys
KERNEL_DEPENDENCIES = ("tensorfold.kernels.qwen.flash_next.v1.prefill_mm",)
KERNEL_VERSION = "v1"
# the storage formats each engine reads: MLX affine on a Mac; that or EXL3 on CUDA
QUANT_METHODS = {"mlx": ("mlx",), "cuda": ("mlx", "exl3")}
CUDA_TP4 = True                    # --tp 4: the CUDA engine splits over four ranks
# EXL3 at any width the shared kernels read (tensorfold.families.EXL3_VARIANT_ANY: require_readable runs
# cuda/exl3/format.require_config); widths come from each trellis' shape, never from config.json
EXL3_VARIANT = "any"
# of those, the codebooks checked on GLM-5.3 (3inst is in the kernels but no GLM checkpoint uses it)
EXL3_CODEBOOKS = ("mcg", "mul1")
# buffers of 200 ops and 200 MB, so a prompt chunk's memory frees as it runs; no TF32: row kernels repeat fp32
MLX_ENV = {"MLX_MAX_OPS_PER_BUFFER": "200", "MLX_MAX_MB_PER_BUFFER": "200", "MLX_ENABLE_TF32": "0"}


def _mac_reads(fmt: tuple) -> bool:
    from tensorfold.families.glm5_next.config import BITS, GROUPS

    return fmt[0] in BITS and fmt[1] in GROUPS


def check(model_dir: str | Path) -> None:
    """Refuse what neither engine reads: MLX affine weights on a Mac; those or EXL3 (mcg or mul1) on two GPUs."""

    import sys

    from tensorfold.families import OWN_MODEL_HELP, describe_quantization, quant_method, quantization, read_config

    config = read_config(model_dir)
    want = config.get("tensorfold_activation_dtype")
    if want not in (None, "bfloat16", "float32"):
        raise ValueError(f"tensorfold_activation_dtype {want!r}: bfloat16 or float32")
    if want == "float32" and sys.platform != "darwin":
        raise ValueError("tensorfold_activation_dtype float32 is the Mac engine; the CUDA engine stays bf16")
    method = quant_method(config)
    if method == "exl3":
        # the CUDA engine's layout; the Mac engine refuses it before this through QUANT_METHODS (require_readable)
        _check_exl3(model_dir, config)
        print("[tensorfold] EXL3 support is experimental: replies are exact, but the MLX checkpoint "
              f"({MODELS[0]}) is tested more and runs faster (docs/recipes/glm-5.3-flash.md)", flush=True)
    elif quantization(config) != (4, 64) and not (sys.platform == "darwin" and _mac_reads(quantization(config))):
        raise ValueError(f"GLM-5.3-Flash's Mac engine reads MLX affine weights of 2 to 8 bits in groups of 32, 64 or "
                         f"128 ({MODELS[0]} is 4-bit in groups of 64), and the CUDA engine 4-bit groups of 64 or "
                         f"EXL3 ({', '.join(MODELS[1:])}); this checkpoint has {describe_quantization(config)}. "
                         f"{OWN_MODEL_HELP}")
    if sys.platform == "darwin":
        from tensorfold.families.glm5_next.config import quant_formats, unreadable

        _require_mlx((0, 32, 2))
        from tensorfold.families.glm5_next.layouts import canonical

        text = config.get("text_config") or config
        mtp_layer = int(text.get("num_hidden_layers", 0))
        # Attention outputs and the MTP input projection already have a dense matmul path.
        def dense_supported(name: str) -> bool:
            short = canonical(name, mtp_layer) or ""
            return (short.startswith("layers.") and short.endswith(".self_attn.o_proj")
                    or short == f"layers.{mtp_layer}.eh_proj")

        bad = sorted(name for name, fmt in quant_formats(config)[1].items()
                     if unreadable(fmt) and not (fmt is None and dense_supported(name)))
        if bad:
            raise ValueError(f"GLM-5.3-Flash's Mac engine reads MLX affine weights of 2 to 8 bits in groups of 32, 64 or "
                             f"128; this checkpoint stores {len(bad)} module(s) otherwise, {bad[0]} first. {OWN_MODEL_HELP}")
        # a mixed-bit conversion's 5-, 6- and 8-bit tensors take row kernels at their width (widths.py), one-row bits
        base = quantization(config)[0]
        overrides = [k for k, v in quant_formats(config)[1].items() if v is not None and v[0] != base]
        if overrides:
            print(f"[tensorfold] {len(overrides)} tensors are not 4-bit (per-tensor quantization overrides): they run "
                  f"through the 8/6/5-bit row kernels; drafting stays exact", flush=True)
        if (method != "exl3" and (Path(model_dir) / "model.safetensors.index.json").is_file()
                and not has_mtp(model_dir)):
            print(f"[tensorfold] this checkpoint has no MTP layer: decoding without MTP drafts ({MODELS[0]} has "
                  f"one)", flush=True)
        return
    print("[tensorfold] GLM-5.3-Flash runs on two NVIDIA GPUs with 128 GB each (two DGX Sparks): pull it on both "
          "and serve with --tp 2 (or a 4-rank build) on both (docs/recipes/glm-5.3-flash.md)", flush=True)


def _check_exl3(model_dir: str | Path, config: dict) -> None:
    """Refuse an EXL3 codebook or width GLM-5.3's CUDA engine does not read: config.json always, and once every shard
    is on disk the safetensors headers too (no weight data); prints what it found."""

    import json

    from tensorfold.cuda.exl3 import format as exl3_format
    from tensorfold.families import OWN_MODEL_HELP

    where, tested = "GLM-5.3-Flash on NVIDIA GPUs (CUDA)", ", ".join(MODELS[1:])
    fields = exl3_format.require_config(config, where=where, tested=tested, help=OWN_MODEL_HELP)

    def refuse(why: str) -> None:
        raise ValueError(f"{where} reads EXL3 checkpoints with the {' or '.join(EXL3_CODEBOOKS)} codebook at "
                         f"{', '.join(f'{b:g}' for b in exl3_format.BITS)} bits a tensor; this one has {why} "
                         f"({exl3_format.describe_config(fields)}). Tested checkpoints: {tested}. {OWN_MODEL_HELP}")

    codebook = fields.get("codebook")
    if codebook is not None and str(codebook).lower() not in EXL3_CODEBOOKS:
        refuse(f"codebook {codebook!r}")
    index = Path(model_dir) / "model.safetensors.index.json"
    if not index.is_file():
        return
    shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
    if not shards or not all((Path(model_dir) / f).is_file() for f in shards):
        return                                           # before the download: config.json was all there was
    if next(Path(model_dir).glob("*.rank*.safetensors"), None) is not None:
        return                                           # split.py's per-rank pieces sit beside the shards: theirs
    ck = exl3_format.scan(model_dir, read_markers=False)
    if ck.bad:
        prefix = sorted(ck.bad)[0]
        refuse(f"{len(ck.bad)} EXL3 tensor(s) that do not parse, {prefix} first: {ck.bad[prefix]}")
    if not ck.groups:
        refuse("no EXL3 tensors (.trellis) in its safetensors files")
    books = sorted({g.codebook for g in ck.groups.values()})
    if [b for b in books if b not in EXL3_CODEBOOKS] or len(books) > 1:
        refuse(f"tensors in the {'/'.join(books)} codebook(s)")
    widths: dict[float, int] = {}
    for g in ck.groups.values():
        widths[g.bits] = widths.get(g.bits, 0) + 1
    layers = int((config.get("text_config") or config).get("num_hidden_layers", 0))
    experts = sum(1 for p in ck.groups if ".mlp.experts." in p)
    head = ck.groups.get("lm_head")
    print(f"[tensorfold] EXL3 tensors: {len(ck.groups)} in the {books[0]} codebook ({experts} routed experts' "
          f"projections, {len(ck.groups) - experts} dense), widths "
          + ", ".join(f"{b:g}-bit x{n}" for b, n in sorted(widths.items()))
          + (f"; head {head.bits:g}-bit" if head else "; head plain")
          + f"; {sum(1 for p in ck.groups if f'layers.{layers}.' in p)} in the MTP layer {layers}", flush=True)


def check_ranks(model_dir: str | Path, ranks: int) -> None:
    """Refuse --tp ``ranks`` for an EXL3 checkpoint with a split tensor whose rank boundaries are not multiples of the
    128-wide Hadamard block (split.check_exl3): RED-SNOW's 8-bit head has 154,880 vocabulary columns, 38,720 a rank over
    four, so it cannot run on four ranks; over two (77,440) it can. Headers only, no weight data. Nothing to check for
    MLX checkpoints, before the download (shards missing) or for a folder of split.py's rank pieces (already cut)."""

    import json

    from tensorfold.families import quant_method, read_config

    if ranks < 2:
        return
    try:
        config = read_config(model_dir)
    except (OSError, ValueError):
        return                              # no readable config.json (before the download): the check runs later
    if quant_method(config) != "exl3":
        return
    root = Path(model_dir)
    index = root / "model.safetensors.index.json"
    if not index.is_file() or next(root.glob("*.rank*.safetensors"), None) is not None:
        return
    shards = sorted(set(json.loads(index.read_text()).get("weight_map", {}).values()))
    if not shards or not all((root / f).is_file() for f in shards):
        return
    from .cuda.split import EXL3_PART, FUSED3_PLAIN, check_exl3, read_header, rule

    import re

    bad: list[str] = []
    for shard in shards:
        header, _ = read_header(root / shard)
        for name, info in header.items():
            if name == "__metadata__" or not (EXL3_PART.search(name) or re.search(FUSED3_PLAIN, name)):
                continue
            kind = rule(name)
            try:
                check_exl3(name, [int(x) for x in info["shape"]], kind, ranks)
            except ValueError:
                bad.append(f"{name} {kind} {list(info['shape'])}")
    if bad:
        groups = sorted({b.split()[0].rsplit(".", 1)[0] for b in bad})
        raise ValueError(f"--tp {ranks}: {len(bad)} EXL3 tensor(s) of this checkpoint ({len(groups)} matrices, "
                         f"{', '.join(groups[:3])}{' ...' if len(groups) > 3 else ''}) cannot be cut into {ranks} "
                         f"pieces at multiples of the 128-wide Hadamard block, first: {sorted(bad)[0]}. Serve it with "
                         f"--tp 2 on two GPUs; no weights have been loaded")


def has_mtp(model_dir: str | Path) -> bool:
    """Whether the checkpoint kept the MTP layer; JSON only, so the CLI's check runs before MLX starts."""

    import json

    config = json.loads((Path(model_dir) / "config.json").read_text())
    text = config.get("text_config") or config
    n = int(text.get("num_hidden_layers", 0))
    if int(text.get("num_nextn_predict_layers", 0)) < 1:
        return False
    index = Path(model_dir) / "model.safetensors.index.json"
    if not index.is_file():
        return False
    from tensorfold.families.glm5_next.layouts import mtp_layer_names

    return mtp_layer_names(json.loads(index.read_text())["weight_map"], n)


def _require_mlx(least: tuple[int, ...]) -> None:
    """Refuse an older MLX, read from the package's metadata (MLX itself is not imported yet)."""

    import re
    from importlib.metadata import PackageNotFoundError, version

    try:
        found = version("mlx")
    except PackageNotFoundError:
        return
    parts = tuple(int(p) for p in re.findall(r"\d+", found)[:3])
    if parts < least:
        need = ".".join(str(p) for p in least)
        raise ValueError(f"GLM-5.3-Flash needs MLX {need} or later (this is {found}): with 0.32.0 its weights dropped "
                         f"out of wired memory on a 256 GB Mac and decoding fell to a few tokens a second. Install it "
                         f"with: python -m pip install \"mlx>={need}\"")


def expert_bytes(model_dir: Path) -> int:
    """Bytes of the decoder layers' routed experts, which --ssd-experts leaves on disk (the MTP layer's stay)."""

    import json

    from tensorfold.families.glm5_next.layouts import routed_expert
    from tensorfold.streaming.checkpoint import tensor_bytes

    config = json.loads((Path(model_dir) / "config.json").read_text())
    layers = int((config.get("text_config") or config).get("num_hidden_layers", 0))
    return tensor_bytes(Path(model_dir), lambda name: routed_expert(name, layers) is not None)


def load(model_dir: Path, *, mtp_drafts: int | None = None, ssd_experts: float | None = None,
         vision: bool = False, vision_urls: bool = False, **_: Any) -> tuple[Any, Any]:
    """The MLX engine; ``mtp_drafts`` caps the MTP drafts a round (0: none); ``ssd_experts``: the expert pool's GiB."""

    import mlx.core as mx

    from tensorfold.families.glm5_next.runtime import load as load_runtime

    # about 170 GB of weights on a 256 GB Mac: keep them wired, or macOS can page them out between steps
    if mx.metal.is_available():
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        limit = int(info.get("max_recommended_working_set_size", 0))
        if limit:
            mx.set_wired_limit(limit)
    family, tokenizer = load_runtime(Path(model_dir), drafts=mtp_drafts, ssd_experts=ssd_experts)
    if vision:
        from tensorfold.vision.glm_mlx import GLMVisionFrontend

        family.vision = GLMVisionFrontend.load(Path(model_dir), family.model.embed_tokens, allow_urls=vision_urls)
        print(f"[tensorfold] GLM image encoder: {family.vision.workspace_bytes / 1024**3:.2f} GiB workspace "
              "measured at the largest admitted image request (4,096 visual tokens)", flush=True)
    return family, tokenizer


def engine_settings(model: Any) -> dict[str, Any]:
    """Rows a round verifies at most: the widest window checked exact at load."""

    width = int(getattr(model, "exact_width", 1) or 1)
    return {"max_rows": width, "max_draft": max(0, width - 1)}


def kernel_version(model: Any) -> str:
    """Names the kernels behind a prefix snapshot: this engine's and its kernels' sources, and MLX's version."""

    import hashlib
    import importlib

    import mlx.core as mx

    digest = hashlib.sha256()
    for module in (__name__, KERNEL_PACKAGE, *KERNEL_DEPENDENCIES):
        source = Path(str(importlib.import_module(module).__file__))
        folder = source.parent
        # a package's folder only (the CUDA engine, cuda/, is not on this path); a module's own file
        for path in sorted(folder.glob("*.py")) if source.name == "__init__.py" else [source]:
            digest.update(path.relative_to(folder).as_posix().encode())
            digest.update(path.read_bytes())
    digest.update(mx.__version__.encode())
    return f"{MODEL_TYPES[0]}-{KERNEL_VERSION}-" + digest.hexdigest()[:12]


def memory_fraction(ram_bytes: int) -> float | None:
    """85% of RAM on a Mac of 256 GB or less, approved for this checkpoint with nothing else loaded."""

    return 0.85 if ram_bytes <= 256 * 1024**3 else None


# the CUDA engine's kernels read MLX affine weights of this (bits, group size); EXL3 checkpoints are checked above
CUDA_QUANTIZATION = (4, 64)
# --kv-dtype on CUDA: bf16, or fp8 (TF_GLM_KV=fp8: the DSA latent cache and indexer pooled keys as e4m3 rows, cuda/kv8.py)
CUDA_KV_DTYPES = ("bf16", "fp8")
# --spill-gib on CUDA: kept prompt states go to a disk tier (cuda/spill.py) and come back instead of a prefill (Mia's 0088)
CUDA_SPILL = True


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None, **options: Any):
    """Build the two-rank engine with adaptive drafting, reusable prompt state, or serial decoding when drafts are disabled."""

    if int(tp) not in (2, 4):
        raise ValueError("GLM-5.3-Flash needs two GPUs, one per machine: run the same `tensorfold serve` command "
                         "with --tp 2/4 --rank R --master ADDRESS on both (non-zero ranks first)")
    if not master:
        raise ValueError("--tp 2/4 needs --master: rank 0's address on the link between the ranks")
    if options.get("kv_dtype", "bf16") not in CUDA_KV_DTYPES:
        raise ValueError(f"--kv-dtype {options['kv_dtype']}: GLM-5.3-Flash on CUDA caches {' or '.join(CUDA_KV_DTYPES)}")
    if options.get("vision_offload"):
        raise ValueError("--vision-offload is for Qwen's CUDA tower; GLM-5.3's tower stays on rank 0's GPU")
    check_ranks(model_dir, int(tp))                 # before the ranks connect or any weight loads
    from .cuda.engine import DEFAULT_POLICY, DFLASH_POLICY, GlmEngine

    if mtp_drafts is None:
        policy = DEFAULT_POLICY
    elif int(mtp_drafts) == 0 and drafter and not no_drafts:
        policy = DFLASH_POLICY          # no MTP drafts: every round still verifies DFlash2's drafts
    else:
        policy = str(int(mtp_drafts))
    return GlmEngine(Path(model_dir), rank=int(rank), master=master, port=int(master_port), policy=policy,
                     drafter=Path(drafter) if drafter and not no_drafts else None,
                     context=options.get("context"), context_explicit=options.get("context_explicit"),
                     kv=options.get("kv_dtype"),
                     serial_only=bool(no_drafts), world=int(tp), parallel=int(options.get("parallel", 1) or 1),
                     vision=bool(options.get("vision", False)), vision_urls=bool(options.get("vision_urls", False)),
                     spill=options.get("spill"))


def __getattr__(name: str) -> Any:
    if name == "CUDA_APP":             # imported on first use, so the Mac side never loads the CUDA server
        from .cuda.app import GlmApp

        return GlmApp
    raise AttributeError(name)

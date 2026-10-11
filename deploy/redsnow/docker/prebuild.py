#!/usr/bin/env python3
"""Compile TensorFold's CUDA extensions (and the RoCE proxy) at image build time, without a GPU.

Every extension is built through the package's own loader (an lru-cached ``_ext*`` function that calls
``tensorfold.cuda.build.load``), so the build directory, the ninja file and the flags are exactly the ones a serving
process computes. At start the ranks then find each module up to date and only run ninja's no-op check.

docker build has no GPU, so torch's capability query answers ``--arch`` (12.1, GB10). build.load turns that into
``-gencode=arch=compute_121,code=sm_121``, the same flag the real GPU gives at start; any other value would make the
container compile again.

Modules come from two places:
  * REQUIRED: the extensions the GLM 5.3 path is known to load (a failure here fails the image build);
  * discovery: every other module under tensorfold.families.glm5_next.cuda and tensorfold.cuda (minus EXCLUDE) that
    defines an ``_ext*`` loader, so kernels merged later (segmented / multi-stream / kv8 variants) are picked up
    without editing this list. A failure there is reported and, with --strict, fails the build.
"""

from __future__ import annotations

import argparse
import importlib
import os
import pkgutil
import sys
import time
import traceback

REQUIRED = [
    "tensorfold.cuda.exl3.linear",              # native EXL3 dense linear (tensorfold_exl3_linear_v3)
    "tensorfold.cuda.exl3.experts",             # routed EXL3 experts, all codebooks + mixed widths
    "tensorfold.families.glm5_next.cuda.exl3_mm",
    "tensorfold.families.glm5_next.cuda.kda",
    "tensorfold.cuda.kernels.qmm",
    "tensorfold.cuda.experts",
    "tensorfold.cuda.roce",                     # GPU side of the RoCE all-gather (+ proxy_library below)
]
# not used by RED-SNOW: other families' kernels and NVFP4 (cluster builds for sm_121a)
EXCLUDE = ("tensorfold.cuda.nvfp4", "tensorfold.families.qwen", "tensorfold.families.nemotron",
           "tensorfold.families.deepseek")
SCAN = ("tensorfold.families.glm5_next.cuda", "tensorfold.cuda")   # CUDA side only (the rest imports mlx)


def fake_gpu(arch: str) -> None:
    import torch

    major, minor = (int(x) for x in arch.split("."))
    if torch.cuda.is_available():
        real = torch.cuda.get_device_capability()
        if real != (major, minor):
            sys.exit(f"a GPU {real} is visible but --arch is {arch}: refusing to build for the wrong target")
        return
    torch.cuda.get_device_capability = lambda *a, **k: (major, minor)
    torch.cuda.get_device_name = lambda *a, **k: f"(image build, sm_{major}{minor})"


def loaders(module) -> list[str]:
    return sorted(n for n, v in vars(module).items()
                  if n.startswith("_ext") and callable(v) and hasattr(v, "cache_clear")
                  and getattr(v, "__module__", "") == module.__name__)


def discover() -> list[str]:
    import tensorfold  # noqa: F401

    found = []
    for root in SCAN:
        pkg = importlib.import_module(root)
        for info in pkgutil.walk_packages(pkg.__path__, root + "."):
            if info.name.startswith(EXCLUDE) or ".tests" in info.name:
                continue
            found.append(info.name)
    return found


def build(name: str) -> tuple[list[str], str | None]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        return [], f"import failed: {exc!r}"
    done = []
    for fn in loaders(module):
        t = time.time()
        try:
            getattr(module, fn)()
        except (ImportError, OSError) as exc:
            # compiled, but the .so cannot be loaded here (no driver in docker build): fine if the module file exists
            done.append(f"{name}.{fn} (built; load skipped without a GPU: {exc.__class__.__name__})")
            continue
        except Exception as exc:  # noqa: BLE001
            return done, f"{fn}() failed: {exc!r}\n{traceback.format_exc()}"
        done.append(f"{name}.{fn} ({time.time() - t:.0f} s)")
    return done, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="12.1")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--strict", action="store_true", help="fail on a discovered module that does not build")
    ap.add_argument("--list", action="store_true", help="print the modules and loaders it would build, build nothing")
    args = ap.parse_args()
    if args.list:
        for name in REQUIRED + [n for n in discover() if n not in REQUIRED]:
            try:
                fns = loaders(importlib.import_module(name))
            except Exception as exc:  # noqa: BLE001
                fns = [f"(import failed: {exc!r})"]
            if fns or name in REQUIRED:
                print(("REQUIRED " if name in REQUIRED else "found    ") + name, " ".join(fns))
        return 0
    if not os.environ.get("TORCH_EXTENSIONS_DIR"):
        sys.exit("TORCH_EXTENSIONS_DIR must be set (the baked directory the containers use)")
    fake_gpu(args.arch)

    lines, failed = [], []
    for name in REQUIRED:
        done, err = build(name)
        if err or not done:
            failed.append(f"REQUIRED {name}: {err or 'no _ext loader found'}")
        lines += done
    for name in discover():
        if name in REQUIRED:
            continue
        done, err = build(name)
        if err:
            (failed.append if args.strict else print)(f"optional {name}: {err}")
        lines += done

    from tensorfold.cuda import roce
    try:
        lib = roce.proxy_library()
        lines.append(f"tensorfold.cuda.roce.proxy_library (abi {lib.roce_abi_version()})")
    except Exception as exc:  # noqa: BLE001
        failed.append(f"REQUIRED RoCE proxy: {exc!r}")

    root = os.environ["TORCH_EXTENSIONS_DIR"]
    built = sorted(e for e in os.listdir(root) if not e.startswith("."))
    report = ["# built for sm_" + args.arch.replace(".", ""), *lines, "# directories:", *built]
    print("\n".join(report))
    if args.manifest:
        with open(args.manifest, "w") as f:
            f.write("\n".join(report) + "\n")
    if failed:
        print("\n".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

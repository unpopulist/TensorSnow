"""Read each rank directly from checkpoint bytes or a saved rank folder, preserving packed groups at every split boundary."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

ROW = (
    r"\.mlp\.experts\.\d+\.(gate|up)_proj\.",
    r"\.mlp\.shared_experts\.(gate|up)_proj\.",
    r"\.mlp\.(gate|up)_proj\.",
    r"\.self_attn\.(q|k|v)_proj\.",
    r"\.self_attn\.(q|k|v)_conv1d\.",
    r"\.self_attn\.(f_b|g_b|b)_proj\.",
    r"\.self_attn\.(A_log|dt_bias)$",
    r"\.self_attn\.(q_b|kv_b)_proj\.",
)
COL = (
    r"\.mlp\.experts\.\d+\.down_proj\.",
    r"\.mlp\.shared_experts\.down_proj\.",
    r"\.mlp\.down_proj\.",
    r"\.self_attn\.o_proj\.",
)
REP = (
    r"^lm_head\.", r"embed_tokens\.", r"^model\.language_model\.norm\.weight$",
    r"_layernorm\.weight$", r"\.hc_(attn|ffn)_(fn|base|scale)$", r"\.mlp\.gate\.(weight|e_score_correction_bias)$",
    r"\.self_attn\.indexer\.", r"\.self_attn\.(q_a_proj|kv_a_proj_with_mqa)\.", r"\.self_attn\.(f_a|g_a)_proj\.",
    r"\.self_attn\.o_norm\.weight$", r"\.(eh_proj)\.", r"\.(enorm|hnorm)\.weight$", r"\.shared_head\.norm\.weight$",
)
RUN = 128 << 20          # most bytes one read of neighbouring tensors takes (``RankReader.prefetch``)
GAP = 1 << 20            # most unused bytes such a read spans between two tensors (more reads other layers twice)
READERS = 8              # reads in flight: past this the layers are built slower than they are read
DTYPE_BYTES = {"U32": 4, "I32": 4, "F32": 4, "BF16": 2, "F16": 2, "I16": 2, "U16": 2, "U8": 1, "I8": 1, "I64": 8,
               "F64": 8}
# the files a rank folder needs besides its weights (the tokenizer, chat template and configs)
SMALL = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
         "processor_config.json", "model.safetensors.index.json")


# EXL3 linears are stored as trellis I16 [K/16, N/16, 16*bits] (input K, output N), suh F16 [K], svh F16 [N] and a
# codebook scalar (mul1 I32 [] or mcg); y = rot_N(rot_K(x * suh) @ W) * svh with the Hadamard rotations applied per
# 128-wide block, so a split is exact iff every rank boundary is a multiple of 128 on the split axis (``check_exl3``).
# Output split ("cols": gate/up, q_b, lm_head vocab): trellis tile columns (dim1) and svh; suh copied.
# Input split ("rows", all-reduced: down, o_proj): trellis tile rows (row) and suh; svh copied.
# The codebook scalar (mul1 / mcg) is the same on every rank and copied; so are the low-rank q_a / kv_a and MTP eh_proj.
EXL3_EXPERT = re.compile(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj\.(trellis|suh|svh|mcg|mul1)$")
EXL3_PART = re.compile(r"\.(trellis|suh|svh|mcg|mul1)$")
EXL3_OUT = (r"\.mlp\.experts\.\d+\.(gate|up)_proj\.", r"\.mlp\.shared_experts\.(gate|up)_proj\.",
            r"\.mlp\.(gate|up)_proj\.", r"\.self_attn\.q_b_proj\.", r"^lm_head\.")
EXL3_IN = (r"\.mlp\.experts\.\d+\.down_proj\.", r"\.mlp\.shared_experts\.down_proj\.", r"\.mlp\.down_proj\.",
           r"\.self_attn\.o_proj\.")
EXL3_REP = (r"\.self_attn\.(q_a_proj|kv_a_proj_with_mqa)\.", r"\.eh_proj\.")
# KDA's q/k/v are fused along the output: qkv_proj's N (and conv1d's rows) is [q | k | v], each heads x 128, and rank
# r takes heads [r*H/ranks, (r+1)*H/ranks) of each of the three. The rank keeps the name qkv_proj (conv1d) for its
# concatenated [q_r | k_r | v_r]: one valid EXL3 linear (K 4096, N 3*H/ranks*128) per rank, in the order KDAW.conv
# already uses, so a rank folder holds the same names as the checkpoint.
EXL3_FUSED3 = r"\.self_attn\.qkv_proj\."
FUSED3_PLAIN = r"\.self_attn\.conv1d\.weight$"
EXL3_KINDS = {"out": {"trellis": "dim1", "suh": "rep", "svh": "row"},
              "in": {"trellis": "row", "suh": "row", "svh": "rep"},
              "fused3": {"trellis": "fused3_dim1", "suh": "rep", "svh": "fused3_row"},
              "rep": {"trellis": "rep", "suh": "rep", "svh": "rep"}}
# what each split kind does to the shape (for size estimates such as ``tensorfold.cuda.geometry.split_weights``)
GEOMETRY_KIND = {"fused3_dim1": "dim1", "fused3_row": "row"}


def rule(name: str) -> str:
    if name.startswith("model.visual."):
        return "drop"
    m = EXL3_PART.search(name)
    if m:
        part = m.group(1)
        groups = [g for g, pats in (("out", EXL3_OUT), ("in", EXL3_IN), ("fused3", (EXL3_FUSED3,)), ("rep", EXL3_REP))
                  if any(re.search(p, name) for p in pats)]
        if len(groups) != 1:
            raise ValueError(f"{name}: EXL3 split rule is ambiguous or missing ({groups})")
        return "rep" if part in ("mul1", "mcg") else EXL3_KINDS[groups[0]][part]
    if re.search(FUSED3_PLAIN, name):
        return "fused3_row"
    hits = [kind for kind, pats in (("row", ROW), ("col", COL), ("rep", REP)) if any(re.search(p, name) for p in pats)]
    if len(hits) != 1:
        raise ValueError(f"{name}: split rule is ambiguous or missing ({hits})")
    return hits[0]


def geometry_rule(name: str) -> str:
    """``rule`` with the fused kinds named by the axis they halve (row / dim1), for shape-only callers."""

    kind = rule(name)
    return GEOMETRY_KIND.get(kind, kind)


def check_exl3(name: str, shape: list[int], kind: str, ranks: int) -> None:
    """Raise unless every rank boundary of an EXL3 tensor (or the fused conv1d) is a multiple of 128 elements, the
    Hadamard block: then each rank's piece is bit-exact with its slice of the full linear."""

    m = EXL3_PART.search(name)
    if kind in ("rep", "drop") or not (m or re.search(FUSED3_PLAIN, name)):
        return
    axis = 1 if kind in ("dim1", "fused3_dim1") else 0
    parts = ranks * (3 if kind.startswith("fused3") else 1)
    tile = 16 if m and m.group(1) == "trellis" else 1        # trellis axes count 16-element tiles
    if len(shape) <= axis or shape[axis] % parts or (shape[axis] // parts * tile) % 128:
        raise ValueError(f"{name}: {kind} split of {shape} over {ranks} ranks leaves a boundary that is not a multiple "
                         "of the 128-wide Hadamard block")


def read_header(path: str | Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def split_bytes(raw: np.ndarray, shape: list[int], itemsize: int, kind: str, rank: int, ranks: int = 2) -> tuple[np.ndarray, list[int]]:
    """A tensor's bytes -> rank's part of them and its shape (ranks: 2 or 4)."""

    if kind == "rep":
        return raw, list(shape)
    if kind == "row":
        rows = shape[0]
        if rows % ranks:
            raise ValueError(f"row split of leading dim {shape} not divisible by {ranks}")
        per = raw.size // rows
        chunk = rows // ranks
        return raw[rank * chunk * per:(rank + 1) * chunk * per], [chunk] + list(shape[1:])
    if kind == "col":
        if len(shape) != 2 or shape[1] % ranks:
            raise ValueError(f"column split needs a 2-D shape divisible by {ranks}, got {shape}")
        view = raw.reshape(shape[0], shape[1] * itemsize)
        chunk = shape[1] // ranks
        part = np.ascontiguousarray(view[:, rank * chunk * itemsize:(rank + 1) * chunk * itemsize])
        return part.reshape(-1), [shape[0], chunk]
    if kind == "dim1":                                   # the second axis of a 2-D or higher tensor
        if len(shape) < 2 or shape[1] % ranks:
            raise ValueError(f"split of the second axis needs a second dim divisible by {ranks}, got {shape}")
        inner = int(np.prod(shape[2:])) * itemsize
        view = raw.reshape(shape[0], shape[1] * inner)
        chunk = shape[1] // ranks
        part = np.ascontiguousarray(view[:, rank * chunk * inner:(rank + 1) * chunk * inner])
        return part.reshape(-1), [shape[0], chunk] + list(shape[2:])
    if kind in FUSED3_AXIS:
        outer, inner, chunk, out = _fused3(shape, itemsize, kind, ranks)
        part = np.ascontiguousarray(raw.reshape(outer, 3, ranks, chunk * inner)[:, :, rank])
        return part.reshape(-1), out
    raise ValueError(kind)


FUSED3_AXIS = {"fused3_row": 0, "fused3_dim1": 1}


def _fused3(shape: list[int], itemsize: int, kind: str, ranks: int) -> tuple[int, int, int, list[int]]:
    """A [.. | 3*n | ..] axis holding [a | b | c] -> rank r's [a_r | b_r | c_r]: (outer count, inner bytes, rows per
    rank and third, the rank's shape)."""

    axis = FUSED3_AXIS[kind]
    if len(shape) <= axis or shape[axis] % (3 * ranks):
        raise ValueError(f"{kind} split needs axis {axis} divisible by 3*{ranks}, got {shape}")
    outer = int(np.prod(shape[:axis]))
    inner = int(np.prod(shape[axis + 1:])) * itemsize
    chunk = shape[axis] // (3 * ranks)
    return outer, inner, chunk, list(shape[:axis]) + [3 * chunk] + list(shape[axis + 1:])


def split_device(raw, shape: list[int], itemsize: int, kind: str, rank: int, ranks: int = 2):
    """split_bytes for a uint8 tensor on the GPU: the rank's part (a new contiguous tensor) and its shape."""

    if kind == "rep":
        return raw.clone(), list(shape)
    if kind == "row":
        if shape[0] % ranks:
            raise ValueError(f"row split of leading dim {shape} not divisible by {ranks}")
        per, chunk = raw.numel() // shape[0], shape[0] // ranks
        return raw[rank * chunk * per:(rank + 1) * chunk * per].clone(), [chunk] + list(shape[1:])
    if kind in ("col", "dim1"):
        if len(shape) < 2 or shape[1] % ranks or (kind == "col" and len(shape) != 2):
            raise ValueError(f"{kind} split needs a second dim divisible by {ranks}, got {shape}")
        inner = int(np.prod(shape[2:])) * itemsize
        chunk = shape[1] // ranks
        part = raw.view(shape[0], shape[1] * inner)[:, rank * chunk * inner:(rank + 1) * chunk * inner]
        return part.contiguous().reshape(-1), [shape[0], chunk] + list(shape[2:])
    if kind in FUSED3_AXIS:
        outer, inner, chunk, out = _fused3(shape, itemsize, kind, ranks)
        return raw.view(outer, 3, ranks, chunk * inner)[:, :, rank].contiguous().reshape(-1), out
    raise ValueError(kind)


def torch_dtype(dtype: str):
    import torch

    return {"U32": torch.uint32, "I32": torch.int32, "F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16,
            "I16": torch.int16, "U16": torch.uint16, "U8": torch.uint8, "I8": torch.int8, "I64": torch.int64,
            "F64": torch.float64}[dtype]


def rank_files(model_dir: str | Path, rank: int) -> list[Path]:
    return sorted(Path(model_dir).glob(f"*.rank{rank}.safetensors"))


class RankReader:
    """Read stored-dtype CPU tensors for one rank from the full checkpoint or its pre-split folder."""

    rule = staticmethod(rule)                             # name -> split kind (another family can pass its own)

    def __init__(self, model_dir: str | Path, rank: int, ranks: int = 2) -> None:
        from tensorfold.cuda.direct_read import ReadAhead, Reader, SafeTensors

        self.dir, self.rank, self.ranks = Path(model_dir), rank, ranks
        self.io = Reader()                                # O_DIRECT reads where the file system allows them
        self.reads = ReadAhead(self.io, READERS, RUN, GAP)
        mine, other = rank_files(self.dir, rank), rank_files(self.dir, 1 - rank)
        if other and not mine:
            raise ValueError(f"{self.dir} holds rank {1 - rank}'s share: give rank {rank} its own folder or the "
                             "full checkpoint")
        self.split = bool(mine)
        if self.split:
            self.folder = SafeTensors(mine, self.io)
            self.index = dict.fromkeys(self.folder.keys())
            return
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            names = json.loads(index.read_text())["weight_map"]
        else:
            names = {k: p.name for p in sorted(self.dir.glob("*.safetensors")) for k in read_header(p)[0]
                     if k != "__metadata__"}
        self.files: dict[str, tuple[dict, int]] = {}
        self.index = dict(names)

    def prefetch(self, names, device=None) -> None:
        """Start reading ``names`` (a layer's hundreds of small expert tensors) in shared reads; with a CUDA ``device``, ``get`` returns them uploaded."""

        items = []
        for name in dict.fromkeys(names):
            span = self._span(name)
            items.append((name, span[0], span[1], span[2], span))
        self.reads.queue(items, device, self._cut)

    @property
    def ahead(self) -> dict:
        return self.reads.ahead

    def get(self, name: str):
        out = self.reads.take(name)
        return out if out is not None else self._read(name)

    def close(self) -> None:
        self.reads.close()

    def _span(self, name: str) -> tuple[str, int, int, str, list[int], str]:
        """(file, first byte, end byte, split kind, shape, dtype) of the bytes this rank reads for ``name``."""

        if self.split:                                    # a rank folder holds the rank's tensors as they are
            path, begin, n, dtype, shape = self.folder.where[name]
            return str(path), begin, begin + n, "rep", list(shape), dtype
        file = str(self.dir / self.index[name])
        if file not in self.files:
            self.files[file] = read_header(file)
        header, base = self.files[file]
        info = header[name]
        kind = self.rule(name)
        if kind == "drop":
            raise KeyError(f"{name} is not used by the engine")
        a, b = info["data_offsets"]
        shape = list(info["shape"])
        check_exl3(name, shape, kind, self.ranks)
        if kind == "row" and shape and shape[0] % self.ranks == 0:   # the rank's rows are one run: read only those
            chunk = shape[0] // self.ranks
            per = (b - a) // shape[0] * chunk
            a, b, kind, shape = a + self.rank * per, a + (self.rank + 1) * per, "rep", [chunk] + shape[1:]
        return file, base + a, base + b, kind, shape, info["dtype"]

    def _tensor(self, raw: np.ndarray, span: tuple, own: bool):
        """The rank's tensor from the span's bytes; ``own``: never a view of ``raw`` (a shared read's buffer)."""

        import torch

        _, _, _, kind, shape, dtype = span
        data, shape = split_bytes(raw, shape, DTYPE_BYTES[dtype], kind, self.rank, self.ranks)
        if own and np.may_share_memory(data, raw):
            data = data.copy()
        return torch.from_numpy(data).view(torch_dtype(dtype)).reshape(shape)

    def _read(self, name: str):
        if self.split:
            return self.folder.get(name)
        span = self._span(name)
        return self._tensor(self.io.read(span[0], span[1], span[2] - span[1]).numpy(), span, own=False)

    def _cut(self, raw, span: tuple):
        """The rank's tensor from its span's bytes, a view of a shared read (so copied), on the host or the device."""

        if not raw.is_cuda:
            return self._tensor(raw.numpy(), span, own=True)
        _, _, _, kind, shape, dtype = span
        data, shape = split_device(raw, shape, DTYPE_BYTES[dtype], kind, self.rank, self.ranks)
        return data.view(torch_dtype(dtype)).reshape(shape)

def write(path: str, tensors: list[tuple[str, str, list[int], np.ndarray]], metadata: dict | None) -> None:
    header: dict = {}
    offset = 0
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + data.size]}
        offset += data.size
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, _, _, data in tensors:
            f.write(memoryview(data))
    os.replace(tmp, path)


def split_file(src: str | Path, out: str | Path, rank: int, ranks: int = 2) -> dict:
    """One checkpoint file -> OUT/<stem>.rank<R>.safetensors with the rank's part of every tensor it keeps."""

    header, base = read_header(src)
    metadata = header.pop("__metadata__", None)
    mm = np.memmap(src, dtype=np.uint8, mode="r")
    stem = os.path.basename(str(src)).replace(".safetensors", "")
    summary = {"rep": 0, "row": 0, "col": 0, "dim1": 0, "fused3_row": 0, "fused3_dim1": 0, "drop": 0}
    part = []
    for name in sorted(header, key=lambda k: header[k]["data_offsets"][0]):
        info = header[name]
        kind = rule(name)
        summary[kind] += 1
        if kind == "drop":
            continue
        a, b = info["data_offsets"]
        itemsize = DTYPE_BYTES[info["dtype"]]
        check_exl3(name, info["shape"], kind, ranks)
        data, shape = split_bytes(mm[base + a:base + b], info["shape"], itemsize, kind, rank, ranks)
        if int(np.prod(shape)) * itemsize != data.size:
            raise ValueError(f"{name}: {shape} does not match {data.size} bytes")
        part.append((name, info["dtype"], shape, data))
    os.makedirs(out, exist_ok=True)
    if part:
        write(os.path.join(str(out), f"{stem}.rank{rank}.safetensors"), part, metadata)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", type=Path, help="the checkpoint (e.g. the snapshot `tensorfold pull` downloaded)")
    p.add_argument("--rank", type=int, choices=(0, 1, 2, 3), required=True, help="the rank this machine serves")
    p.add_argument("--ranks", type=int, choices=(2, 4), default=2, help="total ranks the weights split over")
    p.add_argument("out", type=Path, help="the folder to write (then: tensorfold serve OUT --tp 2 --rank R ...)")
    args = p.parse_args(argv)
    files = sorted(args.model_dir.glob("model-*.safetensors"))
    if not files:
        raise SystemExit(f"{args.model_dir}: no model-*.safetensors files")
    args.out.mkdir(parents=True, exist_ok=True)
    for name in SMALL:
        if (args.model_dir / name).exists():
            shutil.copyfile(args.model_dir / name, args.out / name)
    for src in files:
        stem = src.name.replace(".safetensors", "")
        if (args.out / f"{stem}.rank{args.rank}.safetensors").exists():
            continue
        print(src.name, split_file(src, args.out, args.rank, args.ranks), flush=True)
    print(f"rank {args.rank}'s share of {len(files)} files in {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

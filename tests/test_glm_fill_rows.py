"""FILL_ROWS=2048 (BadAd84's setting in Mia's recipe), CPU only: a port of her ``tools/fill_rows_check.py``.
deploy/redsnow/start.sh gives both ranks TF_GLM_FILL_ROWS=2048 (at most TF_GLM_PREFILL_ROWS), the engine takes it at
its prompt grid and refuses more than its prompt chunk, and the fill rows size no buffer (the same memory)."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("torch")

from tensorfold.families.glm5_next import cuda as glm
from tensorfold.families.glm5_next.cuda import decode, engine, multi

START = Path(__file__).resolve().parents[1] / "deploy/redsnow/start.sh"


def test_start_sh_gives_both_ranks_2048_rows_at_most_the_prompt_chunk():
    text = START.read_text()
    lines = "\n".join(re.findall(r"^(?:_fill_max|FILL_ROWS)=.*$", text, re.M))
    assert '    TF_GLM_FILL_ROWS="$FILL_ROWS"\n' in text            # in COMMON_ENV, which both ranks' env gets

    def rows(**env):
        out = subprocess.run(["bash", "-c", lines + '\necho "$FILL_ROWS"'], env={"PATH": "/usr/bin:/bin", **env},
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()

    assert rows() == "2048"
    assert rows(FILL_ROWS="1024") == "1024"
    assert rows(TF_GLM_PREFILL_ROWS="1024") == "1024"


def test_the_engine_takes_2048_and_the_fill_rows_size_no_buffer():
    assert decode.PREFILL_ROWS == 2048 and multi.fill_rows(2048, "2048", engine.GRID) == 2048
    with pytest.raises(ValueError):
        multi.fill_rows(1024, "2048", engine.GRID)          # past the prompt chunk: refused at startup
    alloc = {"empty", "zeros", "ones", "full", "empty_like", "zeros_like", "Buffers", "reserve", "alloc", "allocate",
             "buffer_rows", "index_ring", "mla_geometry", "CUDAGraph"}
    reads, sized = 0, []
    for f in sorted(Path(glm.__file__).parent.glob("*.py")):
        tree = ast.parse(f.read_text())
        parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and n.attr == "fill_rows" and isinstance(n.ctx, ast.Load):
                reads += 1
                p = n
                while p in parents:
                    p = parents[p]
                    if isinstance(p, ast.Call):
                        fn = p.func.attr if isinstance(p.func, ast.Attribute) else getattr(p.func, "id", "")
                        if fn in alloc:
                            sized.append(f"{f.name}:{n.lineno} {fn}()")
    assert reads and not sized
    assert "self.pbuf = Buffers(w, prefill_rows, capacity, prefill=True" in Path(decode.__file__).read_text()
    assert Path(multi.__file__).read_text().count("stop - start > e.prefill_rows") >= 2

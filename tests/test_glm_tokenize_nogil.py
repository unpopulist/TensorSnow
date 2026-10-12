"""Patch 0099 (MiaAI-Lab's glm-tokenize-nogil, by BadAd84), CPU only: every tokenizer call in the CUDA server goes
through ``App._encode_ids``, which gives ``encode``'s ids through ``encode_batch_fast`` with Python's GIL released.
A port of Mia's ``tools/tokenize_check.py`` over our checkpoint's tokenizer (``TF_TOKENIZER_JSON``, default
``/var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw/tokenizer.json``)."""

from __future__ import annotations

import ast
import os
import random
import threading
import time
from pathlib import Path

import pytest

tokenizers = pytest.importorskip("tokenizers")

import tensorfold
from tensorfold.cuda import server

TOKENIZER = Path(os.environ.get("TF_TOKENIZER_JSON", "/var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw/tokenizer.json"))
needs_tokenizer = pytest.mark.skipif(not TOKENIZER.is_file(), reason=f"no tokenizer at {TOKENIZER}")


def _app(tok):
    app = server.App.__new__(server.App)
    app.tok = tok
    return app


def test_every_server_tokenizer_call_goes_through_the_helper():
    """No ``.tok.encode`` outside ``_encode_ids`` in server.py, the helper at all 7 sites (Mia's 6 plus /decisions),
    and GlmVision.prepare on the batch call."""

    tree = ast.parse(Path(server.__file__).read_text())
    helper = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_encode_ids")
    inside = {id(n) for n in ast.walk(helper)}
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    held = [n.lineno for n in calls if n.func.attr == "encode" and id(n) not in inside
            and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "tok"]
    assert held == []
    assert sum(n.func.attr == "_encode_ids" for n in calls) == 7
    vision = (Path(tensorfold.__file__).parent / "vision/glm.py").read_text()
    assert "self.tok.encode(" not in vision and "self.tok.encode_batch_fast(" in vision


@needs_tokenizer
def test_same_ids_as_encode_and_the_gil_released():
    tok = tokenizers.Tokenizer.from_file(str(TOKENIZER))
    app = _app(tok)
    rnd = random.Random(591)
    words = "lattice probe ferrule datum spindle gauge trellis offset stylus kinematic fixture wafer".split()
    special = [t.content for t in tok.get_added_tokens_decoder().values()]
    ranges = [(0x4E00, 0x9FFF), (0x1F300, 0x1FAFF), (0x0300, 0x036F), (0x0000, 0x001F), (0x0400, 0x04FF)]
    texts = [Path(server.__file__).read_text(), (Path(tensorfold.__file__).parent / "vision/glm.py").read_text(),
             " ".join(rnd.choice(special + words + ["\n", "  ", "\t"]) for _ in range(2000)),
             "".join(chr(rnd.randint(*rnd.choice(ranges))) for _ in range(3000)),
             "", " ", "\n", "\n\n", " " * 1000, "\u00a0\u2003", "<|user|>", "</think>\n"]
    for text in texts:
        for flag in (False, True):
            assert app._encode_ids(text, flag) == tok.encode(text, add_special_tokens=flag).ids

    # a counting thread keeps at least half its idle rate during a long encode (encode leaves it near 1%)
    big = "\n".join(f"{i:06d} " + " ".join(rnd.choice(words) for _ in range(11)) for i in range(20000))
    count, stop = [0], [False]

    def spin():
        while not stop[0]:
            count[0] += 1

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    time.sleep(0.2)
    c0, t0 = count[0], time.perf_counter()
    time.sleep(0.3)
    idle = (count[0] - c0) / (time.perf_counter() - t0)
    c0, t0 = count[0], time.perf_counter()
    app._encode_ids(big, False)
    rate = (count[0] - c0) / (time.perf_counter() - t0) / idle
    stop[0] = True
    assert rate >= 0.5, f"the other thread ran at {rate:.0%} of its idle rate"

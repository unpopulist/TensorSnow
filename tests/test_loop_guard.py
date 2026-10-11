"""Mia's 0091 loop guard (TF_GLM_LOOP_GUARD) with the real ``call_gate.generate_gated``: a clean block, an exact cycle, a
mostly-one-token block, a loop after </think> (not watched), natural text (never fires), two guards, guard plus budget.
Ported from her tools/test_loop_guard.py (first pytest draft by the local Qwen, checked and edited)."""
from tensorfold.engine.loop_guard import LoopGuard
from tensorfold.engine.call_gate import ThinkBudget, generate_gated

# --- Loop Guard Tests (converted from tools/test_loop_guard.py) ---

THINK_END = 7
CLOSE = [10, 7, 11]

def _run_gated(gates, stream, rounds=4, cap=3000):
    """Helper to run generate_gated with a mock generator."""
    reply_out = []
    
    def generate(ids, count, take):
        tail = list(stream)[len(reply_out):]
        if CLOSE[-1] in ids:
            tail = [50, 51, 52, 2]
        while tail and count > 0:
            chunk, tail = tail[:rounds], tail[rounds:]
            chunk = chunk[:count]
            if take(chunk):
                return {}
            count -= len(chunk)
        return {}

    out = []
    def on_tokens(new):
        out.extend(new)
        reply_out[:] = out
        return False

    generate_gated(generate, [0], cap, gates, on_tokens)
    return out

def test_loop_guard_clean():
    import random
    rnd = random.Random(1)
    prose = [rnd.randrange(100, 5000) for _ in range(900)]
    clean = prose[:600] + [THINK_END, 50, 51, 2]
    g = LoopGuard(CLOSE, THINK_END)
    out = _run_gated([g], clean)
    assert out == clean
    assert g.fires == 0

def test_loop_guard_exact_cycle():
    import random
    rnd = random.Random(1)
    prose = [rnd.randrange(100, 5000) for _ in range(900)]
    cyc = prose[:300] + [9, 8, 9, 6] * 400
    g = LoopGuard(CLOSE, THINK_END)
    out = _run_gated([g], cyc)
    assert g.fires == 1
    assert CLOSE == out[-len(CLOSE) - 4:-4]
    assert out[-4:] == [50, 51, 52, 2]
    assert len(out) < 300 + 256 + 4 + 8 + len(CLOSE) + 8

def test_loop_guard_dominant_token():
    import random
    rnd = random.Random(1)
    prose = [rnd.randrange(100, 5000) for _ in range(900)]
    mixed = []
    for i in range(1200):
        mixed += [5, prose[i % len(prose)]] if i % 3 else [5, 5, prose[i % len(prose)]]
    g = LoopGuard(CLOSE, THINK_END)
    out = _run_gated([g], mixed)
    assert g.fires == 1
    assert len(out) < 700

def test_loop_guard_after_think_end():
    import random
    rnd = random.Random(1)
    prose = [rnd.randrange(100, 5000) for _ in range(900)]
    after = prose[:50] + [THINK_END] + [3] * 800
    g = LoopGuard(CLOSE, THINK_END)
    out = _run_gated([g], after)
    assert g.fires == 0
    assert out == after

def test_loop_guard_near_cycle_and_zipf():
    import random
    rnd = random.Random(1)
    prose = [rnd.randrange(100, 5000) for _ in range(900)]
    
    near = []
    for i in range(3000):
        near.append(prose[i % 11] if i % 12 else rnd.randrange(6000, 9000))
    g = LoopGuard(CLOSE, THINK_END)
    _run_gated([g], near, cap=3000)
    assert g.fires == 0
    
    zipf = rnd.choices(range(100, 5100), weights=[1 / r for r in range(1, 5001)], k=5000)
    g2 = LoopGuard(CLOSE, THINK_END)
    _run_gated([g2], zipf, cap=5000)
    assert g2.fires == 0

def test_loop_guard_independence():
    import random
    rnd = random.Random(1)
    prose = [rnd.randrange(100, 5000) for _ in range(900)]
    cyc = prose[:300] + [9, 8, 9, 6] * 400
    clean = prose[:600] + [THINK_END, 50, 51, 2]
    
    a = LoopGuard(CLOSE, THINK_END)
    b = LoopGuard(CLOSE, THINK_END)
    _run_gated([a], cyc)
    _run_gated([b], clean)
    assert a.fires == 1
    assert b.fires == 0

def test_loop_guard_with_budget():
    import random
    rnd = random.Random(1)
    prose = [rnd.randrange(100, 5000) for _ in range(900)]
    cyc = prose[:300] + [9, 8, 9, 6] * 400
    
    tb = ThinkBudget(350, CLOSE, THINK_END)
    g = LoopGuard(CLOSE, THINK_END)
    out = _run_gated([tb, g], cyc)
    assert out.count(THINK_END) == 1


# --- through the CUDA server ---

import json  # noqa: E402

import pytest  # noqa: E402

from tests.test_cuda_admission import http_server, post  # noqa: E402
from tests.test_cuda_tool_choice import app_for  # noqa: E402
from tests.test_glm_tool_stream import END, EOS, GlmTokens  # noqa: E402


class Looper:
    """Thinks "maybe " forever; once the prompt holds a closed think block it answers."""

    eos = (EOS,)

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        tok = GlmTokens()
        reply = tok.encode("Answer.").ids + [EOS] if END in prompt else tok.encode("maybe " * 400).ids
        reply = reply[:max_tokens]
        for at in range(0, len(reply), 4):
            if on_tokens(reply[at:at + 4]):
                break
        return {"rounds": 1 + len(reply) // 4}


@pytest.mark.parametrize("guarded", ["0", "1"])
def test_the_server_closes_a_collapsed_think_block(tmp_path, monkeypatch, guarded):
    monkeypatch.setenv("TF_GLM_LOOP_GUARD", guarded)
    app = app_for(tmp_path, Looper())
    app.tok = GlmTokens()
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2000,
            "chat_template_kwargs": {"enable_thinking": True}}
    with http_server(app) as port:
        status, reply = post(port, body, True)
    assert status == 200, reply
    choice = json.loads(reply)["choices"][0]
    stats = json.loads(reply).get("tensorfold") or {}
    if guarded == "1":
        assert choice["message"]["content"] == "Answer." and choice["finish_reason"] == "stop"
        assert choice["message"]["reasoning_content"].count("maybe") < 300 and stats["loop_guard"] == 1
    else:
        assert not choice["message"]["content"] and choice["finish_reason"] == "length"
        assert "loop_guard" not in stats

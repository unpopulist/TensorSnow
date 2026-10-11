"""Mia's 0061 smooth streaming (TF_GLM_STREAM_SMOOTH): the playout queue paces text, keeps order, flushes at the end,
and the CUDA server streams the same text and calls through it."""

from __future__ import annotations

import json
import threading
import time

import pytest

from tensorfold.cuda.stream_pacing import LONGEST_HOLD, Playout, hold_seconds, smooth_streaming
from tests.test_cuda_admission import http_server, post
from tests.test_cuda_tool_choice import app_for, events
from tests.test_glm_tool_stream import TOOLS, GlmTokens, Writer, glm_call


class Sink:
    def __init__(self, ok=True):
        self.items, self.times, self.ok = [], [], ok
        self.lock = threading.Lock()

    def __call__(self, delta):
        with self.lock:
            self.items.append(delta)
            self.times.append(time.monotonic())
        return self.ok


def test_events_go_out_in_order_and_close_waits_for_them():
    sink = Sink()
    playout = Playout(sink, hold=0.05)
    for i in range(6):
        playout.put({"i": i}, paced=i % 2 == 0)
    playout.close()
    assert [d["i"] for d in sink.items] == list(range(6))
    assert not playout.thread.is_alive()


def test_a_burst_is_spread_over_the_hold_but_never_held_too_long():
    sink, hold = Sink(), 0.2
    playout = Playout(sink, hold=hold)
    start = time.monotonic()
    for i in range(10):
        playout.put({"i": i})
    time.sleep(hold * LONGEST_HOLD + 0.1)        # long enough for every token to be due
    playout.close()
    assert len(sink.items) == 10
    assert sink.times[-1] - sink.times[0] >= 0.1
    assert sink.times[-1] - start <= hold * LONGEST_HOLD + 0.1


def test_flush_sends_what_is_queued_at_once():
    sink = Sink()
    playout = Playout(sink, hold=2.0)
    for i in range(8):
        playout.put({"i": i})
    flushed = time.monotonic()
    playout.flush()
    playout.close()
    assert len(sink.items) == 8 and sink.times[-1] - flushed < 0.2


def test_unpaced_events_are_not_delayed():
    sink = Sink()
    playout = Playout(sink, hold=2.0)
    start = time.monotonic()
    playout.put({"tool_calls": []}, paced=False)
    deadline = start + 1.0
    while not sink.items and time.monotonic() < deadline:
        time.sleep(0.005)
    assert sink.items and sink.times[0] - start < 0.2
    playout.abort()


def test_a_client_that_left_drops_the_rest():
    sink = Sink(ok=False)
    playout = Playout(sink, hold=0.01)
    playout.put({"i": 0}, paced=False)
    playout.thread.join(1.0)
    playout.put({"i": 1}, paced=False)          # after gone: dropped
    playout.close()
    assert sink.items == [{"i": 0}] and playout.gone


def test_abort_drops_what_is_queued():
    sink = Sink()
    playout = Playout(sink, hold=5.0)
    playout.put({"i": 0})                        # the first goes out at once (nothing sent before it)
    for i in range(1, 5):
        playout.put({"i": i})
    time.sleep(0.05)
    playout.abort()
    assert len(sink.items) < 5 and not playout.thread.is_alive()


def test_settings_are_checked():
    assert smooth_streaming("1") is True and smooth_streaming("0") is False and smooth_streaming("") is False
    with pytest.raises(ValueError):
        smooth_streaming("yes")
    assert hold_seconds("400") == 0.4 and hold_seconds("") == 0.4
    for bad in ("0", "abc", "6000"):
        with pytest.raises(ValueError):
            hold_seconds(bad)


@pytest.mark.parametrize("smooth", ["0", "1"])
def test_the_server_streams_the_same_text_and_calls_smoothed(tmp_path, monkeypatch, smooth):
    monkeypatch.setenv("TF_GLM_STREAM_SMOOTH", smooth)
    monkeypatch.setenv("TF_GLM_STREAM_SMOOTH_MS", "20")
    text = "Reading the file now.\n" + glm_call("bash", {"command": "ls"})
    app = app_for(tmp_path, Writer(text))
    app.tok = GlmTokens()
    body = {"messages": [{"role": "user", "content": "go"}], "tools": TOOLS, "stream": True, "max_tokens": 4096}
    with http_server(app) as port:
        status, reply = post(port, body, True)
    assert status == 200, reply
    deltas = [c["choices"][0]["delta"] for c in events(reply) if c.get("choices")]
    content = "".join(d.get("content", "") for d in deltas)
    calls = [(d["tool_calls"][0]["function"]["name"], json.loads(d["tool_calls"][0]["function"]["arguments"]))
             for d in deltas if d.get("tool_calls")]
    assert content.strip() == "Reading the file now." and calls == [("bash", {"command": "ls"})]
    pieces = [d["content"] for d in deltas if d.get("content")]
    if smooth == "1":                            # one event a token: no event carries a whole round
        assert max(len(p) for p in pieces) == 1

"""Mia's server polish: /health's fields as Prometheus families (0045), capacity refusals as 429 + Retry-After on
every route (0081), and the Responses API's ``include`` accepted and ignored (0093)."""

from __future__ import annotations

import http.client
import json
from types import SimpleNamespace

import pytest

from tensorfold.cuda import health
from tensorfold.server import metrics, responses
from tensorfold.server.anthropic_translate import error as anthropic_error
from tensorfold.server.errors import CapacityError, RequestError, admission_response
from tensorfold.server.responses import Wire
from tests.test_cuda_admission import http_server
from tests.test_cuda_tool_choice import Engine, app_for
from tests.test_metrics import sample


# -- 0045: /health under tensorfold_health: -----------------------------------------------------------------------------

def test_health_totals_and_streams_are_prometheus_families():
    decoder = SimpleNamespace(streams={1: SimpleNamespace(prompt=[0] * 30, out=[0] * 10, context=[0] * 10)},
                              filling=[])
    app = SimpleNamespace(engine=SimpleNamespace(scheduler=SimpleNamespace(decoder=decoder, max_streams=4)),
                          effective_context_window=100, context_window=200)
    out: list[int] = []
    with health.of(app).running(5, out) as request:
        out.extend([1, 2])
        request.stats = {"rounds": 3, "prefill_s": 0.5, "cached": 4}
    body = metrics.render(app)
    assert sample(body, f"{metrics.HEALTH}requests_total") == "1"
    assert sample(body, f"{metrics.HEALTH}rounds_total") == "3"
    assert sample(body, f"{metrics.HEALTH}cached_tokens_total") == "4"
    assert sample(body, f"{metrics.HEALTH}context_length") == "100"
    assert sample(body, f"{metrics.HEALTH}streams_max") == "4"
    assert f'{metrics.HEALTH}streams{{state="decoding"}} 1' in body
    assert f'{metrics.HEALTH}streams{{state="filling"}} 0' in body
    # the effective window (100), and the stream's prompt plus reply (40): not the reply alone, not the native window
    assert f'{metrics.PREFIX}kv_cache_usage_ratio{{pool="0"}} 0.4' in body
    assert "# TYPE tensorfold_health:requests_total counter" in body


def test_an_app_without_the_cuda_health_gets_no_health_families_and_none_is_made():
    app = SimpleNamespace()
    assert metrics.HEALTH not in metrics.render(app)
    assert "health" not in vars(app)


def test_a_table_that_changes_under_the_read_is_read_again():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("dictionary changed size during iteration")
        return 7

    assert metrics._read(flaky, 0) == 7
    assert metrics._read(lambda: (_ for _ in ()).throw(RuntimeError("x")), "default") == "default"


# -- 0081: 429 + Retry-After ---------------------------------------------------------------------------------------------

def test_admission_response():
    assert admission_response(CapacityError("busy")) == (429, {"Retry-After": "5"})
    assert admission_response(RequestError("bad")) == (400, None)
    assert anthropic_error("busy", 429)["error"]["type"] == "rate_limit_error"


@pytest.mark.parametrize("stream", [False, True])
def test_the_cuda_server_refuses_capacity_with_429_and_retry_after(tmp_path, stream):
    app = app_for(tmp_path, Engine())

    def busy(body, chat):
        raise CapacityError("the server is full; retry shortly")

    app.prepare = busy
    with http_server(app) as port:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.request("POST", "/v1/chat/completions",
                               json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": stream}),
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            status, retry, body = response.status, response.getheader("Retry-After"), response.read().decode()
        finally:
            connection.close()
    assert status == 429 and retry == "5" and "retry shortly" in body


def test_a_wrapped_chat_refusal_keeps_its_retry_after():
    wire = Wire(lambda: None, lambda event: None)
    wire.write(b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 5\r\nContent-Type: application/json\r\n\r\n{}")
    assert wire.status == 429 and wire.headers() == {"Retry-After": "5"}
    plain = Wire(lambda: None, lambda event: None)
    plain.write(b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n{}")
    assert plain.headers() is None


# -- 0093: Responses include ---------------------------------------------------------------------------------------------

BODY = {"model": "x", "store": False, "input": [{"role": "user", "content": [{"type": "input_text", "text": "OK?"}]}]}


@pytest.mark.parametrize("include", [None, [], ["reasoning.encrypted_content"],
                                     ["reasoning.encrypted_content", "message.output_text.logprobs"]])
def test_known_include_values_are_accepted_and_change_nothing(include):
    asked = responses.translate({**BODY, "include": include}, responses.Store())
    assert asked.chat["messages"] == responses.translate(BODY, responses.Store()).chat["messages"]


@pytest.mark.parametrize("include", [["nope"], "reasoning.encrypted_content", [1], {"a": 1}])
def test_unknown_or_malformed_include_is_refused(include):
    with pytest.raises(RequestError, match="include"):
        responses.translate({**BODY, "include": include}, responses.Store())

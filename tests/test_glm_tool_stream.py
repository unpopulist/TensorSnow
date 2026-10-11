"""GLM tool calls on the CUDA server (Mia's patches 0036, 0051, 0053, 0055): whole calls streamed as each closes, calls
written inside the think block, a call the token limit cuts (never sent) or the end token leaves open (closed when it
parses), a lost ``<arg_key>`` put back, schema typing (unions, nullable text, const), the agent history GLM's template
can render, and the reasoning kept for clients that drop it."""

from __future__ import annotations

import json
import re

import pytest

from tensorfold.cuda import server
from tensorfold.cuda.reply_text import (GlmCallStreamer, ThinkSplit, close_glm_call, parse_tool_calls,
                                        repair_glm_keys)
from tensorfold.engine.tool_draft import ToolCallStreamer
from tensorfold.families.glm5_next.cuda.kept_reasoning import KeptReasoning, signature
from tensorfold.families.glm5_next.prompts import agent_history
from tensorfold.server.messages import normalize_messages
from tensorfold.tool_parameters import decode_parameter, nullable_text, schema_types, typed_parameter
from tests.test_cuda_admission import http_server, post
from tests.test_cuda_tool_choice import app_for, events

TOOLS = [
    {"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "limit": {"type": "integer"}, "note": {"type": ["string", "null"]},
        "tags": {"type": "array"}}}}},
    {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}}}}},
]


def glm_call(name, args):
    return "<tool_call>" + name + "".join(
        f"<arg_key>{k}</arg_key><arg_value>{v if isinstance(v, str) else json.dumps(v)}</arg_value>"
        for k, v in args.items()) + "</tool_call>"


def calls(text, tools=TOOLS, **kw):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in
            parse_tool_calls(text, tools, **kw)[1] or []]


# -- schema typing (0036's tool_parameters) ----------------------------------------------------------------------------

def test_schema_types_read_unions_enums_and_const():
    assert schema_types({"type": ["integer", "null"]}) == {"integer", "null"}
    assert schema_types({"anyOf": [{"type": "string"}, {"type": "null"}]}) == {"string", "null"}
    assert schema_types({"anyOf": [{"type": "integer"}, {}]}) == frozenset()      # an untyped member: untyped
    assert schema_types({"enum": [1, 2]}) == {"integer"} and schema_types({"const": True}) == {"boolean"}
    assert typed_parameter({"type": ["integer", "null"]}) and not typed_parameter({"type": ["string", "null"]})
    assert nullable_text({"type": ["string", "null"]}) and not nullable_text({"type": "string"})


def test_decode_reads_unions_and_nullable_text():
    assert decode_parameter("5", {"type": ["integer", "null"]}) == 5
    assert decode_parameter("null", {"type": ["integer", "null"]}) is None
    assert decode_parameter("null", {"type": ["string", "null"]}) is None
    assert decode_parameter("nullish", {"type": ["string", "null"]}) == "nullish"
    assert decode_parameter("true", {"const": True}) is True
    assert decode_parameter('"5"', {"type": "integer"}, python=False) == '"5"'      # not coerced


def test_qwen_streamer_holds_a_nullable_text_value_while_it_may_be_null():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "a": {"type": ["string", "null"]}, "b": {"type": ["string", "null"]}}}}}]
    text = ("<tool_call>\n<function=f>\n<parameter=a>\nnull\n</parameter>\n<parameter=b>\nnuance\n</parameter>\n"
            "</function>\n</tool_call>")
    streamer, args = ToolCallStreamer(tools), ""
    for at in range(1, len(text) + 1):
        for delta in streamer.feed(text[:at]):
            args += delta["tool_calls"][0]["function"].get("arguments", "")
    assert json.loads(args) == {"a": None, "b": "nuance"}


# -- the call streamer: whole calls (0036 + 0053) ----------------------------------------------------------------------

def feed_by_char(streamer, text):
    out = []
    for at in range(1, len(text) + 1):
        out.append(streamer.feed(text[:at]))
    return out


def test_each_call_goes_out_whole_once_closed_and_matches_the_end_parser():
    first = glm_call("read", {"path": "a.txt", "limit": 5, "note": "null", "tags": ["x"]})
    second = glm_call("bash", {"command": 'echo "hi"\n'})
    text = "Reading.\n" + first + "\n" + second
    streamer = GlmCallStreamer(TOOLS)
    rounds = feed_by_char(streamer, text)
    sent = [(i, d) for i, out in enumerate(rounds) for d in out]
    assert [i for i, _ in sent] == [len("Reading.\n" + first) - 1, len(text) - 1]      # each at its </tool_call>
    got = []
    for _, delta in sent:
        call = delta["tool_calls"][0]
        assert call["id"].startswith("call_") and call["type"] == "function"
        got.append((call["function"]["name"], json.loads(call["function"]["arguments"])))
    assert got == calls(text) == [("read", {"path": "a.txt", "limit": 5, "note": None, "tags": ["x"]}),
                                  ("bash", {"command": 'echo "hi"\n'})]
    assert streamer.sent == 2 and not streamer.open


def test_a_call_still_open_is_never_sent():
    text = glm_call("bash", {"command": "rm -rf build"})[:-len("</arg_value></tool_call>")]
    streamer = GlmCallStreamer(TOOLS)
    assert not any(feed_by_char(streamer, text))
    assert streamer.open and streamer.sent == 0


def test_a_block_it_cannot_follow_is_left_to_the_end_parser():
    text = '<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>'
    streamer = GlmCallStreamer(TOOLS)
    assert not any(feed_by_char(streamer, text))
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in streamer.rest(TOOLS)] == \
        [("bash", {"command": "ls"})]


# -- the think block (0036's ThinkSplit) -------------------------------------------------------------------------------

def test_calls_that_end_the_reply_inside_the_think_block_are_its_calls():
    call = glm_call("bash", {"command": "ls"})
    split = ThinkSplit()
    assert split("let me look " + call, True) == ("let me look ", call)
    assert ThinkSplit()("let me look " + call + "\n</think>", True) == ("let me look ", call)


def test_a_call_followed_by_more_reasoning_is_reasoning():
    call = glm_call("bash", {"command": "ls"})
    reasoning, answer = ThinkSplit()("maybe " + call + " no, wait</think>Done.", True)
    assert reasoning == "maybe " + call + " no, wait" and answer == "Done."


def test_a_held_call_is_not_reasoning_while_it_may_still_be_one():
    call = glm_call("bash", {"command": "ls"})
    split = ThinkSplit()
    assert split("thinking " + call, False) == ("thinking ", "")
    assert split.stream_from is None


def test_a_long_call_in_the_think_block_is_committed_past_the_hold():
    call = glm_call("bash", {"command": "x" * 64})
    split = ThinkSplit(hold=32)
    reasoning, answer = split("plan " + call[:48], False)
    assert reasoning == "plan " and split.committed == len("plan ") == split.stream_from


# -- a lost <arg_key> and a call the end token left open (0055) -------------------------------------------------------

def test_a_missing_arg_key_is_put_back():
    broken = "read path</arg_key><arg_value>a.txt</arg_value>\nlimit</arg_key><arg_value>3</arg_value>"
    assert repair_glm_keys(broken) == ("read <arg_key>path</arg_key><arg_value>a.txt</arg_value>\n"
                                       "<arg_key>limit</arg_key><arg_value>3</arg_value>")
    whole = "read<arg_key>path</arg_key><arg_value>a.txt</arg_value>"
    assert repair_glm_keys(whole) == whole
    assert calls(f"<tool_call>{broken}</tool_call>") == [("read", {"path": "a.txt", "limit": 3})]


def test_the_streamer_sends_a_repaired_call_as_the_end_parser_reads_it():
    # a later key lost: the streamer follows the call and sends it with the end parser's repaired arguments
    text = "<tool_call>read<arg_key>path</arg_key><arg_value>a.txt</arg_value>\nlimit</arg_key><arg_value>3</arg_value></tool_call>"
    streamer = GlmCallStreamer(TOOLS)
    sent = [d for out in feed_by_char(streamer, text) for d in out]
    assert [json.loads(d["tool_calls"][0]["function"]["arguments"]) for d in sent] == [{"path": "a.txt", "limit": 3}]
    # the first key lost: the name reads "read path", which the streamer leaves to the end parser
    text = "<tool_call>read path</arg_key><arg_value>a.txt</arg_value></tool_call>"
    streamer = GlmCallStreamer(TOOLS)
    assert not any(feed_by_char(streamer, text))
    assert [json.loads(c["function"]["arguments"]) for c in streamer.rest(TOOLS)] == [{"path": "a.txt"}]


def test_close_glm_call_closes_only_a_call_that_then_parses():
    assert close_glm_call("ok " + glm_call("bash", {"command": "ls"})[:-len("</arg_value></tool_call>")], TOOLS) == \
        "</arg_value></tool_call>"
    assert close_glm_call("<tool_call>bash<arg_key>command</arg_key><arg_value>ls</arg_value>", TOOLS) == \
        "</tool_call>"
    assert close_glm_call("<tool_call>nosuch<arg_key>a</arg_key><arg_value>1", TOOLS) == ""
    assert close_glm_call("no call at all", TOOLS) == ""
    assert close_glm_call(glm_call("bash", {"command": "ls"}), TOOLS) == ""


# -- the history GLM's template can render (0036 + 0051) --------------------------------------------------------------

def history(arguments, results=None):
    return [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "bash", "arguments": arguments}}]},
            *(results or [{"role": "tool", "tool_call_id": "call_a", "content": "ok"}])]


@pytest.mark.parametrize("empty", ["", "  ", None])
def test_empty_arguments_are_no_arguments(empty):
    out = agent_history(history(empty))
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {}
    assert history(empty)[1]["tool_calls"][0]["function"]["arguments"] == empty     # the caller's list unchanged


def test_garbled_arguments_are_kept_for_the_shared_normaliser_never_refused(capsys):
    out = agent_history(history('{"command": "ls'))
    assert out[1]["tool_calls"][0]["function"]["arguments"] == '{"command": "ls'
    assert out[2] == {"role": "tool", "tool_call_id": "call_a", "content": "ok"}
    assert "not a JSON object" in capsys.readouterr().err


def test_results_whose_ids_match_no_call_render_in_order():
    out = agent_history(history("{}", [{"role": "tool", "tool_call_id": "renumbered", "content": "ok"}]))
    assert out[2] == {"role": "tool", "content": "ok"}
    kept = agent_history(history("{}"))
    assert kept[2]["tool_call_id"] == "call_a"


def test_a_steps_reasoning_named_reasoning_reaches_the_template():
    out = normalize_messages([{"role": "user", "content": "q"},
                              {"role": "assistant", "content": "a", "reasoning": "because"}])
    assert out[1]["reasoning_content"] == "because"
    both = normalize_messages([{"role": "user", "content": "q"},
                               {"role": "assistant", "content": "a", "reasoning": "x", "reasoning_content": "y"}])
    assert both[1]["reasoning_content"] == "y"


# -- kept reasoning (0036's kept_reasoning) ----------------------------------------------------------------------------

def test_kept_reasoning_is_put_back_where_the_client_dropped_it():
    kept = KeptReasoning(entries=4)
    asked = [{"role": "user", "content": "go"}]
    reply_calls = [{"id": "call_" + "0" * 24, "type": "function", "function": {"name": "bash",
                                                                               "arguments": '{"command":"ls"}'}}]
    kept.remember(asked, reply_calls, "I should list the files.")
    # the client renumbers the id and re-serialises the arguments: the signature still finds it
    resent = [*asked, {"role": "assistant", "content": "", "tool_calls": [
        {"id": "abc", "type": "function", "function": {"name": "bash", "arguments": {"command": "ls"}}}]},
              {"role": "tool", "tool_call_id": "abc", "content": "a.txt"}]
    restored = kept.restore(resent)
    assert restored[1]["reasoning_content"] == "I should list the files." and "reasoning_content" not in resent[1]
    # a step that carries its own reasoning is never changed; another conversation never gets it
    own = [resent[0], {**resent[1], "reasoning_content": "mine"}, resent[2]]
    assert kept.restore(own)[1]["reasoning_content"] == "mine"
    other = [{"role": "user", "content": "something else"}, *resent[1:]]
    assert "reasoning_content" not in kept.restore(other)[1]


def test_kept_reasoning_is_bounded():
    kept = KeptReasoning(entries=2)
    for n in range(3):
        kept.remember([{"role": "user", "content": str(n)}], [{"function": {"name": "bash", "arguments": "{}"}}], "r")
    assert len(kept) == 2
    assert signature([{"role": "user", "content": "0"}], [{"function": {"name": "bash", "arguments": "{}"}}]) \
        not in kept.by_key


def test_kept_reasoning_reads_its_env(monkeypatch):
    monkeypatch.setenv("TF_GLM_KEEP_REASONING", "0")
    assert KeptReasoning.from_env() is None
    monkeypatch.setenv("TF_GLM_KEEP_REASONING", "many")
    with pytest.raises(ValueError):
        KeptReasoning.from_env()


# -- through the CUDA server -------------------------------------------------------------------------------------------

EOS, OPEN, CLOSE, THINK, END, KEY, KEY_END, VALUE, VALUE_END = 0, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007
SPECIAL = {"<tool_call>": OPEN, "</tool_call>": CLOSE, "<think>": THINK, "</think>": END, "<arg_key>": KEY,
           "</arg_key>": KEY_END, "<arg_value>": VALUE, "</arg_value>": VALUE_END}


class GlmTokens:
    """Characters as their code points, GLM's markup as single tokens (so the server sees a GLM call format)."""

    def encode(self, text, **kwargs):
        ids = []
        for part in re.split("(" + "|".join(re.escape(s) for s in SPECIAL) + ")", text):
            ids += [SPECIAL[part]] if part in SPECIAL else [ord(c) for c in part]
        return type("Encoding", (), {"ids": ids})()

    def decode(self, ids, **kwargs):
        names = {v: k for k, v in SPECIAL.items()}
        return "".join(names.get(i, "" if i == EOS else chr(i)) for i in ids)

    def token_to_id(self, text):
        return SPECIAL.get(text)


class Writer:
    """Writes ``text`` (then EOS unless ``eos`` is False), three tokens a round."""

    eos = (EOS,)

    def __init__(self, text, eos=True):
        self.reply = GlmTokens().encode(text).ids + ([EOS] if eos else [])

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        reply = self.reply[:max_tokens]
        for at in range(0, len(reply), 3):
            if on_tokens(reply[at:at + 3]):
                break
        return {"rounds": 1 + len(reply) // 3}


def ask(tmp_path, text, *, stream, thinking=False, eos=True, max_tokens=4096):
    app = app_for(tmp_path, Writer(text, eos))
    app.tok = GlmTokens()
    body = {"messages": [{"role": "user", "content": "go"}], "tools": TOOLS, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": thinking}, "max_tokens": max_tokens}
    with http_server(app) as port:
        status, reply = post(port, body, True)
    assert status == 200, reply
    if not stream:
        choice = json.loads(reply)["choices"][0]
        message = choice["message"]
        got = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in message.get("tool_calls") or []]
        return got, message.get("content") or "", choice["finish_reason"], []
    chunks = [c for c in events(reply) if c.get("choices")]
    content, deltas = "", []
    for chunk in chunks:
        delta = chunk["choices"][0]["delta"]
        content += delta.get("content", "") or ""
        deltas += delta.get("tool_calls", [])
    assert all("id" in d and d["function"]["name"] for d in deltas), deltas        # every call delta is whole
    got = [(d["function"]["name"], json.loads(d["function"]["arguments"])) for d in deltas]
    return got, content, chunks[-1]["choices"][0]["finish_reason"], deltas


FIRST = {"path": "a.txt", "limit": 2}
SECOND = {"command": "ls"}
TWO = "Looking.\n" + glm_call("read", FIRST) + "\n" + glm_call("bash", SECOND)


@pytest.mark.parametrize("stream", [False, True])
def test_two_calls_arrive_whole_streamed_or_not(tmp_path, stream):
    got, content, finish, deltas = ask(tmp_path, TWO, stream=stream)
    assert got == [("read", FIRST), ("bash", SECOND)] and finish == "tool_calls"
    assert content.strip() == "Looking." and "<tool_call>" not in content
    if stream:
        assert [d["index"] for d in deltas] == [0, 1]


@pytest.mark.parametrize("stream", [False, True])
def test_a_call_the_token_limit_cuts_is_never_sent(tmp_path, stream):
    text = "Looking.\n" + glm_call("bash", {"command": "rm -rf build"})
    limit = len(GlmTokens().encode(text).ids) - 4                                     # inside the value
    got, content, finish, _ = ask(tmp_path, text, stream=stream, max_tokens=limit)
    assert got == [] and finish == "length"
    assert "<tool_call>" not in content and "rm -rf" not in content and content.strip() == "Looking."


@pytest.mark.parametrize("stream", [False, True])
def test_a_call_the_end_token_leaves_open_is_closed_when_it_parses(tmp_path, stream):
    text = "Looking.\n" + glm_call("bash", SECOND)[:-len("</arg_value></tool_call>")]
    got, content, finish, _ = ask(tmp_path, text, stream=stream)
    assert got == [("bash", SECOND)] and finish == "tool_calls" and "<tool_call>" not in content


@pytest.mark.parametrize("stream", [False, True])
def test_a_call_written_inside_the_think_block_that_ends_the_reply_is_a_call(tmp_path, stream):
    text = "I will list the files.\n" + glm_call("bash", SECOND)
    got, content, finish, _ = ask(tmp_path, text, stream=stream, thinking=True)
    assert got == [("bash", SECOND)] and finish == "tool_calls" and content == ""


def test_an_open_call_gets_empty_keepalive_deltas(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "CALL_KEEPALIVE_S", 0.0)
    app = app_for(tmp_path, Writer(glm_call("bash", {"command": "x" * 30})))
    app.tok = GlmTokens()
    body = {"messages": [{"role": "user", "content": "go"}], "tools": TOOLS, "stream": True, "max_tokens": 4096}
    with http_server(app) as port:
        status, reply = post(port, body, True)
    assert status == 200
    deltas = [c["choices"][0]["delta"] for c in events(reply) if c.get("choices")]
    assert sum(1 for d in deltas if d == {}) >= 5
    assert [d["tool_calls"][0]["function"]["name"] for d in deltas if d.get("tool_calls")] == ["bash"]

"""Mia's 0096 TF_GLM_EFFORT_TAIL: GLM's reasoning-effort line rendered at the prompt's tail, so the conversation
before it renders alike for every effort and with thinking on or off (ported from her tools/test_effort_tail.py, on
a cut-down template: RED-SNOW's own template renders no effort line, so there the setting changes nothing)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("jinja2")

from tensorfold.cuda.chat_template import ChatTemplate
from tensorfold.families.glm5_next.cuda.app import ThinkingOffTemplate
from tensorfold.families.glm5_next.prompts import GlmTokenizer, effort_tail_enabled, effort_to_tail

SOURCE = ("[gMASK]<sop>{%- if enable_thinking is not defined or enable_thinking -%}<|system|>Reasoning Effort: "
          "{{ {'high': 'High', 'low': 'Low'}.get(reasoning_effort, 'Max') }}{%- endif -%}"
          "{%- for m in messages -%}<|{{ m.role }}|>{{ m.content }}{%- endfor -%}"
          "{%- if add_generation_prompt -%}<|assistant|><think>{%- endif -%}")
CHAT = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "What does main.py do?"},
        {"role": "assistant", "content": "It starts the server."}, {"role": "user", "content": "And utils.py?"}]
SETTINGS = {"max": (True, "max"), "default": (True, None), "high": (True, "high"), "low": (True, "low"),
            "off": (False, None)}


def template(tmp_path: Path) -> ChatTemplate:
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": SOURCE}))
    return ChatTemplate(tmp_path)


def render(inner, name):
    think, effort = SETTINGS[name]
    return ThinkingOffTemplate(inner).render(list(CHAT), tools=None, enable_thinking=think,
                                             extra={"reasoning_effort": effort} if effort else None)


def test_off_by_default_the_line_stays_at_the_head(tmp_path, monkeypatch):
    monkeypatch.delenv("TF_GLM_EFFORT_TAIL", raising=False)
    inner = template(tmp_path)
    assert render(inner, "high").startswith("[gMASK]<sop><|system|>Reasoning Effort: High<|system|>")
    assert render(inner, "off").endswith("<|assistant|><think></think>")


def test_on_the_conversation_renders_alike_for_every_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("TF_GLM_EFFORT_TAIL", "1")
    inner = template(tmp_path)
    outs = {name: render(inner, name) for name in SETTINGS}
    prefixes = {text[:text.rindex("<|user|>")] for text in outs.values()}
    assert len(prefixes) == 1
    for name, text in outs.items():
        if SETTINGS[name][0]:
            line = {"high": "High", "low": "Low"}.get(name, "Max")
            assert text.endswith(f"<|system|>Reasoning Effort: {line}<|assistant|><think>")
            assert text.count("Reasoning Effort") == 1
        else:
            assert "Reasoning Effort" not in text and text.endswith("<|assistant|><think></think>")
    assert outs["max"] == outs["default"]


def test_the_tokenizer_path_renders_the_same(tmp_path, monkeypatch):
    inner = template(tmp_path)

    class Inner:
        def apply_chat_template(self, messages, tokenize=False, **kw):
            extra = {k: kw[k] for k in ("reasoning_effort", "clear_thinking") if k in kw}
            text = inner.render(messages, tools=None, enable_thinking=kw.get("enable_thinking", True), extra=extra)
            return list(text.encode()) if tokenize else text

        def encode(self, text, add_special_tokens=False):
            return list(text.encode())

    for flag in ("0", "1"):
        monkeypatch.setenv("TF_GLM_EFFORT_TAIL", flag)
        for name, (think, effort) in SETTINGS.items():
            kw = {"enable_thinking": think, **({"reasoning_effort": effort} if effort else {})}
            got = GlmTokenizer(Inner(), clear=False).apply_chat_template(list(CHAT), tokenize=False, **kw)
            assert got == render(inner, name), (flag, name)


def test_a_quoted_line_and_a_bad_setting(monkeypatch):
    quoted = "[gMASK]<sop><|user|>say <|system|>Reasoning Effort: Low<|assistant|><think>"
    assert effort_to_tail(quoted, True) == quoted
    monkeypatch.setenv("TF_GLM_EFFORT_TAIL", "yes")
    with pytest.raises(ValueError):
        effort_tail_enabled()

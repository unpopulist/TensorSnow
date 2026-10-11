"""GLM-5.3's prompts, the same on both servers: earlier turns keep their reasoning, and with thinking off no effort
line and no opened think block."""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

EFFORT_LINE = "<|system|>Reasoning Effort: Max"
OPENED = "<|assistant|><think>"
CLEAR_THINKING = "TF_GLM_CLEAR_THINKING"


def clear_thinking() -> bool:
    """The template's ``clear_thinking`` when a request leaves it unset: False unless ``TF_GLM_CLEAR_THINKING=1``.

    zai-org's GLM-5.3-Flash template defaults it to false, keeping every assistant turn's reasoning in the prompt
    (its model card: pass ``clear_thinking=true`` for chat). Checkpoints that carry the template's first revision
    (``Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`` among them) clear it unless told otherwise, so the reasoning before the
    last user message renders as ``<think></think>``: each new user message changes the earlier turns' tokens, and
    an agent's whole previous tool loop is prefilled again. Passing false renders those checkpoints as the current
    template does; on the current template it changes nothing."""

    value = os.environ.get(CLEAR_THINKING, "0").strip() or "0"
    if value not in ("0", "1"):
        raise ValueError(f"{CLEAR_THINKING}={value!r}: expected 0 or 1")
    return value == "1"


def _call_id(message: Any) -> Any:
    """The id GLM's template matches a tool result or call by (its ``id_of``)."""

    return message.get("tool_call_id") or message.get("id") if isinstance(message, dict) else None


def agent_history(messages: Any) -> Any:
    """A chat as GLM-5.3's template can render every part of it: a call's arguments sent as "" or null are no
    arguments (the template reads them as a mapping), and results whose ids match none of their calls' ids are
    rendered in the order sent (the template sorts results by id and silently drops one it cannot place).

    A call whose arguments are not a JSON object (a truncated or garbled string from a model, client or proxy) is
    logged and left as sent, never refused: an agent replays its history on every later turn, so a refusal would end
    the conversation. The shared normalisation (``server.messages._normalize_tool_call_arguments``) then renders it
    under ``_invalid_arguments``, with its result kept."""

    if not isinstance(messages, list):
        return messages
    out = list(messages)
    for i, message in enumerate(out):
        calls = message.get("tool_calls") if isinstance(message, dict) and message.get("role") == "assistant" else None
        if not isinstance(calls, list) or not calls:
            continue
        fixed = []
        for j, call in enumerate(calls):
            function = call.get("function") if isinstance(call, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else {}
            if isinstance(function, dict) and (arguments is None
                                               or (isinstance(arguments, str) and not arguments.strip())):
                call = {**call, "function": {**function, "arguments": {}}}
            elif isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except ValueError:
                    parsed = None
                if not isinstance(parsed, dict):
                    print(f"[tensorfold] messages[{i}].tool_calls[{j}] ({_call_id(call)!r}): arguments are not a JSON "
                          f"object ({arguments[:80]!r}); rendered as _invalid_arguments", file=sys.stderr, flush=True)
            fixed.append(call)
        out[i] = {**message, "tool_calls": fixed}
        end = i + 1
        while end < len(out) and isinstance(out[end], dict) and out[end].get("role") == "tool":
            end += 1
        known = {_call_id(call) for call in fixed}
        if any(_call_id(result) and _call_id(result) not in known for result in out[i + 1:end]):
            out[i + 1:end] = [{k: v for k, v in result.items() if k not in ("tool_call_id", "id")}
                              for result in out[i + 1:end]]
    return out


_HEAD_EFFORT = re.compile(r"(\[gMASK\]<sop>)?(<\|system\|>Reasoning Effort: (?:Max|High|Low))")


def effort_tail_enabled() -> bool:
    """``TF_GLM_EFFORT_TAIL=1``: the reasoning-effort line is rendered at the tail of the prompt (Mia's issue #93)."""

    value = os.environ.get("TF_GLM_EFFORT_TAIL", "0").strip() or "0"
    if value not in ("0", "1"):
        raise ValueError(f"TF_GLM_EFFORT_TAIL={value!r}: expected 0 or 1")
    return value == "1"


def effort_to_tail(text: str, thinking: bool) -> str:
    """The checkpoint text with its leading effort line moved to just before the generation prompt (thinking on), or
    gone (thinking off), so the conversation before it renders the same for every effort level and with thinking on or
    off. Text without the line at its head, or thinking on and not ending in the opened think block, is returned as is."""

    head = _HEAD_EFFORT.match(text)
    if head is None:
        return text
    rest = text[head.end():]
    if not thinking:
        return (head.group(1) or "") + rest
    if not rest.endswith(OPENED):
        return text
    return (head.group(1) or "") + rest[:-len(OPENED)] + head.group(2) + OPENED


def thinking_off(text: str) -> str:
    """The checkpoint text as the thinking-off template renders it: no effort line, an empty think block."""

    text = text.replace(EFFORT_LINE, "", 1)
    return text + "</think>" if text.endswith(OPENED) else text


class GlmTokenizer:
    """The checkpoint's tokenizer (the Mac's), its chat template read through ``thinking_off`` when thinking is off
    and given ``clear_thinking`` (``clear_thinking()``) when the call leaves it unset."""

    def __init__(self, inner: Any, clear: bool | None = None) -> None:
        self._inner = inner
        self._clear = clear_thinking() if clear is None else clear

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def apply_chat_template(self, messages: Any, *args: Any, tokenize: bool = True, **kwargs: Any) -> Any:
        kwargs.setdefault("clear_thinking", self._clear)
        messages = agent_history(messages)
        thinking = kwargs.get("enable_thinking", True) is not False
        tail = effort_tail_enabled()
        if thinking and not tail:
            return self._inner.apply_chat_template(messages, *args, tokenize=tokenize, **kwargs)
        text = self._inner.apply_chat_template(messages, *args, tokenize=False, **kwargs)
        if tail:
            text = effort_to_tail(text, thinking)
        if not thinking:
            text = thinking_off(text)
        return list(self._inner.encode(text, add_special_tokens=False)) if tokenize else text

"""GLM request routes validate context before streaming and render an empty think block without a reasoning-effort line when thinking is off."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from tensorfold.cuda.server import App, PreparedRequest, RequestError
from tensorfold.families.glm5_next.cuda.kept_reasoning import KeptReasoning
from tensorfold.families.glm5_next.prompts import (agent_history, clear_thinking, effort_tail_enabled, effort_to_tail,
                                                   thinking_off)
from tensorfold.server.errors import CONTEXT_LIMIT


class ThinkingOffTemplate:
    """The checkpoint template as GLM-5.3's thinking-off template renders it (``prompts.thinking_off``), with earlier
    turns' reasoning kept unless the request's ``chat_template_kwargs.clear_thinking`` says otherwise
    (``prompts.clear_thinking``). An agent's history is made renderable first (``prompts.agent_history``), and the
    reasoning of tool-calling steps the client dropped is put back (``kept``, ``kept_reasoning.KeptReasoning``; None:
    off). ``TF_GLM_EFFORT_TAIL=1`` moves the effort line to the prompt's tail (``prompts.effort_to_tail``)."""

    def __init__(self, inner, clear: bool | None = None, kept: KeptReasoning | None = None) -> None:
        self.inner = inner
        self.efforts = getattr(inner, "efforts", frozenset())
        self.clear = clear_thinking() if clear is None else clear
        self.kept = kept

    def render(self, messages, *, tools, enable_thinking, extra=None, allow_images: bool = False) -> str:
        extra = {"clear_thinking": self.clear, **(extra or {})}       # a request's own value wins
        if self.kept is not None:
            messages = self.kept.restore(messages)
        text = self.inner.render(agent_history(messages), tools=tools, enable_thinking=enable_thinking, extra=extra,
                                 **({"allow_images": True} if allow_images else {}))
        if effort_tail_enabled():
            text = effort_to_tail(text, bool(enable_thinking))
        return text if enable_thinking else thinking_off(text)


# GLM-5.3 Flash's template answers a picture or clip with this reminder; with --vision it writes GLM's markers
# instead (Glm5NextProcessor's layout). RED-SNOW's template already writes the markers (its emit_image / emit_video
# macros). Either way a part's ``tf_mark`` (server.prompts' MediaMarks, taken out again before the prompt is
# encoded) goes right before its markers, and every text-only prompt renders as before.
_NO_MEDIA = ('{{- "<reminder>You are unable to process this " ~ media_type ~ " because you don\'t have multi-modal '
             'input ability. Try different methods.</reminder>" }}')
_MARK = "{%- if item.tf_mark is defined -%}{{- item.tf_mark -}}{%- endif -%}"
_MEDIA = (_MARK + "{%- if media_type == 'image' -%}<|begin_of_image|><|image|><|end_of_image|>"
          "{%- elif media_type == 'video' -%}<|begin_of_video|><|video|><|end_of_video|>"
          "{%- else -%}" + _NO_MEDIA + "{%- endif -%}")
_EMITS = ("{{- emit_image() -}}", "{{- emit_video() -}}")
# what a text-only server puts in place of a tool result's picture (the GLM-5.3 template's own words)
TOOL_MEDIA_REMINDER = ("<reminder>You are unable to process this {media} because you don't have multi-modal input "
                       "ability. Try different methods.</reminder>")


def media_source(source: str) -> str:
    """The checkpoint template's source with each picture and clip marked (``tf_mark``) and, where it only had the
    no-media reminder, GLM's markers written."""

    if source.count(_NO_MEDIA) == 1:
        return source.replace(_NO_MEDIA, _MEDIA)
    if all(source.count(e) == 1 for e in _EMITS):
        for e in _EMITS:
            source = source.replace(e, _MARK + e)
        return source
    raise ValueError("--vision: this checkpoint's chat template has no media branch TensorFold knows how to extend "
                     "(GLM-5.3 Flash's or RED-SNOW's chat_template.jinja)")


def media_template(inner, model_dir) -> None:
    """Swap the checkpoint template for its marked form (``media_source``)."""

    path = model_dir / "chat_template.jinja"
    if not path.is_file():
        raise ValueError("--vision needs the checkpoint's chat_template.jinja")
    inner.template = inner.template.environment.from_string(media_source(path.read_text()))


class GlmApp(App):
    reads_ignore_eos = True             # ``run`` hands it to the engine's request
    # without --vision, a tool result's pictures render as this reminder (Mia's issue #28): a user's are refused
    tool_media_reminder = TOOL_MEDIA_REMINDER

    def __init__(self, engine, model_dir, served: str, **kwargs: Any) -> None:
        super().__init__(engine, model_dir, served, **kwargs)
        effort_tail_enabled()           # a typo in TF_GLM_EFFORT_TAIL fails at load, not at the first request
        if self.vision is not None:
            media_template(self.template, Path(model_dir))
        self.template = ThinkingOffTemplate(self.template, kept=KeptReasoning.from_env())

    def check(self, body: dict[str, Any], *, prepared: PreparedRequest | None = None) -> str | None:
        """Validate the rendered prompt plus max_tokens against the engine context limit before streaming."""

        problem = self._check_fields(body)
        limit = getattr(self.engine, "limit", None)
        if problem or limit is None:
            return problem or super().check(body, prepared=prepared)
        if prepared is None:
            try:
                prepared = self._prepare(body, "messages" in body)
            except RequestError as exc:
                return str(exc)
        prompt = len(prepared.prompt)
        asked = body.get("max_tokens") or body.get("max_completion_tokens")
        need = prompt + (int(asked) if asked else 1)
        if need <= limit:
            return super().check(body, prepared=prepared)
        detail = f"{prompt} prompt tokens plus max_tokens {int(asked)}" if asked else f"a {prompt}-token prompt"
        # OpenAI's wording, so prepare refuses it as context_length_exceeded (clients compact on it)
        return (f"{CONTEXT_LIMIT} {limit} tokens: this request needs a {need}-token context ({detail}), which exceeds "
                f"the context window this server was started for; shorten the prompt or reply"
                f"{self._restart(need, ' both ranks')}")

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool], *,
            prepared: PreparedRequest | None = None, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        model = str(body.get("model") or "")
        self.engine.request.policy = body.get("tf_policy") or (model.split("@", 1)[1] if "@" in model else None)
        self.engine.request.stop_eos = not bool(body.get("ignore_eos", False))
        self.engine.request.cancelled = cancelled          # --parallel: a queued request whose client left is dropped
        result = super().run(body, chat, emit, prepared=prepared, cancelled=cancelled)
        kept = getattr(self.template, "kept", None)
        if chat and kept is not None and result.get("calls") and result.get("finish") == "tool_calls":
            # the calls' reasoning, for a client that will not send it back
            kept.remember(body.get("messages"), result["calls"], result.get("reasoning"))
        return result

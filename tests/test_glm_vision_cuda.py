"""GLM-5.3 vision on CUDA (port of Mia's 0003/0050/0056/0079/0080): processors, prompt layout against the
checkpoint's own template, the rank-0 -> rank-1 feature feed, and (GPU, RED-SNOW checkpoint) the real tower."""

from __future__ import annotations

import ast
import base64
import io
import json
import math
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorfold.vision import glm

CHECKPOINT = Path(os.environ.get("TF_TEST_GLM_VISION_DIR", "/var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw"))
HF_RESIZE = Path(os.environ.get("TF_TEST_HF_TRANSFORMERS_DIR", "/nonexistent/transformers")) / ("models/"
                 "glm5_next/image_processing_glm5_next.py")
needs_checkpoint = pytest.mark.skipif(not (CHECKPOINT / "chat_template.jinja").is_file(),
                                      reason="no GLM-5.3 vision checkpoint")


def _png(width: int, height: int, seed: int = 0) -> str:
    pil = pytest.importorskip("PIL.Image")
    g = torch.Generator().manual_seed(seed)
    pixels = torch.randint(0, 256, (height, width, 3), generator=g, dtype=torch.uint8).numpy()
    out = io.BytesIO()
    pil.fromarray(pixels).save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


# -- processors ------------------------------------------------------------------------------------------------
def test_canvas_keeps_aspect_and_token_cap():
    h, w = glm.smart_resize(2, 1080, 1920, 2048)
    assert h % 28 == 0 and w % 28 == 0 and (h // 28) * (w // 28) <= 2048
    assert (h // 28) * (w // 28) > 1900                  # 1080p: 2,040 tokens (Mia's note)
    ch, cw = glm.content_size(2, 1080, 1920, (h, w))
    assert ch <= h and cw <= w and abs(cw / ch - 1920 / 1080) < 0.01
    assert glm.smart_resize(2, 10, 10, 2048) == (112, 112)   # raised to the 16-token floor


@pytest.mark.skipif(not HF_RESIZE.is_file(), reason="no transformers glm5_next source to compare with")
def test_canvas_equals_transformers_glm5_next_smart_resize():
    tree = ast.parse(HF_RESIZE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "smart_resize")
    space: dict = {"math": math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(HF_RESIZE), "exec"), space)
    hf = space["smart_resize"]
    for frames in (2, 8, 64):
        for height, width in ((1080, 1920), (37, 900), (4000, 3000), (28, 28), (720, 1280), (3, 5000)):
            for cap in (64, 2048, 8000):
                try:
                    want = hf(frames, height, width, min_pixels=16, max_pixels=cap)
                except ValueError:
                    continue
                assert glm.smart_resize(frames, height, width, cap) == tuple(want), (frames, height, width, cap)


def test_fit_resizes_without_torchvision_and_pads_right_and_bottom():
    frame = torch.full((3, 50, 100), 200, dtype=torch.uint8)
    out = glm.fit(frame, (56, 112), (40, 80))
    assert out.shape == (3, 56, 112) and out.dtype == torch.uint8
    assert int(out[:, :40, :80].min()) == 200 and int(out[:, 40:].max()) == 0 and int(out[:, :, 80:].max()) == 0


def test_patch_layout_matches_the_processor_order():
    tower = glm.GlmVisionTower.__new__(glm.GlmVisionTower)
    tower.patch, tower.merge = 14, 2
    tower.mean, tower.std = torch.zeros(3, 1, 1), torch.ones(3, 1, 1)
    frames = torch.arange(2 * 3 * 56 * 56, dtype=torch.float32).view(2, 3, 56, 56) % 251
    p = tower.patches(frames.to(torch.uint8))
    assert p.shape == (16, 3 * 2 * 14 * 14)
    # row 1 is the second patch of the first 2x2 merge block: patch (y 0, x 1); values (channel, time, y, x)
    want = frames.to(torch.uint8).float()[:, :, 0:14, 14:28].permute(1, 0, 2, 3).reshape(-1)
    assert torch.equal(p[1], want)


# -- server side: template, marks, expansion -------------------------------------------------------------------
def test_media_source_marks_both_template_forms():
    from tensorfold.families.glm5_next.cuda import app

    flash = "{%- for item in c -%}" + app._NO_MEDIA + "{%- endfor -%}"
    assert "tf_mark" in app.media_source(flash) and "<|begin_of_image|>" in app.media_source(flash)
    red = "x{{- emit_image() -}}y{{- emit_video() -}}"
    assert app.media_source(red) == f"x{app._MARK}{{{{- emit_image() -}}}}y{app._MARK}{{{{- emit_video() -}}}}"
    with pytest.raises(ValueError, match="media branch"):
        app.media_source("no media here")


def test_tool_media_parts_become_the_reminder_text():
    from tensorfold.families.glm5_next.cuda.app import TOOL_MEDIA_REMINDER
    from tensorfold.server.prompts import tool_media_parts

    shot = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
    messages = [{"role": "user", "content": "look"},
                {"role": "tool", "tool_call_id": "a", "content": [{"type": "text", "text": "ok"}, shot]}]
    out = tool_media_parts(messages, TOOL_MEDIA_REMINDER)
    assert out[1]["content"][1] == {"type": "text", "text": TOOL_MEDIA_REMINDER.format(media="image")}
    assert "unable to process this image" in out[1]["content"][1]["text"]
    assert tool_media_parts([{"role": "user", "content": [shot]}], TOOL_MEDIA_REMINDER) is None
    assert tool_media_parts([{"role": "user", "content": "text"}], TOOL_MEDIA_REMINDER) is None


def _frontend():
    """GlmVision's processor half (no tower), on the checkpoint's tokenizer."""
    from tokenizers import Tokenizer

    v = glm.GlmVision.__new__(glm.GlmVision)
    cfg = json.loads((CHECKPOINT / "config.json").read_text())
    v.image_token = int(cfg["image_token_id"])
    v.allow_urls = False
    v.limits = glm.GlmVisionLimits()
    v.tok = Tokenizer.from_file(str(CHECKPOINT / "tokenizer.json"))
    v._pictures, v._canvases = glm.OrderedDict(), glm.OrderedDict()
    return v


def _templates():
    from tensorfold.cuda.chat_template import ChatTemplate
    from tensorfold.families.glm5_next.cuda.app import media_template

    plain = ChatTemplate(CHECKPOINT)
    marked = ChatTemplate(CHECKPOINT)
    media_template(marked, CHECKPOINT)
    return plain, marked


@needs_checkpoint
def test_prompt_expansion_matches_the_checkpoint_template():
    pytest.importorskip("PIL")
    from tensorfold.server.prompts import prepare_images

    plain, marked = _templates()
    front = _frontend()
    pic_a, pic_b = _png(100, 60, 1), _png(300, 200, 2)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "two calls"},
                                     {"type": "image_url", "image_url": {"url": pic_a}}]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "shot", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "shot", "arguments": "{}"}}]},
        # results out of call order: the template sorts them by call id, and the pictures follow their results
        {"role": "tool", "tool_call_id": "c2", "content": [{"type": "image_url", "image_url": {"url": pic_b}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "no picture; mentions <|image|> as text"},
    ]
    seen = {}

    def render(template, tmpl=marked):
        text = tmpl.render(template, tools=None, enable_thinking=True, allow_images=True)
        seen.setdefault("text", text)
        return text

    rendered = prepare_images(front, messages, render)
    prepared = rendered.vision
    # with the marks taken out, the marked template renders exactly what the checkpoint's own template does
    from tensorfold.vision.images import split_images

    bare, _ = split_images(messages, tool_media=True, allow_videos=True)
    reference = plain.render(bare, tools=None, enable_thinking=True, allow_images=True)
    import re

    assert re.sub("\x00tf-media-[0-9a-f]+:\\d+\x00", "", seen["text"]) == reference
    assert reference.count(glm.IMAGE_SPAN) == 2
    # each picture's span expands to (H/28)(W/28) image tokens between its begin/end tokens
    canvases = [x.shape for _, x in prepared.items]
    counts = [(h // 28) * (w // 28) for _, h, w in canvases]
    assert prepared.item_rows == counts
    ids = prepared.token_ids
    image, begin, end = (front.tok.token_to_id(t) for t in ("<|image|>", "<|begin_of_image|>", "<|end_of_image|>"))
    assert len(prepared.positions) == sum(counts) == ids.count(image)
    starts = [i for i, t in enumerate(ids) if t == begin]
    assert len(starts) == 2
    for s, n in zip(starts, counts):
        assert ids[s + 1:s + 1 + n] == [image] * n and ids[s + 1 + n] == end
    # the expanded text tokenizes to the same ids (the quoted <|image|> stays text: no extra image position)
    pieces = reference.split(glm.IMAGE_SPAN)
    expanded = glm._as_text(pieces[0]) + "".join(
        "<|begin_of_image|>" + glm.IMAGE * n + "<|end_of_image|>" + glm._as_text(rest)
        for n, rest in zip(counts, pieces[1:]))
    assert front.tok.encode(expanded, add_special_tokens=False).ids == ids
    # the user's picture (100x60) comes first, the tool's (300x200) second: rows follow the rendered order
    assert canvases[0][1:] == glm.smart_resize(2, 60, 100, 2048) and canvases[1][1:] == glm.smart_resize(2, 200, 300,
                                                                                                         2048)
    # a remembered picture (0079) gives the same prompt and keys
    again = prepare_images(front, messages, lambda t: marked.render(t, tools=None, enable_thinking=True,
                                                                     allow_images=True)).vision
    assert again.token_ids == ids and again.item_keys == prepared.item_keys
    assert isinstance(again.items[0][1], glm.FittedPicture)


@needs_checkpoint
def test_text_only_prompts_render_unchanged_by_the_marked_template():
    plain, marked = _templates()
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi <|image|> there"},
                {"role": "assistant", "content": "ok"}, {"role": "user", "content": "again"}]
    for thinking in (True, False):
        assert (marked.render(messages, tools=None, enable_thinking=thinking)
                == plain.render(messages, tools=None, enable_thinking=thinking))


# -- the feed: rank 0's rows reach rank 1 ----------------------------------------------------------------------
class _Hub:
    def __init__(self, world: int) -> None:
        self.world, self.barrier, self.slots = world, threading.Barrier(world), [None] * world


class _Comm:
    """all_gather between threads (one per rank) on one device."""

    def __init__(self, hub: _Hub, rank: int) -> None:
        self.hub, self.rank, self.world, self.gathers = hub, rank, hub.world, 0

    def all_gather(self, send, recv) -> None:
        assert recv.numel() == send.numel() * self.world and send.dtype == recv.dtype
        self.hub.slots[self.rank] = send.reshape(-1).clone()
        self.hub.barrier.wait()
        recv.view(-1).copy_(torch.cat(self.hub.slots))
        self.gathers += 1
        self.hub.barrier.wait()

    def barrier(self) -> None:
        self.hub.barrier.wait()


def _threads(world, body):
    out, errors = [None] * world, []

    def run(r):
        try:
            out[r] = body(r, hub)
        except BaseException as exc:          # noqa: BLE001 - reported below
            errors.append(exc)
            hub.barrier.abort()

    hub = _Hub(world)
    ts = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    if errors:
        raise errors[0]
    return out, hub


@pytest.mark.parametrize("world", [2, 4])
def test_vision_feed_hands_every_chunk_its_rows(world):
    from tensorfold.families.glm5_next.cuda.engine import VisionFeed

    width, positions = 64, [3, 4, 5, 9, 20, 21, 22, 23, 40]
    rows = torch.randn(len(positions), width).to(torch.bfloat16)

    def body(rank, hub):
        hub_comm = _Comm(hub, rank)
        engine = SimpleNamespace(torch=torch, comm=hub_comm, world=world, device="cpu",
                                 w=SimpleNamespace(cfg=SimpleNamespace(hidden=width)))
        feed = VisionFeed(engine, positions, rows if rank == 0 else None)
        got = []
        for start in range(0, 48, 8):                         # the prefill's chunks
            got.append(feed.chunk(start, 8))
        return got

    out, _ = _threads(world, body)
    for got in out:
        flat_idx = [int(i) + 8 * c for c, item in enumerate(got) if item is not None for i in item[0]]
        assert flat_idx == positions
        feats = torch.cat([item[1] for item in got if item is not None])
        assert torch.equal(feats, rows)
    assert [item is None for item in out[0]] == [False, False, False, True, True, False]


def test_inject_rows_writes_every_stream():
    from tensorfold.families.glm5_next.cuda.forward import inject_rows

    x = torch.zeros(6, 4 * 8, dtype=torch.bfloat16)
    feats = torch.arange(16, dtype=torch.float32).view(2, 8).to(torch.bfloat16)
    inject_rows(x, (torch.tensor([1, 4]), feats), 4)
    assert torch.equal(x[1].view(4, 8), feats[0].expand(4, 8)) and torch.equal(x[4].view(4, 8), feats[1].expand(4, 8))
    assert int(x[[0, 2, 3, 5]].abs().sum()) == 0


needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@needs_gpu
def test_two_rank_request_carries_positions_and_rows():
    """generate (rank 0, with a tower) and follow (rank 1) agree on the header, the positions and every chunk's rows."""
    from tensorfold.families.glm5_next.cuda import engine as eng

    width = 32
    prompt = list(range(100, 140))
    positions = [5, 6, 7, 8, 30, 31]
    rows = torch.randn(len(positions), width, device="cuda").to(torch.bfloat16)
    prepared = SimpleNamespace(positions=positions, token_ids=prompt)

    class Stop(Exception):
        pass

    def make(rank, comm):
        e = eng.GlmEngine.__new__(eng.GlmEngine)
        e.torch, e.rank, e.world, e.comm = torch, rank, 2, comm
        e.limit, e.serial_only, e.policy = 4096, True, "0"
        e.request = threading.local()
        e.cache, e.drafter = [], None
        e.w = SimpleNamespace(cfg=SimpleNamespace(hidden=width, quant="exl3"), mtp=None)
        e.vision = SimpleNamespace(features=lambda p: rows) if rank == 0 else None
        e.model_dir = Path(".")
        e.eos = (0,)
        seen = {}

        def run(prompt_, max_tokens, sampling, stop_eos, on_tokens, code, hit, draft, constraint=None, feed=None, shared=(), key=None, images=False):
            seen["prompt"], seen["feed"] = list(prompt_), feed
            seen["chunks"] = [feed.chunk(s, 16) for s in range(0, len(prompt_), 16)] if feed is not None else None
            if rank:
                raise Stop
            return {}

        e._run = run
        return e, seen

    def body(rank, hub):
        e, seen = make(rank, _Comm(hub, rank))
        if rank == 0:
            stats = e.generate(prompt, 4, None, lambda t: None, draft=False, vision=prepared)
            assert stats["image_rows"] == len(positions)
        else:
            with pytest.raises(Stop):
                e.follow()
        return seen

    out, _ = _threads(2, body)
    assert out[0]["prompt"] == out[1]["prompt"] == prompt
    assert out[1]["feed"].positions == positions and out[1]["feed"].rows is None
    for a, b in zip(out[0]["chunks"], out[1]["chunks"]):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    got = torch.cat([c[1] for c in out[1]["chunks"] if c is not None])
    assert torch.equal(got, rows)


@needs_gpu
def test_text_request_header_is_unchanged_but_for_the_image_count():
    """A text-only request (--vision off) sends no positions: rank 1 builds no feed and resumes as before."""
    from tensorfold.families.glm5_next.cuda import engine as eng

    class Stop(Exception):
        pass

    def body(rank, hub):
        e = eng.GlmEngine.__new__(eng.GlmEngine)
        e.torch, e.rank, e.world, e.comm = torch, rank, 2, _Comm(hub, rank)
        e.limit, e.serial_only, e.policy = 4096, True, "0"
        e.request, e.cache, e.drafter, e.vision = threading.local(), [], None, None
        e.w = SimpleNamespace(cfg=SimpleNamespace(hidden=8, quant="exl3"), mtp=None)
        e.model_dir, e.eos = Path("."), (0,)
        seen = {}

        def run(*args, **kwargs):
            seen["feed"] = args[9] if len(args) > 9 else kwargs.get("feed")
            if rank:
                raise Stop
            return {}

        e._run = run
        if rank == 0:
            with pytest.raises(ValueError, match="--vision"):
                e.generate([1, 2, 3], 2, None, lambda t: None, draft=False, vision=SimpleNamespace(positions=[1]))
            stats = e.generate([1, 2, 3], 2, None, lambda t: None, draft=False)
            assert "image_rows" not in stats
        else:
            with pytest.raises(Stop):
                e.follow()
        return seen

    out, _ = _threads(2, body)
    assert out[0]["feed"] is None and out[1]["feed"] is None


# -- the real tower (RED-SNOW's bf16 model.visual.*), one GPU ----------------------------------------------------
@needs_gpu
@needs_checkpoint
@pytest.mark.skipif(os.environ.get("TF_TEST_GLM_TOWER", "") != "1", reason="set TF_TEST_GLM_TOWER=1 (1 GiB load)")
def test_real_tower_encodes_a_small_picture():
    device = torch.device("cuda", 0)
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    tower = glm.GlmVisionTower(CHECKPOINT, device)
    loaded = torch.cuda.memory_allocated() - base
    assert abs(loaded - tower.weight_bytes) < 2**20 and 1.0 * 2**30 < tower.weight_bytes < 1.1 * 2**30
    g = torch.Generator().manual_seed(0)
    picture = torch.randint(0, 256, (3, 112, 168), generator=g, dtype=torch.uint8)
    rows = tower.features([("image", picture)])
    assert rows.shape == ((112 // 28) * (168 // 28), 4096) and rows.dtype == torch.bfloat16
    assert torch.isfinite(rows.float()).all() and rows.float().std() > 0
    again = tower.features([("image", picture)])
    assert torch.equal(rows, again)                                  # deterministic
    # a picture alone and inside a batch with another give the same rows (one picture attends within itself)
    other = torch.randint(0, 256, (3, 56, 56), generator=g, dtype=torch.uint8)
    both = tower.features([("image", other), ("image", picture)])
    assert torch.allclose(both[4:].float(), rows.float(), atol=2e-2, rtol=2e-2)
    # a ~2,000-token picture (the default cap is 2,048): the encode's peak above the weights, for the workspace budget
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    big = torch.randint(0, 256, (3, 1036, 1512), generator=g, dtype=torch.uint8)    # 37 x 54 = 1,998 tokens
    before = torch.cuda.memory_allocated()
    out = tower.features([("image", big)])
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    print(f"\n[glm tower] weights {loaded / 2**30:.3f} GiB; 1,998-token picture: encode peak "
          f"{peak / 2**30:.3f} GiB above the weights; rows {tuple(out.shape)}")
    from tensorfold.families.glm5_next.cuda.engine import VISION_WORKSPACE

    assert peak < VISION_WORKSPACE


def test_rank0_admission_counts_the_tower_and_its_workspace(monkeypatch):
    from tensorfold.cuda.capacity import Geometry
    from tensorfold.families.glm5_next.cuda import engine as eng

    split = lambda name, info: (0, 0) if name.startswith("model.visual.") else (100, 0)
    weights, geometry = eng.vision_admission(split, lambda text: Geometry(lambda slots: slots * 10, 3, 7), 1000)
    assert weights("model.visual.blocks.0.attn.qkv.weight", {"shape": [3072, 1024]}) == (3072 * 1024 * 2, 0)
    assert weights("model.language_model.embed_tokens.weight", {"shape": [4, 4]}) == (100, 0)
    g = geometry({})
    assert g.bytes_at(5) == 1050 and g.reserve == 3 and g.minimum_slots == 7
    monkeypatch.delenv("TENSORFOLD_VISION_WORKSPACE_MIB", raising=False)
    monkeypatch.delenv("TENSORFOLD_GLM_IMAGE_TOKENS", raising=False)
    assert eng.vision_workspace() == eng.VISION_WORKSPACE
    monkeypatch.setenv("TENSORFOLD_GLM_IMAGE_TOKENS", "8000")
    assert eng.vision_workspace() == eng.VISION_WORKSPACE * 8000 // 4096
    monkeypatch.setenv("TENSORFOLD_VISION_WORKSPACE_MIB", "512")
    assert eng.vision_workspace() == 512 * 2**20
    monkeypatch.setenv("TENSORFOLD_VISION_WORKSPACE_MIB", "lots")
    with pytest.raises(ValueError, match="WORKSPACE"):
        eng.vision_workspace()


def test_many_media_limits_share_the_request_budget():
    lim = glm.GlmVisionLimits()
    assert lim.picture_tokens(1) == 2048 and lim.picture_tokens(8) == 2048 and lim.picture_tokens(50) == 16384 // 50
    assert lim.clip_tokens(2) == 16384 and lim.clip_tokens(4) == 8192
    v = glm.GlmVision.__new__(glm.GlmVision)
    v.limits = lim
    assert v.image_limits.max_images == 50 and v.image_limits.max_total_encoded_bytes == 64 * 1024 * 1024
    assert v.max_videos == 4


def test_split_images_takes_tool_clips_only_for_a_marking_frontend():
    from tensorfold.vision.images import ImageInputError, split_images

    clip = {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}
    tool = [{"role": "tool", "tool_call_id": "a", "content": [clip]}]
    with pytest.raises(ImageInputError, match="user messages"):
        split_images(tool, allow_videos=True)
    _, sources = split_images(tool, allow_videos=True, tool_media=True)
    assert len(sources) == 1
    many = [{"role": "user", "content": [clip, clip, clip]}]
    with pytest.raises(ImageInputError, match="at most 2 videos"):
        split_images(many, allow_videos=True)
    assert len(split_images(many, allow_videos=True, max_videos=4)[1]) == 3


def test_quoted_markers_stay_text():
    v = glm.GlmVision.__new__(glm.GlmVision)
    text = "see <|image|> and " + glm.IMAGE_SPAN + " end"
    own = len("see <|image|> and ")
    out = v.escape_quoted(text, [own])
    assert out.count(glm.IMAGE_SPAN) == 1 and "<|​image|>" in out
    assert v.escape_quoted(text, []).count(glm.IMAGE_SPAN) == 0       # not this request's: a quotation


@needs_gpu
def test_compute_takes_the_injected_rows_in_place_of_the_token_embeddings():
    """forward.compute(inject=) (no layers): the image rows' hidden state is the feature, the others the embedding."""
    from tensorfold.families.glm5_next.cuda.forward import compute

    D, S, R, V = 128, 4, 6, 50
    table = torch.randn(V, D, device="cuda").to(torch.bfloat16)
    w = SimpleNamespace(cfg=SimpleNamespace(hidden=D, streams=S), embed=table, layers=[])
    b = SimpleNamespace(ids=torch.tensor([3, 7, 9, 9, 9, 11], device="cuda"), x=torch.empty(8, S * D, device="cuda",
                        dtype=torch.bfloat16), hidden=torch.empty(8, D, device="cuda", dtype=torch.bfloat16),
                        tap_at={}, prefill=True, split=None)
    feats = torch.randn(3, D, device="cuda").to(torch.bfloat16)
    compute(w, None, b, R, logits=False, inject=(torch.tensor([2, 3, 4], device="cuda"), feats))
    torch.cuda.synchronize()
    assert torch.equal(b.x[2:5].view(3, S, D), feats[:, None].expand(3, S, D))
    assert torch.equal(b.x[[0, 1, 5]].view(3, S, D)[:, 0], table[[3, 7, 11]])
    assert torch.allclose(b.hidden[2:5].float(), feats.float(), atol=1e-2)

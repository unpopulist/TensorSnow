from __future__ import annotations

import pytest

from tensorfold.vision.config import validate_vision_config
from tensorfold.vision.qwen_checkpoint import vision_key


def test_glm_vision_config_accepts_the_native_checkpoint():
    config = {"model_type": "glm5_next", "image_token_id": 154854, "image_start_token_id": 154830,
              "image_end_token_id": 154831, "text_config": {"hidden_size": 4096},
              "vision_config": {"out_hidden_size": 4096, "hidden_size": 1024}}
    assert validate_vision_config(config, "glm5_next")["hidden_size"] == 1024


def test_glm_vision_config_requires_image_tokens_and_matching_width():
    config = {"model_type": "glm5_next", "text_config": {"hidden_size": 4096},
              "vision_config": {"out_hidden_size": 2048}}
    with pytest.raises(ValueError, match="output width"):
        validate_vision_config(config, "glm5_next")
    config["vision_config"]["out_hidden_size"] = 4096
    with pytest.raises(ValueError, match="image-token configuration"):
        validate_vision_config(config, "glm5_next")


def test_glm_vision_loader_recognizes_legacy_mlx_vision_model_prefix():
    assert vision_key("vision_model.blocks.0.attn.qkv.weight") == "blocks.0.attn.qkv.weight"


def test_glm_vision_is_served_on_cuda_after_its_config_checks(monkeypatch):
    """GLM-5.3 image input runs on the CUDA engine too (port of Mia's 0003): --vision reads and checks the config."""
    from argparse import Namespace
    from types import SimpleNamespace
    from tensorfold import families, serve_options

    read = []
    good = {"model_type": "glm5_next", "image_token_id": 154854, "image_start_token_id": 154830,
            "image_end_token_id": 154831, "text_config": {"hidden_size": 4096},
            "vision_config": {"out_hidden_size": 4096, "hidden_size": 1024}}
    monkeypatch.setattr(families, 'read_config', lambda path: read.append(path) or good)
    serve_options.check(Namespace(vision=True), SimpleNamespace(model_type='glm5_next', package=None), 'cuda', 'ckpt')
    assert read == ['ckpt']
    monkeypatch.setattr(families, 'read_config', lambda path: {**good, "vision_config": {"out_hidden_size": 2048}})
    with pytest.raises(ValueError, match="output width"):
        serve_options.check(Namespace(vision=True), SimpleNamespace(model_type='glm5_next', package=None), 'cuda', 'ckpt')


def test_glm_cuda_refuses_the_qwen_image_budget_flags():
    from argparse import Namespace
    from types import SimpleNamespace
    from tensorfold import serve_options

    family = SimpleNamespace(model_type='glm5_next', package=None)
    for flags in ({"vision_max_images": 8}, {"vision_image_tokens": 2048}):
        with pytest.raises(ValueError, match="TENSORFOLD_GLM_"):
            serve_options.check(Namespace(vision=True, **flags), family, 'cuda', 'ckpt')

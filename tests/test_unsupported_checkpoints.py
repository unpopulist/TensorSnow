"""Checkpoints TensorFold has no recipe for: detected from config.json alone, refused or noted with the way forward."""

import json
from types import SimpleNamespace

import pytest

from tensorfold import cli, families

MLX4 = {"model_type": "glm5_next", "quantization": {"group_size": 64, "bits": 4},
        "quantization_config": {"group_size": 64, "bits": 4}}
EXL3 = {"model_type": "glm5_next", "quantization_config": {"bits": 4, "codebook": "mcg", "quant_method": "exl3",
                                                           "scope": "glm53_routed_experts_only"}}
NVFP4 = {"model_type": "qwen3_5", "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}


def _family(**members):
    package = SimpleNamespace(**{"MODELS": ("owner/tested",), **members})
    return SimpleNamespace(model_type="fake", title="Fake Model", lanes=False, package=package)


def test_quant_method_reads_every_format():
    assert families.quant_method(MLX4) == "mlx"
    assert families.quant_method(EXL3) == "exl3"
    assert families.quant_method(NVFP4) == "modelopt"
    assert families.quant_method({"model_type": "x"}) is None
    assert families.quant_method({"text_config": {"quantization": {"bits": 4, "group_size": 32}}}) == "mlx"
    assert families.quant_method({"quantization": {"bits": 4, "group_size": 32, "mode": "mxfp4"}}) == "mlx-mxfp4"


def test_quantization_reports_mlx_only():
    assert families.quantization(MLX4) == (4, 64)
    assert families.quantization(EXL3) == (None, None)          # not "4-bit, groups of 64"
    assert families.describe_quantization(EXL3) == "exl3 (4-bit)"
    assert families.describe_quantization(MLX4) == "MLX 4-bit, groups of 64"


def test_unreadable_formats_are_refused_with_the_way_forward():
    family = _family(load=lambda *a, **k: None, cuda_engine=lambda *a, **k: None, CUDA_QUANTIZATION=(4, 64))
    families.require_readable(family, MLX4, "cuda")
    families.require_readable(family, MLX4, "mlx")
    for config, backend in ((EXL3, "cuda"), (EXL3, "mlx"), (NVFP4, "cuda")):
        with pytest.raises(ValueError) as refused:
            families.require_readable(family, config, backend)
        message = str(refused.value)
        assert "recipe book" in message and "RUNBOOK.md" in message and "owner/tested" in message
    with pytest.raises(ValueError, match="groups of 64"):
        families.require_readable(family, {"quantization": {"bits": 4, "group_size": 32}}, "cuda")


def test_a_family_can_declare_more_formats():
    family = _family(cuda_engine=lambda *a, **k: None, QUANT_METHODS={"cuda": ("mlx", "exl3")})
    families.require_readable(family, EXL3, "cuda")


def test_a_family_can_read_every_exl3_codebook_and_width():
    family = _family(cuda_engine=lambda *a, **k: None, QUANT_METHODS={"cuda": ("mlx", "exl3")}, EXL3_VARIANT="any")
    families.require_readable(family, EXL3, "cuda")
    for config in ({"model_type": "mimo", "quantization_config": {"quant_method": "exl3", "codebook": "mul1",
                                                                 "bits": 2.5078, "head_bits": 6}},
                   {"model_type": "mimo", "quantization_config": {"quant_method": "exl3", "codebook": "3inst",
                                                                 "bits": 2.5}},
                   {"model_type": "mimo", "quantization_config": {"quant_method": "exl3", "bits": 4.15}}):
        families.require_readable(family, config, "cuda")
    for config in ({"model_type": "mimo", "quantization_config": {"quant_method": "exl3", "codebook": "mcg2"}},
                   {"model_type": "mimo", "quantization_config": {"quant_method": "exl3", "head_bits": 9}}):
        with pytest.raises(ValueError) as refused:
            families.require_readable(family, config, "cuda")
        assert "does not read" in str(refused.value) and "owner/tested" in str(refused.value)
    with pytest.raises(ValueError):                            # EXL3 is not something the Mac engine reads
        families.require_readable(family, EXL3, "mlx")


RED_SNOW = {"model_type": "glm5_next", "quantization_config": {
    "quant_method": "exl3", "version": "1.4.9", "bits": 5, "head_bits": 8, "out_scales": "always", "codebook": "mul1",
    "mtp_bits": 4}}                                      # vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw's config.json


def test_glm_reads_mias_exl3_checkpoint_as_an_experiment(tmp_path, capsys):
    from tensorfold.families import glm5_next

    (tmp_path / "config.json").write_text(json.dumps(EXL3))
    glm5_next.check(tmp_path)
    assert "experimental" in capsys.readouterr().out
    for key, value in (("bits", 3), ("bits", 4.91), ("scope", "all_linear")):    # any width and scope now
        other = {**EXL3, "quantization_config": {**EXL3["quantization_config"], key: value}}
        (tmp_path / "config.json").write_text(json.dumps(other))
        glm5_next.check(tmp_path)


def test_glm_reads_red_snow_and_refuses_what_its_kernels_do_not(tmp_path, capsys):
    from tensorfold.families import glm5_next

    (tmp_path / "config.json").write_text(json.dumps(RED_SNOW))
    glm5_next.check(tmp_path)
    assert "experimental" in capsys.readouterr().out
    families.require_readable(families.detect(tmp_path), RED_SNOW, "cuda")
    for key, value, why in (("codebook", "3inst", "codebook '3inst'"), ("codebook", "mcg2", "codebook 'mcg2'"),
                            ("bits", 9, "average bits 9"), ("head_bits", 9, "head_bits 9"),
                            ("mtp_bits", 4.5, "mtp_bits 4.5"), ("bits", "mixed", "bits 'mixed'")):
        other = {**RED_SNOW, "quantization_config": {**RED_SNOW["quantization_config"], key: value}}
        (tmp_path / "config.json").write_text(json.dumps(other))
        with pytest.raises(ValueError) as refused:
            glm5_next.check(tmp_path)
        message = str(refused.value)
        assert why in message and "RED-SNOW" in message and "recipe book" in message, message


def _exl3_shard(path, groups):
    """A safetensors file of EXL3 groups: prefix -> (bits, codebook marker or None), K = N = 128."""

    import struct

    header, offset, blobs = {}, 0, []
    for prefix, (bits, marker) in groups.items():
        parts = {"trellis": ("I16", [8, 8, 16 * bits]), "suh": ("F16", [128]), "svh": ("F16", [128])}
        if marker:
            parts[marker] = ("I32", [])
        for part, (dtype, shape) in parts.items():
            size = 2 if dtype != "I32" else 4
            for v in shape:
                size *= v
            header[f"{prefix}.{part}"] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
            blobs.append(bytes(size))
            offset += size
    text = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))
    return list(header)


def test_glm_scans_a_complete_exl3_checkpoints_headers(tmp_path, capsys):
    from tensorfold.families import glm5_next

    config = {**RED_SNOW, "text_config": {"num_hidden_layers": 1}}
    (tmp_path / "config.json").write_text(json.dumps(config))

    def write(groups):
        names = _exl3_shard(tmp_path / "model-00001-of-00001.safetensors", groups)
        index = {"weight_map": {n: "model-00001-of-00001.safetensors" for n in names}}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    write({"model.language_model.layers.0.mlp.experts.0.down_proj": (4, "mul1"),
           "model.language_model.layers.0.self_attn.o_proj": (7, "mul1"), "lm_head": (8, "mul1")})
    glm5_next.check(tmp_path)
    out = capsys.readouterr().out
    assert "3 in the mul1 codebook (1 routed experts' projections, 2 dense)" in out
    assert "4-bit x1, 7-bit x1, 8-bit x1; head 8-bit" in out
    for groups, why in (({"lm_head": (8, "mul1"), "model.language_model.layers.0.self_attn.o_proj": (4, None)},
                         "3inst/mul1 codebook"),         # a 3inst tensor beside mul1 ones
                        ({"lm_head": (9, "mul1")}, "do not parse")):
        write(groups)
        with pytest.raises(ValueError, match=why):
            glm5_next.check(tmp_path)
    (tmp_path / "model-00001-of-00001.safetensors").unlink()  # before the download: config.json alone
    glm5_next.check(tmp_path)


def test_unknown_model_types_point_to_the_recipe_book(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "brand_new_arch"}))
    with pytest.raises(ValueError) as refused:
        families.detect(tmp_path)
    assert "no recipe for model_type 'brand_new_arch'" in str(refused.value)
    assert "RUNBOOK.md" in str(refused.value)


@pytest.mark.parametrize("kind", ["qwen4_exp", "qwen3_8_flash_next"])
def test_flash_next_is_found_by_either_model_type_its_exports_carry(tmp_path, kind):
    config = {"model_type": kind, "text_config": {"model_type": kind + "_text"}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert families.detect(tmp_path).module == "tensorfold.families.qwen4_exp"


def test_serve_refuses_an_unreadable_checkpoint_before_downloading(tmp_path, capsys, monkeypatch):
    # a GPTQ Qwen3.8 dense checkpoint: no engine of that family reads it (NVFP4 ModelOpt ones it does, on CUDA)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5",
                                                      "quantization_config": {"quant_method": "gptq", "bits": 4}}))
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setenv("TENSORFOLD_NO_UPDATE_CHECK", "1")
    assert cli.main(["serve", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "gptq" in err and "recipe book" in err


def test_qwen_dense_reads_nvfp4_on_cuda_only():
    family = families.families()["qwen3_5"]
    families.require_readable(family, NVFP4, "cuda")
    with pytest.raises(ValueError, match="modelopt"):
        families.require_readable(family, NVFP4, "mlx")


@pytest.mark.parametrize("config,named", [
    ({"model_type": "qwen3_5"}, "none (unquantized weights)"),
    ({"model_type": "qwen3_5", "tie_word_embeddings": True, "quantization": {"bits": 4, "group_size": 64}},
     "tied embedding"),
    ({"model_type": "nemotron_h", "quantization": {"bits": 8, "group_size": 64}}, "MLX 8-bit, groups of 64"),
])
def test_serve_refuses_what_the_mac_decoders_cannot_read_before_downloading(tmp_path, capsys, monkeypatch, config,
                                                                            named):
    from tensorfold import hub

    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setenv("TENSORFOLD_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(hub, "resolve", lambda *a, **kw: pytest.fail("downloaded before refusing"))
    assert cli.main(["serve", str(tmp_path)]) == 1
    assert named in capsys.readouterr().err


def test_untested_hugging_face_checkpoints_get_a_note(capsys):
    family = _family(DRAFTER="owner/drafter")
    cli._note_untested(family, "someone/other-conversion")
    assert "not a checkpoint TensorFold is tested with" in capsys.readouterr().out
    for model in ("owner/tested", "owner/drafter", "/local/folder"):
        cli._note_untested(family, model)
        assert capsys.readouterr().out == ""

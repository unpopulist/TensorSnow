"""GLM-5.3's CUDA config: MLX affine reads one bit width and refuses a mixed-bit encode by name (#226); EXL3's bits are
informational (each tensor's width comes from its trellis), so a RED-SNOW average, a fraction or "mixed" all load."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")         # the CUDA weights module imports the latent kernels

from tensorfold.families.glm5_next.cuda.weights import bits_of  # noqa: E402


def test_mlx_reads_one_bit_width_and_refuses_a_mixed_encode_by_name():
    assert bits_of({"bits": 4}) == 4 and bits_of({"bits": "3"}) == 3 and bits_of({}) == 4
    for bits in ("mixed_k34_per_tensor", 4.5):
        with pytest.raises(ValueError, match="mixed-bit MLX encodes are not supported"):
            bits_of({"bits": bits})


@pytest.mark.parametrize("bits, want", [(5, 5), ("5", 5), (5.0, 5), (4, 4), (4.91, 0), ("mixed", 0),
                                        ("mixed_k34_per_tensor", 0), (None, 0)])
def test_exl3_bits_never_refuse(bits, want):
    assert bits_of({"quant_method": "exl3", "bits": bits}) == want
    assert bits_of({"quant_method": "EXL3", "bits": bits}) == want


def test_exl3_without_bits_keeps_the_default():
    assert bits_of({"quant_method": "exl3"}) == 4

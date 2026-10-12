# Third-party notices

TensorFold uses [MLX](https://github.com/ml-explore/mlx) and
[mlx-lm](https://github.com/ml-explore/mlx-lm), MIT License, Copyright © 2023 Apple Inc.
They are installed as dependencies.

## MLX and mlx-lm adaptations

The DeltaNet implementations in `src/tensorfold/kernels/qwen/dense/v1/lane_gdn.py` and
`lane_tree.py` adapt mlx-lm's `qwen3_5` and `gated_delta` model math and kernels under its MIT License.

## Qwen Flash Next

The n-gram ID helpers in `src/tensorfold/families/qwen4_exp/model.py` and `cuda/ngram.py` translate
Hugging Face transformers' `models/qwen4_exp/modeling_qwen4_exp.py` into MLX and NumPy with renamed
identifiers. Copyright 2026 The Qwen Team and The HuggingFace Inc. team, Apache License 2.0.
See [the license text](LICENSES/Apache-2.0.txt). The same helpers appear in mlx-vlm's
`models/qwen4_exp/language.py`, MIT License, Copyright © 2025 Prince Canuma.

## GLM-5.3-Flash on Apple Silicon

The MLX engine of `glm5_next` (`src/tensorfold/families/glm5_next/`: the forward pass in `model.py`, `kda.py`,
`mla.py` and `mlp.py`, the draft head in `mtp.py`, `runtime.py`) and its Metal kernels
(`src/tensorfold/kernels/glm/flash/v1/`) are written for TensorFold. What they follow or port:

- The forward pass follows, op for op on its prefill path, the GLM-5.3-Flash (`glm5_next`)
  implementation added to [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) by PR #2030 (by Lazarus-931; MIT License,
  Copyright (c) 2025 Prince Canuma), as vendored by [oMLX](https://github.com/jundot/omlx) (Apache-2.0). Nothing is
  imported from either at runtime.
- `kernels/glm/flash/v1/kda.py` is ported from mlx-vlm PR #2105 ("glm5_next: fuse the KDA decode chain into one
  Metal kernel", by avlp12; `mlx_vlm/models/glm5_next/fused_kda.py`; closed without merging, MIT License,
  Copyright (c) 2025 Prince Canuma): the whole KDA decode step in one Metal kernel. TensorFold runs a window of
  rows in order inside the launch, folds the 4-bit `f_b` / `g_b` projections in with MLX's one-row `qmv_quad`
  arithmetic, and keeps its own rounding points. Its precision rules (precise exp, uncontracted sums of squares)
  are also used in `fused.py`, `moe.py` and `hc.py`.
- `kernels/glm/flash/v1/sparse_attention.py` is mlx-vlm's `indexed_sparse_attention` kernel
  (`mlx_vlm/models/sparse_attention.py`) as extended by mlx-vlm PR #2245 ("Fix GLM-5.3 cached decode batch
  invariance", by raullenchai; closed without merging, MIT License, Copyright (c) 2025 Prince Canuma), adapted to
  TensorFold's single latent cache.
- mlx-vlm PR #2107 (the sparse indexer's incremental decode and a stale-pool fix, by avlp12) needed no code:
  TensorFold's cache already pools once per completed block. Its stale-pool case is pinned by
  `tests/test_glm5_ported_kernels.py`.
- The hyper-connection kernel `_HC_SPLIT` in `kernels/glm/flash/v1/kernels.py`, and the sinkhorn and collapse in
  `hc.py`, repeat the `hc_sinkhorn_collapse` kernel of mlx-vlm's `mlx_vlm/models/deepseek_v4/hyper_connection.py`
  (MIT License, Copyright (c) 2026 Apple Inc.), with its output type set to the input's.
- The 4-bit matvec `_QMV_ROWS` in `kernels.py` is Flash Next's `qmv_rows` with MLX's group-64 scale indexing, and
  the expert kernels (`_EXPERT_GROUP`, `_EXPERT_QMV`) follow Flash Next's `expert_group` / `grouped_gateup`. The
  row kernels in `kernels.py`, `moe.py` and `hc.py` repeat the arithmetic and partitions of MLX 0.32's own kernels (MIT
  License, Copyright © 2023 Apple Inc.): `qmv_fast`, `qmv_quad` and `gather_qmv_fast` (`quantized.h`), `GEMVKernel`
  and `GEMVTKernel` (`gemv.h`) and the `rms_norm` kernels, one row per grid slice with the tiling MLX picks for one
  row, so each row keeps MLX's one-row bits.

## CUDA

CUDA backends use [PyTorch](https://github.com/pytorch/pytorch), BSD-3-Clause, and
[Triton](https://github.com/triton-lang/triton), MIT, supplied by NVIDIA's container rather than bundled.
Dense Qwen implements mlx-lm's model math. CUDA DFlash2 implementations port z-lab's architecture under
the MIT License, Copyright © 2026 Z Lab.

Flash Next implements transformers' model math under the attribution above. Its new DeltaNet kernel
follows flash-linear-attention's numerics, MIT; its NCCL wrapper follows vLLM's stream convention,
Apache-2.0, without copying either implementation.

GLM's CUDA engine implements transformers' `models/glm5_next/modular_glm5_next.py` math, Apache-2.0,
without including that source. Its draft inputs and thinking-off rendering follow the public GLM recipe
from Mia-AiLab without including recipe code.

GLM EXL3 (`families/glm5_next/cuda/exl3.py`, `exl3.cu`, `exl3_mm.py`), the shared EXL3 module
(`src/tensorfold/cuda/exl3/`) and the EXL3 loaders of Qwen3.8-27B and Qwen3.8 Flash Next
(`families/qwen3_5/cuda/exl3_load.py`, `families/qwen4_exp/cuda/exl3.py`) read
[ExLlamaV3](https://github.com/turboderp-org/exllamav3)'s EXL3 format: its trellis layout and bitstream, its
"3inst", "mcg" and "mul1" codebooks, its half-integer bit widths and its tensor-core fragment order. Flash
Next's packs also carry ExLlamaV3's n-gram row codec, read as its `ngram_dequant` reads it. ExLlamaV3 uses the
MIT License, Copyright © 2025 Turboderp. TensorFold's decoders and kernels are separate implementations,
checked bit for bit against ExLlamaV3's dequantization.

Flash Next's optional int8 and int4 KV caches (`families/qwen4_exp/cuda/kvcache.py`) follow the cache quantization scheme of [ExLlamaV3](https://github.com/turboderp-org/exllamav3) `-cq 8` and `-cq 4` (MIT License, Copyright (c) 2025 Turboderp, text below): groups of 32, one fp16 absmax scale per group, the group rotated by a 32-point Hadamard, midpoint-grid codes, `compand_a == 0`. 8-bit stores each code as a signed int8 (`q - 128`). 4-bit stores two unsigned codes per byte, low nibble first (the same bits as ExLlamaV3's little-endian packing, a uint8 tensor rather than their uint32 words). Their dequantizer folds another `1/sqrt(32)` into the scale and applies the unnormalized butterfly on the way out; this cache applies the normalized H32 to the query and to the merged output instead, and leaves the stored codes rotated. Scales match their quantizer bit for bit. Reconstructed values agree within fp16/bf16 rounding (under 0.01 on random groups), not bit for bit. The quantizer and the attention dequant are written for TensorFold and checked against an independent reference of that arithmetic.

## Vendored code and weights

`src/tensorfold/drafters/vendor/z_lab_dflash/model_mlx.py` is the unmodified `dflash/model_mlx.py` from
[z-lab/dflash](https://github.com/z-lab/dflash), MIT License, Copyright © 2026 Z Lab.

`src/tensorfold/families/deepseek_v4/vendor/encoding_dsv4.py` is the unmodified `encoding/encoding_dsv4.py` of
[deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (revision 60d8d70), and
`tests/fixtures/deepseek_v4/` holds two of its test cases, MIT License, Copyright (c) 2023 DeepSeek.
The MTP layer TensorFold drafts with comes from that checkpoint's last shard (MIT), converted by
`families/deepseek_v4/convert.py`.

TensorFold ships no model weights. The `z-lab/Qwen3.8-27B-DFlash2` model card states Apache-2.0.
The optional `incoai/GLM-5.3-Flash-DFlash2` model card states CC BY-NC-ND 4.0, for non-commercial use
without derivatives. Each checkpoint keeps its own license.

## MIT License text

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Flash Next CUDA image integration

The multimodal rotary and image-feature integration is adapted from MiaAI-Lab's
[Flash Next vision patch 0008](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/a3aa89835022c55ca8e55008c37785954834e04f/patches/0008-flash-next-vision.patch),
MIT License, Copyright (c) 2026 MiaAI-Lab. The license is included in `LICENSES/MiaAI-Lab-MIT.txt`.
The port preserves the v0.5 CUDA execution APIs and adds an offline EXL3 vision adapter.

## RoCE all-gather (`TF_COMM_BACKEND=roce`)

`src/tensorfold/cuda/roce_proxy.c` is b12x's RoCE proxy ("RoCEnante",
[b12x](https://github.com/local-inference-lab/b12x) `b12x/comm/roce/_roce_proxy.c` at commit
8a99d639410e39d5f39cb4037675331beceea1d4), Copyright the b12x contributors, Apache License 2.0, unchanged except its
header; `roce.cu` reimplements b12x's CuTe all-gather kernel in CUDA C++. `roce.cpp`, `roce.cu`, `roce.py` and the
proxy come from MiaAI-Lab's
[GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
patches 0006, 0052 and 0040 (Apache License 2.0), adapted to `tensorfold.cuda.comm`'s transport interface.
See [the license text](LICENSES/Apache-2.0.txt).

## RED-SNOW port (GLM-5.3-Flash EXL3 on two DGX Sparks)

The two-rank GLM-5.3-Flash EXL3 work in this tree (native EXL3 dense matrices, the FP8 latent cache, multi-stream
decoding, copy drafts, the DFlash2 policy, prefill and decode kernels, the spill tier, the server additions and
`deploy/redsnow`) ports much of
[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold),
a patch series against TensorFold v0.6.0, Copyright 2026 MiaAI-Lab, Apache License 2.0. Code comments name the
patch a change comes from ("Mia's 0038" is that series' patch 0038). The ported patches include parts contributed to
that project by others, credited there and here:

- 0003 (GLM's image and video processors and vision tower): checked bit for bit against Hugging Face
  [transformers](https://github.com/huggingface/transformers) 5.17 (Apache-2.0), the reference they follow.
- 0013 (verify windows of up to 16 rows, so copy drafts can run long): after the wider copy windows proposed in
  [TensorFold PR #115](https://github.com/ashhart/TensorFold/pull/115) by Ash Hart.
- 0043 (decode scoring and split selection bounded to the pools a row can see): the idea follows
  [TensorFold PR #140](https://github.com/ashhart/TensorFold/pull/140) by [mikolaj92](https://github.com/mikolaj92).
- 0054 (image prompts resume from kept prompt states): [abhicnv007](https://github.com/abhicnv007).
- 0063 (kept-prompt cap drops superseded states first): [Alexbob0](https://github.com/Alexbob0), pull request #32.
- 0071 (shared-prefix copy), 0072 (display reservation in the pool; the technique proposed by
  [gabewillen](https://github.com/gabewillen)) and 0074 (compact before evict): [ezoushen](https://github.com/ezoushen).
- 0073 (queued requests dropped when their client leaves): [desy0305](https://github.com/desy0305), with
  delivery-failure handling from [johnwhited](https://github.com/johnwhited).
- 0077 (shared prefixes evicted by recency): diagnosis by [meleesciony](https://github.com/meleesciony).
- 0078 (take-over decides, then copies): [m-naoki-m](https://github.com/m-naoki-m).
- 0079 (picture cache), 0080 (quoted vision markers) and 0089 (kept-state budgets):
  [ThomasWadeZ](https://github.com/ThomasWadeZ).
- 0081 / 0082 / 0083 (capacity status, admission cap, delivery abort): [johnwhited](https://github.com/johnwhited),
  following vLLM's (Apache-2.0) admission and abort behaviour as a reference.
- 0085 / 0086 (split selection and prompt scoring loops) and 0087 (display reservation through an external lender;
  nothing of that lender ships here): [BadAd84](https://github.com/BadAd84).
- 0088 (spill tier, `src/tensorfold/cuda/spill.py`): Robert Wojciechowski ([wojo](https://github.com/wojo)); its
  multi-rank design follows patch 0250 of jayleaton/glm53-tensorfold-spark, and its conventions credit TensorFold's
  MLX spill ([gilby](https://github.com/gilby), PR #68), [raymondkpwong](https://github.com/raymondkpwong) (issue
  #155), [gabewillen](https://github.com/gabewillen) (the idea) and [jakejharris](https://github.com/jakejharris)'s
  JSpark3 session tier.
- 0097 (kept entries as a share of the pool): diagnosis by [jdecker76](https://github.com/jdecker76).
- 0012 / 0014 / 0039 (chunked KDA prompt kernel): the chunked WY form follows the published chunkwise algorithm as
  implemented in [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT); the kernels are
  written anew.

Everything else in those patches is by MiaAI-Lab. The port to RED-SNOW's mixed-width EXL3 pack, the two-rank split
rules for it, and the launcher in `deploy/redsnow` are this tree's own work under the Apache License 2.0.

### jayleaton/glm53-tensorfold-spark

[jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark), Copyright 2026 Jay Leaton,
Apache License 2.0:

- `src/tensorfold/families/glm5_next/cuda/l2pf.cu`, `l2pf.cpp`, `l2pf.py` (`TF_GLM_L2PF`): the L2 prefetch kernel,
  adapted from that project's patch 0460 by way of MiaAI-Lab's patch 0046 (changes listed in `l2pf.cu`).
- `src/tensorfold/cuda/exl3/experts_grouped.cuh` (`TF_EXL3_LOADS=nc`): 16-byte trellis loads in the routed-expert
  decode kernel, after MiaAI-Lab's patch 0047, the idea adapted from that project's patch 0580.
- `src/tensorfold/tool_parameters.py` and `src/tensorfold/families/glm5_next/cuda/kept_reasoning.py`: parts adapted
  from that project's patch 0620 (listed in each file's docstring).

The b12x RoCE transport used by `TF_COMM_BACKEND=roce` is credited in the section above.
See [the license text](LICENSES/Apache-2.0.txt).

## MiaAI-Lab recipe patches 0090, 0099–0107 and `FILL_ROWS`

These changes are ports of patches in MiaAI-Lab's
[GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
(Apache License 2.0), adapted to this tree. Credit goes to their authors:

- `0090-glm-expert-launch-order` (`TF_GLM_EXL3_DEC_ORDER`, the launch order of the EXL3 expert decode kernel's
  blocks): by Lukasz Raczylo ([lukaszraczylo](https://github.com/lukaszraczylo)), in
  [PR #95](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/95).
- `0099-glm-tokenize-nogil` (prompts tokenized with Python's GIL released), `0100-glm-draft-prelaunch` (the next
  DFlash2 block pass launched at a decode round's end), `0101-glm-prompt-select-floor` (a prompt chunk's pool
  selection in one pass), `0103-glm-qmm-decode-noclusters` (decode 4-bit matmuls without clusters at few rows),
  `0105-glm-prompt-scores-cuda` (a prompt chunk's indexer scoring in CUDA), `0107-glm-decode-indexer` (a decode
  window's indexer scoring in CUDA and selection in one pass), and the `FILL_ROWS=2048` launcher default: by
  [BadAd84](https://github.com/BadAd84).

See [the license text](LICENSES/Apache-2.0.txt).

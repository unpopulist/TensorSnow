"""GLM's universal EXL3 route uses buffer-owned scratch and the existing bf16 expert epilogue."""

import os

from tensorfold.cuda.exl3 import experts as generic

# Decode / verify / MTP windows of up to 16 rows take the decode kernel (16-byte loads), larger windows and prompt
# chunks of 8 rows or more the prompt kernel: the same bits either way (test_exl3_experts), only the speed. Measured on
# RED-SNOW's experts (rank 0): the decode kernel 221-226 GB/s at 8-16 rows on the 5-bit layers where the prompt kernel
# gives ~200, level with it on the 6-bit ones.
DECODE_PROMPT_ROWS = int(os.environ.get("TF_GLM_EXL3_PROMPT_ROWS", "17"))


def dec_order() -> int:
    """TF_GLM_EXL3_DEC_ORDER (Mia's 0090, by Lukasz Raczylo): the decode expert kernel's program launch order, the same
    bits for every value. ``0`` (default): as launched, experts fastest; ``1`` / ``2``: the 128-column blocks of one k
    slice first, then the experts (1) or the matrices, splits and member tiles (2). Not TF_EXL3_PROMPT_ORDER (the
    prompt kernel's pass order)."""

    v = os.environ.get("TF_GLM_EXL3_DEC_ORDER", "0").strip() or "0"
    if v not in ("0", "1", "2"):
        raise ValueError(f"TF_GLM_EXL3_DEC_ORDER: 0, 1 or 2, not {v!r}")
    return int(v)


DEC_ORDER = dec_order()


def routed(x, pick, ex, scratch, rows, limit):
    """Write the live routed pairs into scratch.y; the caller supplies each row's shared-expert slot."""

    return generic.routed(x, pick, None, ex, scratch, None, rows, limit, act_mode=generic.ACT_BF16, order=DEC_ORDER)

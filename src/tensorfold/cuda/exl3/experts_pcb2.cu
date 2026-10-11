// Prompt-window EXL3 expert kernel instances for codebook 2 (mul1).
#include "experts_prompt.cuh"

namespace tf_exl3x {
template void prompt_launch<2>(const PromptArgs&, cudaStream_t);
}  // namespace tf_exl3x

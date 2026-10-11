// Prompt-window EXL3 expert kernel instances for codebook 1 (mcg).
#include "experts_prompt.cuh"

namespace tf_exl3x {
template void prompt_launch<1>(const PromptArgs&, cudaStream_t);
}  // namespace tf_exl3x

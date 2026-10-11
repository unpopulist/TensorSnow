// Prompt-window EXL3 expert kernel instances for codebook 0 (3inst).
#include "experts_prompt.cuh"

namespace tf_exl3x {
template void prompt_launch<0>(const PromptArgs&, cudaStream_t);
}  // namespace tf_exl3x

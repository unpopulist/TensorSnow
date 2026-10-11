// The routed experts' epilogue arithmetic, shared by the split-K epilogue kernels (experts.cu) and the prompt kernel
// (experts_prompt.cuh) so both give the same bits.
#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace tf_exl3x {

constexpr float EP_HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

// Walsh-Hadamard transform of 128 values, 4 a lane, fixed butterfly order (strides 1, 2 in registers, 4..64 across lanes).
__device__ __forceinline__ void ep_fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

__device__ __forceinline__ float ep_bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// gv / uv: this lane's 4 summed gate / up values of one 128-column block (columns n .. n + 3); writes xd_row[n .. n + 3]
// = fp16((SwiGLU(rot(gv) * svh_g, rot(uv) * svh_u) * suh_d) @ H) (act_mode 0: GLM's bf16 roundings, 1: fp32).
__device__ __forceinline__ void gateup_finish(float (&gv)[4], float (&uv)[4], int lane, int e, int N, int n,
                                              const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                              const half* __restrict__ suh_d, float limit, int act_mode,
                                              half* __restrict__ xd_row) {
    ep_fwht128(gv, lane);
    ep_fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float act;
        if (act_mode == 0) {
            float gg = fminf(ep_bf16r(gv[j] * EP_HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
            float uu = fminf(fmaxf(ep_bf16r(uv[j] * EP_HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])),
                                   -limit), limit);
            act = ep_bf16r(ep_bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {
            float gg = fminf(gv[j] * EP_HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]), limit);
            float uu = fminf(fmaxf(uv[j] * EP_HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]), -limit), limit);
            act = gg / (1.f + expf(-gg)) * uu;
        }
        v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
    }
    ep_fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) xd_row[n + j] = __float2half_rn(v[j] * EP_HAD_SCALE);
}

// v: this lane's 4 summed down values of one 128-column block; y_row[n .. n + 3] = rot(v) * svh_d (fp32).
__device__ __forceinline__ void down_finish(float (&v)[4], int lane, int e, int D, int n,
                                            const half* __restrict__ svh_d, float* __restrict__ y_row) {
    ep_fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) y_row[n + j] = v[j] * EP_HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
}

}  // namespace tf_exl3x

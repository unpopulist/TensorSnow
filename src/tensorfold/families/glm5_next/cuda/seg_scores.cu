// Patch 0107: the DSA indexer's decode scoring for a segmented window (sparse._seg_scores as seg_select_tokens launches
// it: FP8 pooled keys, 32 heads x 128, 64-pool blocks, 4 warps) as a CUDA kernel that computes every score with the
// Triton kernel's own instruction sequence (prompt_scores.cu's, whose sequence _seg_scores shares: the same
// tile, warps and epilogue), so every bit is the same, while a CTA converts each key tile once into shared memory for
// up to 8 of a segment's rows. Triton's _seg_scores loads each tile once a program too, but converts it, lays it out
// and masks it again for every row: at 4 streams x 8 rows x 596k it ran at 1.9x the keys' memory time (the tensor
// cores' work alone 0.7x). With few rows the scoring is memory-bound and Triton's programs read faster (measured: 1-3
// rows ~10% faster), so a segment of fewer than ``min_rows`` rows is left to _seg_scores, launched next over the pool
// table this kernel writes. tools/decode_indexer_check.py.
//
// Per score (prompt_scores.cu's header has the Triton PTX it reads): keys e4m3 -> bf16 (exact); dots[32 heads, 64
// pools] = mma.sync m16n8k16 bf16 -> fp32, one chain over the 8 k16 steps from +0, the kWidth = 4 k order; per element
// (dot * ks[pool]) * scale, max(., 0); per pool the head sum fma(w24, x24, fma(w16, x16, fma(w0, x0, w8 * x8))), then
// shfl_xor 16, 8, 4. A row's score is stored for its visible pools only (p < (pos + 1) / 4), as _seg_scores stores
// them; the keys a row does not see change none of its stored columns (a column's mma chain reads its own key only),
// so Triton's per-row key mask is not repeated.
//
// Work: CTA (row group, block g, segment) takes the segment's pool blocks g, g + gridDim.y, .. (below its device-side
// pool count; neighbouring blocks read at once, as Triton's grid does), ST - 1 tiles in flight through cp.async; a
// group's rows share its 8 warps (a row's n8 tiles split over 8 / rows-rounded-up warps).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>

namespace {

constexpr int H = 32, D = 128, BP = 64, KS = D / 16;   // heads, key width, pools a block, k16 steps
constexpr int CH = D * 2 / 16;                          // 16-byte chunks of a bf16 key row (16)
constexpr int RB = 8;                                   // rows a CTA, one a warp

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

// two e4m3 codes (low byte first) -> their bf16 values (low first), exactly: e4m3 -> f16 -> f32 -> bf16
__device__ __forceinline__ uint32_t e4m3x2_bf16x2(uint32_t two) {
    uint32_t h2, r;
    const unsigned short v = (unsigned short)two;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(v));
    asm("{\n .reg .b16 lo, hi;\n .reg .f32 a, b;\n mov.b32 {lo, hi}, %1;\n cvt.f32.f16 a, lo;\n cvt.f32.f16 b, hi;\n"
        " cvt.rn.bf16x2.f32 %0, b, a;\n}" : "=r"(r) : "r"(h2));
    return r;
}

// The physical 16-byte chunk of logical chunk c in key row p (ldmatrix's 8 rows on distinct bank groups).
__device__ __forceinline__ int swz(int p, int c) { return c ^ (p & 7); }

// Key tiles stream through ST stages of raw shared memory (cp.async, ST - 1 tiles in flight while one is converted):
// a stage is a tile's 64 rows of 128 e4m3 codes, then their 64 fp32 scales; rows at or past the segment's pool count
// are zero-filled (their columns are never stored). ``convert`` turns a stage into the bf16 tile, tile[p][chunk] in
// mma slot order, and the scales into ks[p].
constexpr int STAGE = BP * D + BP * 4;                  // bytes a raw stage
constexpr int TPP = RB * 32 / BP, GPT = KS / TPP;       // threads a pool, k16 groups a thread (convert)

__device__ __forceinline__ void cp_async(uint32_t dst, const void* src, int bytes, bool ok) {
    if (bytes == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(ok ? 16 : 0));
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(ok ? 4 : 0));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// tile pb's codes and scales into stage ``raw`` (``any`` false: past the CTA's blocks, an empty group, so the count
// of groups cp_wait sees stays uniform)
__device__ __forceinline__ void issue(const uint8_t* __restrict__ PK, int RS, long long pbase, int NP, int pb,
                                      bool any, uint8_t* raw) {
    if (any) {
        const uint32_t base = (uint32_t)__cvta_generic_to_shared(raw);
#pragma unroll
        for (int k = 0; k < BP * (D / 16) / (RB * 32); ++k) {
            const int c = threadIdx.x + k * RB * 32, i = c / (D / 16), part = c % (D / 16);
            const int p = pb * BP + i;
            const bool ok = p < NP;
            cp_async(base + i * D + 16 * part, PK + (size_t)(pbase + (ok ? p : 0)) * RS + 16 * part, 16, ok);
        }
        if (threadIdx.x < BP) {
            const int p = pb * BP + threadIdx.x;
            const bool ok = p < NP;
            cp_async(base + BP * D + 4 * threadIdx.x, PK + (size_t)(pbase + (ok ? p : 0)) * RS + D, 4, ok);
        }
    }
    cp_commit();
}

__device__ __forceinline__ void convert(const uint8_t* raw, uint4* tile, float* ks) {
    const int tid = threadIdx.x, i = tid / TPP, part = tid % TPP;
#pragma unroll
    for (int gg = 0; gg < GPT; ++gg) {
        const int j = part * GPT + gg;                           // k16 group: k = 16 j .. 16 j + 15
        const uint4 c = *reinterpret_cast<const uint4*>(raw + i * D + 16 * j);
        const uint32_t in[4] = {c.x, c.y, c.z, c.w};
        uint32_t wv[8];                                          // wv[m]: values (2m, 2m + 1) of the group
#pragma unroll
        for (int m = 0; m < 8; ++m) wv[m] = e4m3x2_bf16x2(in[m >> 1] >> (16 * (m & 1)));
        // group j: 32-block b = j / 2, half u = j % 2; word m = 2 t + h goes to step 2 b + h, slots 8 u + 2 t + e
        tile[i * CH + swz(i, 4 * (j >> 1) + (j & 1))] = make_uint4(wv[0], wv[2], wv[4], wv[6]);       // h = 0
        tile[i * CH + swz(i, 4 * (j >> 1) + 2 + (j & 1))] = make_uint4(wv[1], wv[3], wv[5], wv[7]);   // h = 1
    }
    if (part == 0) ks[i] = reinterpret_cast<const float*>(raw + BP * D)[i];
}

template <int ST>
__global__ void __launch_bounds__(RB * 32) seg_scores_kernel(
    const __nv_bfloat16* __restrict__ QI, const __nv_bfloat16* __restrict__ W, int w_stride,
    const uint8_t* __restrict__ PK, int RS, float* __restrict__ OUT, long long o_stride,
    const int32_t* __restrict__ POS, const int32_t* __restrict__ SPR, const int32_t* __restrict__ SEG_START,
    const int32_t* __restrict__ SEG_ROWS, const int32_t* __restrict__ SEG_PBASE, const int32_t* __restrict__ SEG_POOLS,
    int32_t* __restrict__ TRI_POOLS, int min_rows, float scale, float wscale) {
    extern __shared__ __align__(16) uint8_t smem[];
    uint4* tile = reinterpret_cast<uint4*>(smem);                // 16 KB: one key tile, bf16
    float* ks = reinterpret_cast<float*>(smem + BP * CH * 16);   // its 64 scales
    uint8_t* raw = smem + BP * CH * 16 + BP * 4;                 // ST raw stages
    const int s = blockIdx.z, G = gridDim.y;
    const int nr = SEG_ROWS[s];
    // segments of fewer than min_rows rows are the Triton kernel's (launched next, over TRI_POOLS: their pool counts,
    // 0 for the segments scored here); a few rows' scoring is memory-bound, where Triton's programs read faster
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) TRI_POOLS[s] = nr < min_rows ? SEG_POOLS[s] : 0;
    if (nr < min_rows || (int)blockIdx.x * RB >= nr) return;     // the whole CTA: not this kernel's, or past the rows
    const int npm = SEG_POOLS[s];                                // pools the segment's last row sees (0: dense)
    const int nblk = (npm + BP - 1) / BP;
    const int b0 = (int)blockIdx.y;                              // blocks b0, b0 + G, .. (neighbours read together)
    if (b0 >= nblk) return;
    const long long pbase = SEG_PBASE[s];
    // the CTA's rows (up to RB) share its warps: WPR = RB / (its rows rounded up to a power of two) warps a row, each
    // taking every WPR-th n8 tile of a key tile (which warp computes a column changes none of its arithmetic)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int nrc = min(RB, nr - (int)blockIdx.x * RB);
    const int wpr = nrc > 4 ? 1 : nrc > 2 ? 2 : nrc > 1 ? 4 : 8;
    const int slot = warp / wpr, sub = warp % wpr;
    const int i = (int)blockIdx.x * RB + slot;
    const int r = SEG_START[s] + (slot < nrc ? i : 0);
    const bool live = slot < nrc && SPR[r] != 0;
    const int npool = live ? (POS[r] + 1) / 4 : 0;

    // this warp's row: query fragments of both m16 tiles for the 8 k16 steps (prompt_scores.cu's), the 4 head weights
    uint32_t qa[2][KS][4];
    float wh[4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int st = 0; st < KS; ++st) {
            const __nv_bfloat16* q0 = QI + ((size_t)r * H + 16 * mt + g) * D;
            const __nv_bfloat16* q1 = q0 + 8 * D;
            const int k = 32 * (st >> 1) + 4 * t + 2 * (st & 1);
            qa[mt][st][0] = *reinterpret_cast<const uint32_t*>(q0 + k);
            qa[mt][st][1] = *reinterpret_cast<const uint32_t*>(q1 + k);
            qa[mt][st][2] = *reinterpret_cast<const uint32_t*>(q0 + k + 16);
            qa[mt][st][3] = *reinterpret_cast<const uint32_t*>(q1 + k + 16);
        }
    {
        const int hs[4] = {g, g + 8, 16 + g, 24 + g};
#pragma unroll
        for (int x = 0; x < 4; ++x) wh[x] = __fmul_rn(__bfloat162float(W[(size_t)r * w_stride + hs[x]]), wscale);
    }

    float* orow = OUT + (size_t)r * o_stride;
#pragma unroll
    for (int k = 0; k < ST - 1; ++k) issue(PK, RS, pbase, npm, b0 + k * G, b0 + k * G < nblk, raw + k * STAGE);
    for (int pb = b0, jt = 0; pb < nblk; pb += G, ++jt) {
        const int p0 = pb * BP;
        cp_wait<ST - 2>();                                       // this thread's copies of tile pb landed
        __syncthreads();                                         // everyone's; the previous tile's dots are done
        convert(raw + (jt % ST) * STAGE, tile, ks);
        // tile pb + (ST - 1) G into the stage converted last iteration
        const int nx = pb + (ST - 1) * G;
        issue(PK, RS, pbase, npm, nx, nx < nblk, raw + ((jt + ST - 1) % ST) * STAGE);
        __syncthreads();
        if (p0 >= npool) continue;                               // the row sees none of them (or no row)
#pragma unroll 2
        for (int n = sub; n < BP / 8; n += wpr) {                // n8 tile: pools p0 + 8 n .. + 7
            float acc[2][4];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[mt][c] = 0.0f;
#pragma unroll
            for (int j = 0; j < KS; j += 2) {                    // steps j, j + 1 from one ldmatrix.x4
                uint32_t b[4];
                const int pr = 8 * n + (lane & 7), c = 2 * j + (lane >> 3);
                ldsm_x4(b, &tile[pr * CH + swz(pr, c)]);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    mma16816(acc[mt], qa[mt][j], b[0], b[1]);
                    mma16816(acc[mt], qa[mt][j + 1], b[2], b[3]);
                }
            }
            const int pc = 8 * n + 2 * t;
            float sv[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {                        // pool columns pc, pc + 1
                const float k = ks[pc + e];
                const float x0 = fmaxf(__fmul_rn(scale, __fmul_rn(acc[0][e], k)), 0.0f);       // head g
                const float x8 = fmaxf(__fmul_rn(scale, __fmul_rn(acc[0][2 + e], k)), 0.0f);   // g + 8
                const float x16 = fmaxf(__fmul_rn(scale, __fmul_rn(acc[1][e], k)), 0.0f);      // 16 + g
                const float x24 = fmaxf(__fmul_rn(scale, __fmul_rn(acc[1][2 + e], k)), 0.0f);  // 24 + g
                float v = __fmul_rn(wh[1], x8);
                v = __fmaf_rn(wh[0], x0, v);
                v = __fmaf_rn(wh[2], x16, v);
                v = __fmaf_rn(wh[3], x24, v);
                v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 16));
                v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 8));
                v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 4));
                sv[e] = v;
            }
            const int p = p0 + pc;
            if (g == 0) {
                if (p + 1 < npool)
                    *reinterpret_cast<float2*>(orow + p) = make_float2(sv[0], sv[1]);
                else if (p < npool)
                    orow[p] = sv[0];
            }
        }
    }
}

template <int ST>
void launch(dim3 grid, cudaStream_t stream, const __nv_bfloat16* QI, const __nv_bfloat16* W, int w_stride,
            const uint8_t* PK, int RS, float* OUT, long long o_stride, const int32_t* POS, const int32_t* SPR,
            const int32_t* SS, const int32_t* SR, const int32_t* SB, const int32_t* SP, int32_t* TP, int min_rows,
            float scale, float wscale) {
    constexpr int bytes = BP * CH * 16 + BP * 4 + ST * STAGE;
    static bool sized = false;                                   // over 48 KB needs the opt-in (once a process)
    if (!sized) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(seg_scores_kernel<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        sized = true;
    }
    seg_scores_kernel<ST><<<grid, RB * 32, bytes, stream>>>(QI, W, w_stride, PK, RS, OUT, o_stride, POS, SPR, SS, SR,
                                                            SB, SP, TP, min_rows, scale, wscale);
}

}  // namespace

// _seg_scores' stores, bit for bit, for the segments of at least ``min_rows`` rows: OUT[r, p] (row stride o_stride)
// for each of their sparse rows r and its visible pools p < (POS[r] + 1) / 4; nothing else is scored here. TRI_POOLS
// [segs] (int32) <- SEG_POOLS for the other segments, 0 for these: _seg_scores launched after it over TRI_POOLS
// scores exactly the rest. QI [R, 32 * 128] bf16 contiguous rows, W bf16 head weights (row stride w_stride, unit
// column stride), PK an FP8 pooled-key arena (rows of 128 e4m3 codes then the row's fp32 scale, RS bytes apart; a
// segment's pool p at row SEG_PBASE + p), the int32 SegRows tables (POS, SPR per row; SEG_START, SEG_ROWS, SEG_PBASE,
// SEG_POOLS per segment, ``segs`` of them). ``blocks``: CTAs a segment's pool blocks are split over, ``stages``: raw
// key tiles a CTA holds (2-4); ``min_rows`` (speed only, like both).
void seg_scores(const at::Tensor& QI, const at::Tensor& W, int64_t w_stride, const at::Tensor& PK, int64_t RS,
                at::Tensor OUT, int64_t o_stride, const at::Tensor& POS, const at::Tensor& SPR,
                const at::Tensor& SEG_START, const at::Tensor& SEG_ROWS, const at::Tensor& SEG_PBASE,
                const at::Tensor& SEG_POOLS, at::Tensor TRI_POOLS, int64_t segs, double scale, double wscale,
                int64_t blocks, int64_t stages, int64_t min_rows) {
    TORCH_CHECK(QI.is_cuda() && QI.scalar_type() == at::kBFloat16 && W.scalar_type() == at::kBFloat16 &&
                OUT.scalar_type() == at::kFloat && PK.element_size() == 1, "seg scores: dtypes");
    for (const at::Tensor* x :
         std::initializer_list<const at::Tensor*>{&POS, &SPR, &SEG_START, &SEG_ROWS, &SEG_PBASE, &SEG_POOLS, &TRI_POOLS})
        TORCH_CHECK(x->scalar_type() == at::kInt && x->is_contiguous(), "seg scores: int32 SegRows tables");
    TORCH_CHECK(QI.dim() == 2 && QI.is_contiguous() && QI.size(1) == H * D && W.stride(-1) == 1 &&
                OUT.stride(-1) == 1 && SEG_START.numel() >= segs && SEG_ROWS.numel() >= segs &&
                SEG_PBASE.numel() >= segs && SEG_POOLS.numel() >= segs && TRI_POOLS.numel() >= segs &&
                POS.numel() >= QI.size(0) && SPR.numel() >= QI.size(0), "seg scores: shapes");
    TORCH_CHECK(o_stride % 2 == 0 && reinterpret_cast<uintptr_t>(OUT.data_ptr()) % 8 == 0,
                "seg scores: score rows 8-byte aligned (float2 stores)");
    TORCH_CHECK(RS % 16 == 0 && RS >= D + 4 && PK.is_contiguous(), "seg scores: an FP8 key cache of 16-byte rows");
    TORCH_CHECK(blocks >= 1 && blocks <= 65535 && segs >= 1 && segs <= 65535 && stages >= 2 && stages <= 4 &&
                min_rows >= 1, "seg scores: grid / stages / rows");
    const int64_t R = QI.size(0);
    if (R <= 0) return;
    const c10::cuda::CUDAGuard guard(QI.device());
    const dim3 grid((unsigned)((R + RB - 1) / RB), (unsigned)blocks, (unsigned)segs);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto qi = reinterpret_cast<const __nv_bfloat16*>(QI.data_ptr());
    auto w = reinterpret_cast<const __nv_bfloat16*>(W.data_ptr());
    auto pk = reinterpret_cast<const uint8_t*>(PK.data_ptr());
    auto go = [&](auto f) {
        f(grid, stream, qi, w, (int)w_stride, pk, (int)RS, OUT.data_ptr<float>(), (long long)o_stride,
          POS.data_ptr<int32_t>(), SPR.data_ptr<int32_t>(), SEG_START.data_ptr<int32_t>(),
          SEG_ROWS.data_ptr<int32_t>(), SEG_PBASE.data_ptr<int32_t>(), SEG_POOLS.data_ptr<int32_t>(),
          TRI_POOLS.data_ptr<int32_t>(), (int)min_rows, (float)scale, (float)wscale);
    };
    if (stages == 2) go(launch<2>);
    else if (stages == 3) go(launch<3>);
    else go(launch<4>);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("seg_scores", &seg_scores); }

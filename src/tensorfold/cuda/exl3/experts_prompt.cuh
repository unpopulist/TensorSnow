// Prompt-window EXL3 experts, Y^T = W^T X^T (after Mia's 0004/0020 GLM prompt kernels): a block decodes each trellis
// tile of its 16 m tiles once a k step and feeds it to a pass of up to PM member rows; the gate/up and down epilogues run
// in the same block. Every output keeps grouped_kernel's arithmetic: mma chains of CH k tiles from zero (the decode
// kernel's per-warp k ranges), the W chains of a split added in warp order, the splits added in order from 0 -- so a
// row's bits are the decode path's, whatever the window.
#pragma once

#include "experts_grouped.cuh"
#include "experts_epilogue.cuh"

namespace tf_exl3x {

// A stage holds KTS k tiles (PBK = 16 KTS k values) of the members' rows (row stride PXS halves, +16 B so ldmatrix
// rows fall on distinct banks) and of the block's tiles (PTW words reserved a tile: 4 * the widest K2 it serves).
#define PBK (16 * KTS)
#define PKT KTS
#define PXS (PBK + 8)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

struct PromptArgs {
    const half* x0;          // gate (or down) inputs [P, K], rotated per pair
    const half* x1;          // up inputs (down: = x0)
    const int64_t* tp0;
    const int64_t* tp1;
    const int* k2_0;
    const int* k2_1;
    const int* uids;
    const int* items;        // [max_items, 2]: (u, pass)
    const int* nitems;
    const int* members;
    const half* sv0;         // gate/up: svh_g, svh_u, suh_d; down: svh_d
    const half* sv1;
    const half* su2;
    half* xd;                // gate/up output [P, N]
    float* y;                // down output [P, N]
    int K, N, maxm, slots, E;
    int CH, W, SK;           // the decode tile setting's chains: CH k tiles a chain, W chains a split, SK splits
    int act_mode;
    float limit;
    int max_items;
    int mats;
    int pm;
    int kts;
    int hi;
};

// MATS 2: a block is one 128-column Hadamard block of gate and up (8 tiles of each); 1: 256 columns of down.
template <int MATS>
struct PTiles {
    static constexpr int TPM = 16 / MATS;                         // n tiles a matrix a block
};

template <int MATS, int PM, int KTS, int PTW>
__host__ __device__ constexpr size_t prompt_stage_bytes() {
    return (size_t)MATS * PM * PXS * 2 + (size_t)MATS * PKT * PTiles<MATS>::TPM * PTW * 4;
}

template <int MATS, int PM, int KTS, int PTW, int PSTAGES>
__host__ __device__ constexpr size_t prompt_smem_bytes() {
    const size_t st = PSTAGES * prompt_stage_bytes<MATS, PM, KTS, PTW>();
    const size_t ep = MATS == 2 ? (size_t)2 * 16 * (128 + 4) * 4 : (size_t)16 * (256 + 4) * 4;
    return st > ep ? st : ep;
}

// This warp's MT tiles of one stage: decode each k tile's words once (A fragments of W^T) and multiply by every live
// member block (B fragments of X^T from shared memory).
template <int CB, int K2, int MT, int PM, int TPM, int KTS, int PTW>
__device__ __forceinline__ void prompt_stage(const uint32_t* ws, int tile0, const half* xs, int nblocks, int cnt,
                                             int lane, float (&acc)[MT][PM / 8][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
#pragma unroll
    for (int kk = 0; kk < PKT; ++kk) {
        uint32_t a[MT][4];
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            const uint32_t* tp = ws + (size_t)(kk * TPM + tile0 + i) * PTW + lane;
            uint32_t w[LW];
#pragma unroll
            for (int l = 0; l < LW; ++l) {
                if constexpr ((TW % 32) == 0)
                    w[l] = tp[l * 32];
                else
                    w[l] = (l * 32 + lane < TW) ? tp[l * 32] : 0u;
            }
            uint32_t b0[2], b1[2];
            decode_tile<CB, K2>(w, map, lane, b0, b1);
            a[i][0] = b0[0]; a[i][1] = b1[0]; a[i][2] = b0[1]; a[i][3] = b1[1];
        }
#pragma unroll
        for (int jb = 0; jb < PM / 16; ++jb) {
            if (jb < nblocks) {
                uint32_t r[4];
                const int row = jb * 16 + (lane & 7) + ((lane >> 4) & 1) * 8;
                ldsm_x4(r, xs + (size_t)row * PXS + kk * 16 + ((lane >> 3) & 1) * 8);
                const uint32_t bl[2] = {r[0], r[1]}, bh[2] = {r[2], r[3]};
                const bool upper = jb * 16 + 8 < cnt;                 // members 8..15 of the block exist
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    mma16816(acc[i][2 * jb], a[i], bl);
                    if (upper) mma16816(acc[i][2 * jb + 1], a[i], bh);
                }
            }
        }
    }
}

template <int CB, int MATS, int PM, int NW, int MINB, int KTS, int PTW, int PSTAGES, int K2MAX>
__global__ void __launch_bounds__(NW * 32, MINB) prompt_kernel(const PromptArgs a) {
    constexpr int MT = 16 / NW, TPM = PTiles<MATS>::TPM, NB = PM / 8;
    constexpr size_t STAGE = prompt_stage_bytes<MATS, PM, KTS, PTW>();
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int rows_sh[PM];
    if ((int)blockIdx.y >= a.nitems[0]) return;
    const int u = a.items[2 * blockIdx.y], pass = a.items[2 * blockIdx.y + 1];
    const int e = a.uids[u];
    const int cb = blockIdx.x;
    if (threadIdx.x < PM) {
        const int m = pass * PM + threadIdx.x;
        const int code = m < a.maxm ? a.members[(size_t)u * a.maxm + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * a.slots + (code & 31) : -1;
    }
    const int cnt = __syncthreads_count(threadIdx.x < PM && rows_sh[threadIdx.x] >= 0);
    const int nblocks = (cnt + 15) >> 4;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wmat = MATS == 2 ? (warp * MT) / TPM : 0;          // this warp's matrix and first tile in it
    const int tile0 = (warp * MT) % TPM;
    const int K = a.K, KT = K >> 4, NTILES = a.N >> 4;
    const uint32_t* T0 = reinterpret_cast<const uint32_t*>(a.tp0[e]);
    const uint32_t* T1 = reinterpret_cast<const uint32_t*>(MATS == 2 ? a.tp1[e] : a.tp0[e]);
    const int k20 = a.k2_0[e], k21 = MATS == 2 ? a.k2_1[e] : k20;
    const int k2w = wmat ? k21 : k20;
    const int nt0 = cb * TPM;                                     // first n tile of each matrix in this block

    auto load = [&](int chunk) {
        unsigned char* dst = smem + (size_t)(chunk % PSTAGES) * STAGE;
        half* xdst = reinterpret_cast<half*>(dst);
        const int k0 = chunk * PBK;
        for (int idx = threadIdx.x; idx < MATS * PM * (PBK / 8); idx += NW * 32) {
            const int m = idx / (PM * (PBK / 8)), rem = idx % (PM * (PBK / 8)), i = rem / (PBK / 8),
                      j = rem % (PBK / 8);
            const int p = rows_sh[i];
            const half* src = (m ? a.x1 : a.x0) + (size_t)(p < 0 ? 0 : p) * K + k0 + j * 8;
            cp_async16(xdst + ((size_t)m * PM + i) * PXS + j * 8, src, p >= 0);
        }
        uint32_t* wdst = reinterpret_cast<uint32_t*>(dst + (size_t)MATS * PM * PXS * 2);
        // 16-byte pieces: (matrix, k tile, tile, piece), a tile's 4 * K2 words being K2 pieces
        for (int idx = threadIdx.x; idx < MATS * PKT * TPM * K2MAX; idx += NW * 32) {
            const int c = idx % K2MAX, i = (idx / K2MAX) % TPM, kk = (idx / K2MAX) / TPM % PKT, m = (idx / K2MAX) / TPM / PKT;
            const int k2 = m ? k21 : k20;
            if (c >= k2) continue;
            const uint32_t* src = (m ? T1 : T0) + ((size_t)(chunk * PKT + kk) * NTILES + nt0 + i) * (4 * k2) + 4 * c;
            cp_async16(wdst + ((size_t)(m * PKT + kk) * TPM + i) * PTW + 4 * c, src, true);
        }
    };

    float acc[MT][NB][4], S[MT][NB][4], Tt[MT][NB][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NB; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = S[i][j][c] = Tt[i][j][c] = 0.f;

    const int chunks = K / PBK;
#pragma unroll
    for (int c = 0; c < PSTAGES - 1; ++c) {
        if (c < chunks) load(c);
        cp_async_commit();
    }
    for (int c = 0; c < chunks; ++c) {
        cp_async_wait<PSTAGES - 2>();
        __syncthreads();
        if (c + PSTAGES - 1 < chunks) load(c + PSTAGES - 1);
        cp_async_commit();
        const unsigned char* base = smem + (size_t)(c % PSTAGES) * STAGE;
        const half* xs = reinterpret_cast<const half*>(base) + (size_t)wmat * PM * PXS;
        const uint32_t* ws = reinterpret_cast<const uint32_t*>(base + (size_t)MATS * PM * PXS * 2) +
                             (size_t)wmat * PKT * TPM * PTW;
        switch (k2w) {
#define TF_PCASE(K2_) \
    case K2_: if constexpr (K2_ <= K2MAX) prompt_stage<CB, K2_, MT, PM, TPM, KTS, PTW>(ws, tile0, xs, nblocks, cnt, lane, acc); else __trap(); break;
            TF_PCASE(2) TF_PCASE(3) TF_PCASE(4) TF_PCASE(5) TF_PCASE(6) TF_PCASE(7) TF_PCASE(8) TF_PCASE(9)
            TF_PCASE(10) TF_PCASE(11) TF_PCASE(12) TF_PCASE(13) TF_PCASE(14) TF_PCASE(15) TF_PCASE(16)
#undef TF_PCASE
            default: __trap();
        }
        const int done = (c + 1) * PKT;                           // k tiles finished
        if (done % a.CH == 0) {                                   // a chain ends: grouped_kernel's warp and split sums
            const int ci = done / a.CH - 1, w = ci % a.W, s = ci / a.W;
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NB; ++j)
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        S[i][j][q] = w == 0 ? acc[i][j][q] : S[i][j][q] + acc[i][j][q];
                        acc[i][j][q] = 0.f;
                        if (w == a.W - 1) Tt[i][j][q] = s == 0 ? 0.f + S[i][j][q] : Tt[i][j][q] + S[i][j][q];
                    }
        }
    }
    cp_async_wait<0>();
    __syncthreads();                                              // the stages become the epilogue's rows

    const int g = lane >> 2, t = lane & 3;
    float* Es = reinterpret_cast<float*>(smem);
    constexpr int ES = MATS == 2 ? 128 + 4 : 256 + 4;
    for (int jb = 0; jb < nblocks; ++jb) {
        if (jb) __syncthreads();                                  // the previous block of members is read
        // this warp's sums of members jb * 16 .. + 15 -> Es[mat][member][column]
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = (tile0 + i) * 16 + g;
                float* E0 = Es + (size_t)(wmat * 16) * ES;
#pragma unroll
                for (int jj = 0; jj < NB; ++jj) {
                    if (jj / 2 != jb || (jj & 1) != h) continue;
                    const int mloc = h * 8 + 2 * t;
                    E0[(size_t)mloc * ES + col] = Tt[i][jj][0];
                    E0[(size_t)(mloc + 1) * ES + col] = Tt[i][jj][1];
                    E0[(size_t)mloc * ES + col + 8] = Tt[i][jj][2];
                    E0[(size_t)(mloc + 1) * ES + col + 8] = Tt[i][jj][3];
                }
            }
        __syncthreads();
        for (int ml = warp; ml < 16; ml += NW) {
            const int p = rows_sh[jb * 16 + ml < PM ? jb * 16 + ml : 0];
            if (jb * 16 + ml >= cnt) break;
            if constexpr (MATS == 2) {
                float gv[4], uv[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    gv[j] = Es[(size_t)ml * ES + 4 * lane + j];
                    uv[j] = Es[(size_t)(16 + ml) * ES + 4 * lane + j];
                }
                gateup_finish(gv, uv, lane, e, a.N, cb * 128 + 4 * lane, a.sv0, a.sv1, a.su2, a.limit, a.act_mode,
                              a.xd + (size_t)p * a.N);
            } else {
#pragma unroll
                for (int blk = 0; blk < 2; ++blk) {
                    float v[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) v[j] = Es[(size_t)ml * ES + blk * 128 + 4 * lane + j];
                    down_finish(v, lane, e, a.N, cb * 256 + blk * 128 + 4 * lane, a.sv0, a.y + (size_t)p * a.N);
                }
            }
        }
    }
}

// Block-wide exclusive scan of v (1024 threads): the prefix, and the total in `total`.
__device__ __forceinline__ int prompt_scan(int v, int* wsum, int& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inc = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int x = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += x;
    }
    __syncthreads();
    if (lane == 31) wsum[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        const int x = wsum[lane];
        int s = x;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += y;
        }
        wsum[lane] = s - x;
        if (lane == 31) wsum[32] = s;
    }
    __syncthreads();
    total = wsum[32];
    return wsum[warp] + inc - v;
}

// items [max_items, 2] = (u, pass) for each pass of up to pm members of each distinct expert u; one block. order 0: in
// u order; 1 (Mia's 0020): the passes of experts taking several (mma bound, their weights shared through L2) and the
// experts taking one (DRAM bound) merged evenly, each list in u order. Launch order only: every output keeps its bits.
static __global__ void __launch_bounds__(1024) prompt_items_kernel(const int* __restrict__ ucount,
                                                                   const int* __restrict__ members, int maxm, int pm,
                                                                   int* __restrict__ items, int* __restrict__ nitems,
                                                                   int max_items, int order) {
    __shared__ int wsum[33];
    const int n = ucount[0];
    // pass 1: totals of heavy passes and light experts
    int nh = 0, nl = 0;
    auto passes_of = [&](int u) {
        int lo = 0, hi = maxm;                                    // the first `cnt` entries are >= 0
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (members[(size_t)u * maxm + mid] >= 0) lo = mid + 1; else hi = mid;
        }
        return (lo + pm - 1) / pm;
    };
    for (int base = 0; base < n; base += 1024) {
        const int u = base + threadIdx.x;
        const int ps = u < n ? passes_of(u) : 0;
        int th, tl;
        prompt_scan(order && ps > 1 ? ps : 0, wsum, th);
        prompt_scan(order && ps == 1 ? 1 : 0, wsum, tl);
        nh += th;
        nl += tl;
    }
    int ch = 0, cl = 0, call = 0;                                 // running heavy / light / all item counts
    for (int base = 0; base < n; base += 1024) {
        const int u = base + threadIdx.x;
        const int ps = u < n ? passes_of(u) : 0;
        int th, tl, ta;
        const int rh = ch + prompt_scan(ps > 1 ? ps : 0, wsum, th);
        const int rl = cl + prompt_scan(ps == 1 ? 1 : 0, wsum, tl);
        const int ra = call + prompt_scan(ps, wsum, ta);
        for (int q = 0; q < ps; ++q) {
            int at;
            if (!order) {
                at = ra + q;
            } else if (ps > 1) {                                  // heavy rank r at (r + 1/2) / nh: lights strictly before
                const int r = rh + q;
                const long long X = (2LL * r + 1) * nl;
                const int c = X > nh ? (int)min((long long)nl, (X - nh + 2LL * nh - 1) / (2LL * nh)) : 0;
                at = r + c;
            } else {                                              // light rank l at (l + 1/2) / nl: heavies at or before
                const int l = rl;
                const long long Y = (2LL * l + 1) * nh;
                const int c = Y >= nl ? (int)min((long long)nh, (Y - nl) / (2LL * nl) + 1) : 0;
                at = l + c;
            }
            if (at < max_items) {
                items[2 * at] = u;
                items[2 * at + 1] = q;
            }
        }
        ch += th;
        cl += tl;
        call += ta;
    }
    if (threadIdx.x == 0) nitems[0] = min(call, max_items);
}

template <int CB>
void prompt_launch(const PromptArgs& a, cudaStream_t stream) {
#define TF_PL(MATS_, PM_, NW_, MB_, KTS_, NS_, K2M_)                                                              \
    do {                                                                                                          \
        constexpr size_t SM = prompt_smem_bytes<MATS_, PM_, KTS_, 4 * K2M_, NS_>();                              \
        auto fn = prompt_kernel<CB, MATS_, PM_, NW_, MB_, KTS_, 4 * K2M_, NS_, K2M_>;                             \
        static bool once = [&] {                                                                                  \
            cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SM);                       \
            return true;                                                                                          \
        }();                                                                                                      \
        (void)once;                                                                                               \
        dim3 grid((unsigned)(a.N / (MATS_ == 2 ? 128 : 256)), (unsigned)a.max_items);                            \
        fn<<<grid, NW_ * 32, SM, stream>>>(a);                                                                    \
    } while (0)
    // a.pm: pass size; a.kts: k tiles a stage (2, 4 or 8); a.hi: the widest K2 of the layer's matrices
#define TF_PK(MATS_, PM_, NW_, MB_, NS_)                                                                          \
    if (a.hi <= 12) {                                                                                             \
        if (a.kts == 8) TF_PL(MATS_, PM_, NW_, MB_, 8, NS_, 12);                                                  \
        else if (a.kts == 4) TF_PL(MATS_, PM_, NW_, MB_, 4, NS_, 12);                                             \
        else TF_PL(MATS_, PM_, NW_, MB_, 2, 3, 12);                                                               \
    } else {                                                                                                      \
        if (a.kts == 8) TF_PL(MATS_, PM_, NW_, MB_, 4, NS_, 16);                                                  \
        else if (a.kts == 4) TF_PL(MATS_, PM_, NW_, MB_, 4, NS_, 16);                                             \
        else TF_PL(MATS_, PM_, NW_, MB_, 2, 3, 16);                                                               \
    }
    if (a.mats == 2 && a.pm == 32) { TF_PK(2, 32, 8, 1, 2) }
    else if (a.mats == 2 && a.pm == 16) { TF_PK(2, 16, 8, 2, 2) }
    else if (a.mats == 1 && a.pm == 32) { TF_PK(1, 32, 8, 1, 2) }
    else if (a.mats == 1 && a.pm == 16) { TF_PK(1, 16, 8, 2, 2) }
    else TORCH_CHECK(false, "prompt experts: mats 1/2, passes of 16 or 32 members");
#undef TF_PK
#undef TF_PL
}

#undef PBK
#undef PKT
#undef PXS

}  // namespace tf_exl3x

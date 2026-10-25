#include "mmv-p100-v2-q5k.cuh"

// Expected inner work per weight: about 6-9 integer/shuffle instructions plus
// 0.5 hsub2 and 0.5 hfma2, with FP32 scale/min work amortized over eight weights
// per lane. No FP32 weight decode. A row streams exactly 1760 B at k=2560;
// coalesced 16-B loads, R-way activation reuse and native GP100 half2 arithmetic
// should make the 437-MB output head memory-bound. Persistent blocks now convert
// activations once and reuse them across row groups, with the grid capped by
// cached SM count and occupancy. A second register tile prefetches all R rows of
// the next Q5_K block before decoding the current one. Aligned 16-B shared loads
// replace four scalar half2 loads to remove their 4-way bank conflicts. The
// eleven weight-loading lanes and the decode math are unchanged. The <=1150-us
// gate and actual register count still require measurement on the P100.

// HIP/MUSA also glob this directory; only CUDA gets the new implementation.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <map>
#include <set>
#include <tuple>

namespace {

constexpr int q5k_threads = 128;
constexpr unsigned q5k_mask = 0xffffffffu;
static_assert(sizeof(block_q5_K) == 11 * sizeof(uint4), "Q5_K must be eleven aligned vectors");
static_assert(offsetof(block_q5_K, scales) == 4 && offsetof(block_q5_K, qh) == 16 &&
              offsetof(block_q5_K, qs) == 48, "Q5_K header/qh/qs layout changed");

static int q5k_rows_per_warp() {
    static const int rows = [] {
        const char * env = std::getenv("GGML_CUDA_P100_V2_R");
        return env && std::strcmp(env, "2") == 0 ? 2 : env && std::strcmp(env, "4") == 0 ? 4 : 8;
    }();
    return rows;
}

static void q5k_log_launch(int64_t m, int64_t k, int rows) {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_CUDA_P100_LOG");
        return env && std::strcmp(env, "1") == 0;
    }();
    if (enabled) {
        static std::mutex mutex;
        static std::set<std::tuple<int64_t, int64_t, int>> seen;
        const std::lock_guard<std::mutex> lock(mutex);
        if (seen.emplace(m, k, rows).second) {
            std::fprintf(stderr, "p100 v2 q5_k m=%lld k=%lld R=%d\n", (long long) m, (long long) k, rows);
        }
    }
}

// Keep every shuffle converged: the selector belongs to the receiving lane,
// so selecting a vector component before the shuffle would read the wrong word.
static __device__ __forceinline__ uint2 q5k_shuffle_pair(uint4 v, int source, bool upper) {
    const uint32_t x = __shfl_sync(q5k_mask, v.x, source);
    const uint32_t y = __shfl_sync(q5k_mask, v.y, source);
    const uint32_t z = __shfl_sync(q5k_mask, v.z, source);
    const uint32_t w = __shfl_sync(q5k_mask, v.w, source);
    return make_uint2(upper ? z : x, upper ? w : y);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
static __device__ __forceinline__ half2 q5k_half2_bits(uint32_t bits) {
    return __halves2half2(__ushort_as_half(static_cast<unsigned short>(bits)),
                         __ushort_as_half(static_cast<unsigned short>(bits >> 16)));
}

static __device__ __forceinline__ half2 q5k_half2(uint32_t low, uint32_t high, int selector, int group) {
    // Byte-permute two adjacent qs/qh bytes into the low bits of two halfwords.
    // qs stores groups 2*p and 2*p+1 in its low/high nibbles. qh[l] bit g
    // belongs to weight 32*g+l (not to byte g/2). Merge it into bit 4 FIRST.
    const uint32_t q4 = __byte_perm(low >> (4 * (group & 1)), 0, selector) & 0x000f000fu;
    const uint32_t qh = (__byte_perm(high, 0, selector) >> group) & 0x00010001u;
    const uint32_t q5 = q4 | (qh << 4);
    uint32_t magic;
    // (q5 & 0x001f001f) | 0x64006400: each half is exactly 1024 + q5.
    asm("lop3.b32 %0, %1, 0x001f001f, 0x64006400, 0xea;" : "=r"(magic) : "r"(q5));
    const half2 value = __halves2half2(__ushort_as_half(static_cast<unsigned short>(magic)),
                                    __ushort_as_half(static_cast<unsigned short>(magic >> 16)));
    return __hsub2(value, __float2half2_rn(1024.0f));
}
static __device__ __forceinline__ half2 q5k_pair(uint32_t t, uint32_t qrot, uint32_t nib_mask, uint32_t qh_mask) {
    // Even groups: half = 1024 + v. Odd groups: half = 1024 + 16 v (activations
    // are stored /16 for odd groups). qrot already has the qh bit in place.
    uint32_t y, magic;
    asm("lop3.b32 %0, %1, %2, 0x64006400, 0xea;" : "=r"(y) : "r"(qrot), "r"(qh_mask));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(magic) : "r"(t), "r"(nib_mask), "r"(y));
    union { uint32_t u; half2 h; } c;
    c.u = magic;
    return __hsub2(c.h, __float2half2_rn(1024.0f));
}
#endif

template <int R>
static __global__ __launch_bounds__(q5k_threads, (R == 8 ? 2 : (R == 4 ? 4 : 8))) void q5k_mmv(
        const char * __restrict__ weights, const float * __restrict__ x, float * __restrict__ dst,
        int k, int m, size_t row_stride) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
    // 8 resident blocks * 128 threads targets <=64 registers/thread on GP100.
    // The sm_50 pass used by multi-arch builds has an empty, never-launched body.
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    constexpr int warps = q5k_threads / 32;
    const int groups = k / 32;
    extern __shared__ __align__(16) float shared[];
    half2 * input = reinterpret_cast<half2 *>(shared);
    float * sum_x = shared + k / 2;
    int * exponent = reinterpret_cast<int *>(sum_x + groups);

    // Each block reads and converts the input exactly once, and keeps the sums
    // in the original FP32 domain for the minimum term.
    for (int g = warp; g < groups; g += warps) {
        const float value = x[32 * g + lane];
        float maximum = fabsf(value);
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
            maximum = fmaxf(maximum, __shfl_xor_sync(q5k_mask, maximum, offset));
        }
        const int e = max(0, int((__float_as_uint(maximum) >> 23) & 255) - 127 - 5);
        // Finite scaled |x| <64 (or <=64 after half rounding). Even a full
        // 32-weight half2 partial has at most 16 terms/component, hence
        // |partial| <=16*31*64=31744: conversion AND accumulation cannot overflow.
        // No clamping of ordinary activations, and undo 2^-e only in FP32.
        // e is in [0,122] for finite FP32 input, so both powers of two are
        // normal FP32 values; constructing them avoids a libdevice ldexp call.
        const float prescale = __uint_as_float(uint32_t(127 - e) << 23);
        // Layout per 256-weight block (256 halves): [vec v=0 for 16 lanes][vec v=1 ...],
        // lane slot l16 = 2*group+half holds its 8 weights of vector v. Within each
        // 4-weight chunk bits 0 and 1 of the position are swapped so weights are
        // paired (byte0,byte2),(byte1,byte3).
        {
            const int gg = g & 7;
            const int h = lane >> 4;
            const int v = (lane >> 3) & 1;
            const int pos = (lane & 4) | ((lane & 1) << 1) | ((lane >> 1) & 1);
            reinterpret_cast<half *>(input)[(g >> 3) * 256 + v * 128 + (2 * gg + h) * 8 + pos] =
                __float2half_rn(value * prescale * ((g & 1) ? 0.0625f : 1.0f));
        }
        float sum = value;
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
            sum += __shfl_down_sync(q5k_mask, sum, offset);
        }
        if (lane == 0) {
            sum_x[g] = sum;
            exponent[g] = e;
        }
    }
    __syncthreads();

    // Four lanes cover one 32-weight sub-block, eight adjacent weights each.
    // Each half2 partial is reset before crossing a sub-block boundary.
    // A warp handles a PAIR of blocks of one row: lanes 0-15 block 2*ip, lanes
    // 16-31 block 2*ip+1. Two lanes cover one 32-weight sub-block (16 weights
    // each), so the scale work is paid once per 16 weights.
    const int bp = lane >> 4;
    const int l16 = lane & 15;
    const int group = l16 >> 1;
    const int half_sel = l16 & 1;
    const int n_blocks = k / QK_K;
    const int n_pairs = (n_blocks + 1) / 2;
    const int qpos = (group & 1) ? 8 : 4;
    const uint32_t nib_mask = (group & 1) ? 0x00f000f0u : 0x000f000fu;
    const uint32_t qh_mask = (group & 1) ? 0x01000100u : 0x00100010u;
    const uint32_t rot_a = uint32_t(group - qpos) & 31u;
    const uint32_t rot_b = uint32_t(group + 8 - qpos) & 31u;
    const int qs_off = 48 + 32 * (group >> 1) + 16 * half_sel;
    const int qh_off = 16 + 16 * half_sel;
    const int sel = 0x4440 + (group & 3);
    const int64_t n_groups = (int64_t(m) + warps * R - 1) / (warps * R);
    for (int64_t row_group = blockIdx.x; row_group < n_groups; row_group += gridDim.x) {
        const int64_t first_row = (row_group * warps + warp) * R;
        float acc[R] = {};
        const char * rowp[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int64_t row = first_row + r < m ? first_row + r : m - 1;
            rowp[r] = weights + row * row_stride;
        }
        for (int ip = 0; ip < n_pairs; ++ip) {
            const int ib = 2 * ip + bp;
            const bool valid = ib < n_blocks;
            const int nb = valid ? ib : n_blocks - 1;
            uint4 th[R], tl[R], tq[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const char * p = rowp[r] + size_t(nb) * sizeof(block_q5_K);
                th[r] = __ldg(reinterpret_cast<const uint4 *>(p));
                tl[r] = __ldg(reinterpret_cast<const uint4 *>(p + qs_off));
                tq[r] = __ldg(reinterpret_cast<const uint4 *>(p + qh_off));
            }
            const int cb = valid ? ib : n_blocks - 1;
            const int g = 8 * cb + group;
            const half2 * act = input + cb * 128 + l16 * 4;
            uint4 act0, act1;
            const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(act));
            asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(act0.x), "=r"(act0.y), "=r"(act0.z), "=r"(act0.w) : "r"(a_addr) : "memory");
            asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(act1.x), "=r"(act1.y), "=r"(act1.z), "=r"(act1.w) : "r"(a_addr + 256) : "memory");
            const half2 a0 = q5k_half2_bits(act0.x), a1 = q5k_half2_bits(act0.y);
            const half2 a2 = q5k_half2_bits(act0.z), a3 = q5k_half2_bits(act0.w);
            const half2 a4 = q5k_half2_bits(act1.x), a5 = q5k_half2_bits(act1.y);
            const half2 a6 = q5k_half2_bits(act1.z), a7 = q5k_half2_bits(act1.w);
            const float vm = valid ? 1.0f : 0.0f;
            const float rescale = __uint_as_float(uint32_t(127 + exponent[g]) << 23) * vm;
            const float sxh = sum_x[g] * (0.5f * vm);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t dm = th[r].x;
                const uint32_t b0 = __byte_perm(th[r].y, 0, sel);
                const uint32_t b1 = __byte_perm(th[r].z, 0, sel);
                const uint32_t b2 = __byte_perm(th[r].w, 0, sel);
                const int sc = group < 4 ? b0 & 63 : (b2 & 15) | ((b0 >> 6) << 4);
                const int mn = group < 4 ? b1 & 63 : (b2 >> 4) | ((b1 >> 6) << 4);
                const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(dm)));
                const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(dm >> 16)));
                const uint4 lo = tl[r];
                const uint4 hi = tq[r];
                half2 pa = __hmul2(q5k_pair(lo.x, __funnelshift_r(hi.x, hi.x, rot_a), nib_mask, qh_mask), a0);
                half2 pb = __hmul2(q5k_pair(lo.x >> 8, __funnelshift_r(hi.x, hi.x, rot_b), nib_mask, qh_mask), a1);
                pa = __hfma2(q5k_pair(lo.y, __funnelshift_r(hi.y, hi.y, rot_a), nib_mask, qh_mask), a2, pa);
                pb = __hfma2(q5k_pair(lo.y >> 8, __funnelshift_r(hi.y, hi.y, rot_b), nib_mask, qh_mask), a3, pb);
                pa = __hfma2(q5k_pair(lo.z, __funnelshift_r(hi.z, hi.z, rot_a), nib_mask, qh_mask), a4, pa);
                pb = __hfma2(q5k_pair(lo.z >> 8, __funnelshift_r(hi.z, hi.z, rot_b), nib_mask, qh_mask), a5, pb);
                pa = __hfma2(q5k_pair(lo.w, __funnelshift_r(hi.w, hi.w, rot_a), nib_mask, qh_mask), a6, pa);
                pb = __hfma2(q5k_pair(lo.w >> 8, __funnelshift_r(hi.w, hi.w, rot_b), nib_mask, qh_mask), a7, pb);
                const float2 p = __half22float2(__hadd2(pa, pb));
                const float dot = (p.x + p.y) * rescale;
                acc[r] += (d * sc) * dot - (dmin * mn) * sxh;
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int offset = 16; offset; offset >>= 1) {
                acc[r] += __shfl_down_sync(q5k_mask, acc[r], offset);
            }
            if (lane == 0 && first_row + r < m) {
                dst[first_row + r] = acc[r];
            }
        }
    }
#else
    GGML_UNUSED_VARS(weights, x, dst, k, m, row_stride);
#endif
}

static int q5k_persistent_blocks(int device, int rows, size_t shared_bytes) {
    static std::mutex mutex;
    static std::map<int, int> sm_counts;
    static std::map<std::tuple<int, int, size_t>, int> block_counts;
    const std::lock_guard<std::mutex> lock(mutex);
    auto sm = sm_counts.find(device);
    if (sm == sm_counts.end()) {
        int count;
        CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
        sm = sm_counts.emplace(device, count).first;
    }
    // Occupancy also depends on k's dynamic shared allocation. Cache once per
    // R and allocation size on each physical GPU, rather than reusing a count
    // from an unrelated shape or device.
    const auto key = std::make_tuple(device, rows, shared_bytes);
    auto blocks = block_counts.find(key);
    if (blocks == block_counts.end()) {
        int count;
        if (rows == 8) {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&count, q5k_mmv<8>, q5k_threads, shared_bytes));
        } else if (rows == 4) {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&count, q5k_mmv<4>, q5k_threads, shared_bytes));
        } else {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&count, q5k_mmv<2>, q5k_threads, shared_bytes));
        }
        GGML_ASSERT(count > 0);
        blocks = block_counts.emplace(key, count).first;
    }
    return sm->second * blocks->second;
}

} // namespace

bool ggml_cuda_p100_v2_q5k(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t k = src0->ne[0];
    const int64_t m = src0->ne[1];
    // The caller checks type, 2-D, single-column and contiguous F32 eligibility.
    // Restrict shared memory to the portable Pascal 48-KiB per-block limit.
    if (k <= 0 || k > INT_MAX || k % QK_K != 0 || m <= 0 || m > INT_MAX ||
        src1->ne[0] != k || dst->ne[0] != m || src0->nb[0] != sizeof(block_q5_K) ||
        reinterpret_cast<uintptr_t>(src0->data) % 16 != 0 || src0->nb[1] % 16 != 0 ||
        src0->nb[1] < size_t(k / QK_K) * sizeof(block_q5_K) ||
        reinterpret_cast<uintptr_t>(src1->data) % alignof(float) != 0 ||
        reinterpret_cast<uintptr_t>(dst->data) % alignof(float) != 0) {
        return false;
    }
    const size_t shared_bytes = size_t(k) * sizeof(half) + size_t(k / 32) * (sizeof(float) + sizeof(int));
    if (shared_bytes > 48 * 1024) {
        return false;
    }
    const int rows = q5k_rows_per_warp();
    const int64_t rows_per_block = (q5k_threads / 32) * rows;
    const int64_t n_groups = (m + rows_per_block - 1) / rows_per_block;
    const int max_blocks = q5k_persistent_blocks(ggml_cuda_info().devices[ctx.device].physical_device, rows, shared_bytes);
    const dim3 grid(static_cast<unsigned>(n_groups < max_blocks ? n_groups : max_blocks));
    const char * weights = static_cast<const char *>(src0->data);
    const float * input = static_cast<const float *>(src1->data);
    float * output = static_cast<float *>(dst->data);
    if (rows == 8) {
        q5k_mmv<8><<<grid, q5k_threads, shared_bytes, ctx.stream()>>>(weights, input, output, int(k), int(m), src0->nb[1]);
    } else if (rows == 4) {
        q5k_mmv<4><<<grid, q5k_threads, shared_bytes, ctx.stream()>>>(weights, input, output, int(k), int(m), src0->nb[1]);
    } else {
        q5k_mmv<2><<<grid, q5k_threads, shared_bytes, ctx.stream()>>>(weights, input, output, int(k), int(m), src0->nb[1]);
    }
    CUDA_CHECK(cudaGetLastError());
    q5k_log_launch(m, k, rows);
    return true;
}

#else

bool ggml_cuda_p100_v2_q5k(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    return false;
}

#endif

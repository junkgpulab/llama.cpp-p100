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
        return env && std::strcmp(env, "4") == 0 ? 4 : 2;
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
static __device__ __forceinline__ half2 q5k_pair(uint32_t t, uint32_t qh4) {
    uint32_t y, magic;
    asm("lop3.b32 %0, %1, 0x00100010, 0x64006400, 0xea;" : "=r"(y) : "r"(qh4));
    asm("lop3.b32 %0, %1, 0x000f000f, %2, 0xea;" : "=r"(magic) : "r"(t), "r"(y));
    union { uint32_t u; half2 h; } c;
    c.u = magic;
    return __hsub2(c.h, __float2half2_rn(1024.0f));
}
#endif

template <int R>
static __global__ __launch_bounds__(q5k_threads, 8) void q5k_mmv(
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
        // Weights are paired (byte0,byte2),(byte1,byte3) of each 4-weight chunk, so the
        // activations are stored with bits 0 and 1 of the chunk position swapped.
        reinterpret_cast<half *>(input)[32 * g + (lane & ~3) + ((lane & 1) << 1) + ((lane >> 1) & 1)] =
            __float2half_rn(value * prescale);
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
    const int group = lane / 4;
    const int segment = lane & 3;
    const int n_blocks = k / QK_K;
    const int64_t n_groups = (int64_t(m) + warps * R - 1) / (warps * R);
    for (int64_t row_group = blockIdx.x; row_group < n_groups; row_group += gridDim.x) {
        const int64_t first_row = (row_group * warps + warp) * R;
        float acc[R] = {};
        // Every lane loads its own qs/qh bytes (8 B each) plus the 16-B header
        // (broadcast), so no shuffles are needed. Rows are clamped, not masked.
        const char * rowp[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int64_t row = first_row + r < m ? first_row + r : m - 1;
            rowp[r] = weights + row * row_stride;
        }
        const int qs_off = 48 + 32 * (group >> 1) + 8 * segment;
        const int qh_off = 16 + 8 * segment;
        uint4 th[R]; uint2 tl[R]; uint2 tq[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            th[r] = __ldg(reinterpret_cast<const uint4 *>(rowp[r]));
            tl[r] = __ldg(reinterpret_cast<const uint2 *>(rowp[r] + qs_off));
            tq[r] = __ldg(reinterpret_cast<const uint2 *>(rowp[r] + qh_off));
        }
        for (int ib = 0; ib < n_blocks; ++ib) {
            // Fetch all R next-block vectors before consuming any current vector.
            // The final iteration and padded rows keep zero tiles, with no loads.
            uint4 nh[R]; uint2 nl[R]; uint2 nq[R];
            const int nb = ib + 1 < n_blocks ? ib + 1 : ib;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const char * p = rowp[r] + size_t(nb) * sizeof(block_q5_K);
                nh[r] = __ldg(reinterpret_cast<const uint4 *>(p));
                nl[r] = __ldg(reinterpret_cast<const uint2 *>(p + qs_off));
                nq[r] = __ldg(reinterpret_cast<const uint2 *>(p + qh_off));
            }
            const int g = 8 * ib + group;
            const int input_pair = 128 * ib + 16 * group + 4 * segment;
            // Every lane starts on a 16-B boundary. Force a single vector LDS so
            // the compiler cannot split this into four bank-conflicting loads.
            uint4 activation;
            const uint32_t input_address = static_cast<uint32_t>(__cvta_generic_to_shared(input + input_pair));
            asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(activation.x), "=r"(activation.y), "=r"(activation.z), "=r"(activation.w)
                         : "r"(input_address) : "memory");
            const half2 a0 = q5k_half2_bits(activation.x);
            const half2 a1 = q5k_half2_bits(activation.y);
            const half2 a2 = q5k_half2_bits(activation.z);
            const half2 a3 = q5k_half2_bits(activation.w);
            const float rescale = __uint_as_float(uint32_t(127 + exponent[g]) << 23);
            const float sx = segment == 0 ? sum_x[g] : 0.0f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                // Header and payload are shuffled from the registers that received
                // the vectors; no per-weight byte loads, and no header rereads.
                const uint32_t dm = th[r].x;
                const uint32_t s0 = th[r].y;
                const uint32_t s1 = th[r].z;
                const uint32_t s2 = th[r].w;
                const int shift = 8 * (group & 3);
                const uint32_t b0 = (s0 >> shift) & 255;
                const uint32_t b1 = (s1 >> shift) & 255;
                const uint32_t b2 = (s2 >> shift) & 255;
                // Exactly get_scale_min_k4: bytes 0..3, 4..7, 8..11 of scales.
                const int sc = group < 4 ? b0 & 63 : (b2 & 15) | ((b0 >> 6) << 4);
                const int mn = group < 4 ? b1 & 63 : (b2 >> 4) | ((b1 >> 6) << 4);
                const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(dm)));
                const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(dm >> 16)));
                const uint2 lo = tl[r];
                const uint2 hi = tq[r];
                half2 partial = __float2half2_rn(0.0f);
                const int qsh = 4 * (group & 1);
                const uint32_t xl = hi.x >> group;
                const uint32_t xh = hi.y >> group;
                partial = __hfma2(q5k_pair(lo.x >> qsh, xl << 4), a0, partial);
                partial = __hfma2(q5k_pair((lo.x >> qsh) >> 8, xl >> 4), a1, partial);
                partial = __hfma2(q5k_pair(lo.y >> qsh, xh << 4), a2, partial);
                partial = __hfma2(q5k_pair((lo.y >> qsh) >> 8, xh >> 4), a3, partial);
                const float2 p = __half22float2(partial);
                const float dot = (p.x + p.y) * rescale;
                acc[r] += (d * sc) * dot - (dmin * mn) * sx;
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                th[r] = nh[r]; tl[r] = nl[r]; tq[r] = nq[r];
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
        if (rows == 4) {
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
    if (rows == 4) {
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

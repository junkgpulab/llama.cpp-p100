#include "mmv-p100-v2-q6k.cuh"

// Q6_K n=1 mat-vec for P100. Blocks are 210 B (2 mod 4 for odd blocks), so every
// weight read is a 4-byte word, realigned with a funnel shift for odd blocks.
// A warp takes a block pair of a row (16 lanes per block); a lane owns 4 adjacent
// columns l..l+3 of one 128-weight half: ql[l], ql[l+32], qh[l] words -> 16 weights.
// FP16 magic decode (1024+q-1056 = q-32), FP16 activations in shared memory,
// scale applied in FP32 once per 4 weights.

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include <climits>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>

namespace {

constexpr int q6k_threads = 128;
constexpr unsigned q6k_mask = 0xffffffffu;
static_assert(sizeof(block_q6_K) == 210, "Q6_K block size");
static_assert(offsetof(block_q6_K, qh) == 128 && offsetof(block_q6_K, scales) == 192 &&
              offsetof(block_q6_K, d) == 208, "Q6_K layout changed");

static int q6k_rows_per_warp() {
    static const int rows = [] {
        const char * env = std::getenv("GGML_CUDA_P100_V2_R");
        return env && std::strcmp(env, "2") == 0 ? 2 : env && std::strcmp(env, "8") == 0 ? 8 : 4;
    }();
    return rows;
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
static __device__ __forceinline__ half2 q6k_half2_bits(uint32_t bits) {
    return __halves2half2(__ushort_as_half(static_cast<unsigned short>(bits)),
                         __ushort_as_half(static_cast<unsigned short>(bits >> 16)));
}

// 4-byte load at a 2-byte aligned address; sh = 16 when the address is 2 mod 4.
template <bool LAST = false>
static __device__ __forceinline__ uint32_t q6k_ldw(const char * p, int sh) {
    const uint32_t * a = reinterpret_cast<const uint32_t *>(reinterpret_cast<uintptr_t>(p) & ~uintptr_t(3));
    const uint32_t w0 = __ldg(a);
    uint32_t w1 = w0;
    if (!LAST || sh) {
        w1 = __ldg(a + 1);   // LAST: the word after an aligned scale tail would pass the block end
    }
    return __funnelshift_r(w0, w1, sh);
}

template <int XS, int HS>
static __device__ __forceinline__ half2 q6k_pair(uint32_t X, uint32_t H) {
    const uint32_t t1 = X >> XS;
    const uint32_t t2 = HS >= 0 ? (H >> HS) : (H << (-HS));
    uint32_t y, m;
    asm("lop3.b32 %0, %1, 0x00300030, 0x64006400, 0xea;" : "=r"(y) : "r"(t2));
    asm("lop3.b32 %0, %1, 0x000f000f, %2, 0xea;" : "=r"(m) : "r"(t1), "r"(y));
    union { uint32_t u; half2 h; } c;
    c.u = m;
    return __hsub2(c.h, __float2half2_rn(1056.0f));
}

template <int P>
static __device__ __forceinline__ float q6k_dot(uint32_t qx, uint32_t qh, uint32_t aA, uint32_t aB) {
    // P: 0..3 -> ql from qa(P even)/qb(P odd): caller passes the right word
    constexpr int XS = (P >> 1) ? 4 : 0;
    constexpr int HSA = 2 * P - 4;
    constexpr int HSB = 2 * P + 4;
    half2 pa = __hmul2(q6k_pair<XS, HSA>(qx, qh), q6k_half2_bits(aA));
    pa = __hfma2(q6k_pair<XS + 8, HSB>(qx, qh), q6k_half2_bits(aB), pa);
    const float2 f = __half22float2(pa);
    return f.x + f.y;
}
#endif

template <int R>
static __global__ __launch_bounds__(q6k_threads, (R == 8 ? 2 : (R == 4 ? 4 : 8))) void q6k_mmv(
        const char * __restrict__ weights, const float * __restrict__ x, float * __restrict__ dst,
        int k, int m, size_t row_stride) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    constexpr int warps = q6k_threads / 32;
    const int n_blocks = k / QK_K;
    extern __shared__ __align__(16) float shared[];
    half * input = reinterpret_cast<half *>(shared);
    int * exponent = reinterpret_cast<int *>(shared + k / 2);

    // One warp converts one 256-weight block: lane reads x[32*i+lane], i=0..7.
    for (int cb = warp; cb < n_blocks; cb += warps) {
        float v[8];
        float maximum = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = x[cb * 256 + 32 * i + lane];
            maximum = fmaxf(maximum, fabsf(v[i]));
        }
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
            maximum = fmaxf(maximum, __shfl_xor_sync(q6k_mask, maximum, offset));
        }
        const int e = max(0, int((__float_as_uint(maximum) >> 23) & 255) - 127 - 5);
        const float prescale = __uint_as_float(uint32_t(127 - e) << 23);
        const int gq = lane >> 2, t = lane & 3;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int h = i >> 2, p = i & 3;
            const int pos = p * 4 + (t & 1) * 2 + (t >> 1);
            input[cb * 256 + (pos >> 3) * 128 + (h * 8 + gq) * 8 + (pos & 7)] = __float2half_rn(v[i] * prescale);
        }
        if (lane == 0) {
            exponent[cb] = e;
        }
    }
    __syncthreads();

    const int bp = lane >> 4;
    const int l16 = lane & 15;
    const int h = l16 >> 3;
    const int g = l16 & 7;
    const int is = g >> 2;
    const int sh = bp * 16;
    const int n_pairs = n_blocks / 2;
    const int o_qa = 64 * h + 4 * g;
    const int o_qb = o_qa + 32;
    const int o_qh = 128 + 32 * h + 4 * g;
    const int o_sc = 192 + 8 * h;
    const int sa = 24 - 8 * is, sb = 8 - 8 * is;
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
            uint32_t qa[R], qb[R], qh[R], s0[R], s1[R], dh[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const char * p = rowp[r] + size_t(ib) * sizeof(block_q6_K);
                qa[r] = q6k_ldw(p + o_qa, sh);
                qb[r] = q6k_ldw(p + o_qb, sh);
                qh[r] = q6k_ldw(p + o_qh, sh);
                s0[r] = q6k_ldw(p + o_sc, sh);
                s1[r] = q6k_ldw<true>(p + o_sc + 4, sh);
                dh[r] = *reinterpret_cast<const unsigned short *>(p + 208);
            }
            const half * act = input + ib * 256 + l16 * 8;
            uint4 a0, a1;
            const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(act));
            asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(a0.x), "=r"(a0.y), "=r"(a0.z), "=r"(a0.w) : "r"(a_addr) : "memory");
            asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(a1.x), "=r"(a1.y), "=r"(a1.z), "=r"(a1.w) : "r"(a_addr + 256) : "memory");
            const float rescale = __uint_as_float(uint32_t(127 + exponent[ib]) << 23);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int sc0 = int(s0[r] << sa) >> 24;
                const int sc1 = int(s0[r] << sb) >> 24;
                const int sc2 = int(s1[r] << sa) >> 24;
                const int sc3 = int(s1[r] << sb) >> 24;
                float s = sc0 * q6k_dot<0>(qa[r], qh[r], a0.x, a0.y);
                s = fmaf(float(sc1), q6k_dot<1>(qb[r], qh[r], a0.z, a0.w), s);
                s = fmaf(float(sc2), q6k_dot<2>(qa[r], qh[r], a1.x, a1.y), s);
                s = fmaf(float(sc3), q6k_dot<3>(qb[r], qh[r], a1.z, a1.w), s);
                const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(dh[r])));
                acc[r] = fmaf(d * rescale, s, acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int offset = 16; offset; offset >>= 1) {
                acc[r] += __shfl_down_sync(q6k_mask, acc[r], offset);
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

static int q6k_persistent_blocks(int device, int rows, size_t shared_bytes) {
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
    const auto key = std::make_tuple(device, rows, shared_bytes);
    auto blocks = block_counts.find(key);
    if (blocks == block_counts.end()) {
        int count;
        if (rows == 8) {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&count, q6k_mmv<8>, q6k_threads, shared_bytes));
        } else if (rows == 4) {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&count, q6k_mmv<4>, q6k_threads, shared_bytes));
        } else {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&count, q6k_mmv<2>, q6k_threads, shared_bytes));
        }
        GGML_ASSERT(count > 0);
        blocks = block_counts.emplace(key, count).first;
    }
    return sm->second * blocks->second;
}

} // namespace

bool ggml_cuda_p100_v2_q6k(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t k = src0->ne[0];
    const int64_t m = src0->ne[1];
    if (k <= 0 || k > INT_MAX || k % QK_K != 0 || (k / QK_K) % 2 != 0 || m <= 0 || m > INT_MAX ||
        src1->ne[0] != k || dst->ne[0] != m || src0->nb[0] != sizeof(block_q6_K) ||
        reinterpret_cast<uintptr_t>(src0->data) % 4 != 0 || src0->nb[1] % 4 != 0 ||
        src0->nb[1] < size_t(k / QK_K) * sizeof(block_q6_K) ||
        reinterpret_cast<uintptr_t>(src1->data) % alignof(float) != 0 ||
        reinterpret_cast<uintptr_t>(dst->data) % alignof(float) != 0) {
        return false;
    }
    const size_t shared_bytes = size_t(k) * sizeof(half) + size_t(k / QK_K) * sizeof(int);
    if (shared_bytes > 48 * 1024) {
        return false;
    }
    const int rows = q6k_rows_per_warp();
    const int64_t rows_per_block = (q6k_threads / 32) * rows;
    const int64_t n_groups = (m + rows_per_block - 1) / rows_per_block;
    const int max_blocks = q6k_persistent_blocks(ggml_cuda_info().devices[ctx.device].physical_device, rows, shared_bytes);
    const dim3 grid(static_cast<unsigned>(n_groups < max_blocks ? n_groups : max_blocks));
    const char * weights = static_cast<const char *>(src0->data);
    const float * input = static_cast<const float *>(src1->data);
    float * output = static_cast<float *>(dst->data);
    if (rows == 8) {
        q6k_mmv<8><<<grid, q6k_threads, shared_bytes, ctx.stream()>>>(weights, input, output, int(k), int(m), src0->nb[1]);
    } else if (rows == 4) {
        q6k_mmv<4><<<grid, q6k_threads, shared_bytes, ctx.stream()>>>(weights, input, output, int(k), int(m), src0->nb[1]);
    } else {
        q6k_mmv<2><<<grid, q6k_threads, shared_bytes, ctx.stream()>>>(weights, input, output, int(k), int(m), src0->nb[1]);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_p100_v2_q6k(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    return false;
}

#endif

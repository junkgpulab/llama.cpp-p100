#include "mmv-p100.cuh"
#include "mmv-p100-v2-q5k.cuh"
#include "mmv-p100-v2-q4k.cuh"

// HIP and MUSA also glob this directory. They must retain their existing paths.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "mmv-p100-types.cuh"
#include "unary.cuh"

#include <array>
#include <climits>
#include <cstring>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>

namespace {

struct p100_geometry {
    int threads_per_row;
    int rows_per_block;
};

static bool p100_log_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_CUDA_P100_LOG");
        return env && std::strcmp(env, "1") == 0;
    }();
    return enabled;
}

static bool p100_decline(const char * reason, ggml_type type = GGML_TYPE_COUNT) {
    if (p100_log_enabled()) {
        static std::mutex mutex;
        static std::set<std::pair<std::string, ggml_type>> seen;
        const std::lock_guard<std::mutex> lock(mutex);
        if (seen.emplace(reason, type).second) {
            if (type == GGML_TYPE_COUNT) {
                std::fprintf(stderr, "GGML_CUDA_P100 decline: %s\n", reason);
            } else {
                std::fprintf(stderr, "GGML_CUDA_P100 decline: %s type=%s\n", reason, ggml_type_name(type));
            }
        }
    }
    return false;
}

static bool p100_format_enabled(ggml_type type) {
    static const auto enabled = [] {
        std::array<bool, GGML_TYPE_COUNT> formats{};
        const char * env = std::getenv("GGML_CUDA_P100_FORMATS");
        // Measured: all ten quantized formats were 2.5-6x slower; enable them only by explicit opt-in.
        if (!env || std::strcmp(env, "default") == 0) {
            formats[GGML_TYPE_BF16] = true;
            return formats;
        }
        if (std::strcmp(env, "all") == 0) {
            for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
                formats[i] = p100_supported(static_cast<ggml_type>(i));
            }
            return formats;
        }
        // Match ggml_type_name() exactly (including the capital K in q4_K).
        // Empty/unknown tokens enable nothing; allow whitespace around list entries.
        const std::string list(env);
        size_t begin = 0;
        while (begin < list.size()) {
            const size_t comma = list.find(',', begin);
            const size_t end = comma == std::string::npos ? list.size() : comma;
            const std::string token = list.substr(begin, end - begin);
            const size_t first = token.find_first_not_of(" \t\r\n\f\v");
            if (first != std::string::npos) {
                const size_t last = token.find_last_not_of(" \t\r\n\f\v");
                const std::string name = token.substr(first, last - first + 1);
                for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
                    const auto candidate = static_cast<ggml_type>(i);
                    if (p100_supported(candidate) && name == ggml_type_name(candidate)) {
                        formats[i] = true;
                    }
                }
            }
            if (comma == std::string::npos) {
                break;
            }
            begin = comma + 1;
        }
        return formats;
    }();
    const size_t index = static_cast<size_t>(type);
    return index < enabled.size() && enabled[index];
}

static int p100_variant() {
#if defined(GGML_CUDA_P100_ALL_VARIANTS)
    static const int variant = [] {
        const char * env = std::getenv("GGML_CUDA_P100_VARIANT");
        if (env && std::strcmp(env, "0") == 0) {
            return 0;
        }
        if (env && std::strcmp(env, "1") == 0) {
            return 1;
        }
        return 2; // Unset, "2", and invalid values use the measured default.
    }();
    return variant;
#else
    // Measured: variant 2 wins BF16 512x2560 (8.7 us vs MMVF's 10.5 us); ignore the env in this build.
    return 2;
#endif
}

static void p100_log_launch(
        ggml_type type, int64_t m, int64_t k, int64_t n, bool ids, bool fused, p100_geometry geometry) {
    if (!p100_log_enabled()) {
        return;
    }
    using key = std::tuple<ggml_type, int64_t, int64_t, int64_t, bool, bool>;
    static std::mutex mutex;
    static std::set<key> seen;
    const std::lock_guard<std::mutex> lock(mutex);
    if (seen.emplace(type, m, k, n, ids, fused).second) {
        // variant=-1 is the unchanged shared-input short-row kernel.
        std::fprintf(stderr, "GGML_CUDA_P100 take: type=%s m=%lld k=%lld ncols=%lld ids=%d fused=%d "
                            "variant=%d lanes/row=%d rows/block=%d\n",
                     ggml_type_name(type), (long long) m, (long long) k, (long long) n, int(ids), int(fused),
                     k > 1024 ? p100_variant() : -1, geometry.threads_per_row, geometry.rows_per_block);
    }
}

struct p100_bias {
    const char * data;
    int64_t ne[4];
    size_t nb[4];
};

struct p100_mmv_args {
    const char * x;
    const char * gate;
    const char * y;
    const int32_t * ids;
    char * dst;
    int64_t k;
    int64_t m;
    int64_t ids_stride;
    size_t x_row, x_channel, x_sample;
    size_t y_col, y_channel, y_sample;
    size_t dst_col, dst_channel, dst_sample;
    uint3 channel_ratio, sample_ratio, channels_y;
    p100_bias bias, gate_bias;
    ggml_glu_op glu_op;
    float glu_limit;
};

// Transpose each 32-by-8 input tile in shared memory. Consecutive lanes decode
// eight consecutive weights each; this makes their FP32 input reads bank-conflict free.
static __device__ __forceinline__ int p100_shared_index(int i) {
    return (i & ~255) | ((i & 7) << 5) | ((i >> 3) & 31);
}

template <bool has_ids>
static __device__ __forceinline__ float p100_bias_value(
        const p100_bias & bias, int64_t row, int col, uint32_t channel, uint32_t sample, int expert) {
    if (!bias.data) {
        return 0.0f;
    }
    const size_t offset = row * sizeof(float) + (has_ids
        ? expert * bias.nb[1]
        : (col % bias.ne[1]) * bias.nb[1] + (channel % bias.ne[2]) * bias.nb[2] +
          (sample % bias.ne[3]) * bias.nb[3]);
    return *reinterpret_cast<const float *>(bias.data + offset);
}

template <ggml_type type, int ncols, int threads_per_row, int block_size, int tile_k, bool has_fusion, bool has_ids>
static __global__ void p100_mmv(p100_mmv_args a) {
    using weight = p100_w<type>;
    constexpr int dw = weight::decode_weights;
    static_assert(dw > 0 && dw % 8 == 0 && 512 % dw == 0, "decoder chunks must divide the input tile");
    constexpr int rows_per_block = block_size / threads_per_row;
    [[maybe_unused]] constexpr int warps = block_size / WARP_SIZE;
    constexpr int reduce_width = threads_per_row < WARP_SIZE ? threads_per_row : WARP_SIZE;
    const int tid = threadIdx.x;
    const int lane = tid % threads_per_row;
    const int64_t row = int64_t(blockIdx.x) * rows_per_block + tid / threads_per_row;
    const uint32_t channel = blockIdx.y;
    const uint32_t sample = has_ids ? 0 : blockIdx.z;
    const uint32_t token = has_ids ? blockIdx.z : 0;
    // Expert selection stays on the device, including strided/offset ID views.
    const int expert = has_ids ? a.ids[channel + int64_t(token) * a.ids_stride]
                              : fastdiv(channel, a.channel_ratio);
    const uint32_t channel_y = has_ids ? fastmodulo(channel, a.channels_y) : channel;
    const uint32_t sample_x = fastdiv(sample, a.sample_ratio);
    const size_t x_offset = sample_x * a.x_sample + expert * a.x_channel + row * a.x_row;
    const char * x = a.x + x_offset;
    const char * gate = has_fusion && a.gate ? a.gate + x_offset : nullptr;
    const char * y = a.y + sample * a.y_sample + channel_y * a.y_channel + token * a.y_col;
    char * dst = a.dst + sample * a.dst_sample + channel * a.dst_channel + token * a.dst_col;

    extern __shared__ float input[];
    float sum[ncols] = {};
    [[maybe_unused]] float gate_sum[ncols] = {};

    // Each FP32 input is staged once per block, in bounded tiles so ncols=8
    // never exceeds Pascal's shared-memory limit. Weights are reused across columns.
    for (int64_t tile = 0; tile < a.k; tile += tile_k) {
        const int count = int(a.k - tile < tile_k ? a.k - tile : tile_k);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            for (int i = tid; i < count; i += block_size) {
                input[j * tile_k + p100_shared_index(i)] =
                    *reinterpret_cast<const float *>(y + j * a.y_col + (tile + i) * sizeof(float));
            }
        }
        __syncthreads();

        if (row < a.m) {
            for (int i = lane * dw; i < count; i += threads_per_row * dw) {
                const int valid = count - i < dw ? count - i : dw;
                float w[dw];
                [[maybe_unused]] float g[dw];
                weight::decode(x, tile + i, valid, w);
                if constexpr (has_fusion) {
                    if (gate) {
                        weight::decode(gate, tile + i, valid, g);
                    }
                }
#pragma unroll
                for (int v = 0; v < dw; ++v) {
                    if (v < valid) {
#pragma unroll
                        for (int j = 0; j < ncols; ++j) {
                            const float activation = input[j * tile_k + p100_shared_index(i + v)];
                            sum[j] = fmaf(w[v], activation, sum[j]);
                            if constexpr (has_fusion) {
                                if (gate) {
                                    gate_sum[j] = fmaf(g[v], activation, gate_sum[j]);
                                }
                            }
                        }
                    }
                }
            }
        }
        // All lanes, including those beyond an incomplete row tile, reach both barriers.
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        float value = warp_reduce_sum<reduce_width>(sum[j]);
        [[maybe_unused]] float gate_value = 0.0f;
        if constexpr (has_fusion) {
            if (gate) {
                gate_value = warp_reduce_sum<reduce_width>(gate_sum[j]);
            }
        }
        if constexpr (threads_per_row > WARP_SIZE) {
            __shared__ float partial[ncols * warps * (has_fusion ? 2 : 1)];
            if (tid % WARP_SIZE == 0) {
                partial[j * warps + tid / WARP_SIZE] = value;
                if constexpr (has_fusion) {
                    partial[ncols * warps + j * warps + tid / WARP_SIZE] = gate_value;
                }
            }
            __syncthreads();
            if (tid < WARP_SIZE) {
                value = warp_reduce_sum<WARP_SIZE>(tid < warps ? partial[j * warps + tid] : 0.0f);
                if constexpr (has_fusion) {
                    if (gate) {
                        gate_value = warp_reduce_sum<WARP_SIZE>(
                            tid < warps ? partial[ncols * warps + j * warps + tid] : 0.0f);
                    }
                }
            }
        }

        if (lane == 0 && row < a.m) {
            if constexpr (has_fusion) {
                value += p100_bias_value<has_ids>(a.bias, row, j, channel, sample, expert);
                if (gate) {
                    gate_value += p100_bias_value<has_ids>(a.gate_bias, row, j, channel, sample, expert);
                    switch (a.glu_op) {
                        case GGML_GLU_OP_SWIGLU:
                            value *= ggml_cuda_op_silu_single(gate_value);
                            break;
                        case GGML_GLU_OP_GEGLU:
                            value *= ggml_cuda_op_gelu_single(gate_value);
                            break;
                        case GGML_GLU_OP_SWIGLU_OAI:
                            value = ggml_cuda_op_swiglu_oai_single(gate_value, value);
                            break;
                        case GGML_GLU_OP_SWIGLU_CLAMP:
                            value = ggml_cuda_op_swiglu_clamp_single(gate_value, value, a.glu_limit);
                            break;
                        default:
                            break;
                    }
                }
            }
            *reinterpret_cast<float *>(dst + j * a.dst_col + row * sizeof(float)) = value;
        }
    }
}

// The long-row loop reads the FP32 input through the read-only cache. Each full,
// aligned group is one 16-byte load; tensor views and the last 1..3 values use
// scalar loads, without reading padding or changing the weight decoder contract.
static __device__ __forceinline__ float4 p100_load_activation4(const char * column, int64_t first, int valid) {
    if (valid <= 0) {
        return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const float * p = reinterpret_cast<const float *>(column + first * sizeof(float));
    if (valid >= 4 && (reinterpret_cast<uintptr_t>(p) & 15) == 0) {
        return __ldg(reinterpret_cast<const float4 *>(p));
    }
    return make_float4(__ldg(p), valid > 1 ? __ldg(p + 1) : 0.0f,
                      valid > 2 ? __ldg(p + 2) : 0.0f, valid > 3 ? __ldg(p + 3) : 0.0f);
}

template <bool has_fusion, bool has_ids>
static __device__ __forceinline__ void p100_store_long_result(
        const p100_mmv_args & a, char * dst, int64_t row, int col, uint32_t channel, uint32_t sample,
        int expert, float value, float gate_value) {
    if constexpr (has_fusion) {
        value += p100_bias_value<has_ids>(a.bias, row, col, channel, sample, expert);
        if (a.gate) {
            gate_value += p100_bias_value<has_ids>(a.gate_bias, row, col, channel, sample, expert);
            switch (a.glu_op) {
                case GGML_GLU_OP_SWIGLU:
                    value *= ggml_cuda_op_silu_single(gate_value);
                    break;
                case GGML_GLU_OP_GEGLU:
                    value *= ggml_cuda_op_gelu_single(gate_value);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    value = ggml_cuda_op_swiglu_oai_single(gate_value, value);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    value = ggml_cuda_op_swiglu_clamp_single(gate_value, value, a.glu_limit);
                    break;
                default:
                    break;
            }
        }
    }
    *reinterpret_cast<float *>(dst + col * a.dst_col + row * sizeof(float)) = value;
    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(channel, sample, expert, gate_value);
    }
}

template <ggml_type type, int ncols, int threads_per_row, int rows_per_block, bool has_fusion, bool has_ids>
static __global__ void p100_mmv_long(p100_mmv_args a) {
    using weight = p100_w<type>;
    constexpr int dw = weight::decode_weights;
    constexpr int block_size = threads_per_row * rows_per_block;
    constexpr int reduce_width = threads_per_row < WARP_SIZE ? threads_per_row : WARP_SIZE;
    static_assert(dw > 0 && dw % 8 == 0 && 512 % dw == 0, "decoder chunks must divide 512");
    static_assert(rows_per_block >= 2 && block_size % WARP_SIZE == 0 && block_size <= 512,
                  "long-row blocks must contain multiple rows and complete warps");
    const int tid = threadIdx.x;
    const int lane = tid % threads_per_row;
    const int row_in_block = tid / threads_per_row;
    const int64_t row = int64_t(blockIdx.x) * rows_per_block + row_in_block;
    const uint32_t channel = blockIdx.y;
    const uint32_t sample = has_ids ? 0 : blockIdx.z;
    const uint32_t token = has_ids ? blockIdx.z : 0;
    const int expert = has_ids ? a.ids[channel + int64_t(token) * a.ids_stride]
                              : fastdiv(channel, a.channel_ratio);
    const uint32_t channel_y = has_ids ? fastmodulo(channel, a.channels_y) : channel;
    const uint32_t sample_x = fastdiv(sample, a.sample_ratio);
    const size_t x_offset = sample_x * a.x_sample + expert * a.x_channel + row * a.x_row;
    const char * x = a.x + x_offset;
    const char * gate = has_fusion && a.gate ? a.gate + x_offset : nullptr;
    const char * y = a.y + sample * a.y_sample + channel_y * a.y_channel + token * a.y_col;
    char * dst = a.dst + sample * a.dst_sample + channel * a.dst_channel + token * a.dst_col;
    // Two independent FMA chains accompany the two independent decoded loads.
    float sum0[ncols] = {};
    float sum1[ncols] = {};
    [[maybe_unused]] float gate_sum0[ncols] = {};
    [[maybe_unused]] float gate_sum1[ncols] = {};

    if (row < a.m) {
        // No staging or barriers in this loop. Every weight is decoded once and
        // reused across all ncols before advancing to the next pair of chunks
        // (an explicit factor-two unroll of the weight-load loop).
        for (int64_t first = int64_t(lane) * dw; first < a.k; first += 2 * threads_per_row * dw) {
            const int64_t second = first + threads_per_row * dw;
            const int valid0 = int(a.k - first < dw ? a.k - first : dw);
            const int valid1 = second < a.k ? int(a.k - second < dw ? a.k - second : dw) : 0;
            float w0[dw], w1[dw];
            [[maybe_unused]] float g0[dw], g1[dw];
            // Issue both independent weight loads before consuming either one.
            weight::decode(x, first, valid0, w0);
            if (valid1) {
                weight::decode(x, second, valid1, w1);
            }
            if constexpr (has_fusion) {
                if (gate) {
                    weight::decode(gate, first, valid0, g0);
                    if (valid1) {
                        weight::decode(gate, second, valid1, g1);
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                const char * column = y + j * a.y_col;
#pragma unroll
                for (int v = 0; v < dw; v += 4) {
                    if (v < valid0) {
                        const float4 y0 = p100_load_activation4(column, first + v, valid0 - v);
                        sum0[j] = fmaf(w0[v + 0], y0.x, sum0[j]);
                        sum0[j] = fmaf(w0[v + 1], y0.y, sum0[j]);
                        sum0[j] = fmaf(w0[v + 2], y0.z, sum0[j]);
                        sum0[j] = fmaf(w0[v + 3], y0.w, sum0[j]);
                        if constexpr (has_fusion) {
                            if (gate) {
                                gate_sum0[j] = fmaf(g0[v + 0], y0.x, gate_sum0[j]);
                                gate_sum0[j] = fmaf(g0[v + 1], y0.y, gate_sum0[j]);
                                gate_sum0[j] = fmaf(g0[v + 2], y0.z, gate_sum0[j]);
                                gate_sum0[j] = fmaf(g0[v + 3], y0.w, gate_sum0[j]);
                            }
                        }
                    }
                    if (v < valid1) {
                        const float4 y1 = p100_load_activation4(column, second + v, valid1 - v);
                        sum1[j] = fmaf(w1[v + 0], y1.x, sum1[j]);
                        sum1[j] = fmaf(w1[v + 1], y1.y, sum1[j]);
                        sum1[j] = fmaf(w1[v + 2], y1.z, sum1[j]);
                        sum1[j] = fmaf(w1[v + 3], y1.w, sum1[j]);
                        if constexpr (has_fusion) {
                            if (gate) {
                                gate_sum1[j] = fmaf(g1[v + 0], y1.x, gate_sum1[j]);
                                gate_sum1[j] = fmaf(g1[v + 1], y1.y, gate_sum1[j]);
                                gate_sum1[j] = fmaf(g1[v + 2], y1.z, gate_sum1[j]);
                                gate_sum1[j] = fmaf(g1[v + 3], y1.w, gate_sum1[j]);
                            }
                        }
                    }
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        sum0[j] = warp_reduce_sum<reduce_width>(sum0[j] + sum1[j]);
        if constexpr (has_fusion) {
            if (gate) {
                gate_sum0[j] = warp_reduce_sum<reduce_width>(gate_sum0[j] + gate_sum1[j]);
            }
        }
    }
    if constexpr (threads_per_row > WARP_SIZE) {
        constexpr int warps = block_size / WARP_SIZE;
        constexpr int warps_per_row = threads_per_row / WARP_SIZE;
        __shared__ float partial[ncols * warps * (has_fusion ? 2 : 1)];
        if (tid % WARP_SIZE == 0) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                partial[j * warps + tid / WARP_SIZE] = sum0[j];
                if constexpr (has_fusion) {
                    partial[ncols * warps + j * warps + tid / WARP_SIZE] = gate_sum0[j];
                }
            }
        }
        // One final barrier for all columns, after the complete dot product.
        // Incomplete row tiles participate too; no thread returns before it.
        __syncthreads();
        if (lane < WARP_SIZE) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                const int index = j * warps + row_in_block * warps_per_row + lane;
                const float value = warp_reduce_sum<WARP_SIZE>(lane < warps_per_row ? partial[index] : 0.0f);
                float gate_value = 0.0f;
                if constexpr (has_fusion) {
                    if (gate) {
                        gate_value = warp_reduce_sum<WARP_SIZE>(
                            lane < warps_per_row ? partial[ncols * warps + index] : 0.0f);
                    }
                }
                if (lane == 0 && row < a.m) {
                    p100_store_long_result<has_fusion, has_ids>(
                        a, dst, row, j, channel, sample, expert, value, gate_value);
                }
            }
        }
    } else if (row < a.m && lane == 0) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            p100_store_long_result<has_fusion, has_ids>(
                a, dst, row, j, channel, sample, expert, sum0[j], gate_sum0[j]);
        }
    }
}

template <ggml_type type, int ncols, bool has_fusion, bool has_ids, int variant>
static p100_geometry p100_launch_long(
        const p100_mmv_args & a, int64_t channels, int64_t samples_or_tokens, cudaStream_t stream) {
    using weight = p100_w<type>;
    // Measured: variant 2 wins BF16 512x2560; variants 0/1 are built only for explicit tuning builds.
    // BF16, n=1..3, >=1024 row instances (m * channels * samples/tokens):
    // variant 0:           64 lanes/row, 4 rows/256-thread block.
    // variant 1:          128 lanes/row, 2 rows/256-thread block.
    // variant 2 (default): 32 lanes/row, 8 rows/256-thread block.
    // Below 1024 row instances use respectively 256/128/64 lanes, 2 rows/block,
    // spreading a thin matrix across more SMs while retaining several rows/block.
    // For n=4..8 halve lanes/row to limit accumulator register pressure.
    // Compressed (<1 byte/weight) formats halve lanes again; larger decoder chunks
    // reduce lanes in proportion to decode_weights/8 and target 128-thread blocks
    // for decoder register pressure. All choices are powers of two.
    constexpr bool compressed = weight::block_bytes < weight::block_size;
    constexpr int target_threads = compressed || weight::decode_weights > 8 ? 128 : 256;
    constexpr int pressure = (ncols >= 4 ? 2 : 1) * (compressed ? 2 : 1) *
                             (weight::decode_weights / 8);
    constexpr int regular_raw = (variant == 0 ? 64 : variant == 1 ? 128 : 32) / pressure;
    constexpr int thin_raw = (variant == 0 ? 256 : variant == 1 ? 128 : 64) / pressure;
    constexpr int regular = regular_raw < 8 ? 8 : regular_raw;
    constexpr int thin = thin_raw < 8 ? 8 : thin_raw;
    const auto launch = [&](auto row_threads, auto row_count) {
        constexpr int r = decltype(row_threads)::value;
        constexpr int rows = decltype(row_count)::value;
        const dim3 grid(static_cast<unsigned int>((a.m + rows - 1) / rows),
                        static_cast<unsigned int>(channels), static_cast<unsigned int>(samples_or_tokens));
        p100_mmv_long<type, ncols, r, rows, has_fusion, has_ids>
            <<<grid, r * rows, 0, stream>>>(a);
        return p100_geometry{r, rows};
    };
    if (uint64_t(a.m) * uint64_t(channels) * uint64_t(samples_or_tokens) < 1024) {
        constexpr int rows = thin < 64 ? 128 / thin : 2;
        return launch(std::integral_constant<int, thin>{}, std::integral_constant<int, rows>{});
    } else {
        return launch(std::integral_constant<int, regular>{}, std::integral_constant<int, target_threads / regular>{});
    }
}

template <ggml_type type, int ncols, bool has_fusion, bool has_ids>
static p100_geometry p100_launch(const p100_mmv_args & a, int64_t channels, int64_t samples_or_tokens, cudaStream_t stream) {
    // Measured: BF16 10240x320 improved from 66.4 to 19.3 us; preserve the short-row kernel/geometries.
    const auto launch = [&](auto row_threads, auto threads, auto tile) {
        constexpr int r = decltype(row_threads)::value;
        constexpr int b = decltype(threads)::value;
        constexpr int t = decltype(tile)::value;
        const dim3 grid(static_cast<unsigned int>((a.m + b/r - 1) / (b/r)),
                        static_cast<unsigned int>(channels), static_cast<unsigned int>(samples_or_tokens));
        p100_mmv<type, ncols, r, b, t, has_fusion, has_ids>
            <<<grid, b, ncols * t * sizeof(float), stream>>>(a);
        return p100_geometry{r, b/r};
    };
    if (a.k <= 512) {
        return launch(std::integral_constant<int, 8>{}, std::integral_constant<int, 128>{}, std::integral_constant<int, 512>{});
    } else if (a.k <= 1024) {
        return launch(std::integral_constant<int, 16>{}, std::integral_constant<int, 128>{}, std::integral_constant<int, 1024>{});
    } else {
        switch (p100_variant()) {
#if defined(GGML_CUDA_P100_ALL_VARIANTS)
            case 0: return p100_launch_long<type, ncols, has_fusion, has_ids, 0>(a, channels, samples_or_tokens, stream);
            case 1: return p100_launch_long<type, ncols, has_fusion, has_ids, 1>(a, channels, samples_or_tokens, stream);
#endif
            case 2: return p100_launch_long<type, ncols, has_fusion, has_ids, 2>(a, channels, samples_or_tokens, stream);
            default: GGML_ABORT("invalid cached P100 variant");
        }
    }
}

template <ggml_type type, bool has_fusion>
static p100_geometry p100_launch_columns(
        const p100_mmv_args & a, int ncols, int64_t channels, int64_t samples, cudaStream_t stream) {
    if (a.ids) {
        // Different tokens can select different experts, so each token has its own grid plane.
        // NVIDIA's existing unfused BF16 ID dispatcher bypasses MMVF; fused ID calls
        // can reach this hook. Routing and fusion eligibility are deliberately left to the caller.
        return p100_launch<type, 1, has_fusion, true>(a, channels, ncols, stream);
    }
    switch (ncols) {
#define P100_COLUMNS(N) case N: return p100_launch<type, N, has_fusion, false>(a, channels, samples, stream);
        P100_COLUMNS(1)
        P100_COLUMNS(2)
        P100_COLUMNS(3)
        P100_COLUMNS(4)
        P100_COLUMNS(5)
        P100_COLUMNS(6)
        P100_COLUMNS(7)
        P100_COLUMNS(8)
#undef P100_COLUMNS
        default: GGML_ABORT("invalid P100 column count");
    }
}

static bool p100_float_layout(const ggml_tensor * t) {
    if (t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float) ||
        reinterpret_cast<uintptr_t>(t->data) % alignof(float) != 0) {
        return false;
    }
    for (int d = 1; d < 4; ++d) {
        if (t->nb[d] % sizeof(float) != 0) {
            return false;
        }
    }
    return true;
}

static bool p100_set_bias(p100_bias & out, const ggml_tensor * bias, const ggml_tensor * dst,
                   const ggml_tensor * src0, bool has_ids) {
    if (!bias) {
        return true;
    }
    if (!p100_float_layout(bias) || bias->ne[0] != dst->ne[0]) {
        return false;
    }
    if (has_ids) {
        if (bias->ne[1] != src0->ne[2] || bias->ne[2] != 1 || bias->ne[3] != 1) {
            return false;
        }
    } else {
        for (int d = 1; d < 4; ++d) {
            if (bias->ne[d] <= 0 || dst->ne[d] % bias->ne[d] != 0) {
                return false;
            }
        }
    }
    out.data = static_cast<const char *>(bias->data);
    for (int d = 0; d < 4; ++d) {
        out.ne[d] = bias->ne[d];
        out.nb[d] = bias->nb[d];
    }
    return true;
}

} // namespace

bool ggml_cuda_p100_mmv(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_CUDA_P100");
        return env && std::strcmp(env, "1") == 0;
    }();
    // These two exits precede tensor inspection, logging, stream access and allocations.
    // They stay silent even with LOG=1 to honor the mandatory no-work opt-out contract.
    if (!enabled) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc != GGML_CUDA_CC_PASCAL) {
        return false;
    }
    static const bool v2_enabled = [] {
        const char * env = std::getenv("GGML_CUDA_P100_V2");
        return env && std::strcmp(env, "1") == 0;
    }();
    // GLU is active only with a gate; glu_op alone may be uninitialized.
    if (v2_enabled && (src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q4_K) && !ids &&
        (!fusion || (!fusion->gate && !fusion->x_bias && !fusion->gate_bias && !fusion->x_scale &&
                     !fusion->gate_scale && !fusion->shared_up && !fusion->shared_gate && !fusion->shared_dst)) &&
        dst->ne[1] == 1 && src1->ne[1] == 1 &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        dst->ne[2] == 1 && dst->ne[3] == 1 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(src1) && ggml_is_contiguous(dst)) {
        if (reinterpret_cast<uintptr_t>(src0->data) % 16 != 0 || src0->nb[1] % 16 != 0) {
            return p100_decline("v2 Q5_K requires 16-byte aligned weight base/row stride", src0->type);
        }
        if (src0->type == GGML_TYPE_Q4_K ? ggml_cuda_p100_v2_q4k(ctx, src0, src1, dst)
                                         : ggml_cuda_p100_v2_q5k(ctx, src0, src1, dst)) {
            return true;
        }
    }
    if (!p100_supported(src0->type)) {
        return p100_decline("no weight decoder");
    }
    if (!p100_format_enabled(src0->type)) {
        return p100_decline("format not enabled", src0->type);
    }
    // Measured: k>4096 lost at 320x10240 and was near MMVF at 4096x14336; retain the old path.
    if (src0->type == GGML_TYPE_BF16 && src0->ne[0] > 4096) {
        return p100_decline("BF16 long rows: old path is as fast");
    }
    if (!p100_float_layout(src1) || !p100_float_layout(dst)) {
        return p100_decline("src1/dst must have contiguous aligned F32 rows");
    }
    const int64_t k = src0->ne[0];
    const int64_t m = src0->ne[1];
    const size_t weight_alignment = p100_weight_alignment(src0->type);
    if (src0->nb[0] != ggml_type_size(src0->type) || k % ggml_blck_size(src0->type) != 0 ||
        reinterpret_cast<uintptr_t>(src0->data) % weight_alignment != 0) {
        return p100_decline("weight rows are not contiguous or aligned");
    }
    for (int d = 0; d < 4; ++d) {
        if (src0->ne[d] <= 0 || src1->ne[d] <= 0 || dst->ne[d] <= 0) {
            return p100_decline("empty tensor");
        }
        if (d > 0 && src0->nb[d] % weight_alignment != 0) {
            return p100_decline("weight strides are not block aligned");
        }
    }
    const int64_t ncols = ids ? dst->ne[2] : dst->ne[1];
    const int64_t channels = ids ? dst->ne[1] : dst->ne[2];
    const int64_t samples_or_tokens = ids ? ncols : dst->ne[3];
    if (ncols < 1 || ncols > 8) {
        return p100_decline("column count outside 1..8");
    }
    // Pascal grid.y/z are limited to 65535; grid.x is limited to INT_MAX.
    if (k > INT_MAX || m > INT_MAX || channels > 65535 || samples_or_tokens > 65535 ||
        src0->ne[2] > INT_MAX || src0->ne[3] > UINT_MAX) {
        return p100_decline("shape exceeds launch/index limits");
    }
    if (src1->ne[0] != k || dst->ne[0] != m) {
        return p100_decline("matrix shape mismatch");
    }
    if (ids) {
        if (ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) ||
            ids->nb[1] % sizeof(int32_t) != 0 ||
            reinterpret_cast<uintptr_t>(ids->data) % alignof(int32_t) != 0 ||
            ids->ne[0] != channels || ids->ne[1] != ncols || ids->ne[2] != 1 || ids->ne[3] != 1 ||
            src0->ne[3] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1 ||
            src1->ne[2] != ncols || channels % src1->ne[1] != 0) {
            return p100_decline("unsupported expert ID shape/layout");
        }
    } else if (src1->ne[1] != ncols || src1->ne[2] != channels || src1->ne[3] != dst->ne[3] ||
               channels % src0->ne[2] != 0 || dst->ne[3] % src0->ne[3] != 0) {
        return p100_decline("unsupported channel/sample broadcast");
    }

    // Measured in round 1: unbatched BF16 48x2560 lost to MMVF; retain the underfill fallback.
    // Even with only two rows/block, unbatched 48x2560 has just 24 blocks on
    // P100's 56 SMs; round 1 already lost to MMVF there. Keep the existing path.
    // Count expert/token and broadcast planes too, so otherwise useful batches
    // of small matrices are not rejected solely because m is small.
    if (src0->type == GGML_TYPE_BF16 && k > 1024) {
        const uint64_t two_row_blocks = uint64_t((m + 1) / 2) * uint64_t(channels) * uint64_t(samples_or_tokens);
        if (two_row_blocks < uint64_t(ggml_cuda_info().devices[ctx.device].nsm)) {
            return p100_decline("BF16 long rows: fewer than one two-row block per SM");
        }
    }

    p100_mmv_args a{};
    a.glu_op = GGML_GLU_OP_SWIGLU;
    if (fusion) {
        if (fusion->x_scale || fusion->gate_scale || fusion->shared_up || fusion->shared_gate || fusion->shared_dst) {
            return p100_decline("scale/shared-expert fusion");
        }
        if (fusion->gate) {
            if (fusion->gate->type != src0->type || !ggml_are_same_shape(fusion->gate, src0) ||
                !ggml_are_same_stride(fusion->gate, src0) ||
                reinterpret_cast<uintptr_t>(fusion->gate->data) % weight_alignment != 0) {
                return p100_decline("gate shape/type/stride mismatch");
            }
            switch (fusion->glu_op) {
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    break;
                default: return p100_decline("unsupported GLU operation");
            }
            a.gate = static_cast<const char *>(fusion->gate->data);
            a.glu_op = fusion->glu_op;
            a.glu_limit = fusion->glu_limit;
        }
        if (!p100_set_bias(a.bias, fusion->x_bias, dst, src0, ids != nullptr) ||
            !p100_set_bias(a.gate_bias, fusion->gate_bias, dst, src0, ids != nullptr)) {
            return p100_decline("unsupported bias shape/type/stride");
        }
    }

    a.x = static_cast<const char *>(src0->data);
    a.y = static_cast<const char *>(src1->data);
    a.ids = ids ? static_cast<const int32_t *>(ids->data) : nullptr;
    a.dst = static_cast<char *>(dst->data);
    a.k = k;
    a.m = m;
    a.ids_stride = ids ? ids->nb[1] / sizeof(int32_t) : 0;
    a.x_row = src0->nb[1]; a.x_channel = src0->nb[2]; a.x_sample = src0->nb[3];
    a.y_col = src1->nb[ids ? 2 : 1]; a.y_channel = src1->nb[ids ? 1 : 2]; a.y_sample = src1->nb[3];
    a.dst_col = dst->nb[ids ? 2 : 1]; a.dst_channel = dst->nb[ids ? 1 : 2]; a.dst_sample = dst->nb[3];
    a.channel_ratio = ids ? make_uint3(0, 0, 0) : init_fastdiv_values(channels / src0->ne[2]);
    a.sample_ratio = init_fastdiv_values(dst->ne[3] / src0->ne[3]);
    a.channels_y = ids ? init_fastdiv_values(src1->ne[1]) : make_uint3(0, 0, 0);
    const bool fused = a.gate || a.bias.data || a.gate_bias.data;
    const cudaStream_t stream = ctx.stream();
    p100_geometry geometry{};
    switch (src0->type) {
#define P100_DISPATCH(T) \
        case T: \
            if (fused) { geometry = p100_launch_columns<T, true>(a, int(ncols), channels, dst->ne[3], stream); } \
            else       { geometry = p100_launch_columns<T, false>(a, int(ncols), channels, dst->ne[3], stream); } \
            break;
        GGML_CUDA_P100_TYPES(P100_DISPATCH)
#undef P100_DISPATCH
        default: GGML_ABORT("missing P100 decoder dispatch");
    }
    CUDA_CHECK(cudaGetLastError());
    p100_log_launch(src0->type, m, k, ncols, ids != nullptr, fused, geometry);
    return true;
}

#else

bool ggml_cuda_p100_mmv(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_UNUSED_VARS(ctx, src0, src1, ids, dst, fusion);
    return false;
}

#endif

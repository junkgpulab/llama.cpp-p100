#include "mmv-p100.cuh"

// HIP and MUSA also glob this directory. They must retain their existing paths.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "mmv-p100-types.cuh"
#include "unary.cuh"

#include <climits>
#include <cstring>
#include <set>
#include <tuple>
#include <type_traits>

namespace {

bool p100_log_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_CUDA_P100_LOG");
        return env && std::strcmp(env, "1") == 0;
    }();
    return enabled;
}

bool p100_decline(const char * reason) {
    if (p100_log_enabled()) {
        static std::mutex mutex;
        static std::set<std::string> seen;
        const std::lock_guard<std::mutex> lock(mutex);
        if (seen.emplace(reason).second) {
            std::fprintf(stderr, "GGML_CUDA_P100 decline: %s\n", reason);
        }
    }
    return false;
}

void p100_log_launch(ggml_type type, int64_t m, int64_t k, int64_t n, bool ids, bool fused) {
    if (!p100_log_enabled()) {
        return;
    }
    using key = std::tuple<ggml_type, int64_t, int64_t, int64_t, bool, bool>;
    static std::mutex mutex;
    static std::set<key> seen;
    const std::lock_guard<std::mutex> lock(mutex);
    if (seen.emplace(type, m, k, n, ids, fused).second) {
        std::fprintf(stderr, "GGML_CUDA_P100 take: type=%s m=%lld k=%lld ncols=%lld ids=%d fused=%d\n",
                     ggml_type_name(type), (long long) m, (long long) k, (long long) n, int(ids), int(fused));
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

template <ggml_type type, int ncols, bool has_fusion, bool has_ids>
void p100_launch(const p100_mmv_args & a, int64_t channels, int64_t samples_or_tokens, cudaStream_t stream) {
    // PROPOSED geometry, to be measured on P100:
    // k<=512 (including k=320): 128 threads, 8 lanes/row, 16 rows/block (4 rows/warp).
    // k<=1024: 128 threads, 16 lanes/row, 8 rows/block.
    // k>1024 (including k=10240): 256 threads/row, 1 row/block, 2048-weight input tiles.
    // For 5..8 columns use 128 threads/row and 1024-weight tiles, capping shared input at 32 KiB.
    const auto launch = [&](auto row_threads, auto threads, auto tile) {
        constexpr int r = decltype(row_threads)::value;
        constexpr int b = decltype(threads)::value;
        constexpr int t = decltype(tile)::value;
        const dim3 grid(static_cast<unsigned int>((a.m + b/r - 1) / (b/r)),
                        static_cast<unsigned int>(channels), static_cast<unsigned int>(samples_or_tokens));
        p100_mmv<type, ncols, r, b, t, has_fusion, has_ids>
            <<<grid, b, ncols * t * sizeof(float), stream>>>(a);
    };
    if (a.k <= 512) {
        launch(std::integral_constant<int, 8>{}, std::integral_constant<int, 128>{}, std::integral_constant<int, 512>{});
    } else if (a.k <= 1024) {
        launch(std::integral_constant<int, 16>{}, std::integral_constant<int, 128>{}, std::integral_constant<int, 1024>{});
    } else if constexpr (ncols <= 4) {
        launch(std::integral_constant<int, 256>{}, std::integral_constant<int, 256>{}, std::integral_constant<int, 2048>{});
    } else {
        launch(std::integral_constant<int, 128>{}, std::integral_constant<int, 128>{}, std::integral_constant<int, 1024>{});
    }
}

template <ggml_type type, bool has_fusion>
void p100_launch_columns(
        const p100_mmv_args & a, int ncols, int64_t channels, int64_t samples, cudaStream_t stream) {
    if (a.ids) {
        // Different tokens can select different experts, so each token has its own grid plane.
        // NVIDIA's existing unfused BF16 ID dispatcher bypasses MMVF; fused ID calls
        // can reach this hook. Routing and fusion eligibility are deliberately left to the caller.
        p100_launch<type, 1, has_fusion, true>(a, channels, ncols, stream);
        return;
    }
    switch (ncols) {
#define P100_COLUMNS(N) case N: p100_launch<type, N, has_fusion, false>(a, channels, samples, stream); break;
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

bool p100_float_layout(const ggml_tensor * t) {
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

bool p100_set_bias(p100_bias & out, const ggml_tensor * bias, const ggml_tensor * dst,
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
    if (!p100_supported(src0->type)) {
        return p100_decline("no weight decoder");
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
    switch (src0->type) {
#define P100_DISPATCH(T) \
        case T: \
            if (fused) { p100_launch_columns<T, true>(a, int(ncols), channels, dst->ne[3], stream); } \
            else       { p100_launch_columns<T, false>(a, int(ncols), channels, dst->ne[3], stream); } \
            break;
        GGML_CUDA_P100_TYPES(P100_DISPATCH)
#undef P100_DISPATCH
        default: GGML_ABORT("missing P100 decoder dispatch");
    }
    CUDA_CHECK(cudaGetLastError());
    p100_log_launch(src0->type, m, k, ncols, ids != nullptr, fused);
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

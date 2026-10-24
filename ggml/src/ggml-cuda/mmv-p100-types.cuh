#pragma once

#include "common.cuh"

// p100_w<T> is the only device-side interface a new weight format implements.
// block_size/block_bytes describe storage blocks; alignment is their minimum byte alignment.
// decode_weights is a multiple of eight, at most 512, dividing 512.
// decode(row, first, valid, out) decodes decode_weights consecutive logical weights.
// row points to the start of one row; first is a multiple of decode_weights.
// valid is in [1, decode_weights]; read only valid weights and zero the remaining out entries.
// out is a float[decode_weights]; decode must use FP32 and no activation quantization.
// BF16 example: first=8, valid=8 loads bytes [16,32) with one aligned uint4 load.
// Each BF16 bits value becomes __uint_as_float(uint32_t(bits) << 16), e.g. 0x3f80 -> 1.0f.
// Add one specialization and one GGML_CUDA_P100_TYPES entry; dispatch needs no other edits.
template <ggml_type T> struct p100_w;

template <> struct p100_w<GGML_TYPE_BF16> {
    static constexpr int block_size = 1;
    static constexpr int block_bytes = 2;
    static constexpr int alignment = alignof(uint16_t);
    static constexpr int decode_weights = 8;

    static __device__ __forceinline__ void decode(
            const char * row, int64_t first, int valid, float (&out)[decode_weights]) {
        const char * p = row + first * block_bytes;
        if (valid == decode_weights && (reinterpret_cast<uintptr_t>(p) & 15) == 0) {
            const uint4 v = *reinterpret_cast<const uint4 *>(p);
            out[0] = __uint_as_float(v.x << 16);
            out[1] = __uint_as_float(v.x & 0xffff0000u);
            out[2] = __uint_as_float(v.y << 16);
            out[3] = __uint_as_float(v.y & 0xffff0000u);
            out[4] = __uint_as_float(v.z << 16);
            out[5] = __uint_as_float(v.z & 0xffff0000u);
            out[6] = __uint_as_float(v.w << 16);
            out[7] = __uint_as_float(v.w & 0xffff0000u);
        } else {
            const uint16_t * bits = reinterpret_cast<const uint16_t *>(p);
#pragma unroll
            for (int i = 0; i < decode_weights; ++i) {
                out[i] = i < valid ? __uint_as_float(uint32_t(bits[i]) << 16) : 0.0f;
            }
        }
    }
};

// Single format registry, shared by the support check and the launch dispatch.
#define GGML_CUDA_P100_TYPES(X) \
    X(GGML_TYPE_BF16)

static inline bool p100_supported(ggml_type type) {
    switch (type) {
#define P100_SUPPORTED(T) case T: return true;
        GGML_CUDA_P100_TYPES(P100_SUPPORTED)
#undef P100_SUPPORTED
        default: return false;
    }
}

static inline size_t p100_weight_alignment(ggml_type type) {
    switch (type) {
#define P100_ALIGNMENT(T) case T: return p100_w<T>::alignment;
        GGML_CUDA_P100_TYPES(P100_ALIGNMENT)
#undef P100_ALIGNMENT
        default: return 1;
    }
}

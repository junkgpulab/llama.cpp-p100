#pragma once

#include "mmv-p100-types.cuh"

// iq2_xs: block_iq2_xs = { ggml_half d; uint16_t qs[32]; uint8_t scales[8]; } = 74 bytes per 256 weights.
// Byte layout inside a block: d at 0..1, qs[i] at 2 + 2*i (i = 0..31), scales[s] at 66 + s.
// Each 32-weight sub-block s has one scale byte: the low nibble scales its first 16 weights,
// the high nibble its last 16. So one decode group is 16 weights and decode_weights = 16.
// Math follows dequantize_row_iq2_xs (ggml-quants.c) in the same order: db = d * (0.5f + nib) * 0.25f,
// then value = db * grid[j] * sign. All FP32, no shared memory, no cross-thread ops.
template <> struct p100_w<GGML_TYPE_IQ2_XS> {
    static constexpr int block_size = 256;
    static constexpr int block_bytes = sizeof(block_iq2_xs);   // 74
    static constexpr int alignment = alignof(block_iq2_xs);    // 2: only 1- and 2-byte loads are used
    static constexpr int decode_weights = 16;

    static __device__ __forceinline__ void decode(
            const char * row, int64_t first, int valid, float (&out)[decode_weights]) {
        const char * blk = row + (first / block_size) * block_bytes;
        const int off  = int(first % block_size);   // multiple of 16, 0..240
        const int ib32 = off >> 5;                   // 32-weight sub-block, 0..7
        const int half = (off >> 4) & 1;             // 0: low scale nibble, 1: high scale nibble

        const float d    = __half2float(*reinterpret_cast<const ggml_half *>(blk));
        const uint8_t sc = (uint8_t) blk[66 + ib32];
        const int nib    = half ? (sc >> 4) : (sc & 0xf);
        const float db   = d * (0.5f + float(nib)) * 0.25f;

#pragma unroll
        for (int g = 0; g < 2; ++g) {
            // Two 8-weight groups per decode call. Group g uses qs[4*ib32 + 2*half + g].
            if (8 * g < valid) {
                const int qi = 4 * ib32 + 2 * half + g;
                const uint16_t q = *reinterpret_cast<const uint16_t *>(blk + 2 + 2 * qi);
                const uint8_t * grid = reinterpret_cast<const uint8_t *>(iq2xs_grid + (q & 511));
                const uint8_t signs = ksigns_iq2xs[q >> 9];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int i = 8 * g + j;
                    const float s = (signs & kmask_iq2xs[j]) ? -1.f : 1.f;
                    out[i] = i < valid ? db * float(grid[j]) * s : 0.0f;
                }
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    out[8 * g + j] = 0.0f;
                }
            }
        }
    }
};

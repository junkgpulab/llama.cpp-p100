#pragma once
#include "mmv-p100-types.cuh"

// iq2_s: block_iq2_s is 82 bytes for 256 weights: d (fp16) | qs[64] | qh[8] | scales[8].
// Sub-block ib32 (32 weights) uses qs[4*ib32 .. +3] for grid low bits, qh[ib32] for grid high bits,
// scales[ib32] for two 4-bit scales (one per 16 weights), and qs[32 + 4*ib32 .. +3] for sign bytes.
// Each 8-weight group l of a sub-block is one iq2s_grid entry, expanded as in dequantize_row_iq2_s.
// Every byte is loaded on its own, so the decoder needs only 1-byte alignment.
template <> struct p100_w<GGML_TYPE_IQ2_S> {
    static constexpr int block_size = QK_K;
    static constexpr int block_bytes = static_cast<int>(sizeof(block_iq2_s));
    static constexpr int alignment = alignof(uint8_t);
    static constexpr int decode_weights = 16;

    static __device__ __forceinline__ void decode(
            const char * row, int64_t first, int valid, float (&out)[decode_weights]) {
        const uint8_t * blk = reinterpret_cast<const uint8_t *>(row + (first / block_size) * block_bytes);
        const int wb = int(first % block_size);                    // weight offset in block, multiple of 16

        const uint16_t dbits = uint16_t(blk[0]) | uint16_t(uint16_t(blk[1]) << 8);
        const float d = __half2float(__ushort_as_half(dbits));
        const uint8_t * qs     = blk + 2;                          // qs[QK_K/4]
        const uint8_t * qh     = blk + 2 + QK_K/4;                 // qh[QK_K/32]
        const uint8_t * scales = qh + QK_K/32;                     // scales[QK_K/32]

        const int ib32 = wb / 32;                                  // sub-block, 0..7
        const int sel  = (wb % 32) / 16;                           // 0: low scale nibble, 1: high
        const uint8_t sc = scales[ib32];
        const int nib = sel ? (sc >> 4) : (sc & 0xf);
        const float dl = d * (0.5f + float(nib)) * 0.25f;

#pragma unroll
        for (int g8 = 0; g8 < 2; ++g8) {
            const int i0 = 8 * g8;                                 // first output slot of this 8-weight group
            if (i0 >= valid) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    out[i0 + j] = 0.0f;
                }
                continue;
            }
            const int l = 2 * sel + g8;                            // 8-weight group in the sub-block, 0..3
            const int gi = qs[4 * ib32 + l] | ((qh[ib32] << (8 - 2 * l)) & 0x300);
            const uint8_t * grid = reinterpret_cast<const uint8_t *>(iq2s_grid + gi);
            const uint8_t signs = qs[QK_K/8 + 4 * ib32 + l];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float v = dl * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                out[i0 + j] = i0 + j < valid ? v : 0.0f;
            }
        }
    }
};

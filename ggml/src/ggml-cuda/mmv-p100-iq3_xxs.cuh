#pragma once
#include "mmv-p100-types.cuh"

// GGML_TYPE_IQ3_XXS decoder for the P100 path. Bit-exact with dequantize_row_iq3_xxs (ggml-quants.c).
//
// block_iq3_xxs is 98 bytes for 256 weights. Byte offsets inside one block:
//   [0, 2)     d                ggml_half super-block scale (little endian, two bytes)
//   [2, 66)    qs[64]           grid indices, 8 bytes per 32-weight sub-block
//   [66, 98)   scales_signs[32] 4 bytes per sub-block, read as one little-endian uint32 aux
//
// Sub-block s (0..7) covers weights [32s, 32s + 32) of the block with the scale
//   db = d * (0.5 + (aux >> 28)) * 0.5
// where aux is the uint32 assembled from sub-block s's four scale_signs bytes.
// For l in 0..3 the 8 weights 8l..8l+7 of the sub-block come from two grid entries:
//   grid1 = iq3xxs_grid[qs[2l]]    -> weights 8l + j,     j = 0..3
//   grid2 = iq3xxs_grid[qs[2l+1]]  -> weights 8l + 4 + j, j = 0..3
//   sign  = ksigns_iq2xs[(aux >> 7l) & 127]; bit j of sign covers weight 8l + j, bit j + 4 covers 8l + 4 + j
//   value = db * byte_j(grid) * (sign bit set ? -1 : +1)
// decode_weights divides block_size, so a 32-weight window never straddles a block.

template <> struct p100_w<GGML_TYPE_IQ3_XXS> {
    static constexpr int block_size = 256;
    static constexpr int block_bytes = sizeof(block_iq3_xxs);
    // Every field is assembled from single bytes, so the widest load is one byte.
    static constexpr int alignment = 1;
    static constexpr int decode_weights = 32;

    static_assert(QK_K == block_size, "iq3_xxs super-block must be QK_K weights");
    static_assert(block_bytes == 98, "block_iq3_xxs layout changed");

    static __device__ __forceinline__ void decode(
            const char * row, int64_t first, int valid, float (&out)[decode_weights]) {
#pragma unroll
        for (int i = 0; i < decode_weights; ++i) {
            out[i] = 0.0f;
        }
        if (valid <= 0) {
            return;
        }

        const uint8_t * x = reinterpret_cast<const uint8_t *>(row) + (first / block_size) * block_bytes;
        const int sub = int(first % block_size) / decode_weights;   // sub-block index 0..7

        const uint16_t d_bits = uint16_t(x[0]) | uint16_t(uint16_t(x[1]) << 8);
        const float d = __half2float(__ushort_as_half(d_bits));

        const uint8_t * sc = x + 66 + 4 * sub;
        const uint32_t aux = uint32_t(sc[0]) | (uint32_t(sc[1]) << 8) |
                             (uint32_t(sc[2]) << 16) | (uint32_t(sc[3]) << 24);
        const float db = d * (0.5f + float(aux >> 28)) * 0.5f;

        const uint8_t * qs = x + 2 + 8 * sub;

#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int k0 = 8 * l;   // first weight of this 8-weight group, relative to the window
            if (k0 >= valid) {
                break;
            }
            const uint8_t sg = ksigns_iq2xs[(aux >> (7 * l)) & 127];
            const uint32_t g1 = iq3xxs_grid[qs[2 * l + 0]];
            uint32_t g2 = 0;
            if (k0 + 4 < valid) {
                g2 = iq3xxs_grid[qs[2 * l + 1]];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (k0 + j < valid) {
                    const float b = float((g1 >> (8 * j)) & 0xff);
                    out[k0 + j] = db * b * ((sg & (1 << j)) ? -1.f : 1.f);
                }
                if (k0 + 4 + j < valid) {
                    const float b = float((g2 >> (8 * j)) & 0xff);
                    out[k0 + 4 + j] = db * b * ((sg & (1 << (j + 4))) ? -1.f : 1.f);
                }
            }
        }
    }
};

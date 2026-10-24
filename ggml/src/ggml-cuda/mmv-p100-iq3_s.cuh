#pragma once
#include "mmv-p100-types.cuh"

// GGML_TYPE_IQ3_S decoder for the P100 path. Bit-exact with dequantize_row_iq3_s (ggml-quants.c).
//
// block_iq3_s is 110 bytes for 256 weights. Byte offsets inside one block:
//   [0]       d            ggml_half super-block scale (little endian, two bytes)
//   [2, 66)   qs[64]       low 8 bits of grid indices, 8 bytes per 32-weight sub-block
//   [66, 74)  qh[8]        high (9th) bit of grid indices, one byte per sub-block
//   [74, 106) signs[32]    4 bytes per sub-block; bit j = sign of weight j, bit j+4 = weight j+4
//   [106,110) scales[4]    two 4-bit scales per byte; sub-block s uses nibble (s & 1) of byte s >> 1
//
// Sub-block s (0..7) covers weights [32s, 32s + 32) of the block with the scale
//   db = d * (1 + 2 * nibble)
// For l in 0..3 the 8 weights 8l..8l+7 come from two grid entries:
//   grid1 = iq3s_grid[qs[2l]   | ((qh << (8 - 2l)) & 256)]   -> weights 8l + j,     j = 0..3
//   grid2 = iq3s_grid[qs[2l+1] | ((qh << (7 - 2l)) & 256)]   -> weights 8l + 4 + j, j = 0..3
//   value = db * byte_j(grid) * (sign bit set ? -1 : +1)
// decode_weights divides block_size, so a 32-weight window never straddles a block.

template <> struct p100_w<GGML_TYPE_IQ3_S> {
    static constexpr int block_size = 256;
    static constexpr int block_bytes = sizeof(block_iq3_s);
    // Every field is assembled from single bytes, so the widest load is one byte.
    static constexpr int alignment = 1;
    static constexpr int decode_weights = 32;

    static_assert(QK_K == block_size, "iq3_s super-block must be QK_K weights");
    static_assert(block_bytes == 110, "block_iq3_s layout changed");

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
        const int nib = (x[106 + sub / 2] >> (4 * (sub & 1))) & 0xf;
        const float db = d * (1 + 2 * nib);

        const uint8_t qh = x[66 + sub];
        const uint8_t * qs = x + 2 + 8 * sub;
        const uint8_t * signs = x + 74 + 4 * sub;

#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int k0 = 8 * l;   // first weight of this 8-weight group, relative to the window
            if (k0 >= valid) {
                break;
            }
            const uint8_t sg = signs[l];
            const uint32_t g1 = iq3s_grid[qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 256)];
            uint32_t g2 = 0;
            if (k0 + 4 < valid) {
                g2 = iq3s_grid[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)];
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

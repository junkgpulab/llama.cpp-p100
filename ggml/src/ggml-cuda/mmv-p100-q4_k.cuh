#pragma once

#include "mmv-p100-types.cuh"

// q4_K decoder for the P100 mat-vec path. Layout of one 256-weight block (144 bytes):
//   bytes [0,2)    d     (half, super-block scale)
//   bytes [2,4)    dmin  (half, super-block min scale)
//   bytes [4,16)   scales[12] (6-bit scale/min pairs for the 8 sub-blocks)
//   bytes [16,144) qs[128]  (4-bit quants)
// A 32-weight sub-block s (0..7) shares one (scale, min) pair from get_scale_min_k4(s).
// Its quants are qs[32*(s/2) + l] for l in [0,32): low nibble when s is even, high nibble when s is odd.
// The math mirrors dequantize_row_q4_K in ggml-quants.c: d1 = d*sc, m1 = dmin*m, y = d1*q - m1.
// __fmul_rn/__fsub_rn keep each operation separately rounded (no FMA contraction), as on the CPU reference.
template <> struct p100_w<GGML_TYPE_Q4_K> {
    static constexpr int block_size = QK_K;
    static constexpr int block_bytes = static_cast<int>(sizeof(block_q4_K));
    static constexpr int alignment = alignof(uint16_t);
    static constexpr int decode_weights = 32;

    static __device__ __forceinline__ void decode(
            const char * row, int64_t first, int valid, float (&out)[decode_weights]) {
        const uint8_t * p = reinterpret_cast<const uint8_t *>(row + (first / block_size) * block_bytes);
        const int sub = int(first % block_size) / decode_weights;  // sub-block 0..7 inside the block
        const int c   = sub / 2;                                    // 64-weight chunk 0..3
        const bool hi = (sub & 1) != 0;                             // odd sub-blocks use high nibbles

        // Only the two super-block halves and the scale bytes this sub-block needs.
        const float d    = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t *>(p + 0)));
        const float dmin = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t *>(p + 2)));

        const uint8_t * sc_bytes = p + 4;
        uint8_t sc, m;
        if (sub < 4) {
            sc = sc_bytes[sub] & 63;
            m  = sc_bytes[sub + 4] & 63;
        } else {
            sc = (sc_bytes[sub + 4] & 0xF) | ((sc_bytes[sub - 4] >> 6) << 4);
            m  = (sc_bytes[sub + 4] >> 4)  | ((sc_bytes[sub]     >> 6) << 4);
        }
        const float d1 = d * float(sc);
        const float m1 = dmin * float(m);

        const uint8_t * qs = p + 16 + 32 * c;
#pragma unroll
        for (int l = 0; l < decode_weights; ++l) {
            if (l < valid) {
                const int nib = hi ? (qs[l] >> 4) : (qs[l] & 0xF);
                out[l] = __fsub_rn(__fmul_rn(d1, float(nib)), m1);
            } else {
                out[l] = 0.0f;
            }
        }
    }
};

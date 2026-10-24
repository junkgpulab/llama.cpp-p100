#pragma once

#include "mmv-p100-types.cuh"

// q5_K decoder. One call decodes one 32-weight sub-block: the 8 sub-blocks of a 256-weight block each
// have their own 6-bit scale and min (get_scale_min_k4), and sub-block s is bit s of qh, with the low
// nibble of qs for even s and the high nibble for odd s. Same float ops as dequantize_row_q5_K:
//   y = d1 * (q + 16*hbit) - m1,  d1 = d * sc,  m1 = dmin * m
// Only the bytes of the 'valid' weights are read: qs[32*(s/2) + i] and qh[i] for i < valid.
template <> struct p100_w<GGML_TYPE_Q5_K> {
    static constexpr int block_size = QK_K;
    static constexpr int block_bytes = int(sizeof(block_q5_K));
    static constexpr int alignment = alignof(uint16_t);  // the widest load used: the 2-byte d/dmin pair
    static constexpr int decode_weights = 32;

    static __device__ __forceinline__ void decode(
            const char * row, int64_t first, int valid, float (&out)[decode_weights]) {
        const char * b = row + (first / block_size) * block_bytes;
        // block_q5_K layout: d (0), dmin (2), scales[12] (4), qh[32] (16), qs[128] (48)
        const int s  = int(first % block_size) / decode_weights;  // sub-block 0..7
        const int j  = s >> 1;                                     // 64-weight group 0..3
        const bool hi = (s & 1) != 0;                              // odd sub-blocks use high nibbles

        const ggml_half * hp = reinterpret_cast<const ggml_half *>(b);
        const float dall = __half2float(hp[0]);
        const float dmin = __half2float(hp[1]);
        const uint8_t * sc = reinterpret_cast<const uint8_t *>(b + 4);
        const uint8_t * qh = reinterpret_cast<const uint8_t *>(b + 16);
        const uint8_t * qs = reinterpret_cast<const uint8_t *>(b + 48) + 32 * j;

        // get_scale_min_k4(s, scales, &d, &m)
        int d8, m8;
        if (s < 4) {
            d8 = sc[s] & 63;
            m8 = sc[s + 4] & 63;
        } else {
            d8 = (sc[s + 4] & 0xF) | ((sc[s - 4] >> 6) << 4);
            m8 = (sc[s + 4] >> 4) | ((sc[s] >> 6) << 4);
        }
        const float d1 = __fmul_rn(dall, float(d8));
        const float m1 = __fmul_rn(dmin, float(m8));
        const int bit = 1 << s;

#pragma unroll
        for (int i = 0; i < decode_weights; ++i) {
            if (i < valid) {
                const int byte = qs[i];
                const int nib  = hi ? (byte >> 4) : (byte & 0xF);
                const int hb   = (qh[i] & bit) ? 16 : 0;
                out[i] = __fsub_rn(__fmul_rn(d1, float(nib + hb)), m1);
            } else {
                out[i] = 0.0f;
            }
        }
    }
};

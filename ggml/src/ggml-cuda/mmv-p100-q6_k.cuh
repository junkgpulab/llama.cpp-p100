#pragma once

#include "mmv-p100-types.cuh"

// q6_K super-block (block_q6_K, 210 bytes): ql[128] (low/high nibbles), qh[64] (top 2 bits),
// scales[16] (int8), d (fp16) at byte 208. For weight w in [0, 256) of a block:
//   ip = w / 128, j = (w % 128) / 32, l = w % 32, is = l / 16
//   nibble = (j >= 2) ? ql[64*ip + 32*(j&1) + l] >> 4 : ql[64*ip + 32*(j&1) + l] & 0xF
//   q      = (nibble | (((qh[32*ip + l] >> 2*j) & 3) << 4)) - 32
//   value  = (d * scales[8*ip + 2*j + is]) * q      (same float order as dequantize_row_q6_K)
// A decode_weights=16 chunk starts at a multiple of 16, so all 16 weights share ip, j and is.
template <> struct p100_w<GGML_TYPE_Q6_K> {
    static constexpr int block_size = QK_K;
    static constexpr int block_bytes = sizeof(block_q6_K);
    static constexpr int alignment = alignof(uint16_t);
    static constexpr int decode_weights = 16;

    static constexpr int OFF_QH = QK_K / 2;
    static constexpr int OFF_SC = OFF_QH + QK_K / 4;
    static constexpr int OFF_D  = OFF_SC + QK_K / 16;
    static_assert(OFF_D + 2 == block_bytes, "q6_K byte layout");

    static __device__ __forceinline__ void decode(
            const char * row, int64_t first, int valid, float (&out)[decode_weights]) {
        const char * b = row + (first / block_size) * block_bytes;
        const int offset = int(first % block_size);  // multiple of 16
        const int ip = offset >> 7;                  // 128-weight half of the block
        const int j  = (offset >> 5) & 3;            // 32-weight quarter inside the half
        const int l0 = offset & 31;                  // first lane: 0 or 16
        const int is = l0 >> 4;                      // scale sub-group inside the quarter

        const float d  = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t *>(b + OFF_D)));
        const float ds = d * float(static_cast<int8_t>(b[OFF_SC + 8*ip + 2*j + is]));

        const uint8_t * ql = reinterpret_cast<const uint8_t *>(b) + 64*ip + 32*(j & 1) + l0;
        const uint8_t * qh = reinterpret_cast<const uint8_t *>(b) + OFF_QH + 32*ip + l0;
        const int nib_shift = (j >> 1) ? 4 : 0;
        const int hi_shift  = 2*j;

#pragma unroll
        for (int i = 0; i < decode_weights; ++i) {
            if (i < valid) {
                const int nib = (ql[i] >> nib_shift) & 0xF;
                const int hi  = (qh[i] >> hi_shift) & 3;
                const int q   = (nib | (hi << 4)) - 32;
                out[i] = ds * float(q);
            } else {
                out[i] = 0.0f;
            }
        }
    }
};

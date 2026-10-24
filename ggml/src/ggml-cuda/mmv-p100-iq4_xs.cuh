#pragma once
#include "mmv-p100-types.cuh"

// iq4_xs super-block (QK_K = 256 weights, 136 bytes):
//   [0,2)   ggml_half d
//   [2,4)   uint16 scales_h   (2 high bits per sub-block)
//   [4,8)   uint8  scales_l[4] (4 low bits per sub-block, two per byte)
//   [8,136) uint8  qs[128]
// Sub-block ib (32 weights) has ls = low4(scales_l[ib/2] >> 4*(ib%2)) | (2 bits of scales_h >> 2*ib) << 4,
// dl = d * (ls - 32). Within it, weight r in [0,16) is qs[16*ib + r] & 0xf and r in [16,32) is qs[16*ib + r - 16] >> 4.
// Same tables and float op order as dequantize_row_iq4_xs (ggml-quants.c) and dequantize_iq4_xs (dequantize.cuh).

template <> struct p100_w<GGML_TYPE_IQ4_XS> {
    static_assert(QK_K == 256, "iq4_xs P100 decoder assumes QK_K == 256");

    static constexpr int block_size = 256;
    static constexpr int block_bytes = sizeof(block_iq4_xs);
    static constexpr int alignment = alignof(uint16_t);
    static constexpr int decode_weights = 32;

    static __device__ __forceinline__ void decode(
            const char * row, int64_t first, int valid, float (&out)[decode_weights]) {
        const char * blk = row + (first / block_size) * block_bytes;
        const uint8_t * p = reinterpret_cast<const uint8_t *>(blk);

        // first is a multiple of 32, so the 32 weights are exactly one sub-block of this block.
        const int ib = int(first % block_size) / 32;

        const float d = (float)*reinterpret_cast<const ggml_half *>(blk + 0);
        const uint16_t scales_h = *reinterpret_cast<const uint16_t *>(blk + 2);
        const int ls = ((p[4 + ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
        const float dl = d * (float)(ls - 32);

        const uint8_t * qs = p + 8 + 16 * ib;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            out[j]      = j      < valid ? dl * (float)kvalues_iq4nl[qs[j] & 0xf] : 0.0f;
            out[j + 16] = j + 16 < valid ? dl * (float)kvalues_iq4nl[qs[j] >> 4]  : 0.0f;
        }
    }
};

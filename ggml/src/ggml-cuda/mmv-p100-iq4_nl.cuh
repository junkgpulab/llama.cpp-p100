#pragma once

#include "mmv-p100-types.cuh"

// iq4_nl: 32 weights per block. block_iq4_nl is { ggml_half d; uint8_t qs[16]; } = 18 bytes.
// Weight i (0..15) of a block is d * kvalues_iq4nl[qs[i] & 0xf]; weight i + 16 is d * kvalues_iq4nl[qs[i] >> 4].
// Same table and same single float multiply as dequantize_row_iq4_nl, so the output is bit-exact.
// The only multi-byte field is the 2-byte scale, so the widest load is 2 bytes; qs is read byte by byte.
// Row starts are multiples of 18 bytes (even), so 2-byte alignment holds for every block.
template <> struct p100_w<GGML_TYPE_IQ4_NL> {
    static constexpr int block_size = QK4_NL;
    static constexpr int block_bytes = static_cast<int>(sizeof(block_iq4_nl));
    static constexpr int alignment = alignof(uint16_t);
    static constexpr int decode_weights = QK4_NL;

    static __device__ __forceinline__ void decode(
            const char * row, int64_t first, int valid, float (&out)[decode_weights]) {
        // first is a multiple of decode_weights == block_size, so the window is exactly one block at offset 0.
        const int64_t block = first / block_size;
        const block_iq4_nl * b = reinterpret_cast<const block_iq4_nl *>(row + block * block_bytes);
        const float d = (float)b->d;

#pragma unroll
        for (int i = 0; i < 16; ++i) {
            // Weight i + 16 needs the same byte as weight i, and i + 16 < valid implies i < valid,
            // so qs[i] is read only when weight i is inside the valid window.
            const bool lo = i < valid;
            const bool hi = i + 16 < valid;
            const uint8_t q = lo ? b->qs[i] : (uint8_t) 0;
            out[i]      = lo ? d * (float) kvalues_iq4nl[q & 0xf] : 0.0f;
            out[i + 16] = hi ? d * (float) kvalues_iq4nl[q >> 4]  : 0.0f;
        }
    }
};

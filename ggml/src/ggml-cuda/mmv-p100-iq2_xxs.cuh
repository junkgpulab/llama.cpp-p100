#pragma once
#include "mmv-p100-types.cuh"

// iq2_xxs: a block is 256 weights = 8 sub-blocks of 32, and each sub-block shares one scale.
// Block layout (66 bytes): ggml_half d; uint16_t qs[32]. For sub-block ib32 the four qs words
// s[0..3] = qs[4*ib32 .. 4*ib32+3] hold:
//   grid indices: s[0] gives bytes 0,1 and s[1] gives bytes 2,3 (weights 8*l .. 8*l+7 use byte l)
//   aux1 = s[2] | s[3] << 16: scale in bits 28..31, sign index for group l in bits 7*l .. 7*l+6
// Every load is 2 bytes, because block_bytes is 66 and the alignment is 2.
static_assert(QK_K == 256, "iq2_xxs decoder assumes 256-weight blocks");

template <> struct p100_w<GGML_TYPE_IQ2_XXS> {
    static constexpr int block_size = QK_K;
    static constexpr int block_bytes = sizeof(block_iq2_xxs);
    static constexpr int alignment = alignof(uint16_t);
    static constexpr int decode_weights = 32;

    static __device__ __forceinline__ void decode(
            const char * row, int64_t first, int valid, float (&out)[decode_weights]) {
        const int64_t ib    = first / block_size;
        const int     ib32  = int(first % block_size) / 32;
        const uint16_t * q  = reinterpret_cast<const uint16_t *>(row + ib * block_bytes);
        const uint16_t * s  = q + 1 + 4 * ib32; // qs[4*ib32 ...]

        const float d = (float) *reinterpret_cast<const ggml_half *>(q);
        const uint32_t aux1 = uint32_t(s[2]) | (uint32_t(s[3]) << 16);
        const float db = d * (0.5f + float(aux1 >> 28)) * 0.25f;

        const uint16_t g0 = s[0];                  // grid indices for weights 0..15
        const uint16_t g1 = valid > 16 ? s[1] : 0; // grid indices for weights 16..31

#pragma unroll
        for (int i = 0; i < decode_weights; ++i) {
            if (i < valid) {
                const int l = i >> 3; // 8-weight group inside the sub-block, 0..3
                const int j = i & 7;  // byte inside the 8-byte grid entry
                const uint32_t gi = l == 0 ? (g0 & 0xffu) : l == 1 ? (g0 >> 8) : l == 2 ? (g1 & 0xffu) : (g1 >> 8);
                const uint8_t gj = uint8_t(iq2xxs_grid[gi] >> (8 * j));
                const uint8_t signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                const float mag = db * float(gj);
                out[i] = (signs & kmask_iq2xs[j]) ? -mag : mag;
            } else {
                out[i] = 0.0f;
            }
        }
    }
};

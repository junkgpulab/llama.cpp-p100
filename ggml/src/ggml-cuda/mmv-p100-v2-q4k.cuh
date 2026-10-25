#pragma once

#include "common.cuh"

bool ggml_cuda_p100_v2_q4k(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

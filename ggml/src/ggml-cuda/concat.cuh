#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_op_concat_cpy(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * cpy);

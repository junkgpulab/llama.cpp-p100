#pragma once

#include "common.cuh"

// Consecutive CPY nodes with the same shape and strides, whose sources are views of one tensor and
// whose destinations are views of another, run as one launch. The delta-net writes its conv-state
// rollback snapshots this way: K = n_rs_seq + 1 small copies per layer, ~4 us each as separate
// launches. Returns the number of nodes to skip after i (0 = not fused).
int ggml_cuda_try_cpy_batch(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

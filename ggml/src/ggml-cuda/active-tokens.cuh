#pragma once

#include <cstdint>

// Fixed-width speculative verify (the fork's server pads a short draft to a fixed width so the
// verify graph never changes shape and is always reused). While a padded graph is enqueued the
// caller declares how many of its n_full tokens are real; ops whose cost scales with the token
// count (the matvecs, flash attention) then process only the first n_active tokens. Those are
// exactly the results a graph of width n_active computes: every per-token result depends only on
// itself (matvec) or on earlier tokens (causal attention). The padded tokens' results are garbage
// and are rolled back like rejected draft tokens. Set from the host thread that enqueues the graph,
// and cleared after it.
extern int ggml_cuda_active_tokens_n;    // 0: off
extern int ggml_cuda_active_tokens_full; // the padded width it applies to

static inline int64_t ggml_cuda_active_cols(const int64_t n) {
    return ggml_cuda_active_tokens_n > 0 && n == ggml_cuda_active_tokens_full && ggml_cuda_active_tokens_n < n
        ? ggml_cuda_active_tokens_n : n;
}

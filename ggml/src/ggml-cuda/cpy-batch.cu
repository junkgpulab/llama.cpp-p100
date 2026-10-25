#include "cpy-batch.cuh"

#define CPY_BATCH_MAX 8

struct cpy_batch_args {
    const char * src[CPY_BATCH_MAX];
    char *       dst[CPY_BATCH_MAX];
    int64_t ne0, ne1, ne2, ne3;
    int64_t snb0, snb1, snb2, snb3;
};

static __global__ void k_cpy_batch_f32(const cpy_batch_args a, const int64_t n) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const int k = blockIdx.y;
    const int64_t i0 = idx % a.ne0;
    int64_t r = idx / a.ne0;
    const int64_t i1 = r % a.ne1; r /= a.ne1;
    const int64_t i2 = r % a.ne2;
    const int64_t i3 = r / a.ne2;
    const float v = *(const float *) (a.src[k] + i0*a.snb0 + i1*a.snb1 + i2*a.snb2 + i3*a.snb3);
    ((float *) a.dst[k])[idx] = v; // contiguous destination, in the source's element order (ggml_cpy)
}

int ggml_cuda_try_cpy_batch(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool enabled = [] {
        const char * s = getenv("GGML_CUDA_CPY_BATCH");
        return !s || atoi(s) != 0;
    }();
    const ggml_tensor * c0 = cgraph->nodes[i];
    if (!enabled || c0->op != GGML_OP_CPY) {
        return 0;
    }
    const ggml_tensor * s0 = c0->src[0];
    const ggml_tensor * d0 = c0->src[1];
    if (s0->type != GGML_TYPE_F32 || d0->type != GGML_TYPE_F32 || s0->view_src == nullptr || d0->view_src == nullptr ||
            (c0->flags & GGML_TENSOR_FLAG_OUTPUT) || s0->view_src == d0->view_src) {
        return 0;
    }
    cpy_batch_args a = {};
    int n = 0;
    int last = i;
    for (int j = i; j < cgraph->n_nodes && n < CPY_BATCH_MAX; ++j) {
        const ggml_tensor * c = cgraph->nodes[j];
        if (j > i && (c->op == GGML_OP_VIEW || c->op == GGML_OP_RESHAPE || c->op == GGML_OP_NONE)) {
            continue; // views between the copies
        }
        if (c->op != GGML_OP_CPY || (c->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            break;
        }
        const ggml_tensor * s = c->src[0];
        const ggml_tensor * d = c->src[1];
        if (s->type != GGML_TYPE_F32 || d->type != GGML_TYPE_F32 || s->view_src != s0->view_src || d->view_src != d0->view_src ||
                !ggml_are_same_shape(s, s0) || !ggml_are_same_shape(d, d0) || !ggml_is_contiguous(d) ||
                ggml_nelements(d) != ggml_nelements(s) ||
                memcmp(s->nb, s0->nb, sizeof(s->nb)) != 0 || memcmp(d->nb, d0->nb, sizeof(d->nb)) != 0) {
            break;
        }
        a.src[n] = (const char *) s->data;
        a.dst[n] = (char *) d->data;
        ++n;
        last = j;
    }
    if (n < 2) {
        return 0;
    }
    a.ne0 = s0->ne[0]; a.ne1 = s0->ne[1]; a.ne2 = s0->ne[2]; a.ne3 = s0->ne[3];
    a.snb0 = s0->nb[0]; a.snb1 = s0->nb[1]; a.snb2 = s0->nb[2]; a.snb3 = s0->nb[3];
    const int64_t ne = ggml_nelements(s0);
    const dim3 grid((unsigned) ((ne + 255)/256), n);
    k_cpy_batch_f32<<<grid, 256, 0, ctx.stream()>>>(a, ne);
    CUDA_CHECK(cudaGetLastError());
    return last - i;
}

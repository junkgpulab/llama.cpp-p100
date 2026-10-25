# P100 fork: patch series and build notes

Upstream `ggml-org/llama.cpp` master `6050b467c` (b11887, 2026-10-25) = tag `p7-base`, plus:

- Kmic-68/llama.cpp `p100-optimizations` `d327bf1f0..7461904d` (MIT), ported as 5 squashed commits P7-2..P7-6,
  each with an `Origin:` trailer. Kernel design and tuning are theirs (see `NOTICE-P100.md`).
- P7-7: padded speculative-verify fix (zeroed pad columns, scale-by-0 writes 0), so a NaN in an unused
  verify column cannot poison the recurrent state.
- P7-8 (a): compact KQ mask only on the qwen35 arch, so other hybrid graphs (Qwen3.8-Flash-Next) load.
- P7-8 (b): GDN in-place state gather copies its row index into a private device buffer (fixes an
  out-of-bounds read / Xid 13 on Flash-Next).
- Flash-Next / MoE patches (BF16 and K-quant P100 mat-vec kernels, pinned host experts, per-card MoE cache
  budget), all opt-in by env or compile flag. Merge notes: `P7-CONFLICTS.md`.

## Patch series (`git log --oneline p7-base..p100`, oldest last; P7-1 = docs/preset/notice; P7-8 (a)/(b) = the two later fixes on top)

```
d25e1b9db P100-X1: shinbunbun patch 01 (vmad dp4a on sm_60, MIT) behind compile flag GGML_CUDA_P100_VMAD (default off = unchanged)
e89f7c9b6 P100-K4 sonnet loop Q6_K: neighbour shuffle replaces realign loads (71/46/162 us vs 111/61/212)
51947f13b P100-K4 sonnet loop Q6_K: 4B-word realigned loads, FP16 magic decode, R=4 (83/55/190 us vs 110/61/212)
9bb035a5a P100-K3 sonnet loop Q4_K: Q5_K kernel minus qh, R=8 (e.g. 4096x14336 133us vs 157)
fd66c0bca P100-K2 sonnet loop: R=8 option, default R=8 (1109 us)
95715897b P100-K2 sonnet loop: bounds R4 4 / R2 8 (1123 us)
aaaad9a0a P100-K2 sonnet loop: drop register prefetch, load at top (R4 1124 us)
15c502300 P100-K2 sonnet loop: byte_perm scale extraction (1216 us)
ab0801a66 P100-K2 sonnet loop: launch bounds R2 6 / R4 3 (1233 us)
0dd8254e9 P100-K2 sonnet loop: 2 lanes per sub-block (16 weights/lane), block-pair warp, 16B loads all lanes (1294 us)
a2e32db58 P100-K2 sonnet loop: mask-in-place nibbles, funnelshift qh, odd-group acts /16 (1480 us)
cfd4ffb17 P100-K2 sonnet loop: R4 launch_bounds(128,6) no spill (1524 us)
767387044 P100-K2 sonnet loop: pair decode (b0,b2)/(b1,b3), permuted smem activations, 2 lop3 per half2 (1669 us)
7fe224ccd P100-K2 sonnet loop: no shuffles, per-lane 8B qs/qh + broadcast header loads (2004 us)
6b7d41c1d P100-K2 codex r2: Q5_K v2 persistent blocks + prefetch + 16-byte LDS
57a085377 P100-K2 codex: Q5_K FP16/half2 mat-vec v2 (GGML_CUDA_P100_V2=1), output-head gate
6bb1d672b P100-2C codex: per-device MoE cache budget (LLAMA_MOE_CACHE_MIB_DEV)
583025604 P100-2A codex: pin MoE-cache host experts in place (GGML_CUDA_REGISTER_HOST=1) + per-call wait/plan/upload timing (LLAMA_MOE_CACHE_STATS=N)
d2021f477 P100-1 codex round 3: BF16-only default (GGML_CUDA_P100_FORMATS), BF16 k>4096 declines, variant 2 only (measured)
d84c4b244 P100-1 codex round 2: barrier-free long-row kernel, 3 geometry variants (GGML_CUDA_P100_VARIANT), BF16 underfill decline, static helpers
e641494eb P100-1: 10 quantized decoders (Haiku, each exact vs ggml CPU dequant + memcheck, rechecked by Claude) + registry + model-shape tests
4f9e38d5a P100-1 codex round 1: mmv-p100 framework + BF16 decoder + tests (unbuilt)
c3a623672 P7-7: padded-verify fix: finite pad columns + scale-by-0 clears
092fad32f P7-6: server: padded verify, n_active, checkpoints, slot carry-over (Kmic port)
c63e2997c P7-5: common: MTP fixed-width verify, sampled drafts, block verification, checkpoint buffers (Kmic port)
0ae0cfbe2 P7-4: llama core, compact mask, fixed-width verify, MTP draft vocab (Kmic port)
3d0185826 P7-3: CUDA kernels for sm_60 (Kmic port)
25467f2fa P7-2: ggml core + backend plumbing (Kmic port)
```

Each ported commit carries an `Origin:` trailer (`Kmic-68/llama.cpp p100-optimizations d327bf1f0..7461904d`,
our padded-verify fix; the Flash-Next / MoE commits keep their original messages).

## Build (sm_60 ONLY: a multi-arch build silently gets upstream's kernels)

```
cmake --preset p100
cmake --build --preset p100        # llama-server + llama-bench in build-p100/
```

The same without the preset:

```
cmake -G Ninja -B build-p100 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build-p100 --target llama-server llama-bench
```

For the kernel tests configure with `-DLLAMA_BUILD_TESTS=ON` and build `test-backend-ops`.
The Pascal paths select themselves by `__CUDA_ARCH_LIST__ == 600` (`GGML_CUDA_MMVQ_PASCAL` in mmvq.cu).
Flash-Next's own switches (`-DGGML_CUDA_P100_VMAD`, `GGML_CUDA_P100*` env) are off by default.

## Run-time switches

Kmic's run-time switches, useful for bisecting a speed difference on one binary: `GGML_CUDA_GRAPHS_PRE_VOLTA`,
`GGML_CUDA_FUSE_ADD_NORM=0`, `GGML_CUDA_FUSE_NORM_GATE=0`, `GGML_CUDA_GDN_GATHER=0`,
`GGML_CUDA_AR_P2P=0`, `GGML_CUDA_XCHG_NOP2P=0`, `GGML_CUDA_FOLD_GLU=0`, `LLAMA_SPEC_PAD=0`,
`LLAMA_MTP_KV_ONLY=0`, `LLAMA_MTP_DRAFT_VOCAB=0`, `LLAMA_GRAPH_REUSE_DISABLE=1`.

## lab/

`lab/kmic-tools/`: reference copies of Kmic's `depth-bench.py` and `gate.sh` (see `ORIGIN.md` there).

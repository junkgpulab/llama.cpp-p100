This tree is a fork of ggml-org/llama.cpp (MIT, Copyright (c) 2023-2026 The ggml authors).
It carries, as patches, code from:

- Kmic-68/llama.cpp, branch p100-optimizations (MIT). Copyright (c) 2026 Kmic-68 and
  contributors (git authors Kaden, Tyler/Keoian, kane/KaneQuah). Pascal (sm_60) matvec,
  flash-attention, fold-GEMM and gated-delta-net kernels, the fixed-width speculative verify,
  MTP draft handling and the tensor-parallel exchange. Ported from commits d327bf1f0..7461904d
  Kernel design and tuning are theirs.
- shinbunbun/llama-cpp-p100-patches (MIT), patch 01 (vmad dp4a on sm_60), behind
  GGML_CUDA_P100_VMAD.
- JonathanColetti: the DFlash2 draft head for Qwen3.8-27B used with --spec-type draft-dflash
  (weights, not code; license per the model card).
- ISTA-DASLab: the GSQ-RCO quantisation of Qwen3.8-Flash-Next served by this tree (weights).
- ggml-org: llama.cpp and ggml themselves, including the upstream DFlash, MTP and Qwen3.8
  hybrid implementations this fork builds on.

The MIT permission notice in LICENSE applies to all of the above code.

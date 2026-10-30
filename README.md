# llama.cpp-p100: llama.cpp tuned for NVIDIA Tesla P100 (sm_60)

> **This is a fork.** Upstream llama.cpp's own README follows below the line.

**What it is:** upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) (master `6050b467c`, b11887, 2026-10-25)
plus the Pascal / sm_60 kernels from **[Kmic-68/llama.cpp](https://github.com/Kmic-68/llama.cpp)** (`p100-optimizations`)
plus our fixes:

- **Padded speculative-verify fix:** unused verify columns could hold garbage, `0 x NaN` then poisoned the recurrent
  state and the model answered "4444..." from the second request on. Pad columns are now zeroed and scale-by-0 writes 0.
- **GDN in-place gather fix:** the gated-delta-net kernel could read a row index whose buffer had already been reused
  (Xid 13 on Qwen3.8-Flash-Next). The index now lives in a private device buffer.
- **Compact KQ mask limited to the qwen35 arch**, so other hybrid graphs (Qwen3.8-Flash-Next) load.
- Opt-in MoE / Flash-Next work: BF16 and K-quant P100 mat-vec kernels, pinned host experts
  (`GGML_CUDA_REGISTER_HOST=1`), per-card MoE cache budget (`LLAMA_MOE_CACHE_MIB_DEV`). All off by default.
- A one-line build preset for sm_60.

The speed comes from Kmic-68 and co's kernels. Patch series and notes: [docs/P100.md](docs/P100.md).

## Speed: it ties Kmic's fork, it does not beat it

Measured by us on one box, 2026-10-27: 2x Tesla P100 16 GB (PCIe), power cap 175 W, tensor split (`-sm tensor`),
P2P on, Qwen3.8-27B Q6_K, MTP speculative decoding with 4 drafts, p-min 0.2, plus one tuned draft setting (explained in our next video, link here when it is out).
"Kmic" = Kmic-68's `p100-optimizations` tip plus our padded-verify fix, same box, same flags, same session.

| test | this fork | Kmic's fork | reading |
|---|---|---|---|
| 22-prompt head-to-head, overall t/s | 45.2 | 45.4 / 45.9 (two arms) | tie: median per-prompt ratio 0.999, inside Kmic's own run-to-run drift (0.990) |
| 11-question bench, prose / code / 30K-context t/s | 58.9 / 70.5 / 56.5 | 59.8 / 71.4 / 57.3 | ~1.5% behind |
| DFlash draft entry at 30K context, t/s | 44.5 | 41.4 | +7%, from draft acceptance (upstream numerics), not a kernel win |

Single box, few runs; treat differences under ~3% as noise. No numbers at 62K+ context are claimed for this build.

## Build

sm_60 **only**: a multi-arch build silently falls back to upstream's kernels.

**You need** (Linux x86_64; tested on Ubuntu 24.04):

| what | version | why |
|---|---|---|
| NVIDIA driver | 580 series | the last driver branch that supports Pascal; 590+ drops the P100 |
| CUDA toolkit | 12.x (tested 12.9) | **not 13.x**: CUDA 13 cannot compile for sm_60 |
| CMake | 3.23 or newer (tested 3.28) | the build uses `CMakePresets.json` |
| Ninja | any recent (tested 1.12) | the preset's generator |
| gcc / g++ | up to 14 (tested 13.3) | host compiler for nvcc 12.x |
| git | any | |

On Ubuntu: `sudo apt install build-essential cmake ninja-build git`, then CUDA 12.x from NVIDIA
(the `cuda-toolkit-12-9` package, or the runfile with the driver box unticked if you already have driver 580).
Make sure `nvcc` is on your `PATH` and is 12.x:

```bash
export PATH=/usr/local/cuda-12.9/bin:$PATH   # adjust to where CUDA 12.x is installed
nvcc --version                               # must say release 12.x
nvidia-smi                                   # must show your P100(s) and driver 580.x
```

Then:

```bash
git clone https://github.com/junkgpulab/llama.cpp-p100
cd llama.cpp-p100
cmake --preset p100                # CUDA on, CMAKE_CUDA_ARCHITECTURES=60, Release, Ninja
cmake --build --preset p100        # llama-server + llama-bench in build-p100/bin/
./build-p100/bin/llama-server --version
```

A full build takes about 15 minutes on a 6-core desktop CPU. nvcc prints
"offline compilation for architectures prior to sm_75 will be removed" for every CUDA file: that is a notice, not an error.

**If it fails:**
- `CMake was unable to find a build program corresponding to "Ninja"`: install Ninja (`ninja-build`).
- `No CMAKE_CUDA_COMPILER could be found`: `nvcc` is not on your `PATH`; add the CUDA 12.x `bin` directory, or pass
  `-DCMAKE_CUDA_COMPILER=/path/to/cuda-12.x/bin/nvcc` to the first `cmake` command.
- `Unsupported gpu architecture 'compute_60'`: that `nvcc` is CUDA 13; use CUDA 12.x.
- Link errors at the very end (`libcudart.so.12 ... not found`, `undefined reference to cudaMemcpy` / `cuMemMap`):
  the linker cannot find the CUDA libraries. Register them once:
  `echo /usr/local/cuda-12.9/lib64 | sudo tee /etc/ld.so.conf.d/cuda-12.conf && sudo ldconfig`
  (adjust the path), then run `cmake --build --preset p100` again. `libcuda.so.1` missing means the NVIDIA driver is not installed.
- If you have several CUDA versions installed, put the 12.x one first on your `PATH`.

## Recipe

Settings, launch flags and the full story: **https://youtu.be/JvLUoJYUjDM**.
Channel: **[Junk GPU Lab](https://www.youtube.com/@junkgpulab)**.

## Credits

- **llama.cpp / ggml** ([ggml-org](https://github.com/ggml-org/llama.cpp)), MIT, Copyright (c) 2023-2026 The ggml authors.
  The `LICENSE` file is upstream's, unchanged.
- **[Kmic-68](https://github.com/Kmic-68/llama.cpp) and co-authors (Kaden, Tyler / [Keoian](https://github.com/Keoian), kane)**, MIT:
  the sm_60 mat-vec kernels, fold-GEMM prefill, Pascal flash-attention, gated-delta-net / recurrent / norm kernels,
  compact KQ mask, fixed-width speculative verify, MTP draft vocab, server padded verify and checkpoints, tensor-parallel
  P2P exchange. All kernel design and tuning are theirs.
- **[shinbunbun](https://github.com/shinbunbun/llama-cpp-p100-patches)**, MIT: patch 01 (vmad dp4a), behind `-DGGML_CUDA_P100_VMAD` (off by default).
- Thanks to **fallentree** and **qr0n** for the P100 community work (no code of theirs is in this repo).
- No model weights are shipped here; see each model card for its licence.

Full notice: [NOTICE.md](NOTICE.md).

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain

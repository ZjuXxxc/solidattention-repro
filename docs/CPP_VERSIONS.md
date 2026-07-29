# C++ / CUDA / SYCL / liburing version line

This line is independent from Python V0–V13. A `C` version is not promoted
until its measured backend is real; unavailable backends remain explicitly
marked build-pending.

| Version | Shared C++ | SSD I/O | Accelerator | Status |
|---|---|---|---|---|
| C0 | KV block format, metrics and Perfetto trace | liburing registered file + fixed pinned buffers | CUDA NVRTC transform correctness kernel | implemented |
| C0-SYCL | same host path and metrics schema | same liburing reader | oneAPI SYCL USM/event implementation | source ready; compiler validation pending |
| C1 | representative selection and sparse attention | batched fixed reads | CUDA and SYCL equivalent kernels | planned |
| C2 | cross-layer SSD/H2D/FFN DAG and correction | queued reads, buffer ownership | streams/queues with no global sync | planned |

## C0 measured result

RTX 4080 Laptop, 512 operations, 128 KiB per operation:

| Stage | Mean |
|---|---:|
| registered-buffer `io_uring` read | 0.105731 ms |
| pinned DRAM → VRAM H2D | 0.015386 ms |
| NVRTC CUDA transform kernel | 0.004846 ms |
| VRAM → host correctness D2H | 0.026889 ms |

All 512 device outputs exactly match the CPU byte reference. The checksum field
is an XOR fold across per-operation checksums and may be zero; correctness is
asserted before folding, not inferred from that final value.

## Why C0 is deliberately narrow

C0 proves that four boundaries work together in one native process:

1. `O_DIRECT` SSD storage with 128 KiB blocks;
2. liburing registered-file, registered-buffer reads;
3. CUDA page-locked host memory and asynchronous H2D;
4. a CUDA kernel compiled at runtime with NVRTC, followed by audited D2H.

Its device kernel is a deterministic byte transform, **not sparse attention**.
Calling C0 a SolidAttention quality/performance reproduction would therefore
be incorrect. C1 replaces it with the shared attention contract and validates
CUDA/SYCL output against a CPU reference.

## File ownership

| File | Responsibility |
|---|---|
| `cpp/include/solidattention/backend.hpp` | backend-neutral accelerator contract and timings |
| `cpp/include/solidattention/uring_reader.hpp` | registered-buffer SSD reader contract |
| `cpp/src/uring_reader.cpp` | actual `O_DIRECT` + liburing implementation |
| `cpp/src/cuda_backend.cpp` | pinned allocation, CUDA events, NVRTC compilation and launch |
| `cpp/src/sycl_backend.cpp` | oneAPI USM, queue dependencies and event profiling |
| `cpp/src/trace.cpp` | backend-neutral Chrome/Perfetto trace writer |
| `cpp/src/main.cpp` | C0 benchmark composition; no backend implementation logic |
| `scripts/build_cpp.sh` | reproducible user-space CUDA build |
| `scripts/build_cpp_sycl.sh` | oneAPI build with an explicit compiler check |

Run C0 CUDA:

```bash
./scripts/run_cpp_c0.sh --operations 512
```

The output includes `c0-metrics.json` and a Perfetto-compatible
`c0-trace.json`. The trace has separate SSD read, PCIe H2D, GPU compute and
PCIe D2H lanes.

The current machine has no system C++ compiler, CMake, nvcc or DPC++ compiler.
The CUDA target uses the repository's Zig/Clang compiler plus the CUDA
runtime/NVRTC installed with PyTorch. liburing 2.5 development files are
locally extracted and ignored by Git. The SYCL source is not reported as
passing until a pinned oneAPI compiler build succeeds.

## Build/debug log

1. The first CUDA compile failed because NVIDIA's Python runtime wheel exposes
   `cuda_runtime_api.h` but not the nested `crt/host_defines.h` path expected
   by that header.
2. Triton's pinned CUDA include tree contains the complete matching headers.
   Adding it before the runtime include directory fixed the build without
   copying headers or installing a system toolkit.
3. The fixed-read submission originally used a null buffer address. Registered
   buffers still require the registered virtual address in
   `io_uring_prep_read_fixed`; C0 now retains and supplies that address.
4. CPU reference comparison was added before publishing metrics, so successful
   transport alone cannot hide a CUDA kernel or buffer-indexing error.

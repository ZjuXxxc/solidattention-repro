# C++ / CUDA / SYCL / liburing version line

This line is independent from Python V0–V13. A `C` version is not promoted
until its measured backend is real; unavailable backends remain explicitly
marked build-pending.

| Version | Shared C++ | SSD I/O | Accelerator | Status |
|---|---|---|---|---|
| C0 | KV block format, metrics and Perfetto trace | liburing registered file + fixed pinned buffers | CUDA NVRTC transform correctness kernel | implemented |
| C0-SYCL-CPU | same host path and metrics schema | same liburing reader | oneAPI 2026.1.1 SYCL USM/event on i9-14900HX | implemented |
| C0-SYCL-NVIDIA | same host path and metrics schema | same liburing reader | Codeplay oneAPI NVIDIA plugin | plugin pending |
| C1 | FP16 GQA sparse attention + CPU oracle | 512 KiB fixed read | CUDA NVRTC and oneAPI SYCL CPU | implemented |
| C1.1 | block/work-group reductions | same C1 input | CUDA shared memory + SYCL local memory | implemented |
| C1.2 | persistent resources + wall audit | same C1 input | reused CUDA events and SYCL USM | implemented |
| C1.3 | device-resident output + sampled D2H | same C1 input | CUDA/SYCL persistent output | implemented |
| C1.4 | device-side output consumer | same C1 input | ordered CUDA/SYCL dependency | implemented |
| C1.5 | tiled output projection | same C1 input | device-resident projected output | planned |
| C1.6 | online softmax | same C1 input | warp/sub-group tuned kernels | planned |
| C2.0 | fixed-offset representative transport smoke | batched fixed reads | selected attention | failed quality |
| C2.1 | InfLLM local-causal representatives and packing | batched fixed reads | CUDA/SYCL selected attention | implemented on synthetic Q/K |
| C3 | real Qwen Q/K/V + representative manifest | Python-exported main store | native/Python layer parity | planned |
| C4 | cross-layer SSD/H2D/FFN DAG and correction | queued reads, buffer ownership | streams/queues with no global sync | planned |

The pipeline scheduler has a separate `P` line so synthetic timing tests cannot
be confused with C2 selection correctness or real-model results:

| Version | Scope | Status |
|---|---|---|
| P0 | liburing split submit/wait, two pinned buffers/VRAM slots, CUDA copy/compute streams | implemented |
| P1.0 | controlled history prediction + physical miss read/VRAM overwrite | implemented |
| P1.1 | C2.1 InfLLM selector, dense-mass audit, history correction | implemented |
| P1.2a | real Qwen3-0.6B layer-0 CUDA/cuBLAS teacher parity | implemented |
| P1.2b | real post-RoPE Qwen FP16 KV via liburing into real sparse layer | implemented |
| P1.2c | persistent ring/slots and next-KV overlap around real layer replica | implemented |
| P1.2d | 28 distinct real layers, per-layer KV offsets and teacher audits | implemented |
| P1.2e | 28-layer native sparse hidden chain with chain-specific selection | implemented |
| P1.2f | single-process 28-layer chain with persistent resources | implemented |
| P1.2g | all used FP32 weights resident in VRAM; remove timed weight streaming | implemented |
| P1.2g.1 | two KV slots; L+1 liburing read during L compute and H2D during L MLP | implemented |
| P1.2g.2 | production timer excludes per-layer D2H/file teacher audit | implemented |
| P1.2g.3 | all selected KV in pinned DRAM; SSD-free decode upper bound | implemented |
| P1.2h | bounded independent-ring read-ahead depth 2/4/8/16 | implemented |
| P1.3a | token/layer/selection-generation-safe continuous correction tickets | implemented |
| P1.3b.0 | native 32-token sealing into the main store and liburing readback | implemented |
| P1.3b.1 | real 28-layer Qwen projected FP16 KV through native lifecycle | implemented |
| P1.3b.2 | native RMSNorm/KV/K-Norm/RoPE projection into native lifecycle | implemented |
| P1.3c.0 | native final RMSNorm/LM head and next-token parity | implemented |
| P1.3c.1 | native token embedding feedback for two positions | implemented |
| P1.3 | packed INT4/AWQ resident weights and continuous-token correction | planned |
| P2 | block lifecycle/writeback and 512-token continuous decode | planned |
| P-SYCL | same DAG using oneAPI queues/events | planned; NVIDIA plugin pending |

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

The equivalent oneAPI SYCL CPU run uses the same 512 operations and reference:

| Stage | Mean |
|---|---:|
| registered-buffer `io_uring` read | 0.121729 ms |
| SYCL host USM → device USM | 0.073440 ms |
| SYCL transform kernel | 0.064103 ms |
| device USM → host reference | 0.021966 ms |

This is a CPU OpenCL device result. It is useful for API/queue correctness, but
must not be compared with CUDA PCIe/VRAM timings.

## C1 measured result

C1 runs stable-softmax GQA attention over four selected 32-token blocks. Both
CUDA and SYCL match the C++ CPU oracle with maximum absolute error `1.49e-8`
and cosine `1.0`. CUDA's intentionally serial correctness kernel takes
2.263 ms; the SYCL CPU kernel takes 0.430 ms. These diagnose kernel structure
and are not performance claims. See
[the full C1 record](cpp-versions/C1-gqa-sparse-attention.md).

## C1.1 measured result

C1.1 maps one selected token and one output dimension to each of 128 lanes.
Across three paired runs, CUDA kernel time drops from `2.178537 ± 0.065752 ms`
to `0.039970 ± 0.000016 ms` (54.50x); SYCL CPU drops from
`0.420021 ± 0.011418 ms` to `0.070215 ± 0.005198 ms` (5.98x). Both retain
cosine `1.0`. See
[the full C1.1 record](cpp-versions/C1.1-parallel-attention.md).

## C1.2 measured result

C1.2 adds complete backend wall timing and persistent resources. CUDA wall time
improves slightly from `0.092245 ± 0.000124 ms` to
`0.090433 ± 0.000206 ms` because buffers were already retained in C1.1. SYCL
CPU improves from `0.468826 ± 0.049722 ms` to
`0.229556 ± 0.016728 ms` (2.04x) after removing per-call USM allocation. See
[the full C1.2 record](cpp-versions/C1.2-persistent-resources.md).

## C1.3 measured result

C1.3 audits five of every 64 outputs and leaves the rest device-resident. CUDA
backend wall time falls from `0.090433 ± 0.000206 ms` to
`0.084125 ± 0.000318 ms` (-6.98%); SYCL CPU falls from
`0.229556 ± 0.016728 ms` to `0.191941 ± 0.022294 ms` (-16.39%). Every sampled
audit retains cosine `1.0`. See
[the full C1.3 record](cpp-versions/C1.3-device-resident-output.md).

## C1.4 measured result

C1.4 adds an ordered downstream checksum consumer. CUDA measures
`0.024496 ± 0.000021 ms` consumer time and `0.108761 ± 0.000514 ms` total
backend wall. SYCL CPU measures a `0.007418 ± 0.000495 ms` consumer event but
`0.235857 ± 0.007578 ms` wall, exposing substantial queue/control overhead.
See [the full C1.4 record](cpp-versions/C1.4-device-consumer.md).

## C2.1 measured result

C2.1 implements local-causal mass representatives and one-submission,
four-request liburing packing. It selects `[0,8,13,15]` identically on both
backends, produces bit-exact packed KV, and improves synthetic
selected-vs-dense cosine from failed C2.0's `0.456345` to `0.987693`. See
[the full C2.1 record](cpp-versions/C2.1-infllm-selection-uring.md).

## P0 measured result

P0 implements the cross-layer resource DAG with native split-phase liburing,
a real FP16 GQA attention kernel and a clearly marked synthetic FFN scheduling
window. Across ten 16-step × 28-layer repeats, median layer time falls from
`0.301737 ms` serial to `0.217985 ms` pipelined (1.383603×), with bit-identical
output. See [the P0 record](native-pipeline/P0-cuda-dual-pipeline.md).

## P1.0 measured result

P1.0 adds history prediction, actual miss-only SSD reads and partial VRAM slot
overwrites. At an intentionally controlled 80.692% history block hit rate it
corrects 346 misses per 16×28-layer repeat, obtains 100% oracle block recall and
bit-identical output. Median speedup over an equal-compute serial oracle is
1.112365×. See
[the P1.0 record](native-pipeline/P1.0-history-correction.md).

## P1.1 measured result

P1.1 feeds the C2.1 local-causal representative selector into the P1 correction
pipeline and uses the same query for selection and CUDA attention. It obtains
100% corrected-selector recall, but only 80.9152% dense-oracle block recall and
58.5938% history hit. The 742 miss reads make the pipeline slower than serial:
median speedup is `0.849826×`. See
[the P1.1 record](native-pipeline/P1.1-infllm-selection.md).

## P1.2a measured result

P1.2a executes a complete real Qwen3-0.6B layer using cuBLAS projections and
CUDA RMSNorm/RoPE/GQA/SwiGLU kernels. All audited boundaries have cosine 1.0;
final layer maximum error is `8.64e-7` against the PyTorch FP32 teacher. See
[the P1.2a record](native-pipeline/P1.2a-real-qwen-layer.md).

## P1.2b measured result

P1.2b reads four real post-RoPE Qwen FP16 KV blocks with liburing and executes
the current-token Q path, sparse attention, O projection and MLP natively.
Final error against the matching sparse teacher is `1.80e-4` with cosine
`0.999999988`; sparse-vs-dense layer cosine is `0.999383555`. See
[the P1.2b record](native-pipeline/P1.2b-real-qwen-ssd-sparse.md).

## P1.2c measured result

P1.2c retains the liburing ring, CUDA/cuBLAS resources and real layer executor.
Persistent SSD read median is `0.147124 ms`; next-read exposed wait falls to
`0.000199 ms`, and next-slot H2D is issued under the current real MLP. It uses
a real layer-0 replica, not 28 distinct weights. See
[the P1.2c record](native-pipeline/P1.2c-persistent-real-pipeline.md).

## P1.2d measured result

P1.2d streams all 28 distinct Qwen3-0.6B layer bundles and a shared per-layer
FP16 KV store. All 28 native layers pass their matching sparse teacher;
maximum final error is `6.1035e-5` and minimum cosine is 1.0. The worst
sparse-vs-dense layer cosine is `0.990931404`. See
[the P1.2d record](native-pipeline/P1.2d-28-distinct-layers.md).

## P1.2e measured result

P1.2e feeds every native sparse output into the next distinct native layer and
recomputes selection from the chained query. All 28 layers pass; maximum final
error is `9.1553e-5`, minimum cosine is 1.0, and layer 11 changes its block set
to `[0,2,14,15]`. See
[the P1.2e record](native-pipeline/P1.2e-native-hidden-chain.md).

## P1.2f measured result

P1.2f runs the full native chain in one process without intermediate hidden
files. Five-run median wall is `816.670344 ms`: FP32 weight read/H2D consumes
`755.678503 ms`, selected KV transport `16.439455 ms`, and real sparse compute
only `7.024637 ms`. Correctness remains unchanged. See
[the P1.2f record](native-pipeline/P1.2f-single-process-chain.md).

## P1.2g / P1.2g.1 measured result

P1.2g preloads the 28 layers' used FP32 tensors (`1,526,970,368` bytes) into
VRAM outside the timed chain. Its five-run median falls to `20.410660 ms`.
P1.2g.1 then double-buffers selected KV, submitting layer L+1 SSD read during
layer L and its copy on a separate CUDA stream before the L MLP tail. The
five-run median is `16.179101 ms`, with `10.663527 ms` of prefetch wait still
exposed: these tiny reads remain longer than a Qwen3-0.6B sparse layer. Output
parity is unchanged. See
[the P1.2g record](native-pipeline/P1.2g-resident-weights-kv-pipeline.md).

P1.2g.2 moves per-layer teacher comparison outside the production timing path;
its five-run median is `15.095921 ms`. P1.2g.3 preloads the complete 14 MiB
selected-KV working set into pinned DRAM and measures `7.197472 ms` decode wall
after `11.970140 ms` setup. It is a DRAM-capacity upper bound, not the paper's
default next-layer policy, and isolates SSD small-read waiting as the remaining
major cost in P1.2g.2.

## P1.2h measured result

P1.2h keeps only `depth x 512 KiB` selected KV in pinned DRAM and allows that
many liburing batches to remain outstanding. Five-run median wall decreases
from `15.095921 ms` at depth 1 to `13.430375/12.880146/12.181188/10.167049 ms`
at depths 2/4/8/16. Depth 16 uses 8 MiB pinned DRAM and leaves `2.910755 ms`
exposed wait. Final error remains `9.1553e-5`. See
[the P1.2h record](native-pipeline/P1.2h-bounded-read-ahead.md).

## P1.3a measured result

P1.3a attaches `{token, layer, selection_generation, block_set}` to every
continuous-step prefetch and validates it before the VRAM slot is consumed.
Across 16 steps and 28 layers, all 448 tickets validate; a controlled stale
generation is rejected by the startup self-test. Corrected oracle recall is
1.0 and serial/pipeline maximum error is zero. Performance remains negative
(`0.775908x` median) because the controlled InfLLM workload has only 58.5938%
history hit and 742 correction misses. See
[the P1.3a record](native-pipeline/P1.3a-generation-safe-correction.md).

P1.3b.0 runs the native lifecycle for 512 output tokens: 448 sealed blocks are
written to the main store and all 448 pass byte-exact liburing readback. Every
layer ends with a 32-token local tail and generation 16. Its FP16 KV patterns
are deterministic lifecycle fixtures; real Qwen projection is the next step.

P1.3b.1 replaces deterministic KV bytes with 544 real Qwen3-0.6B token K/V
tensors (28 layers, post-RoPE K, FP16 store contract). The native lifecycle
seals and byte-verifies all 448 blocks; five-run full verification median is
`160.127803 ms`. Projection/export is currently PyTorch, while sealing, main
store writes and liburing verification are native.

P1.3b.2 performs the projection itself with native CUDA/cuBLAS for all 28
layers and 512 tokens. Across 29,360,128 FP16 elements, mismatch rate is
0.142142%, mean absolute error `2.9334e-7`, and cosine `0.999999999987` versus
the PyTorch teacher. All 420 sealed blocks pass physical readback.

P1.3c.0 sends the depth-16 native chain hidden through final RMSNorm and the
151,936-way LM head. Native and teacher argmax are both token 50 and their top-5
orders match exactly. Five-run medians are `10.144562 ms` for the sparse chain
and `2.696542 ms` for the resident LM-head computation.

P1.3c.1 feeds token 50 through native embedding lookup and produces token 271
at position 512. Both steps match teacher LM-head argmax and have logit cosine
above `0.9999999999999`. The second step deliberately retains the fixed prompt
selection plan, so it is feedback plumbing rather than complete sparse decode.

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
| `cpp/src/attention_main.cpp` | C1 liburing + GQA attention benchmark composition |
| `cpp/src/attention_reference.cpp` | backend-independent FP16 conversion and CPU oracle |
| `cpp/src/pipeline_main.cpp` | P0 serial and cross-layer CUDA/liburing scheduling DAG |
| `scripts/build_cpp.sh` | reproducible user-space CUDA build |
| `scripts/build_cpp_sycl.sh` | oneAPI build with an explicit compiler check |
| `scripts/run_cpp_c0_sycl.sh` | build and run the SYCL backend |
| `scripts/benchmark_cpp_p0.py` | repeated P0 runs, percentiles and published trace |

Run C0 CUDA:

```bash
./scripts/run_cpp_c0.sh --operations 512
./scripts/run_cpp_c0_sycl.sh --operations 512
```

The output includes `c0-metrics.json` and a Perfetto-compatible
`c0-trace.json`. The trace has separate SSD read, PCIe H2D, GPU compute and
PCIe D2H lanes.

The current machine has no system C++ compiler, CMake or nvcc. The CUDA target
uses the repository's Zig/Clang compiler plus the CUDA runtime/NVRTC installed
with PyTorch. liburing 2.5 development files are locally extracted and ignored
by Git. Intel oneAPI 2026.1.1 and a GCC 15.2 host toolchain are installed in an
ignored micromamba prefix. `sycl-ls` currently exposes the Intel CPU OpenCL
device only; NVIDIA execution remains pending the Codeplay plugin.

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
5. The first micromamba extraction failed because the host has no `bzip2`
   executable. Python's standard `tarfile` module extracted the same archive.
6. Intel's conda DPC++ package installed without host C++ standard-library
   headers. Adding the pinned conda GCC 15.2 package and explicitly passing its
   include/link directories fixed `cstddef` and `crtbegin.o` failures.

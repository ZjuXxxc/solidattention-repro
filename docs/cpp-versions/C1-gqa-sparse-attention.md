# C1 — Native FP16 GQA sparse-attention correctness

C1 replaces C0's byte transform with the actual attention equation while
retaining the same native SSD and accelerator boundaries.

## Shared contract

- 128 selected tokens = four 32-token blocks.
- Token-major `[K heads][V heads]` interleave.
- FP16 K/V, FP32 query, softmax and accumulation.
- 16 query heads, 8 KV heads, head dimension 128.
- GQA mapping: `kv_head = query_head / (query_heads / kv_heads)`.
- Stable softmax subtracts the maximum logit.

`attention_reference.cpp`, the NVRTC CUDA source and the oneAPI SYCL kernel use
the same indexing and numerical contract. The benchmark rejects an output when
maximum absolute error exceeds `2e-5` or cosine is below `0.999999`.

## 64-operation result

Each operation reads a 512 KiB four-block selection through a liburing
registered fixed buffer, transfers it to the selected device, runs attention
and copies the 8 KiB result back for audit.

| Backend | io_uring read | H2D/USM | attention kernel | audit D2H | max error | cosine |
|---|---:|---:|---:|---:|---:|---:|
| CUDA NVRTC, RTX 4080 | 0.143797 ms | 0.037113 ms | 2.262930 ms | 0.009258 ms | 1.49e-8 | 1.0 |
| oneAPI SYCL, i9 CPU | 0.277938 ms | 0.202481 ms | 0.430380 ms | 0.030018 ms | 1.49e-8 | 1.0 |

These are correctness-kernel timings, not optimized SolidAttention numbers.
The CUDA kernel intentionally assigns one serial thread to each query head.
That makes the equation and indexing easy to audit, but wastes the GPU and
explains why the CPU SYCL kernel is faster. C1.1 must parallelize token dot
products, reductions and value accumulation before performance claims.

## Failed SYCL run and stale-artifact guard

The first 64-operation SYCL attempt hit an intermittent Intel DPC++ 2026.1.1
frontend segmentation fault at `-O2`. The outer ad-hoc shell sequence did not
use `set -e`, so it continued and copied the preceding four-operation output
under a new name. Those invalid generated files were deleted before commit.

C1-SYCL now builds at `-O1`; the replacement artifact explicitly reports
`operations=64`. Published experiment wrappers should use `set -euo pipefail`
and validate the version/operation count before copying immutable evidence.

## Remaining gap

- Synthetic deterministic K/V validates math but does not yet read Qwen prompt
  blocks produced by the Python store.
- Selection is fixed at four blocks; InfLLM representative selection is C2.
- The kernels materialize logits privately and have no warp/sub-group
  reduction, tensor cores or fused online softmax.
- NVIDIA SYCL execution still needs the Codeplay plugin.
- Reads, H2D and attention remain serial in C1; cross-layer scheduling follows
  only after the optimized kernel is trustworthy.

# V13 — Cross-layer predicted H2D pipeline

V13 advances the predicted KV transfer for layer L+1 into layer L's FFN
window. It preserves V11 selection and V12 storage semantics: all three
standard variants generate identical token IDs and logits quality.

## Intended dependency graph

```text
prefetch worker:  SSD read L+1 ─────┐
copy stream:                         └─ H2D predicted L+1 ───────┐
compute stream: attention L ─────────── FFN L ───────────────────┼─ attention L+1
                                                                └─ miss correction
```

The next layer consumes the predicted VRAM buffer only after its CUDA event.
Miss correction is ordered on the same copy stream, so corrected slots cannot
race the full predicted transfer.

## V13.0: retained failed schedule

The first implementation called `ssd_future.result()` on the main thread after
attention L and only then submitted H2D L+1. A trace showed about 0.215 ms per
SSD read versus about 0.098 ms per attention task. The uncovered SSD tail was
therefore inserted immediately before FFN instead of hidden behind it.

Three paired runs, Qwen3-0.6B, prompt 512, output 16, budget 128:

| Version | tok/s (mean ± sample std) | mean latency | P50 | P95 |
|---|---:|---:|---:|---:|
| V11 | 55.601 ± 0.139 | 17.985 ± 0.045 ms | 17.474 ms | 18.665 ms |
| V13.0 | 51.116 ± 1.589 | 19.576 ± 0.620 ms | 19.010 ms | 20.826 ms |

V13.0 is 8.07% slower in throughput and 8.85% worse in mean latency than V11.
This negative result is kept because it identifies a dependency placement bug,
not an SSD bandwidth limit.

## V13.1: asynchronous dependency continuation

V13.1 gives the executor a second worker. That worker waits for the L+1 SSD
future and submits its H2D to the CUDA copy stream; the main thread submits FFN
L immediately. At layer L+1 the main thread only consumes the worker result and
waits on the CUDA event when attention needs the buffer.

| Version | tok/s (mean ± sample std) | mean latency | P50 | P95 |
|---|---:|---:|---:|---:|
| V13.1 | 53.811 ± 0.498 | 18.585 ± 0.173 ms | 18.094 ms | 19.957 ms |

V13.1 recovers 5.27% throughput over V13.0, but remains 3.22% below V11. Every
run has exact prefix 12/16 and logits cosine 0.972023, identical to V11.

## V13.2: scheduler audit

V13.2 adds explicit worker-wait, consumer-wait, H2D, attention and FFN ranges.
The extra events perturb performance, so its 50.314 tok/s is diagnostic rather
than a headline comparison.

Standard 16-token trace:

| Range | Operations | Total | Mean |
|---|---:|---:|---:|
| worker SSD completion wait | 378 | 24.340 ms | 0.0644 ms |
| L+1 consumer wait | 378 | 0.359 ms | 0.0009 ms |
| predicted H2D | 378 | 12.328 ms | 0.0326 ms |
| FFN overlap window | 420 | 57.552 ms | 0.1370 ms |
| sparse attention | 420 | 41.743 ms | 0.0994 ms |
| miss-correction H2D | 99 | 1.655 ms | 0.0167 ms |

Using the recorded host submission timestamp plus CUDA-event duration as an
approximate common axis, 9.894/12.328 ms (80.25%) of predicted H2D intersects a
producer FFN window. This is a diagnostic estimate, not an Nsight-grade device
timeline. The 0.359 ms total consumer wait is the stronger evidence that nearly
all predicted transfers are ready before layer L+1 needs them.

## V12 lifecycle composition

A 33-token run enables both `--pipeline-h2d` and
`--managed-kv-lifecycle`:

- 837 pipelined transfers, 418.5 MiB;
- one sealed 32-token block per layer;
- 28 main-store writes, 3,670,016 bytes;
- logical store length 512 and resident tail 32 tokens;
- 49.994 tok/s, P50/P95 19.675/20.575 ms;
- exact prefix 12/33, logits cosine 0.922738.

This confirms the new dependency schedule does not break the V12 KV lifecycle.
The main-store write is still synchronous before FFN and costs 2.345 ms total.

## Remaining gap to Figure 5

- Python futures, callbacks, CUDA API calls and per-layer bookkeeping dominate
  at Qwen3-0.6B batch=1.
- Dynamic selection still uses `.tolist()`, forcing a GPU-to-CPU dependency.
- Selection correction is not overlapped with the next layer KV projection.
- Managed-store D2H conversion and `pwrite` remain before FFN.
- PyTorch eager attention is not the paper's fused sparse CUDA/SYCL kernel.

The next scheduling version should move sealed-block staging/write submission
off the main path, remove CPU materialization from selection, and validate the
same dependency graph with Nsight Systems on a paper-scale model/context.

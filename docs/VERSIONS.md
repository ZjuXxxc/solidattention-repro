# Experiment version registry

All headline comparisons use Qwen3-0.6B BF16, RTX 4080 Laptop, batch 1,
512-token prompt, 16 generated tokens, 32-token KV blocks, greedy decoding.
Raw per-run outputs are kept under `artifacts/runs/`; version descriptions below
are append-only records. A new optimization gets a new version instead of
silently changing an old result.

Run a version without overwriting earlier evidence:

```bash
./scripts/run_versioned_decode.sh V2-history-prefetch \
  --prompt-tokens 512 --new-tokens 16 --budget-tokens 128 --prefetch-history
```

| Version | Decode path | KV budget | tok/s | P50 / P95 | Exact prefix | Logits cosine |
|---|---|---:|---:|---:|---:|---:|
| V0 | Transformers dense DynamicCache | 512 | 105.50 | 8.73 / 8.96 ms | 16/16 reference | 1.0 |
| V1 | synchronous dynamic select → pread → H2D | 128 | 31.87 | 29.99 / 30.64 ms | 12/16 | 0.97253 |
| V2 | previous-token selection prefetch + miss correction | 128 | 49.84 | 17.99 / 19.12 ms | 12/16 | 0.97253 |
| V3 | V2 + native io_uring + O_DIRECT | 128 | 48.50 | 19.02 / 20.99 ms | 12/16 | 0.97253 |
| V4 | V3 + pinned DRAM double buffers + fixed VRAM slots | 128 | 57.08 | 15.62 / 16.71 ms | 12/16 | 0.97253 |
| V5 | V4 + CUDA copy stream/events, no per-layer global sync | 128 | 55.79 | 16.30 / 17.30 ms | 12/16 | 0.97253 |
| V6 | V5 + missing-block in-place VRAM slot overwrite | 128 | 53.53 | 17.17 / 17.41 ms | 12/16 | 0.97253 |
| V7 | V6 + async new-KV writeback, 32 KiB/layer buffers | 128 | 51.46 | 18.23 / 19.17 ms | 12/16 | 0.97253 |
| V8.0 | fused K/V prototype with retained autograd refs (invalid memory result) | 128 | 54.71 | 16.87 / 18.51 ms | 12/16 | 0.97260 |
| V8.1 | corrected one-GEMM fused K/V projection | 128 | 51.02 | 17.94 / 18.76 ms | 12/16 | 0.97260 |
| V9 | legacy tail-observed landmarks (later audited as incorrect) | 128 | 50.63 | 18.31 / 19.25 ms | 12/16 | 0.97165 |
| V10 | Qwen3-8B-AWQ INT4 + FP16 KV, V9 sparse policy | 128 | 25.23 | 38.13 / 39.89 ms | 3/16 | 0.37901* |
| V11 | InfLLM local-causal representatives | 128 | 55.48 | 17.44 / 18.73 ms | 12/16 | 0.97202 |
| V12.1 | V11 + managed main-store KV lifecycle | 128 | 53.22 | 17.83 / 21.99 ms | 12/33 | 0.92274 |
| V13.0 | main-thread SSD dependency → L+1 H2D during L FFN (failed) | 128 | 51.12† | 19.01 / 20.83 ms† | 12/16 | 0.97202 |
| V13.1 | worker-driven L+1 H2D during L FFN | 128 | 53.81† | 18.09 / 19.96 ms† | 12/16 | 0.97202 |

V2 versus V1 improves decode throughput by 1.564x and reduces mean latency by
36.0%. Selection and math are unchanged, so token IDs and logits similarity are
identical; the difference is scheduling only.

V3 improves the isolated read microtasks but is 2.68% slower end to end than the
single recorded V2 run. The raw backend still allocates an aligned mmap and
copies into Python bytes for every request; V4 is explicitly assigned to remove
that buffer-management cost rather than hiding this negative result.

V4 removes the identified allocation/copy cost and improves V3 throughput by
17.67% (V2 by 14.52%). Two 512 KiB pinned buffers and two 512 KiB VRAM buffers
are reused across every layer and token.

V5 is a retained negative result: it is 2.25% slower than V4. Dynamic top-k
still materializes indices on the CPU once per layer, and CUDA event/stream
bookkeeping is larger than the tiny per-layer copy/attention tasks at this scale.

V6 validates order-independent GPU-slot correction but is 4.06% slower than V5:
66 corrected 128 KiB blocks add exactly 8.25 MiB of H2D. The predicted 210 MiB
still has to move, so correction is additive until prefetch H2D is moved earlier.

V7 adds the previously omitted durability path. It is 3.85% slower than V6,
while asynchronously persisting one full 32 KiB chunk for every layer. Seven
tail tokens remain buffered by design rather than issuing small writes.

V8.0 is retained as a failed implementation audit: fused tensors kept autograd
references to both original projections, inflating VRAM by exactly 112 MiB.
V8.1 fixes ownership with `no_grad/detach` and releases both modules. At decode
M=1, the larger fused GEMM is 0.86% slower than two projections in this run.

V9 correctly implements the published *total* budget and selected-half rule,
but incorrectly claimed that init and local were each one quarter. Its
representatives are not InfLLM-compatible: all blocks were scored by the same
prompt-tail queries after averaging GQA query heads. V11 supersedes that
algorithm while retaining V9 metrics as historical evidence.

V10 validates the 8B INT4 execution and storage path but rejects the 25% sparse
configuration on the current synthetic prompt: free-running generation diverges
after token 3. A full-SSD control is exact for 8/8 tokens with logits cosine
0.999945, proving the FP16 SSD/AWQ path itself is correct. `*` is post-divergence
free-running cosine and therefore not a teacher-forced accuracy metric.

V11 extracts representatives from accumulated local causal attention
probabilities per query head, matching the public InfLLM dataflow. It makes
`repr_topk`, init and local allocations explicit and adds teacher-forced
per-layer hidden cosine, selected dense-attention mass and oracle block recall.
The corrected audit measures 0.99501 mean hidden cosine, 96.80% selected
attention mass and 78.57% oracle block recall. A small number of query heads
still have very low selected mass.

V12.1 seals decode blocks after preserving the true local window, persists them
into reserved segments of the main selectable KV store, and creates new
InfLLM-style representatives. The 33-token boundary run performs exactly one
128 KiB write per layer and returns to a 32-token resident tail. Its synchronous
write is still a correctness path, not the Figure 5 overlap schedule.

V13.0 is a retained scheduler failure: waiting for L+1 SSD completion on the
main thread before FFN exposed the I/O tail and was 8.07% slower than V11.
V13.1 moves that dependency wait to a worker; the layer-entry consumer wait is
only 0.0009 ms per operation and about 80.25% of predicted H2D intersects the
producer FFN window. It recovers 5.27% over V13.0 but remains 3.22% below V11
because Python/CUDA launch bookkeeping dominates this small eager workload.
`†` is the mean of three runs; detailed standard deviations and the V13.2
instrumented trace are in
[the V13 record](versions/V13-cross-layer-h2d-pipeline.md).

## Measurement caveats

- `cold_io_requested=true` uses `posix_fadvise(DONTNEED)`, which is a kernel hint,
  not the strict guarantee of aligned `O_DIRECT`.
- Most historical results are single recorded runs. V11/V13 paired scheduler
  comparisons use three runs and report sample standard deviations.
- Token 1 comes from dense prefill and is excluded from per-token decode latency.
- Peak VRAM is dominated by the 0.6B model at this context length; long-context
  memory scaling must be measured separately.

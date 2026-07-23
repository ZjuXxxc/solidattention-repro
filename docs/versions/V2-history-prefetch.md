# V2 — Historical selection speculative prefetch

For each layer, V2 predicts that the current token will select the same blocks
as the preceding token. While layer L computes, a background thread prefetches
layer L+1's predicted blocks. Once the real query is available, correctly
prefetched blocks are retained; wrong blocks are logically overwritten and only
missing blocks are read synchronously. Global token ordering is not required by
attention, matching SolidAttention's out-of-order overwrite observation.

## End-to-end metrics

- Mean decode: 20.0647 ms; 49.8389 token/s
- P50 / P95: 17.9882 / 19.1210 ms
- Peak VRAM: 1355.71 MiB
- H2D over 15 measured decode steps: 210 MiB
- Dense-equal prefix: 12/16 tokens
- Mean logits cosine: 0.97252984
- V1 → V2 throughput: 1.564x
- V1 → V2 mean-latency reduction: 36.0%

## Predictor and microtask metrics

- Predicted blocks: 1568; hits: 1502; hit rate: 95.7908%
- Missing blocks / wrong blocks: 66 / 66
- First-token demand reads: 28 calls, 14.031 ms total
- Speculative reads: 392 calls, 150.521 ms cumulative worker time
- Miss corrections: 66 calls, 8.933 ms total
- H2D: 420 calls, 24.840 ms total
- Sparse attention: 420 calls, 44.365 ms total

The cumulative speculative time is not wall time: it overlaps layer compute and
host work. The first sparse decode token has no history and therefore uses V1's
demand path. Per-token hit rates after that range from 92.0% to 98.2%.

Raw trace: `artifacts/runs/20260722T165049Z-V2-trace.json`
Visual trace: `artifacts/qwen-decode-v2-history.html`

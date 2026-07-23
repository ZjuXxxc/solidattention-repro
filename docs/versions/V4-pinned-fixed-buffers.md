# V4 — Pinned DRAM double buffers and fixed VRAM slots

V4 retains V3's raw `io_uring + O_DIRECT` backend but submits reads directly to
the address of two preallocated CUDA-pinned CPU tensors. Two fixed GPU uint8
buffers receive alternating layers. There is no per-request mmap, Python bytes
object, `torch.frombuffer`, or dynamic GPU KV allocation.

Incorrect predictions are corrected by reading missing blocks directly into
the pinned slots occupied by wrong predictions. Attention does not require
global token ordering, only K/V pair alignment, so the physical slot order is
left unchanged.

## End-to-end metrics

- Mean decode: 17.5204 ms; 57.0765 token/s
- P50 / P95: 15.6224 / 16.7130 ms
- V3 → V4 throughput: +17.67%; mean latency: -15.02%
- V2 → V4 throughput: +14.52%
- Pinned DRAM buffers: 1.0 MiB; fixed VRAM buffers: 1.0 MiB
- Peak VRAM: 1356.21 MiB; H2D over measured steps: 210 MiB
- Exact prefix: 12/16; logits cosine: 0.97252984
- Predictor: 95.7908%, 1502/1568 hits, 66 corrections

## Microtasks

| Task | Count | Total | Median | P95 |
|---|---:|---:|---:|---:|
| First-token direct demand | 28 | 6.359 ms | 0.2145 ms | 0.3214 ms |
| Direct speculative prefetch | 392 | 82.092 ms | 0.2014 ms | 0.2656 ms |
| Direct slot correction | 66 | 12.419 ms | 0.1949 ms | 0.2191 ms |
| Pinned H2D | 420 | 16.437 ms | 0.0335 ms | 0.0673 ms |
| Sparse attention | 420 | 41.347 ms | 0.0835 ms | 0.1087 ms |

Raw metrics: `artifacts/runs/20260722T170854Z-V4-pinned-fixed-buffers-metrics.json`
Raw trace: `artifacts/runs/20260722T170854Z-V4-pinned-fixed-buffers-trace.json`
Dashboard: `artifacts/runs/20260722T170854Z-V4-pinned-fixed-buffers-dashboard.html`

# V5 — CUDA copy stream and event dependencies

V5 removes V4's `torch.cuda.synchronize()` after every H2D and attention task.
Pinned H2D runs on a dedicated copy stream. CUDA events connect copy completion
to default-stream attention and prevent a double-buffer slot from being reused
until its previous consumer completes. GPU durations are measured with CUDA
events rather than host enqueue time.

## Metrics

- Mean decode: 17.9232 ms; 55.7935 token/s
- P50 / P95: 16.3023 / 17.3043 ms
- V4 → V5 throughput: -2.25%; mean latency: +2.30%
- Peak VRAM: 1356.21 MiB; pinned/fixed buffers: 1 MiB each
- Exact prefix: 12/16; logits cosine: 0.97252984
- Predictor: 95.7908%; 66 missing/wrong blocks

This is a negative result, not evidence that CUDA streams are generally harmful.
Current selection calls `argsort(...).tolist()` for every layer, forcing a
GPU-to-CPU dependency, while each 512 KiB copy and sparse attention task is very
small. Creating and timing 420 pairs of CUDA events also adds host overhead.
V6 will use prediction to move blocks into VRAM before final selection and copy
only miss corrections into wrong slots.

Raw metrics: `artifacts/runs/20260722T171031Z-V5-cuda-streams-metrics.json`
Raw trace: `artifacts/runs/20260722T171031Z-V5-cuda-streams-trace.json`
Dashboard: `artifacts/runs/20260722T171031Z-V5-cuda-streams-dashboard.html`

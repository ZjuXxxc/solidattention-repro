# V6 — In-place missing-block overwrite in VRAM

V6 changes miss handling. V5 first corrected the pinned host buffer and then
copied the complete selected buffer. V6 copies predicted blocks to the fixed
VRAM buffer first. Missing blocks are read into the alternate pinned buffer and
copied only into GPU rows occupied by wrong predictions. K and V remain paired,
but global token ordering is deliberately not restored.

This validates the paper's observation that attention permits out-of-order
replacement. Generated tokens, dense prefix, and logits cosine are identical to
V2–V5.

## Metrics

- Mean decode: 18.6825 ms; 53.5260 token/s
- P50 / P95: 17.1686 / 17.4108 ms
- V5 → V6 throughput: -4.06%; mean latency: +4.24%
- H2D: 218.25 MiB versus 210 MiB in V5
- Added correction H2D: 66 × 128 KiB = 8.25 MiB exactly
- Peak VRAM: 1356.21 MiB
- Exact prefix: 12/16; logits cosine: 0.97252984
- Predictor hit rate: 95.7908%; 66 missing/wrong blocks

The feature is functionally correct but not independently profitable here.
Predicted blocks still require the full 210 MiB H2D; in-place correction is an
additional transfer. A complete scheduler must launch predicted H2D earlier and
hide it under preceding projection/MLP work.

Raw metrics: `artifacts/runs/20260722T171625Z-V6-gpu-slot-overwrite-metrics.json`
Raw trace: `artifacts/runs/20260722T171625Z-V6-gpu-slot-overwrite-trace.json`
Dashboard: `artifacts/runs/20260722T171625Z-V6-gpu-slot-overwrite-dashboard.html`

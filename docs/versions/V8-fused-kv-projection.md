# V8 — Consolidated K/V projection weights

V8 concatenates each layer's K and V projection weights during initialization.
Decode executes one `F.linear` and views its output directly as
`[batch, token, K_or_V, kv_head, head_dim]`. Original `k_proj` and `v_proj`
modules are removed after dense prefill/reference execution.

## V8.0 failed ownership attempt

The initial `torch.cat` was executed with autograd tracking. Its graph retained
references to both original parameter tensors after their modules were removed.
Peak sparse VRAM became 1468.21 MiB—exactly 112 MiB above the expected result,
equal to the fused weights. This run is computationally valid but its memory
result is invalid and must not be used as the V8 headline.

- Throughput: 54.7075 token/s; mean 18.2790 ms
- Raw trace: `artifacts/runs/20260722T172236Z-V8-fused-kv-projection-trace.json`

## V8.1 corrected implementation

Fusion now runs under `torch.no_grad()`, detaches the result, deletes both
modules, releases the final temporary reference, runs Python GC, and then clears
the CUDA allocator cache.

- Mean decode: 19.6004 ms; 51.0195 token/s
- P50 / P95: 17.9424 / 18.7585 ms
- V7 → V8.1 throughput: -0.86%; mean latency: +0.87%
- Peak sparse VRAM: 1356.21 MiB (ownership leak fixed)
- Fused live weight size: 112 MiB, replacing—not supplementing—112 MiB originals
- Exact prefix: 12/16; mean logits cosine: 0.97260163
- Predictor: 1499/1568 hits (95.5995%), 69 corrections

The slight numerical change comes from a different GEMM shape and changes three
borderline block-selection decisions. Generated tokens remain identical to V7.
For batch-1, one-token decode, a single wider GEMM is not faster on this GPU;
the consolidation remains useful because it emits storage-compatible layout and
removes the separate post-projection K/V packing requirement in a fused writer.

Raw metrics: `artifacts/runs/20260722T172321Z-V8.1-fused-kv-projection-metrics.json`
Raw trace: `artifacts/runs/20260722T172321Z-V8.1-fused-kv-projection-trace.json`
Dashboard: `artifacts/runs/20260722T172321Z-V8.1-fused-kv-projection-dashboard.html`

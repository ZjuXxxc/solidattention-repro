# V7 — Asynchronous new-KV writeback with 32 KiB buffers

V7 keeps V6's read path and adds the non-critical write path. Every generated
token contributes 4 KiB of BF16 K/V per Qwen layer. Two banks of CUDA-pinned
32 KiB buffers aggregate eight tokens before a background worker issues one
aligned `O_DIRECT pwritev` per layer. The decode tail remains immediately
available in VRAM; storage persistence is not on the attention dependency path.

## Metrics

- Mean decode: 19.4309 ms; 51.4645 token/s
- P50 / P95: 18.2332 / 19.1688 ms
- V6 → V7 throughput: -3.85%; mean latency: +4.01%
- Exact prefix: 12/16; logits cosine: 0.97252984
- New-KV D2H staging: 15 events, 5.565 ms host enqueue total
- Full async writeback: 28 × 32 KiB = 917,504 bytes, 1.821 ms
- Full chunks persisted: 1; remaining buffered tail: 7 tokens/layer
- Preallocated file: 1,835,008 bytes for two possible chunks

The write is asynchronous, but staging 28 small GPU K/V tensors into pinned
buffers still adds overhead. A later fused K/V projection can emit interleaved
data directly and remove much of this packing cost.

Raw metrics: `artifacts/runs/20260722T171759Z-V7-async-writeback-metrics.json`
Raw trace: `artifacts/runs/20260722T171759Z-V7-async-writeback-trace.json`
Dashboard: `artifacts/runs/20260722T171759Z-V7-async-writeback-dashboard.html`

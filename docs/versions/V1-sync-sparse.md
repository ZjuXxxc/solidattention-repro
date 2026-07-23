# V1 — Synchronous SSD sparse decode

Each layer computes the current query, dynamically selects four 32-token prompt
blocks, synchronously executes `pread`, copies 128-token KV to VRAM, and runs
sparse attention. The generated decode tail remains resident as the local block.

- Mean decode: 31.3735 ms; 31.8740 token/s
- P50 / P95: 29.9942 / 30.6415 ms
- Peak VRAM: 1355.71 MiB
- H2D over 15 measured decode steps: 210 MiB
- Dense-equal prefix: 12/16 tokens
- Mean logits cosine: 0.97252984
- Limitation: 28 synchronous reads per token leave GPU and SSD mostly serialized.

Raw metrics: `artifacts/qwen-decode-budget128.txt`
Raw trace: `artifacts/qwen-decode-budget128.json`

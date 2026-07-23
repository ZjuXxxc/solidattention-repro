# V10 — Qwen3-8B AWQ INT4 and FP16 KV

## Environment and compatibility

- Official model: `Qwen/Qwen3-8B-AWQ`, revision `4da05a8edb55c6046cce958586c33b61da07bb79`
- AWQ: 4-bit GEMM, group size 128, zero point enabled
- Model: 36 layers, hidden 4096, 32 Q heads, 8 KV heads
- Runtime: AutoAWQ 0.2.9, Transformers 4.51.3, Torch 2.5.1+cu121
- AutoAWQ is deprecated and its last tested Transformers release is 4.51.3.
- Triton requires a first-run C helper build. With no system compiler or sudo,
  V10 uses the user-space Zig compiler plus locally extracted Ubuntu Python 3.12
  development headers through `scripts/zig-cc`.

V10 adds dtype-aware KV metadata. The 0.6B path stores BF16; AWQ activations and
KV use FP16. Treating AWQ KV as BF16 was caught by a Triton dtype assertion before
the standard run and is not included as a benchmark version.

## Dense short-prompt baseline

- Input/output: 26/16 tokens
- Throughput: 37.8896 token/s
- Peak VRAM: 5833.33 MiB

## Standard 512-token, 25% sparse run

- Dense: 37.6655 token/s; sparse SSD: 25.2323 token/s
- Sparse mean/P50/P95: 39.6317 / 38.1269 / 39.8926 ms
- Dense peak: 6060.19 MiB; sparse peak: 6057.19 MiB
- Budget: 128 tokens = 1 init + 1 local + 2 dynamic blocks
- Predictor: 1874/2016 = 92.9563%; 142 corrections
- H2D: 287.75 MiB; writeback: 1,179,648 bytes
- Exact dense prefix: 3/16
- Free-running mean logits cosine after divergence: 0.37901478

The sparse output falls into repetition. This configuration fails the quality
criterion and must not be reported as a successful SolidAttention result.

## Full-SSD equivalence control

With all 512 prompt tokens loaded from SSD:

- Exact prefix: 8/8
- Mean logits cosine: 0.99994497
- Dense: 36.4950 token/s; SSD path: 22.5544 token/s
- Predictor: 100%, no corrections

This isolates the failure: packed AWQ projection, FP16 KV serialization,
O_DIRECT reads, manual layer execution and logits are correct. The quality loss
comes from the 25% selection policy/representative on this workload, not an SSD
dtype or decode implementation bug.

Sparse raw trace: `artifacts/runs/20260722T174429Z-V10-qwen3-8b-awq-trace.json`
Sparse dashboard: `artifacts/runs/20260722T174429Z-V10-qwen3-8b-awq-dashboard.html`
Full control: `artifacts/runs/20260722T174445Z-V10-full-ssd-equivalence-trace.json`

# V9 — Legacy tail-observed landmarks and budget-policy audit

V9 is retained as a historical failed implementation. A later source audit
showed that its representative extraction does not follow InfLLM, and that its
init/local split was presented with more certainty than the paper supports.

## Confirmed policy implemented

- Input shorter than 4096: context budget = 25% of input, block-aligned.
- Input at least 4096: fixed 1024-token context budget.
- Half of the budget is init+local; the paper does **not** state how that half
  is divided internally.
- Remaining half consists of dynamically selected blocks.
- Block size remains 32 tokens.

## Incorrect V9 inference

For every layer and KV head, V9 takes the final 32 post-RoPE prefill queries as
an observation window. Within each block, the real key token with the maximum
observed dot-product attention score becomes that block/head's representative.
This is **not InfLLM-compatible**. InfLLM accumulates the normalized attention
score a token actually receives while it is in its local causal window, keeps
query-head granularity under GQA, and supports multiple representative tokens.
V9 instead overfits every block to the same prompt-tail observation queries and
averages GQA query heads before ranking.

The measured V9 split (32 init / 32 local / 64 selected) is an experiment
configuration, not a confirmed SolidAttention parameter.

## Standard 512-token metrics

- Budget: 128 tokens = 1 init + 1 local + 2 selected blocks
- Mean decode: 19.7506 ms; 50.6313 token/s
- P50 / P95: 18.3124 / 19.2525 ms
- V8.1 → V9 throughput: -0.76%
- Exact prefix: 12/16; logits cosine: 0.97165005
- History hit rate: 90.625%, 1421/1568
- Missing/wrong blocks: 147; H2D: 228.375 MiB

On this repetitive prompt, the incorrect landmarks are less temporally stable than mean keys:
V8.1 hit 95.60% with 69 corrections. The result is workload-specific and does
not constitute a LongBench accuracy comparison. History hit rate measures
selection-to-selection stability, not dense-attention mass or selection quality.

## 4096-token policy-boundary validation

- Budget: 1024 = 8 init + 8 local + 16 selected blocks
- Dense: 69.16 token/s; sparse: 22.90 token/s (4-token smoke run)
- Dense peak VRAM: 4259.08 MiB; sparse peak: 2816.82 MiB
- Peak reduction: 1442.26 MiB / 33.86%
- Exact prefix: 4/4; logits cosine: 0.98935789
- History hit rate: 85.21%; 265 corrections
- Pinned/VRAM fixed buffers: 8 MiB each

Standard raw trace: `artifacts/runs/20260722T173251Z-V9-attention-landmarks-budget-trace.json`
Standard dashboard: `artifacts/runs/20260722T173251Z-V9-attention-landmarks-budget-dashboard.html`
4096 trace: `artifacts/runs/20260722T173304Z-V9-long-policy-validation-trace.json`

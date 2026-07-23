# V0 — Dense Transformers baseline

Unmodified Qwen3 `DynamicCache`: the complete prompt and decode KV cache remains
in VRAM. This is the correctness and latency reference.

- Mean decode: 9.4783 ms; 105.50 token/s
- P50 / P95: 8.7323 / 8.9589 ms
- Peak VRAM: 1360.24 MiB
- Accuracy reference: exact 16/16, logits cosine 1.0

The mean exceeds P95 because the first measured decode iteration includes a
one-time warm-up outlier; all raw samples should be retained in future runs.

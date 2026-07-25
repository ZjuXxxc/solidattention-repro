# V11 — InfLLM local-causal representatives and teacher-forced audit

## Corrected algorithm

V11 follows the public `thunlp/InfLLM` representative dataflow:

1. Expand each GQA KV head to its query heads; do not average query heads.
2. During dense prefill, compute causal attention within a configurable local
   window and accumulate the normalized attention probability received by
   every key token.
3. For every layer, query head and block, retain the `repr_topk` real keys with
   the highest accumulated score.
4. Average those retained keys to form the per-query-head block identifier.
5. Compute retrieval similarity per query head and only then reduce head scores
   to the shared physical block set.

The default `repr_topk=4` and reference local window of 4096 follow InfLLM's
public Qwen configuration. They are exposed for 1/2/4 ablation rather than
claimed as unpublished SolidAttention parameters.

## Corrected budget evidence

The SolidAttention paper specifies total budget and says init+local jointly use
half, with selected blocks using the other half. It does not specify the split
inside the mandatory half. V11 requires block-aligned `--init-tokens` and
`--local-tokens`; their sum must equal the mandatory allocation. A 32/32 split
is therefore labeled an experiment configuration.

## New quality audit

`--teacher-forced-audit` feeds the same dense token at every decode step and
records, for every layer:

- cosine similarity of the sparse and dense layer output hidden state;
- fraction of dense attention probability covered by selected tokens;
- minimum selected mass over query heads;
- recall of oracle dynamic blocks ranked by dense block attention mass.

This distinguishes temporal history-hit rate from actual selection quality and
locates the first layer where a free-running failure is introduced. Audit runs
retain the dense cache and hidden references, so their VRAM and throughput are
diagnostic metrics, not performance results.

## Validation status

- CPU selection tests: 2 passed.
- Python compile and shell syntax checks: passed.
- First GPU measurement: not started. CUDA failed before model loading because
  the loaded 535.288.01 kernel module does not match the installed 535.309
  userspace libraries. A reboot or matching driver reload is required.

No empty/partial GPU run is retained as an experiment artifact.

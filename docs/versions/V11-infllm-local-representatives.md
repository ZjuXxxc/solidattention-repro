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

- Unit tests: 5 passed after V12 storage/lifecycle tests were added.
- Python compile and shell syntax checks: passed.
- GPU was restored non-destructively by loading locally extracted 535.288.01
  libcuda/NVML with the already-loaded 535.288.01 kernel module. The system
  535.309 module build remains incomplete because the 6.17.0-40 headers are
  missing; this workaround is local and excluded from Git.

## Standard free-running result

Qwen3-0.6B BF16, prompt 512, output 16, budget 128, init/local 32/32,
`repr_topk=4`, local representative window 512:

- Dense / sparse: 110.73 / 55.48 token/s.
- Sparse P50 / P95: 17.44 / 18.73 ms.
- Exact prefix: 12/16; free-running logits cosine: 0.972023.
- History hit: 1468/1568 = 93.622%; corrections: 100.
- H2D: 222.5 MiB.

V11 is 9.58% faster than V9 in these single recorded runs, but version changes
and run noise prevent attributing that difference only to representatives.

## Corrected teacher-forced audit

V11.1 fixes the final-layer comparison: Transformers exposes the last hidden
state after final RMSNorm, so the sparse final-layer output must be normalized
before cosine calculation. The earlier V11 audit remains historical but its
aggregate hidden cosine is invalid.

- Mean layer hidden cosine: 0.995012.
- Selected dense-attention mass: 0.968039.
- Oracle dynamic-block recall: 0.785714.
- Teacher-forced logits cosine: 0.997085.
- Exact argmax prefix remains 12/16.
- Worst individual query-head mass is 0.169381 at token 15/layer 2.

The large gap between mean selected mass and the worst head demonstrates why
history stability or head-mean mass cannot by itself establish quality.

## Ablations

At budget 128, `repr_topk=1/2/4` all produce 12/16 exact and approximately
0.9722 free-running cosine. History hit improves from 91.65% to 93.62% from
topk 1 to 4, without moving the first divergence.

The requested 32/96, 64/64 and 96/32 splits require budget 256 because their
mandatory allocations sum to 128. All three remain 12/16 exact; 32/96 has the
highest cosine (0.974074) and throughput (49.91 token/s), but this repetitive
prompt is not evidence of a generally optimal split.

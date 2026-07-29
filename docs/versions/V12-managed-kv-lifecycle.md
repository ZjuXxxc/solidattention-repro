# V12 — Managed decode KV lifecycle and main-store writeback

V12 replaces the unbounded decode tail and V7 side-file writeback with an
appendable per-layer main KV store.

## Layout and lifecycle

- The store preallocates an independent capacity segment for each layer, so an
  appended block cannot overwrite the next layer.
- The prompt's final local window starts resident in VRAM and is removed from
  the SSD-selectable representatives, avoiding duplicate attention.
- Each new token appends K/V and its normalized local attention mass to a
  `LayerTail`.
- A block seals only when removing its oldest 32 tokens still leaves the full
  configured local window resident.
- Sealing chooses per-query-head representatives from accumulated attention
  mass, writes interleaved K/V into the same main store, updates logical length,
  and appends the representative to future selection.

## Boundary validation

Qwen3-0.6B, prompt 512, output 33, budget 128, local 32:

- Initial main-store logical length: 480 tokens.
- Initial resident local tail: 32 tokens.
- After 32 decode inputs: tail reaches 64, seals one 32-token block, then
  returns to exactly 32 resident tokens.
- Final main-store logical length: 512 tokens.
- 28 main-store writes, one per layer.
- Total persisted: 3,670,016 bytes (128 KiB per layer).
- Measured synchronous write section: 2.142 ms total.
- Sparse throughput: 53.22 token/s; P50/P95: 17.83/21.99 ms.
- Exact free-running prefix: 12/33; cosine: 0.922738.

The 40-token validation independently ends with one sealed block and a
39-token resident tail, confirming that the post-local incomplete remainder is
retained rather than dropped.

## Remaining scheduling limitation

V12 is a correctness version. D2H conversion and `pwrite` occur after attention
and before the layer FFN, so the sealed block write is still on the critical
path. V13 must stage the block into pinned DRAM and submit its main-store write
during FFN. The predicted L+1 H2D transfer also remains to be advanced into the
current layer's FFN interval.

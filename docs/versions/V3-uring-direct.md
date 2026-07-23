# V3 — Native io_uring and O_DIRECT

V3 changes only the V2 storage backend. `io_uring_setup` and `io_uring_enter`
are invoked directly through Linux syscalls; SQ/CQ rings and SQEs are shared
mmaps. The KV file is opened with `O_DIRECT`, and each 128 KiB KV block is read
into a page-aligned anonymous mapping. No liburing, compiler, page cache or
background `pread` emulation is involved.

Byte correctness was verified for four blocks (524,288 bytes): the direct-ring
result exactly matches buffered V2 and has SHA-256
`f826d52920a45193647e1c6148b945bc0e9f58b9d91895b5a003eeec86e1b980`.

## End-to-end metrics

- Mean decode: 20.6164 ms; 48.5050 token/s
- P50 / P95: 19.0241 / 20.9868 ms
- Peak VRAM: 1355.71 MiB; H2D: 210 MiB
- Exact dense prefix: 12/16; logits cosine: 0.97252984
- Predictor hit rate: 95.7908% (1502/1568); misses: 66

## V2 versus V3 microtasks

| Microtask | V2 total | V3 total | Change |
|---|---:|---:|---:|
| First-token demand read | 14.031 ms | 8.145 ms | -41.9% |
| Historical speculative read | 150.521 ms | 109.376 ms | -27.3% |
| Miss correction | 8.933 ms | 10.980 ms | +22.9% |
| H2D | 24.840 ms | 25.394 ms | +2.2% |
| Sparse attention | 44.365 ms | 46.174 ms | +4.1% |

Despite faster direct reads, V3 is 2.68% slower than the single V2 end-to-end
run. Each ring call currently creates and destroys an aligned mmap and copies
completion data into Python `bytes`. This negative result motivates V4's pinned,
reusable double buffers. It must not be presented as an end-to-end speedup.

Raw metrics: `artifacts/runs/20260722T170118Z-V3-uring-direct-metrics.json`
Raw trace: `artifacts/runs/20260722T170118Z-V3-uring-direct-trace.json`
Dashboard: `artifacts/runs/20260722T170118Z-V3-uring-direct-dashboard.html`

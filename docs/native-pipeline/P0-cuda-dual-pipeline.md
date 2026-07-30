# P0: CUDA dual-stream cross-layer pipeline

P0 starts a separate native pipeline version line. It does not replace C2.1:
C2.1 remains the representative/selection correctness baseline, while P0
isolates the scheduling DAG from Figure 5 of the paper.

## Implemented dependency graph

```text
copy stream:     [ L+1 SSD→pinned DRAM ]  [ L+1 pinned DRAM→VRAM ]
                                      \       overlaps L FFN       /
compute stream:              [ L attention ] [ L synthetic FFN ]
```

- The SSD store contains 28 layers, 16 blocks/layer and 128 KiB/block.
- The C2.1 selected-block contract `[0, 8, 13, 15]` packs 512 KiB/layer.
- Two CUDA `cudaHostAlloc` buffers are registered with liburing.
- Two VRAM KV slots alternate by layer.
- `submit_blocks_fixed()` submits four fixed-buffer SQEs before current-layer
  attention; `wait_blocks_fixed()` only collects the CQEs afterward.
- A non-blocking CUDA copy stream stages `L+1` while a separate compute stream
  executes the current FFN window.
- The next layer is entered only after both streams complete.

The attention is the real C1.1 FP16 GQA stable-softmax CUDA kernel
(`16 Q heads / 8 KV heads / head_dim 128 / 128 selected tokens`). The FFN is
an explicitly named synthetic arithmetic window. P0 therefore validates the
pipeline mechanics but is **not** an end-to-end Qwen or paper throughput result.
P1 must replace this window with the real quantized layer and integrate C2.1
selection plus prediction correction.

## Debug correction made during P0

The first implementation wrapped blocking `read_blocks_fixed()` in one
`std::async` per layer. It overlapped I/O, but thread creation polluted every
layer measurement. The final P0 adds a native two-phase liburing API:

1. prepare and submit the batch directly on the scheduler;
2. run CUDA attention;
3. collect already-submitted CQEs and report only the exposed tail.

## Formal result

RTX 4080 Laptop, NVMe `O_DIRECT`, 10 independent repeats, 16 scheduler
steps/repeat, 28 layers/step, 512 FFN-window iterations:

| Metric | Result |
|---|---:|
| serial layer latency, median (P10–P90) | 0.301737 ms (0.294239–0.312153) |
| pipeline layer latency, median (P10–P90) | 0.217985 ms (0.212702–0.221189) |
| speedup, median (P10–P90) | **1.383603×** (1.378848–1.420864×) |
| exposed SSD wait / 448-layer repeat, median | 40.130913 ms |
| serial vs pipeline max absolute error | **0** |

The SSD read is longer than this small attention kernel, so it is only partly
hidden. H2D is hidden under the synthetic FFN window. The published trace has
separate SSD read, PCIe H2D and GPU compute lanes.

Evidence:

- `artifacts/runs/P0-cuda-dual-pipeline-metrics.json`: all ten raw runs.
- `artifacts/runs/P0-cuda-dual-pipeline-trace.json`: Perfetto/Chrome trace.

## File map

| File | Why it exists |
|---|---|
| `cpp/src/pipeline_main.cpp` | P0 scheduler, buffers/slots/streams, kernels, serial control, trace and equivalence check |
| `cpp/include/solidattention/uring_reader.hpp` | synchronous and split-phase registered-I/O contract |
| `cpp/src/uring_reader.cpp` | `O_DIRECT`, registered buffers/files, SQE submit and CQE collection |
| `cpp/src/trace.cpp` | writes resource lanes consumable by Perfetto |
| `scripts/build_cpp_p0.sh` | reproducible native build without system nvcc |
| `scripts/run_cpp_p0.sh` | build and run with compatible driver library |
| `scripts/benchmark_cpp_p0.py` | repetitions, percentiles and evidence publication |

Run:

```bash
./scripts/run_cpp_p0.sh --steps 16 --ffn-iterations 512
.venv/bin/python scripts/benchmark_cpp_p0.py \
  --repeats 10 --steps 16 --ffn-iterations 512
```

Open the trace in Perfetto. For layer `L`, verify that `next-layer KV read`
overlaps `sparse attention`, and `next-layer H2D` overlaps
`current-layer FFN window`. A long `overlap barrier` identifies exposed work on
the layer-entry critical path.

# SolidAttention reproduction & observability lab

这是一个 **SolidAttention-inspired、可执行且可观测的功能原型**，用于研究显存受限 PC
上的 SSD KV cache、历史选择预取和 SSD/DRAM/VRAM 调度。它不是上交 IPADS 团队未公开的
约 25K 行官方 C++/CUDA/SYCL 实现，也不宣称复现论文端到端结果。

设计依据：[SolidAttention: Low-Latency SSD-based Serving on
Memory-Constrained PCs (FAST '26)](https://www.usenix.org/conference/fast26/presentation/zheng)

## 已实现

- Qwen3-0.6B BF16 与 Qwen3-8B-AWQ INT4 的真实连续 autoregressive decode；
- prompt KV 按 token-major K/V interleave 写入 SSD，decode 结果真实影响 logits；
- 32-token block、init/local/dynamic budget、历史选择预取和 miss correction；
- Linux `io_uring` + `O_DIRECT`、pinned DRAM 双缓冲、固定 VRAM slots；
- CUDA copy stream/events、乱序 slot 修正、异步新 KV 写回；
- BF16 路径的 fused K/V projection、论文 context-budget policy；
- V11 的 InfLLM 局部因果 attention-score representative 与逐层质量审计；
- V12 的 decode KV 封块、主 store 写回和固定 local tail 生命周期；
- V13 的跨层 H2D/FFN pipeline、worker 依赖和关键路径等待审计；
- 独立的 C++/CUDA/SYCL/liburing `C0+` 版本线，不与 Python 版本号混用；
- C1 的 FP16 KV、FP32 stable-softmax GQA attention 与 CPU/CUDA/SYCL 对照；
- P0 原生跨层流水线：liburing split submit/wait、双 pinned buffer、双 VRAM
  slot、CUDA copy/compute 双流与逐资源 trace；
- P1.0 历史集合预取、miss-only SSD read、局部 VRAM slot overwrite 与
  correction/QKV overlap trace；
- P1.1 将 C2.1 InfLLM representative selector 接入同一流水线，并分别记录
  history hit、dense attention mass 和 dense-oracle block recall；
- P1.2a 使用真实 Qwen3-0.6B layer-0 权重，以 cuBLAS + CUDA 执行完整
  RMSNorm/QKV/RoPE/GQA/O-proj/SwiGLU/MLP 并逐算子对齐 PyTorch teacher；
- P1.2b 将真实 post-RoPE FP16 KV 按 block 写入 SSD，经 liburing+pinned
  H2D 送入真实 Qwen sparse layer，并分别对照 sparse/dense teacher；
- P1.2c 保持 liburing/CUDA/cuBLAS 资源常驻，将下一批 SSD read 与当前真实
  attention/O-proj 重叠、下一 slot H2D 与当前真实 MLP 重叠；
- P1.2d 流式导出并审计 28 个不同 Qwen 层、独立权重和逐层 FP16 KV offset，
  所有 native sparse layer 均与对应 teacher 对齐；
- P1.2e 将 native layer L 的实际 sparse output 写回并作为 L+1 输入，完成
  28 层 hidden-state chain 和 chain-specific selection audit；
- P1.2f 在单一C++进程中保持CUDA/cuBLAS/liburing和固定buffers，直接以
  device-to-device hidden递推执行28层，并分解权重/KV/compute瓶颈；
- P1.2g 将 28 层已使用的 FP32 权重全部常驻 VRAM；P1.2g.1 用双
  pinned/VRAM slot 将 L+1 KV 的 liburing 读与 L 计算重叠，H2D 与 L MLP 重叠；
- Chrome/Perfetto trace 与独立 HTML dashboard，统一展示 SSD、DRAM、PCIe 和 GPU 时间线；
- V0–V13 逐版本、不可覆盖的指标与失败实验记录。

尚未实现官方自定义 CUDA sparse-attention kernel、GPU-native block selection、
native AWQ/INT4 packed kernel、跨 token correction pipeline、LongBench/OpenCompass，
以及论文正式 baseline/长上下文统计。

## 目录

| 路径 | 用途 |
|---|---|
| `src/solidattention_lab/qwen_decode.py` | 端到端 dense/sparse 连续 decode 主路径 |
| `src/solidattention_lab/io_uring_backend.py` | 可审计的 raw `io_uring` + `O_DIRECT` reader |
| `src/solidattention_lab/qwen_kv_lab.py` | Q/K/V 捕获、SSD 布局与 attention replay |
| `src/solidattention_lab/qwen_pipeline.py` | 多层 serial/overlap 调度实验 |
| `src/solidattention_lab/simulator.py` | 不依赖模型的 SSD→DRAM→VRAM 调度模拟器 |
| `src/solidattention_lab/trace.py` | Chrome/Perfetto Trace Event 输出 |
| `src/solidattention_lab/dashboard.py` | trace 到独立 HTML 可视化 |
| `scripts/` | 环境检查和可复现实验入口 |
| `cpp/` | 原生 C++ 调度核心、CUDA/SYCL 后端和 liburing I/O |
| `docs/DEBUGGING_ZH.md` | AI Infra 分层监测与逐文件调试教程 |
| `docs/VERSIONS.md` | V0–V13 总指标和改进/负优化说明 |
| `docs/CPP_VERSIONS.md` | C++/CUDA/SYCL/liburing 独立版本线 |
| `docs/versions/` | 每个版本的实现边界与详细指标 |
| `artifacts/runs/` | 已发布的不可变 metrics、trace、dashboard |

模型权重、虚拟环境、临时 KV 文件和本地编译依赖不会提交到 Git。模型 revision 与
SHA256 写在对应版本文档中。

## 环境与模型

参考测量环境：Ubuntu 24.04、RTX 4080 Laptop 12GB、32GB DRAM、NVMe SSD、
Python 3.12、PyTorch 2.5.1+cu121。

```bash
git clone <this-repository>
cd solidattention-repro
uv venv --python 3.12
uv pip install --python .venv/bin/python \
  torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python -r requirements.txt

hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir models/Qwen3-0.6B
hf download Qwen/Qwen3-8B-AWQ \
  --revision 4da05a8edb55c6046cce958586c33b61da07bb79 \
  --local-dir models/Qwen3-8B-AWQ
```

AutoAWQ 已停止维护，本项目锁定 Transformers 4.51.3。V10 首次运行还需要可用的
C/C++ compiler 与 Python 3.12 development headers；本机无 root 的 Zig workaround
记录在 [V10 文档](docs/versions/V10-qwen3-8b-awq.md)。

## 运行与看图

```bash
./scripts/check_env.sh
./scripts/run_simulation.sh
./scripts/run_qwen_baseline.sh --new-tokens 32
./scripts/run_qwen_kv_lab.sh --tokens 2048 --block-tokens 32 --budget-tokens 512
./scripts/run_qwen_pipeline.sh --tokens 2048 --block-tokens 32 --budget-tokens 512

# 原生 C0：liburing fixed buffer + CUDA pinned/H2D/NVRTC kernel
./scripts/run_cpp_c0.sh --operations 512
# 同一 contract 的 oneAPI SYCL 后端（当前设备为 Intel CPU OpenCL）
./scripts/run_cpp_c0_sycl.sh --operations 512
# C1：真实 GQA attention correctness kernel
./scripts/run_cpp_c1.sh --operations 64
./scripts/run_cpp_c1_sycl.sh --operations 64
# C1.1：128-lane block/work-group parallel attention
./scripts/run_cpp_c1.sh --operations 64 --optimized
./scripts/run_cpp_c1_sycl.sh --operations 64 --optimized
# C1.2：persistent device resources + complete backend wall time
./scripts/run_cpp_c1.sh --operations 64 --persistent
./scripts/run_cpp_c1_sycl.sh --operations 64 --persistent
# C1.3：输出留在 device，每 16 次采样一次 CPU oracle
./scripts/run_cpp_c1.sh --operations 64 --resident --audit-every 16
./scripts/run_cpp_c1_sycl.sh --operations 64 --resident --audit-every 16
# C1.4：device-side downstream consumer，只采样回传 4-byte checksum
./scripts/run_cpp_c1.sh --operations 64 --consume --audit-every 16
./scripts/run_cpp_c1_sycl.sh --operations 64 --consume --audit-every 16
# C2.1：InfLLM representative → liburing 四块 packing → sparse attention
./scripts/run_cpp_c2.sh
./scripts/run_cpp_c2_sycl.sh
# P0：L+1 SSD→DRAM 与 L attention、L+1 H2D 与 L FFN window 重叠
./scripts/run_cpp_p0.sh --steps 16 --ffn-iterations 512
.venv/bin/python scripts/benchmark_cpp_p0.py --repeats 10 --steps 16
# P1.0：加入历史预测和物理 miss correction
./scripts/run_cpp_p1.sh --steps 16 --ffn-iterations 512
.venv/bin/python scripts/benchmark_cpp_p0.py \
  --history-correction --repeats 10 --steps 16
# P1.1：真实 native InfLLM selection → history correction → CUDA attention
./scripts/run_cpp_p1_1.sh --steps 16 --ffn-iterations 512
.venv/bin/python scripts/benchmark_cpp_p0.py \
  --infllm-selection --repeats 10 --steps 16
# P1.2a：真实 Qwen 单层导出、原生 CUDA/cuBLAS 执行和 teacher parity
./scripts/run_cpp_p1_2.sh
# P1.2b：真实 512-token KV → InfLLM selection → SSD → native sparse layer
./scripts/run_cpp_p1_2b.sh
.venv/bin/python scripts/benchmark_cpp_p1_2b.py --repeats 10
# P1.2c：持久 ring，64 次稳态读取及真实 compute/copy overlap
.venv/bin/python scripts/benchmark_cpp_p1_2b.py \
  --pipeline-next --io-repeats 64 --repeats 10
# P1.2d：28 个不同真实层与共享逐层 KV store
.venv/bin/python scripts/export_qwen_28_layers.py
.venv/bin/python scripts/run_cpp_p1_2d.py
# P1.2e：28 层 native sparse hidden recurrence
.venv/bin/python scripts/run_cpp_p1_2d.py --chain
# P1.2f：单进程28层真实chain；不再逐层启动进程或落盘hidden
./scripts/build_cpp_p1_2f.sh
.venv/bin/python scripts/run_cpp_p1_2f.py --repeats 5
# P1.2g/g.1：权重常驻，再加跨层 KV SSD→DRAM→VRAM 双缓冲管线
.venv/bin/python scripts/run_cpp_p1_2f.py --resident-weights --repeats 5
.venv/bin/python scripts/run_cpp_p1_2f.py \
  --resident-weights --pipeline-kv --repeats 5

# 新实验使用新版本名，不覆盖历史证据
./scripts/run_versioned_decode.sh EXP-budget128 \
  --prompt-tokens 512 --new-tokens 16 --budget-tokens 128 \
  --prefetch-history --uring-direct --pinned-buffers
```

打开生成的 `artifacts/runs/*-dashboard.html`，或把 `*-trace.json` 放入
[Perfetto UI](https://ui.perfetto.dev/)。调试时重点观察：

`NVMe SSD → kernel I/O → pinned DRAM → PCIe H2D → VRAM → sparse attention`

以及异步回写：

`new KV in VRAM → D2H → 32 KiB DRAM buffer → NVMe SSD`

完整的 lane 含义、`iostat` / `nvidia-smi dmon` / `pidstat` / Nsight Systems
用法见 [调试指南](docs/DEBUGGING_ZH.md)。

## 当前结果摘要

标准 0.6B 配置为 batch 1、prompt 512、decode 16、block 32、sparse budget 128。

| 版本 | 核心变化 | tok/s | P50 / P95 | Exact prefix |
|---|---|---:|---:|---:|
| V0 | dense DynamicCache | 105.50 | 8.73 / 8.96 ms | 16/16 |
| V1 | sync SSD sparse | 31.87 | 29.99 / 30.64 ms | 12/16 |
| V2 | 历史选择预取 + 修正 | 49.84 | 17.99 / 19.12 ms | 12/16 |
| V4 | pinned DRAM + fixed VRAM | **57.08** | **15.62 / 16.71 ms** | 12/16 |
| V9 | landmarks + paper budget | 50.63 | 18.31 / 19.25 ms | 12/16 |
| V10 | 8B AWQ，25% sparse | 25.23 | 38.13 / 39.89 ms | **3/16（失败）** |
| V11 | InfLLM local-causal representatives | 55.60† | 17.47 / 18.66 ms | 12/16 |
| V13.0 | 主线程等待 SSD 后 pipeline H2D（失败） | 51.12† | 19.01 / 20.83 ms | 12/16 |
| V13.1 | worker 驱动 L+1 H2D 与 L FFN 重叠 | 53.81† | 18.09 / 19.96 ms | 12/16 |

V2 相比 V1 为 1.564×；V4 是当前 0.6B sparse 最快完整版本。V5–V8 的负优化均被
保留，没有筛掉失败数据。V10 的 25% sparse 自由生成质量不合格；full-SSD control
达到 8/8 exact、logits cosine 0.999945，说明错误来自选择策略/预算而非 SSD dtype
或 AWQ decode 链路。V13.1 已将下一层入口等待压到约 0.0009 ms/次，但在当前
Python eager 0.6B workload 上仍比 V11 低 3.22%。`†` 为三次运行均值；完整数据、
标准差与 caveats 见 [版本总表](docs/VERSIONS.md)。

## 许可证与引用

本仓库代码使用 [MIT License](LICENSE)。论文和 Qwen 模型分别遵循其原始许可；
仓库不重新分发模型权重或论文 PDF。使用本项目时，请同时引用 SolidAttention 原论文。

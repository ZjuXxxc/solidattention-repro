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
- BF16 路径的 fused K/V projection、论文 budget policy 和推断式 attention landmarks；
- Chrome/Perfetto trace 与独立 HTML dashboard，统一展示 SSD、DRAM、PCIe 和 GPU 时间线；
- V0–V10 逐版本、不可覆盖的指标与失败实验记录。

尚未实现官方自定义 CUDA sparse-attention kernel、GPU-native block selection、
AWQ packed fused K/V、LongBench/OpenCompass，以及论文正式 baseline/长上下文统计。

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
| `docs/DEBUGGING_ZH.md` | AI Infra 分层监测与逐文件调试教程 |
| `docs/VERSIONS.md` | V0–V10 总指标和改进/负优化说明 |
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

V2 相比 V1 为 1.564×；V4 是当前 0.6B sparse 最快完整版本。V5–V8 的负优化均被
保留，没有筛掉失败数据。V10 的 25% sparse 自由生成质量不合格；full-SSD control
达到 8/8 exact、logits cosine 0.999945，说明错误来自选择策略/预算而非 SSD dtype
或 AWQ decode 链路。完整数据与 caveats 见 [版本总表](docs/VERSIONS.md)。

## 许可证与引用

本仓库代码使用 [MIT License](LICENSE)。论文和 Qwen 模型分别遵循其原始许可；
仓库不重新分发模型权重或论文 PDF。使用本项目时，请同时引用 SolidAttention 原论文。

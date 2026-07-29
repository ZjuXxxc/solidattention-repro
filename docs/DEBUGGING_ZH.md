# SolidAttention / AI Infra 系统调试指南

## 先画清数据流

每个 decode layer 的关键路径是：

`NVMe SSD → 内核 I/O → DRAM staging buffer → PCIe H2D → VRAM → sparse attention`

同时还有非关键写回路径：

`新 KV（VRAM）→ PCIe D2H → DRAM 32KiB write buffer → NVMe SSD`

论文的三个重点分别位于：32-token block/interleaved KV（改善 SSD 粒度）、历史选择结果推测预取（提前 I/O）、SSD-aware scheduler（把 I/O 与 GPU microtask 重叠并复用同步点）。

## 看图时问四个问题

1. SSD read lane 是否有较长空洞？有则预取发得太晚或请求生成不足。
2. GPU compute lane 是否在 `top-k` 和 `attention` 之间等待？有则关键路径受 I/O 限制。
3. PCIe H2D 是否与 GPU projection/FFN 重叠？没有则调度过于串行。
4. SSD write 是否阻塞下一层？它应当是非关键任务；阻塞意味着同步点或 buffer 生命周期设计错误。

## 真实进程的分层工具

```bash
# 每秒查看 SSD 吞吐、队列长度、await、util
iostat -xz 1

# VRAM、GPU 利用率、功率（采样式，适合先找大问题）
nvidia-smi dmon -s pucvmet -d 1

# CPU、上下文切换和 major page fault
pidstat -rudw -p <PID> 1

# 打开生成的跨资源时间线
xdg-open artifacts/dashboard.html
```

有 CUDA 源码后，再用 Nsight Systems 捕获 CUDA API、kernel、memcpy 和 NVTX range：

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=cpu --output=artifacts/run ./solidattention ...
```

在 C++ 关键 task 周围加 `nvtxRangePushA("prefetch layer=7")` / `nvtxRangePop()`；I/O 提交和完成事件则写入相同的 Chrome Trace 时钟域。只有统一时间轴，才能判断“SSD 忙但 GPU 等待”究竟是带宽不够、请求粒度太小，还是依赖/同步写错。

## 指标到故障的映射

| 现象 | 先查 | 常见原因 |
|---|---|---|
| SSD `rMB/s` 低、`r/s` 高 | block size / `fio` 对照 | 细粒度随机 I/O |
| SSD `await` 高且 `%util≈100` | 队列深度、后台 I/O | 带宽或 IOPS 饱和 |
| GPU 利用率呈锯齿 | trace 中 attention 前空洞 | 预取错误或同步过多 |
| DRAM 高但 VRAM 低 | H2D lane、pinned memory | pageable staging 导致额外拷贝 |
| VRAM 稳步增长 | 每层 buffer lifetime | KV/临时 tensor 未释放或 stream 尚未完成 |
| P99 突增但均值正常 | block I/O tail、major fault | SSD 干扰、page cache、内存压力 |

## 文件阅读顺序

先读 `simulator.py` 的 `build_schedule()`：这里是任务 DAG。再读 `Resource.reserve()`：它模拟单资源串行与跨资源并行。接着读 `trace.py`：这里定义可观测性数据格式。最后读 `dashboard.py`：它只负责展示，不参与调度，因此不会污染被测逻辑。

真实 Qwen 路径按以下顺序读：

1. `qwen_baseline.py`：不改模型的官方 Transformers 对照组。
2. `qwen_kv_lab.py:capture_queries()`：用 hook 捕获各层 RoPE 后 Q，同时由 `DynamicCache` 保存真实 K/V。
3. `write_store()`：把 GPU KV 转为论文所述 token-major K/V interleave，并持久化到 SSD。
4. `read_blocks()`：用显式 offset 和 `pread` 读取被选 block；这里是替换成 `io_uring` 的边界。
5. `attention()`：GQA KV heads 展开后，在 GPU 上分别执行 dense reference 和 SSD-backed sparse attention。

`qwen_pipeline.py` 是下一层调度实现：

- `choose_blocks()`：每层根据真实 post-RoPE query 和 block representative 选块。
- `load()`：独立 `kv-prefetch` 线程执行 SSD `pread`。
- `replay(overlap=False)`：严格按 read → H2D → attention 串行，作为对照组。
- `replay(overlap=True)`：第 L 层 H2D 完成后，先提交 L+1 的 SSD read，再启动 L 的 GPU attention。

对照时间线：

```bash
./scripts/run_qwen_pipeline.sh --tokens 2048 --budget-tokens 512
xdg-open artifacts/qwen-pipeline-serial.html
xdg-open artifacts/qwen-pipeline-overlap.html
```

读图时，overlap 图中 `SSD read(layer=L+1)` 应与 `GPU compute(layer=L)` 横向相交。若没有相交，检查 future 提交位置、Python 线程是否真正在 `pread` 中释放 GIL，以及 GPU task 后是否过早调用全局 synchronize。

`qwen_decode.py` 则是真实连续出词路径。它不调用 replay 输出作为旁路指标，而是让 SSD-backed attention 的输出继续经过 `o_proj → residual → MLP → LM head`，所得 logits 决定下一个 token。`tails[layer]` 保存 decode 阶段新产生的 local KV；prompt KV 则保持在 SSD。`exact_token_prefix` 表示自由生成与 dense baseline 在分叉前连续一致多少 token，`mean_logits_cosine` 衡量更细粒度的数值差异。

V12 后，`LayerTail` 不再无限增长：它保存 local window 和不足一个 block
的 remainder；当最老 32 tokens 真正离开 local window 时，使用累计的局部
causal attention mass 生成 query-head representative，并写回主 KV store 的
per-layer reserved segment。时间线上应看到
`sealed decode block → main KV store`；它目前位于 attention 与 FFN 之间，
因此仍是 V13 要隐藏的关键路径任务。

`--cold-io` 会调用 `posix_fadvise(..., DONTNEED)`，尽量避免刚写入的数据直接从 Linux page cache 命中。但这是 hint，不等于具有严格保证的 direct I/O。严谨 SSD benchmark 下一步应实现 aligned `O_DIRECT`/`io_uring` 并同时观察块设备计数器。

## 本机驱动升级

本机当前 535 驱动可运行本项目锁定的 PyTorch 2.5.1+cu121。Ubuntu 当前推荐 595-open；升级会替换内核模块并需要重启：

```bash
sudo apt update
sudo apt install nvidia-driver-595-open
sudo reboot
```

重启后先验证 `nvidia-smi` 和 `.venv/bin/python -c 'import torch; print(torch.cuda.is_available())'`，再安装 CUDA toolkit。运行 PyTorch wheel 不要求系统安装 `nvcc`；只有编译自定义 CUDA kernel、FlashAttention 或 Nsight/CUPTI 实验才需要 toolkit。

## 复现边界

当前代码已经执行真实 Qwen3-0.6B BF16 和 Qwen3-8B-AWQ INT4 连续 decode，
记录 token/s、逐 token latency、exact prefix、logits cosine 和资源 trace；V10
full-SSD control 也验证了 FP16 KV 序列化与 AWQ 手工 decode 链路。

它仍不是论文官方复现：作者的 C++/CUDA/SYCL 源码、自定义 sparse kernel、
精确代表元参数和 benchmark harness 尚未公开。本仓库的 landmarks 是根据论文对
InfLLM 的描述推断的实现，V10 25% sparse 质量实验明确失败。正式对论文数字前还需
加入官方 baseline、真实长上下文数据集、重复实验/置信区间和 GPU-native selection。

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

## V13：判断 H2D 是否真的被 FFN 隐藏

V13 的 timeline 增加了三类关键事件：

- `pipeline SSD completion wait (worker)`：依赖线程等待 L+1 的 SSD→DRAM；
  它可以很长，但不能让主线程停住。
- `pipeline predicted H2D (L+1 during L FFN)`：copy stream 上的下一层完整预测
  buffer 搬运。
- `pipeline consume wait at L+1`：主线程进入下一层后等待 worker 提交完成；
  这是判断依赖是否仍在关键路径上的直接指标。

V13.0 的错误是把 `future.result()` 放在 attention L 后、FFN L 前。虽然 H2D
在 FFN 期间执行，SSD 的剩余等待却先阻塞了主线程，因此三次平均吞吐反而下降
8.07%。V13.1 由第二个 worker 等 SSD 并提交 H2D，主线程直接进入 FFN；V13.2
记录到 378 次 consumer wait 合计仅 0.359 ms。

在 dashboard 中逐 token、逐 layer 检查：

```text
Scheduler worker: SSD completion wait ─┐
PCIe H2D:                              └─ predicted L+1 ─────┐
GPU compute:          attention L ─────── FFN L ─────────────┼─ attention L+1
                                                            └─ correction
```

仓库的 80.25% overlap 是用 host submission timestamp 加 CUDA event duration
投到近似统一时间轴后求区间交集：

```text
sum(intersection(predicted_H2D, producer_FFN)) / sum(predicted_H2D)
```

这个数适合发现明显的调度空洞，但不是严格的 GPU 时钟相关结果。最终验证应使用
Nsight Systems 的 CUDA memcpy/kernel timeline；而 `pipeline consume wait`
接近零是当前更可靠的关键路径证据。可用脚本对不可变 metrics 做重复统计：

```bash
python scripts/summarize_version_runs.py \
  artifacts/runs/*V13.1-async-pipeline-repeat*-metrics.json
```

## 原生 C++/CUDA/SYCL/liburing 版本线

原生版本使用 `C0、C1…`，不继承 Python `V` 版本号。四层边界如下：

```text
C++ scheduler
  ├─ liburing: registered file + registered pinned buffers
  ├─ CUDA: stream/event/NVRTC kernels
  └─ SYCL: queue/event/USM kernels
```

C0 trace 的四条 lane 分别是 `SSD read`、`PCIe H2D`、`GPU compute` 和
`PCIe D2H`。其中 D2H 仅用于逐字节 CPU reference 审计；正式 decode 不应在
每个 block 后执行它。若 SSD read 报 `EINVAL`，依次检查文件是否用
`O_DIRECT` 打开、offset/长度是否 4 KiB 对齐，以及
`io_uring_prep_read_fixed` 是否传入注册时相同的虚拟地址。仅传 buffer index、
却把地址设为 null，同样会失败。

```bash
./scripts/run_cpp_c0.sh --operations 512
xdg-open artifacts/cpp-c0/c0-trace.json  # 推荐直接拖入 Perfetto
```

CUDA C0 使用 NVRTC 是为了在没有系统 `nvcc` 时仍执行真实 CUDA kernel。SYCL
文件与 CUDA 实现共享 `AcceleratorBackend`，但只有
`scripts/build_cpp_sycl.sh` 成功、设备 kernel 通过相同 CPU reference 后，才会
在版本表中从 build-pending 改为 implemented。

当前 `sycl-ls` 只枚举到 i9-14900HX 的 Intel OpenCL CPU。因此
`C0-SYCL-CPU` 的 USM copy 不是 PCIe/VRAM 搬运，不能与 CUDA H2D 横向比较。
接入 Codeplay NVIDIA plugin 后必须生成新的 `C0-SYCL-NVIDIA` 指标，不能覆盖
CPU 结果。

C1 开始执行真实 attention 方程。调试时先看质量，再看速度：

1. CPU、CUDA、SYCL 必须使用同一份 FP16 interleaved K/V 和 FP32 query。
2. 先检查 GQA 的 `query_head → kv_head` 映射，再检查 token-major K/V offset。
3. softmax 必须先减最大值；否则较大 query 很容易溢出。
4. `max_absolute_error` 与 cosine 同时过阈值后，才分析 kernel timeline。

C1 CUDA 每个 query head 目前只有一个 thread，2.263 ms 很慢是已知结构结果，
不是 CUDA 硬件比 CPU 慢。下一版会把 token/dimension reduction 映射到 warp。

另一个实验管理教训来自 C1-SYCL：DPC++ `-O2` 曾在编译器内部崩溃，而未启用
`set -e` 的临时外层 shell 继续复制了旧的四次运行 metrics。发布 artifact 前必须
校验其中的 `version`、`operations` 和 backend，不能只相信文件名。

C1.1 的 lane 映射为：

```text
query head → CUDA block / SYCL work-group
token 0..127 → lane 0..127 → Q·K
group max/sum reduction → stable softmax
dimension 0..127 → lane 0..127 → Σ(probability × V)
```

CUDA kernel 从 2.179 ms 降到 0.0400 ms 后，关键路径占比已经转向 SSD read
和 H2D。此时继续只优化 attention kernel 收益很小，应先复用 device buffer/event，
再把 SSD、H2D 和 compute 放进跨层 DAG。文档里的 `stage sum` 只是四个已插桩
区间之和，不含 allocation、driver API、QKV projection 和 FFN，不能称为
end-to-end decode latency。

C1.2 新增 `mean_device_call_wall_ms`，它包住完整 backend 调用。CUDA/SYCL
event 只度量设备队列中的 copy/kernel，观察不到以下成本：

- `cudaEventCreate/Destroy`；
- `sycl::malloc_device/free`；
- runtime/JIT 初始化；
- queue submission 与 host 等待；
- host output vector 的分配。

短实验必须显式 warmup。C1.2 保存 `warmup_operations=1`，而不是把首轮悄悄
删掉。结果显示 CUDA C1.1 已经复用了 device buffer，所以继续复用 event 只改善
1.96%；SYCL 移除每次 USM allocation 后，完整调用墙钟下降 51.04%。这类差异
正是只看 GPU event 会漏掉的 AI Infra 控制面瓶颈。

C1.3 将 attention output 留在 device，只每 16 次采样审计，并强制审计最后一次。
64 次运行应当准确出现 5 个 `C1.3 sampled output audit D2H`。看 trace 时检查：

```text
每次：SSD read → H2D → attention
采样：SSD read → H2D → attention → D2H audit
```

若非采样 operation 仍出现 D2H，说明上层接口无意中访问了 host output；若最后一次
没有 D2H，则质量指标可能来自旧结果。metrics 中的
`device_resident_output/audit_every/audit_operations` 必须与 trace 对得上。

“device-resident”目前只表示 buffer 生命周期正确，还没有证明下游依赖。下一步要在
同一 stream/queue 上提交 output consumer 或 `o_proj`，用 event dependency 证明它
读取的是本轮 attention 输出。

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

# C++ / CUDA / liburing / SYCL 版本总账

更新日期：2026-08-03。本文是原生版本线的快速入口；不可变原始数据位于
`artifacts/runs/`，逐版本设计与失败记录位于 `docs/native-pipeline/` 和
`docs/cpp-versions/`。Python V0–V13 是另一条版本线，不能混用编号或指标。

## 1. 基础算子与后端 C0–C2

| 版本 | 完成内容 | 核心结果 | 限制 |
|---|---|---|---|
| C0 CUDA | liburing fixed read、pinned DRAM、H2D、NVRTC kernel | SSD 0.105731 ms；H2D 0.015386 ms；512/512 正确 | 仅传输/变换 smoke test |
| C0 SYCL CPU | 同一 DAG 的 oneAPI USM/event | I/O 0.121729 ms；kernel 0.064103 ms | 是 CPU OpenCL，不是 NVIDIA SYCL |
| C1 | FP16 GQA sparse attention + CPU oracle | CUDA/SYCL max error 1.49e-8 | 初版 correctness kernel 很慢 |
| C1.1 | 128-lane reduction | CUDA 2.178537→0.039970 ms | 仍非论文 fused kernel |
| C1.2–C1.4 | 常驻资源、device output、下游 consumer | CUDA backend wall 0.084125 ms（C1.3） | C1.5/1.6 尚未实现 |
| C2.0 | 错误 representative smoke | cosine 0.456345 | 已判失败，不复用 |
| C2.1 | InfLLM 局部因果 score representative | selected/dense cosine 0.987693 | synthetic Q/K |

## 2. 调度管线 P0–P1.1

| 版本 | 关键变化 | 指标/结论 |
|---|---|---|
| P0 | L+1 SSD read 与 L attention；L+1 H2D 与 L FFN 重叠 | 0.301737→0.217985 ms/layer，1.383603x |
| P1.0 | 历史预测、miss-only read、局部 VRAM overwrite | hit 80.692%，346 miss，1.112365x |
| P1.1 | 接入 InfLLM selector 和 dense quality audit | hit 58.5938%，742 miss，0.849826x；负优化 |
| P1.3a | token/layer/generation/block-set ticket | 448/448 ticket 验证；0 stale；0.775908x，仍为负优化 |

P1.3a 的 QKV/FFN window 仍是 synthetic；它证明异步 slot 身份和 correction
正确性，不是 Qwen 端到端性能。

## 3. 真实 Qwen 单层到 28 层 P1.2a–P1.2h

| 版本 | 实现边界 | 关键结果 |
|---|---|---|
| P1.2a | 单个真实 Qwen3-0.6B layer，CUDA/cuBLAS 全算子 | final max error 8.64e-7 |
| P1.2b | 真实 FP16 KV 经 SSD/liburing 进入 sparse layer | sparse teacher error 1.80e-4 |
| P1.2c | 常驻 ring/slot，真实 layer replica overlap | exposed next read 0.000199 ms |
| P1.2d | 28 个不同层与独立权重/KV offset | 28/28 通过，max error 6.1035e-5 |
| P1.2e | 28 层 hidden recurrence | max error 9.1553e-5，cosine 1.0 |
| P1.2f | 单进程常驻 CUDA/cuBLAS/liburing | 816.670 ms；FP32 权重流式传输是主瓶颈 |
| P1.2g | 1.527 GB FP32 权重常驻 VRAM | 20.411 ms，约 40x 改善 |
| P1.2g.1 | 双 KV slot 跨层 pipeline | 16.179 ms，仍暴露 10.664 ms read wait |
| P1.2g.2 | 修正 production timer 边界 | 15.096 ms |
| P1.2g.3 | 14 MiB selected KV 全部预置 DRAM | 7.197 ms 上界，不是默认策略 |
| P1.2h | 有界 read-ahead 深度 2/4/8/16 | D16 10.167 ms、8 MiB pinned、2.911 ms exposed wait |

## 4. KV 生命周期 P1.3b

| 版本 | 输入/实现 | 结果 |
|---|---|---|
| P1.3b.0 | synthetic FP16；native 32-token seal/O_DIRECT/main store | 512 token；448/448 block 回读；tail=32 |
| P1.3b.1 | PyTorch 真实 Qwen post-RoPE FP16 KV | 448/448 byte-exact；fixture SHA 已记录 |
| P1.3b.2 | native RMSNorm/KV SGEMM/K-Norm/RoPE | 29,360,128 元素；cosine 0.999999999987；420/420 seal |

生命周期已经解决“tail 永久 torch.cat、旁路文件不可再次选择”的原始错误。
尚缺的是把连续 native decode 每轮产生的 KV 直接送进这一生命周期。

## 5. Native token feedback P1.3c

| 版本 | 新闭环 | 结果 |
|---|---|---|
| P1.3c.0 | final RMSNorm + 151,936-way LM head | token 50；teacher top-5 完全一致；LM 2.697 ms |
| P1.3c.1 | token 50 native embedding 回灌 position 512 | 第二 token 271；两步 logits cosine >0.9999999999999 |
| P1.3c.2 | 每层 current-token K/V 加入 attention | 128+1 token；57,344 FP16 pack 0 mismatch；step-2 chain 10.782 ms |

P1.3c.2 仍只有 current token，没有跨轮保存 1–32 token tail；prompt dynamic block
仍使用固定 plan。P1.3c.3 已补齐跨轮 1–32 token tail，并通过
160-token attention 的 31→32 容量边界；尚未在第 32 token 后在线封块。

## 6. 当前真实性边界

已经真实：Qwen3-0.6B 权重和 hidden、CUDA/cuBLAS layer math、FP16 KV、
liburing/O_DIRECT、pinned DRAM/VRAM 双缓冲、native embedding/LM head、主 store
封块回读、generation ticket。

尚未完成：完整 1–32 decode tail、tail 满 32 后在线封块并进入 selection、真实
连续 token 的 InfLLM selection/correction、packed AWQ/INT4 native kernel、论文的
定制 CUDA/SYCL fused sparse kernel、Qwen2.5-7B/LongBench/OpenCompass/1k–128k
正式实验。

因此仓库仍是逐边界可审计复现，不宣称已达到 FAST'26 论文端到端性能。

## 7. 主要文件怎么读

| 文件 | 作用 |
|---|---|
| `cpp/src/qwen_layer_main.cpp` | CUDA/NVRTC/cuBLAS Qwen 算子与 FP16 sparse attention |
| `cpp/src/qwen_chain_main.cpp` | 28 层常驻权重、read-ahead、hidden recurrence、current KV |
| `cpp/src/pipeline_main.cpp` | history prediction/correction 和 generation ticket 调度器 |
| `cpp/src/kv_lifecycle_main.cpp` | tail、representative index、seal、主 store 写入/回读 |
| `cpp/src/qwen_lm_head_main.cpp` | embedding lookup、final norm、LM head、argmax |
| `cpp/src/uring_reader.cpp` | O_DIRECT + registered-file/fixed-buffer liburing reader |
| `scripts/run_cpp_p1_3*.py` | 正式重复实验、teacher audit、不可变 metrics 发布 |
| `artifacts/runs/*-trace.json` | Perfetto 的 SSD/DRAM/H2D/GPU/host-wait 泳道 |

完整逐版本细节见 [CPP_VERSIONS.md](CPP_VERSIONS.md)。

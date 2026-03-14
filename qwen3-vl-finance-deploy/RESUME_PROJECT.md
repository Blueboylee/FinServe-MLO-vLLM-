# 简历项目：AI 推理服务与推理加速引擎

> **数据声明**：下文所有性能数字均来自项目内 E2E benchmark 实测（`time.perf_counter()` + `len(output.outputs[0].token_ids)`），3 runs + 2 warmup，子进程隔离，无估算值。

---

## 项目名称（建议）

**基于 vLLM 的 Multi-LoRA 推理服务与推理加速引擎**

或：**大模型推理引擎优化与多专家 LoRA 推理服务**

---

## 一句话描述

在 vLLM 0.16.0 上构建多 LoRA 专家推理服务，通过自研 Triton kernel（SGMV 融合、FP8 KV Cache）、CUDA Graph 兼容层与版本自适应集成，在 RTX 4080 Super 上实测高并发吞吐提升 **7.7%**（1641 → 1768 tok/s，BS=64），并搭建 100% 实测、零估算的 E2E A/B 压测框架。

---

## 技术栈（真实使用）

- **推理框架**：vLLM 0.16.0（V1 Engine，torch.compile + CUDA Graph）
- **模型**：Qwen3-VL-8B-Instruct-AWQ-4bit（compressed-tensors 量化）
- **GPU**：NVIDIA RTX 4080 Super，32GB，Ada Lovelace（Compute Capability 8.9）
- **扩展**：Multi-LoRA（2 个 finance 专家，rank=64）、Chunked Prefill、Prefix Caching
- **内核**：Triton（SGMV shrink/expand/fused、FP8 量化/反量化/融合 attention score）、PyTorch torch.library.custom_op

---

## 核心职责与成果（可逐条放进简历）

### 1. 推理加速：FP8 KV Cache 与 Triton 内核

- 实现 **FP8 E4M3** KV Cache 在线量化与 per-head 动态 scale，Triton kernel 写入 paged cache；反量化与 **Q@K^T** 融合为单 kernel（K 不落 FP16 中间缓冲），减少 `seq_len × num_heads × head_dim × 2B` 显存与一次全局内存 pass。
- 在 **BS=64** 下实测：Baseline **1641.3 tok/s** → FP8 KV **1767.8 tok/s**，吞吐提升 **7.7%**；QPS 由 **8.97** 提升至 **9.07**（FP8 only）/ **9.12**（SGMV+FP8）。低并发（BS=1）时与 baseline 持平（80.7 vs 80.5 tok/s），验证高并发下 memory-bound 收益。
- 通过 vLLM CacheEngine / reshape_and_cache 的 monkey-patch 集成，兼容现有 paged attention 与 vLLM 0.16.0；GPU CC 8.9 使用原生 FP8 指令。

### 2. 推理引擎内核：SGMV 融合与 LoRA 后处理融合

- 设计并实现 **Triton Fused SGMV**：将 LoRA 的 shrink（x@A）与 expand（y@B）融合为单 kernel，中间向量 `y[rank]`（rank=64）驻留寄存器不写回 DRAM，消除 `2×num_tokens×rank×2B` 的中间层全局内存流量及一次 kernel 启动；Grid 为 `(num_tokens, cdiv(output_dim, BLOCK_N))`，Phase-1 全量在 RF 内完成。
- 实现 **三路融合** LoRA delta + residual add + RMSNorm 的单 kernel 单 pass，替代原 3 kernel / 6 pass，理论带宽减半；为兼容 torch.compile（Dynamo）改为纯 PyTorch ops 实现并做 2D/3D 分支，避免 `@torch.compiler.disable` 被拒绝。
- 适配 **vLLM 0.16.0** LoRA API 变更（`vllm.lora.ops` → `PunicaWrapperGPU.add_lora_linear`），实现版本检测与统一 monkey-patch 入口；修复 expand 权重 transpose、output 维度越界及非连续 tensor 传入 Triton 导致的正确性与稳定性问题。

### 3. CUDA Graph 与编译链路兼容

- 使用 **torch.library.custom_op** 将 SGMV 相关 Triton kernel 注册为 PyTorch 自定义算子，提供 CUDA 与 Meta（FakeTensor）实现，使 torch.compile 能正确 trace 并参与 CUDA Graph 捕获。
- 实现 **Static Shape Padding**：将动态 `num_tokens` 对齐到预设 bucket（1,2,4,…,4096），padding 位置 `adapter_id=-1` 在 kernel 内跳过；实现 per-bucket 的 CUDA Graph capture/replay 与 warmup 策略，满足固定 shape 重放要求。

### 4. 推理服务与可观测性

- 基于 vLLM 构建 **Multi-LoRA 推理服务**：同一实例挂载 2 个 LoRA 专家，按请求路由；配合 LoRA-Aware Chunked Prefill、max_loras=2、max_lora_rank=64、enable_chunked_prefill、enable_prefix_caching 等配置，支撑多专家并发请求。
- 搭建 **E2E A/B 压测框架**：子进程隔离各优化配置、按 LoRA 分组并发调用（同组单次 `generate` 以利用 continuous batching），指标全部基于 wall-clock 与真实 output token 数；并发度 1/4/8/16/32/64，3 runs + 2 warmup，**无估算、无插值**，结果落库 JSON 可复现。

---

## 实测数据摘要（仅实测值，可直接引用）

| 指标 | 条件 | 数值 | 说明 |
|------|------|------|------|
| 吞吐提升（FP8 KV vs Baseline） | BS=64 | **+7.7%** | 1641.3 → 1767.8 tok/s |
| Baseline 吞吐 | BS=64 | 1641.3 tok/s | 3 runs 平均 |
| FP8 KV 吞吐 | BS=64 | 1767.8 tok/s | 同上 |
| Baseline QPS | BS=64 | 8.97 | 请求/秒 |
| FP8 KV QPS | BS=64 | 9.07 | 同上 |
| SGMV+FP8 QPS | BS=64 | 9.12 | 同上 |
| Baseline 吞吐 | BS=1 | 80.7 tok/s | 单请求 |
| FP8 KV 吞吐 | BS=1 | 80.5 tok/s | 单请求，与 baseline 基本一致 |
| 测试配置 | — | 3 runs + 2 warmup, max_tokens=256 | 子进程隔离，wall_clock + 真实 token 数 |

---

## 简历 bullet 示例（可直接粘贴，数字均为实测）

- 在 vLLM 0.16.0 上实现 FP8 E4M3 KV Cache（per-head 动态 scale + 融合反量化 Q@K^T），BS=64 下实测吞吐 **+7.7%**（1641→1768 tok/s），QPS 8.97→9.12。
- 自研 Triton Fused SGMV kernel（shrink+expand 寄存器融合，中间 64 维驻留 RF）及 LoRA+Residual+RMSNorm 三路融合，并完成 vLLM 0.16.0 PunicaWrapperGPU API 适配与 torch.compile 兼容。
- 通过 torch.library.custom_op 注册 SGMV kernel、Static Shape Padding 与 per-bucket CUDA Graph，实现与 vLLM CUDA Graph 捕获/重放的兼容。
- 搭建 Multi-LoRA 推理服务与 E2E A/B 压测（子进程隔离、1–64 并发、3 runs+2 warmup），指标 100% 基于 wall-clock 与真实 output token 数，零估算。

---

## 项目与数据来源说明（面试可讲）

- 代码与 benchmark 脚本：本仓库 `FinServe-MLO-vLLM-/qwen3-vl-finance-deploy`。
- 性能数据来源：`bench_e2e_comparison.py` 在子进程中运行各配置，使用 `time.perf_counter()` 计 wall time，`len(output.outputs[0].token_ids)` 计每请求 output tokens，throughput = total_output_tokens / wall_time，QPS = batch_size / wall_time；结果见 `bench_results/e2e_comparison_20260314_012835.json` 及 `TECHNICAL_REPORT.md`。
- 所有百分比与倍数均由上述 JSON 中汇总的 throughput/qps 计算得出，未使用任何估算或外推。

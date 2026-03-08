# 优化前 / 优化后 怎么跑

## 做了哪些优化（服务端实际开关）

| 优化 | 未优化时 | 优化后 | 作用 |
|------|----------|--------|------|
| **Chunked Prefill** | 关 | 开 | 长 Prefill 拆成小块与 Decode 交替跑，避免一个长请求把后面短请求堵死（队头阻塞） |
| **Prefix Caching** | 关 | 开 | 相同前缀只算一次 KV 并复用，相同/相似 prompt 的请求更快 |
| **LoRA Reorder** | 关 | 开 | 等待队列按 adapter 聚批，同一批尽量用同一 LoRA，减少切换 |

未优化 = 用 `serve_multi_lora_unopt.sh` 或设 `ENABLE_CHUNKED_PREFILL=false`、`ENABLE_PREFIX_CACHING=false`、`FINSERVE_LORA_REORDER=0` 再起服务。

**关于 SGMV**：本仓库里有 **`sgmv_lora_triton.py`**，用 Triton 实现了「多专家 LoRA 在一个 kernel 里做 X@B@A」的 SGMV 风格计算。vLLM 服务未接入，但可用 **`bench_sgmv_compare.py`** 做「使用 SGMV 前 vs 使用 SGMV 后」的压测对比（纯 LoRA 矩阵乘层，不经过 vLLM）：

```bash
python bench_sgmv_compare.py --batch 256 --iterations 500
# 可调: --in-dim 4096 --r 64 --out-dim 4096 --num-adapters 3
```

**为什么「全相同长 prompt」压测时差异很小？**  
- 全是同一种长 prompt：没有「长短混合」，Chunked Prefill 的「长不堵短」用不上。  
- 只打一个 expert：没有多 adapter 混合，LoRA Reorder 用不上。  
- 总时间里 decode 占大头时，prefill 优化在整体数字里被摊薄。

**要看出明显差异**：用 `--mixed`（50% 短 + 50% 长 prompt），或同时打 expert-a 和 expert-b。

---

## 一、服务端：两种启动方式

### 优化后（默认，推荐日常使用）

```bash
cd /root/下载/FinServe-MLO-vLLM-/qwen3-vl-finance-deploy
conda activate qwen3-vllm
bash serve_multi_lora.sh
```

- 开启：Chunked Prefill、Prefix Caching、LoRA-Aware Scheduler
- 服务地址：`http://127.0.0.1:8000`

---

### 优化前（未优化，仅用于对比压测）

```bash
cd /root/下载/FinServe-MLO-vLLM-/qwen3-vl-finance-deploy
conda activate qwen3-vllm
bash serve_multi_lora_unopt.sh
```

- 关闭：Chunked Prefill、Prefix Caching、LoRA Reorder
- 同样监听 8000 端口

**注意**：同一时间只能跑一个服务（都占 8000）。要对比时，先跑「优化前」压测完再停掉，再起「优化后」压测；或直接用下面的自动对比脚本。

---

## 二、压测：三种用法

### 1）只压测当前服务（不区分优化前/后）

当前已经起了一个服务（优化前或优化后都行），只关心这一种配置下的指标：

```bash
python bench_load_test.py --total-requests 200 --concurrency 32
```

---

### 2）自动对比「优化前 vs 优化后」（推荐）

脚本会**自动**：先起未优化服务 → 压测 → 停掉 → 再起优化后服务 → 压测 → 停掉 → 打印对比表。

```bash
python bench_optimization_compare.py
# 可加参数，例如：
python bench_optimization_compare.py --total-requests 300 --concurrency 64
```

无需手动切换服务，跑完就能看到「未优化 vs 优化后」的 TTFT、Latency、Throughput 对比。

---

### 3）同一服务下对比「Prefix Cache 命中 vs 冷缓存」

服务必须是**优化后**（`serve_multi_lora.sh`），且已开启 Prefix Caching。脚本在同一服务上跑两轮：

- **优化后（命中）**：所有请求用**相同** prompt → Prefix Cache 命中
- **优化前（冷）**：每个请求**不同** prompt → 无缓存复用

```bash
# 先启动优化后服务
bash serve_multi_lora.sh

# 另开终端，跑对比
python bench_load_test.py --compare --rounds 5 --total-requests 1000 --concurrency 160
```

这里「优化前/后」指的是**请求模式**（冷 vs 命中），不是服务配置。

---

## 三、对照表

| 你想做的事 | 怎么跑 |
|------------|--------|
| 日常用服务 | `bash serve_multi_lora.sh` |
| 看「未优化服务」的指标 | `bash serve_multi_lora_unopt.sh`，再 `python bench_load_test.py ...` |
| 看「优化后服务」的指标 | `bash serve_multi_lora.sh`，再 `python bench_load_test.py ...` |
| 一次跑出「未优化 vs 优化后」对比 | 只跑 `python bench_optimization_compare.py`（自动启停两次服务） |
| 对比时想看出明显优化效果 | 加 `--mixed`：`python bench_optimization_compare.py --mixed`（50% 短 + 50% 长，体现 Chunked Prefill） |
| 看「Prefix Cache 命中 vs 冷」对比 | 先 `bash serve_multi_lora.sh`，再 `python bench_load_test.py --compare ...` |

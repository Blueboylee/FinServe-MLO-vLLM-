# FinServe LoRA-Aware Scheduler Plugin

vLLM 插件：**多专家感知的分块预填充调度器 (LoRA-Aware Chunked Prefill Scheduler)**

在 Multi-LoRA serving 场景下，对 vLLM Scheduler 的 waiting queue 执行 LoRA 亲和性重排，
使同一 batch 优先聚合相同 adapter 的请求，减少 LoRA 权重频繁切换的开销。

## 原理

```
默认 FCFS 调度:                     LoRA-Aware 调度:
┌──────────────────────┐           ┌──────────────────────┐
│ Batch #1             │           │ Batch #1             │
│  req1 → Expert-A     │           │  req1 → Expert-A     │
│  req2 → Expert-B  ←切换│         │  req3 → Expert-A     │
│  req3 → Expert-A  ←切换│         │  req5 → Expert-A     │ ← 同adapter聚合
│  req4 → Expert-B  ←切换│         │                      │
│                      │           │ Batch #2             │
│ 4次adapter切换       │           │  req2 → Expert-B     │
│ BGMV kernel无法批量  │           │  req4 → Expert-B     │ ← 同adapter聚合
└──────────────────────┘           │                      │
                                   │ 0次batch内切换       │
                                   │ BGMV kernel高效批量  │
                                   └──────────────────────┘
```

## 技术实现

- **不 Fork vLLM**：通过 `vllm.general_plugins` 入口点注入，pip 安装即生效
- **进程安全**：Plugin 在所有 vLLM 进程（主进程/EngineCore/Worker）启动前加载
- **零侵入**：仅在 `Scheduler.schedule()` 前增加一步 waiting queue 重排
- **可降级**：如果重排失败，自动 fallback 到默认 FCFS 行为
- **可关闭**：设置 `FINSERVE_LORA_REORDER=0` 完全禁用

## 安装

```bash
# 在 vLLM 环境中安装（开发模式）
pip install -e .

# 验证安装
python -c "import finserve_lora_scheduler; print('OK')"
```

## 使用

安装后，启动 vLLM 时 Plugin 自动加载，无需额外参数：

```bash
# 直接用 serve_multi_lora.sh（已集成自动安装）
bash serve_multi_lora.sh

# 或手动启动
vllm serve ./models/Qwen3-VL-8B-Instruct-AWQ-4bit \
    --enable-lora --max-loras 2 \
    --enable-chunked-prefill --max-num-batched-tokens 512
```

启动日志中会看到：
```
FinServe LoRA-Aware Scheduler Plugin Initializing
LoRA-Aware Chunked Prefill patch applied successfully. Config: max_wait=10.0s, group_cap=0, enabled=True
```

## 配置（环境变量）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FINSERVE_LORA_REORDER` | `1` | 启用/禁用重排。设为 `0` 禁用 |
| `FINSERVE_LORA_MAX_WAIT_SEC` | `10` | 反饥饿阈值（秒）。等待超过此时间的请求无条件置顶 |
| `FINSERVE_LORA_GROUP_CAP` | `0` | 同一 adapter 连续调度上限。`0` = 不限制 |

### 调参建议

- **高并发场景**（QPS > 50）：建议 `MAX_WAIT_SEC=5`，降低尾延迟
- **少量专家**（2-3 个）：`GROUP_CAP=0` 即可，不需要限制
- **大量专家**（10+ 个）：建议 `GROUP_CAP=8`，防止单一 adapter 霸占 batch

## 调度算法

每次 `schedule()` 调用前，对 waiting queue 执行稳定排序：

```
排序键 = (饥饿优先级, GPU亲和优先级, LoRA分组ID, 原始FCFS序号)

饥饿优先级:  0 = 等待超过阈值（强制置顶）  1 = 正常
GPU亲和:    0 = adapter 已在 running 请求中（大概率在GPU buffer上）
            1 = adapter 不在 GPU 上
            0 = 无 LoRA 请求（中性）
LoRA分组ID: 相同 adapter 的请求排在一起
FCFS序号:   组内保持原始到达顺序
```

## 与现有系统的关系

```
┌─ web_proxy_server.py ─────────────────────┐
│  RequestPriorityScheduler (代理层)         │
│  控制哪些请求被允许发给 vLLM              │
│  Expert 信号量 + Base 快速通道            │
└────────────────┬──────────────────────────┘
                 │
┌─ vLLM Engine ──┴──────────────────────────┐
│  LoRA-Aware Scheduler (本 Plugin)         │
│  控制 GPU batch 内部如何组织请求           │
│  同 adapter 请求聚合 → BGMV 高效批量计算  │
├───────────────────────────────────────────┤
│  Chunked Prefill (vLLM 原生)              │
│  512 token 分块 → Decode 可在块间穿插     │
├───────────────────────────────────────────┤
│  Prefix Caching (vLLM 原生)               │
│  System prompt KV Cache 复用              │
└───────────────────────────────────────────┘
```

## 兼容性

- vLLM >= 0.8.0（需要 V1 Engine 的 `SchedulerInterface`）
- Python >= 3.10
- 与 Chunked Prefill、Prefix Caching、PD 分离架构完全兼容
- 升级 vLLM：`pip install --upgrade vllm` 后测试 Plugin 即可

# FinServe-MLO-vLLM

使用 vLLM 部署 **Qwen2.5-32B** 基座，并挂载两个 QLoRA 微调专家（专家 A、专家 B），支持按需选择基座或任一专家进行推理。

## 环境要求

- Python 3.10+
- CUDA（vLLM 需 GPU）
- 足够显存：Qwen2.5-32B 建议 24GB+，可按需使用 `--tensor-parallel-size` 多卡

## 安装

```bash
pip install -r requirements.txt
```

## 启动

```bash
# 1. 创建环境（Python 3.10 或 3.11，按你本机 CUDA 选）
conda create -n finserve python=3.10 -y

# 2. 激活
conda activate finserve

# 3. 若用 GPU，在 conda 里装 CUDA 驱动（可选，若系统已有可跳过）
# conda install cuda -c nvidia

# 4. 进项目目录，装依赖
cd /Users/lixingchen/Documents/GitHub/FinServe-MLO-vLLM-
pip install -r requirements.txt

# 5. 按之前的启动顺序来
python scripts/download_experts.py
python -m server.run_grpc_server
# 另开终端
conda activate finserve
python -m gateway.run_gateway
```

## 步骤一：下载专家 LoRA 模型

从 ModelScope 下载两个专家适配器，并生成 `lora_paths.json` 供 vLLM 使用：

```bash
# 使用脚本（推荐）
python scripts/download_experts.py

# 或手动使用 ModelScope CLI 下载后，在项目根目录创建 lora_paths.json，内容示例：
# {"expert-a": "/path/to/qwen25-32b-expert-a-qlora", "expert-b": "/path/to/qwen25-32b-expert-b-qlora"}
```

手动下载示例：

```bash
modelscope download --model GaryLeenene/qwen25-32b-expert-a-qlora
modelscope download --model GaryLeenene/qwen25-32b-expert-b-qlora
```

下载完成后，将两个模型所在目录路径填入项目根目录的 `lora_paths.json`（若用脚本则自动生成）。

## 步骤二：启动 vLLM 服务

在项目根目录执行：

```bash
python scripts/run_serve.py
```

默认行为：

- **基座模型**：`Qwen/Qwen2.5-32B-Instruct`（HuggingFace，首次会自动下载）
- **LoRA**：从 `lora_paths.json` 读取 `expert-a`、`expert-b` 路径并挂载
- **服务地址**：`http://0.0.0.0:8000`

常用参数：

```bash
# 指定基座（本地或 HF 模型 ID）
python scripts/run_serve.py --base-model /path/to/Qwen2.5-32B-Instruct

# 多卡张量并行
python scripts/run_serve.py --tensor-parallel-size 2

# 修改端口 / LoRA rank
python scripts/run_serve.py --port 8001 --max-lora-rank 64
```

## 步骤三：推理

服务启动后，可通过请求中的 `model` 选择**基座**或**专家**：

- 基座：`model="Qwen/Qwen2.5-32B-Instruct"`
- 专家 A：`model="expert-a"`
- 专家 B：`model="expert-b"`

### 使用 curl

```bash
# 专家 A
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "expert-a", "prompt": "你好", "max_tokens": 128}'

# 专家 B
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "expert-b", "prompt": "你好", "max_tokens": 128}'
```

### 使用 OpenAI 兼容客户端

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# 专家 A
r = client.completions.create(model="expert-a", prompt="你好", max_tokens=128)
print(r.choices[0].text)

# 专家 B
r = client.completions.create(model="expert-b", prompt="你好", max_tokens=128)
print(r.choices[0].text)
```

## 项目结构

```
FinServe-MLO-vLLM-/
├── requirements.txt          # 依赖
├── lora_paths.json           # 专家路径配置（由 download_experts 生成）
├── lora_loader.py            # 多 LoRA 挂载：路径 → LoRARequest
├── README.md
├── grpc_gen/                 # gRPC 生成代码与 proto
│   └── proto/
│       ├── chat.proto        # StreamChat 定义
│       ├── chat_pb2.py       # 消息类
│       └── chat_pb2_grpc.py  # Servicer/Stub 桩代码
├── server/                   # 内网 gRPC 服务（The Internal Server）
│   ├── stream_bridge.py      # 流式响应桥接：vLLM AsyncIterator → gRPC StreamChatChunk
│   ├── scheduler.py          # 业务感知调度：按 expert_id 聚类的异步优先级队列
│   ├── rate_limiter.py       # 请求拦截：并发限制（Rate Limiting）
│   ├── internal_servicer.py  # StreamChat Servicer（接入调度层 + 限流 + 流式截断）
│   ├── run_grpc_server.py    # 服务端启动脚本（监听 + CLI + 调度/限流参数）
│   └── grpc_client_example.py # 客户端示例
├── gateway/                  # 对外协议转换（Web 调用）
│   ├── app.py                # FastAPI 应用：/v1/chat/finance + 协议转换与流映射
│   ├── grpc_pool.py          # gRPC 异步连接池
│   ├── router.py             # 关键词 → expert_id 业务路由
│   └── run_gateway.py        # 网关启动脚本
└── scripts/
    ├── download_experts.py   # 下载专家 LoRA 并生成 lora_paths.json
    ├── run_serve.py          # 启动 vLLM 服务（基座 + 双 LoRA）
    └── async_inference.py    # AsyncLLMEngine 异步推理（按 expert_id 选 LoRA）
```

## 多 LoRA 挂载（路径 → LoRARequest）

`lora_loader.py` 将微调好的 LoRA 文件夹路径映射为 vLLM 可识别的 `LoRARequest` 对象：

- **`load_lora_paths(config_path)`**：从 `lora_paths.json` 加载 `expert_id -> 本地路径`。
- **`get_lora_request(expert_id, lora_paths=None, config_path=None)`**：根据 `expert_id` 返回 `LoRARequest`；`expert_id` 为 `None` 或 `"base"` 时返回 `None`（使用基座）。
- **`list_expert_ids(config_path)`**：返回当前配置中所有 expert_id。

示例：

```python
from lora_loader import get_lora_request, load_lora_paths, list_expert_ids

# 获取专家 A 的 LoRARequest（供 vLLM 使用）
lora_req = get_lora_request("expert-a")  # 或 expert_id=None 表示基座

# 列出所有已配置专家
print(list_expert_ids())
```

## 异步推理（AsyncLLMEngine + expert_id）

`scripts/async_inference.py` 基于 **AsyncLLMEngine** 封装了按 `expert_id` 选择 LoRA 的异步生成逻辑（不依赖 HTTP 服务）：

- **`create_engine(...)`**：创建已启用 LoRA 的 `AsyncLLMEngine`。
- **`generate_with_expert(engine, prompt, sampling_params, expert_id=..., ...)`**：**异步生成器**，根据 `expert_id` 挂载对应 LoRA，逐次 yield `RequestOutput`。
- **`generate_with_expert_text(...)`**：同上，但收集完整输出并返回字符串。

示例（在项目根目录运行）：

```bash
# 使用专家 A 生成（默认）
python scripts/async_inference.py "你的问题" --expert-id expert-a

# 使用专家 B
python scripts/async_inference.py "你的问题" --expert-id expert-b

# 使用基座
python scripts/async_inference.py "你的问题" --expert-id base

# 流式输出
python scripts/async_inference.py "你的问题" --expert-id expert-a --stream
```

在代码中复用：

```python
import asyncio
from scripts.async_inference import create_engine, generate_with_expert, generate_with_expert_text
from vllm import SamplingParams

async def run():
    engine = create_engine(model="Qwen/Qwen2.5-32B-Instruct")
    sp = SamplingParams(max_tokens=128, temperature=0.7)
    # 异步生成器，按 expert_id 调用对应 LoRA
    async for out in generate_with_expert(engine, "你好", sp, expert_id="expert-a"):
        if out.outputs:
            print(out.outputs[0].text, end="", flush=True)
            if out.finished:
                break
    # 或直接取完整文本
    text = await generate_with_expert_text(engine, "你好", expert_id="expert-b")
    print(text)

asyncio.run(run())
```

## 内网 gRPC 服务（The Internal Server）

将推理引擎包装成内网可调用的 gRPC 服务，提供 **StreamChat** 流式 RPC。

### Proto 与桩代码

- **`grpc_gen/proto/chat.proto`**：定义 `StreamChatRequest`（prompt、expert_id、max_tokens、temperature）与 `StreamChatChunk`（text_delta、finished），以及服务 `InternalChat.StreamChat`。
- **`grpc_gen/proto/chat_pb2_grpc.py`**：生成的 Servicer 基类 `InternalChatServicer` 与 Stub `InternalChatStub`，以及 `add_InternalChatServicer_to_server`。

### gRPC Servicer 实现（StreamChat 逻辑）

**`server/internal_servicer.py`** 中 **`InternalChatServicerImpl`** 继承生成的 **`InternalChatServicer`**，在服务端内接入 **业务感知调度层**（Infra 优化）：

- **请求进入**：先经 **RateLimiter** 限制并发（`acquire`），再构造 **StreamChatJob**（含 expert_id、prompt、sampling_params、response_queue、cancelled_ev）。
- **流式截断**：`context.add_done_callback(cancelled_ev.set)`，RPC 断开时置位；worker 内检测到后中止推理并向 response_queue 放入 SENTINEL，调用方从 `consume_response_queue_until_sentinel` 正常退出。
- **调度**：job 放入 **ExpertClusterQueue**（按 expert_id 聚类），由 **worker** 从队列取 job、调用 vLLM、通过 **stream_bridge** 将 chunk 写入 response_queue；Servicer 从 response_queue 读取并 yield 给 gRPC 客户端。
- **退出**：`finally` 中 `rate_limiter.release(expert_id)`。

### 流式响应桥接（vLLM → gRPC）

**`server/stream_bridge.py`** 将 vLLM 的异步迭代器与 gRPC 的 yield 流式输出打通：

- **`bridge_vllm_to_grpc_stream(vllm_output_stream)`**：入参为 `AsyncIterator[RequestOutput]`（即 `generate_with_expert` 的返回值），返回 `AsyncIterator[StreamChatChunk]`。
- 每个 `RequestOutput` 转为 `StreamChatChunk(text_delta=..., finished=...)` 并 yield，Servicer 中直接 `async for chunk in bridge_vllm_to_grpc_stream(...): yield chunk`。

### 服务端启动脚本（监听逻辑与 CLI）

**`server/run_grpc_server.py`** 实现服务的监听逻辑，并启动 **业务感知调度层**（队列 + worker + 限流），支持命令行配置：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--host` | `0.0.0.0` | 监听 host |
| `--port` | `50051` | 监听 port |
| `--bind` | - | 直接指定地址（覆盖 host/port），如 `0.0.0.0:50051` |
| `--model` | `Qwen/Qwen2.5-32B-Instruct` | 基座模型 ID 或本地路径 |
| `--lora-config` | 项目根 `lora_paths.json` | LoRA 路径配置 |
| `--max-lora-rank` | `64` | LoRA 最大 rank |
| `--max-loras` | `2` | 最大同时 LoRA 数 |
| `--max-cpu-loras` | `2` | CPU 侧 LoRA 缓存数 |
| **`--max-concurrent`** | **`8`** | **全局最大并发 StreamChat 数（Rate Limiting）** |
| **`--per-expert-limit`** | - | **每个 expert_id 最大并发数，不设则不限制** |
| **`--workers`** | **`1`** | **调度层 worker 数量（从队列取 job 并调用 vLLM）** |

环境变量：`GRPC_HOST`、`GRPC_PORT`、`VLLM_BASE_MODEL` 可覆盖默认。启动后监听指定地址，收到 **SIGTERM/SIGINT** 时优雅关闭（取消 worker、`server.stop(grace=5)` 后退出）。

---

## 第四阶段：业务感知调度层（Infra 优化）

在 gRPC 服务端内加入“Infra 优化”逻辑：**异步优先级队列**（按 expert_id 聚类）与 **请求拦截与预处理**（流式截断 + 并发限制）。

### 9. 异步优先级队列（请求聚类）

**`server/scheduler.py`**

- **ExpertClusterQueue**：在 gRPC 进入 vLLM 前，根据 **expert_id** 做简单请求聚类。
  - **put(job)**：按 `job.expert_id` 分片到子队列（基座用 `_base_`），并 `notify` 等待中的 `get()`。
  - **get()**：优先返回 **当前正在服务的 expert** 的请求（聚类），否则从任意非空子队列取一个并更新 `_current_expert`；无请求时阻塞直到有 `put()`。
- **StreamChatJob**：调度单元，包含 expert_id、prompt、sampling_params、response_queue、cancelled_ev、request_id。
- **run_worker(scheduler, engine, lora_paths)**：单 worker 循环，`await scheduler.get()` 取 job，调用 `generate_with_expert` + `bridge_vllm_to_grpc_stream`，将 chunk 写入 `job.response_queue`；若 `job.cancelled_ev.is_set()` 则中止并仍放入 **STREAM_SENTINEL**，保证 Servicer 侧能退出。

效果：同一 expert 的请求尽量连续被 worker 处理，减少 vLLM 侧 LoRA 切换，提升吞吐。

### 10. 请求拦截与预处理

**流式截断（RPC 断开时取消推理）**

- Servicer 为每个请求创建 **cancelled_ev**，并 **context.add_done_callback(cancelled_ev.set)**；客户端断开或 RPC 结束时回调执行，worker 内 `async for chunk in ...` 时检测 `cancelled_ev.is_set()` 即 break，并在 finally 中向 response_queue 放入 **STREAM_SENTINEL**。
- Servicer 侧通过 **consume_response_queue_until_sentinel(response_queue, SENTINEL, cancelled_ev)** 读取 chunk 并 yield；若已置位 cancelled_ev 或收到 SENTINEL 则结束迭代，实现流式截断。

**并发限制（Rate Limiting）**

- **`server/rate_limiter.py`** 中 **RateLimiter**：
  - **max_concurrent**：全局最大并发 StreamChat 数，使用 **asyncio.Semaphore**，超过则 **acquire()** 阻塞直到有空位。
  - **per_expert_limit**（可选）：每个 expert_id 单独一个 Semaphore，限制该 expert 的并发。
- Servicer 在进入推理前 **await rate_limiter.acquire(expert_id)**，在 **finally** 中 **rate_limiter.release(expert_id)**。

启动时通过 **--max-concurrent**、**--per-expert-limit**、**--workers** 配置限流与 worker 数。

### 启动 gRPC 服务端

在项目根目录执行：

```bash
python -m server.run_grpc_server
```

指定端口、模型路径、调度与限流：

```bash
python -m server.run_grpc_server --port 50052 --model /path/to/Qwen2.5-32B-Instruct
python -m server.run_grpc_server --host 0.0.0.0 --port 50051 --lora-config ./lora_paths.json --max-lora-rank 64
# 调度层：全局并发 16，每 expert 最多 4 并发，2 个 worker
python -m server.run_grpc_server --max-concurrent 16 --per-expert-limit 4 --workers 2
# 或直接指定 bind
python -m server.run_grpc_server --bind 0.0.0.0:50052
```

### 调用 StreamChat（客户端示例）

```bash
# 使用默认 localhost:50051
python -m server.grpc_client_example "你的问题" --expert-id expert-a

# 指定服务地址
GRPC_TARGET=192.168.1.100:50051 python -m server.grpc_client_example "你好" --expert-id expert-b
```

或在代码中：

```python
import grpc
from grpc_gen.proto import chat_pb2, chat_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = chat_pb2_grpc.InternalChatStub(channel)
req = chat_pb2.StreamChatRequest(prompt="你好", expert_id="expert-a", max_tokens=128, temperature=0.7)
for chunk in stub.StreamChat(req):
    print(chunk.text_delta, end="", flush=True)
    if chunk.finished:
        break
```

## 对外协议转换（Web 调用）

让系统可被 Web 调用：FastAPI 提供 REST 接口，内部通过 gRPC 连接池调用内网 StreamChat，并做协议转换与简单业务路由。

### 11. FastAPI 接口：POST /v1/chat/finance

**`gateway/app.py`** 定义 **POST /v1/chat/finance**（请求体 JSON，流式返回 SSE）与 **GET /health**。请求体：`prompt`（必填）、`expert_id`（可选，不传则按关键词路由）、`max_tokens`、`temperature`。响应为 **text/event-stream**，每行 SSE 的 `data` 为 JSON：`{"text_delta": "...", "finished": bool}`。

### 12. gRPC 客户端集成：异步连接池

**`gateway/grpc_pool.py`** 中 **GrpcChatPool** 维护多个 **grpc.aio.insecure_channel**，轮询取 Stub，提供 **stream_chat(...)** 调用内网 StreamChat；FastAPI 在 startup/shutdown 时 connect/close 连接池。

### 13. 协议转换与流映射

外部 JSON 映射为 gRPC **StreamChatRequest**；gRPC 返回的 **StreamChatChunk** 流包装为 FastAPI **StreamingResponse**（media_type=text/event-stream，每 chunk 一行 SSE）。

### 14. 简单业务路由：关键词 → expert_id

**`gateway/router.py`** 中 **route_expert_id(prompt)**：根据 prompt 关键词返回 expert_id（如「账单/消费/支出」→ expert-a，「理财/投资/基金」→ expert-b，无匹配 → base）。网关在 expert_id 未传时调用此函数。

### 启动网关

先启动内网 gRPC 服务，再在项目根目录执行：

```bash
python -m gateway.run_gateway
```

默认网关监听 **0.0.0.0:8000**，内网 gRPC 为 **localhost:50051**。可选：`--host`、`--port`、`--grpc-target`、`--pool-size`；环境变量 **GATEWAY_HOST**、**GATEWAY_PORT**、**GRPC_TARGET**。

### Web 调用示例

```bash
curl -X POST http://localhost:8000/v1/chat/finance \
  -H "Content-Type: application/json" \
  -d '{"prompt": "我的账单怎么查？"}' \
  --no-buffer
```

---

## 说明

- **基座**：需要使用modalscope上的 `Qwen/Qwen2.5-32B-Instruct`，与 QLoRA 微调时的基座一致即可。
- **LoRA rank**：若训练时使用了不同 rank，请将 `run_serve.py` 中的 `--max-lora-rank` 设为所有专家中的最大 rank。
- **显存不足**：可考虑使用 vLLM 的量化参数（如 `--quantization awq`）或增大 `--tensor-parallel-size`。

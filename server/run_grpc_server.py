#!/usr/bin/env python3
"""
The Internal Server：将推理引擎包装成内网可调用的 gRPC 服务。
服务端启动脚本：实现监听逻辑，支持命令行参数配置端口、模型路径、调度与限流等。
内建业务感知调度层：异步优先级队列（expert 聚类）、请求拦截（流式截断 + 并发限制）。
"""
from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grpc
from grpc_gen.proto import chat_pb2_grpc
from lora_loader import load_lora_paths

from scripts.async_inference import create_engine
from server.internal_servicer import InternalChatServicerImpl
from server.rate_limiter import RateLimiter
from server.scheduler import ExpertClusterQueue, run_worker

# 默认配置（可通过环境变量覆盖）
DEFAULT_HOST = os.environ.get("GRPC_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("GRPC_PORT", "50051"))
DEFAULT_MODEL = os.environ.get("VLLM_BASE_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ")
DEFAULT_MAX_LORA_RANK = 64
DEFAULT_MAX_LORAS = 2
DEFAULT_MAX_CONCURRENT = 8
DEFAULT_WORKERS = 1


async def serve(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    bind: str | None = None,
    base_model: str = DEFAULT_MODEL,
    lora_config_path: Path | str | None = None,
    max_lora_rank: int = DEFAULT_MAX_LORA_RANK,
    max_loras: int = DEFAULT_MAX_LORAS,
    max_cpu_loras: int = 2,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    per_expert_limit: int | None = None,
    num_workers: int = DEFAULT_WORKERS,
) -> None:
    """
    启动 gRPC 内网服务：创建引擎、调度队列、限流器与 worker，注册 StreamChat Servicer，监听并等待终止。

    - host/port / bind：监听地址。
    - base_model / lora_config_path / max_lora_rank / max_loras / max_cpu_loras：引擎与 LoRA 配置。
    - max_concurrent：全局最大并发 StreamChat 数（Rate Limiting）。
    - per_expert_limit：若设置，每个 expert_id 单独限制并发。
    - num_workers：调度层 worker 数量（从 ExpertClusterQueue 取 job 并调用 vLLM）。
    """
    listen_addr = bind if bind else f"{host}:{port}"
    lora_config_path = lora_config_path or ROOT / "lora_paths.json"
    lora_paths = load_lora_paths(lora_config_path)
    if not lora_paths:
        print("警告: 未找到 lora_paths.json，仅支持基座（expert_id 为空/base）", file=sys.stderr)
    else:
        print(f"已加载 LoRA 配置: {list(lora_paths.keys())}")

    print("正在创建推理引擎（首次会加载模型）...")
    engine = create_engine(
        model=base_model,
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
        max_cpu_loras=max_cpu_loras,
    )

    scheduler = ExpertClusterQueue()
    rate_limiter = RateLimiter(
        max_concurrent=max_concurrent,
        per_expert_limit=per_expert_limit,
    )
    lora_paths_or_none = lora_paths or None

    worker_tasks = [
        asyncio.create_task(run_worker(scheduler, engine, lora_paths_or_none))
        for _ in range(num_workers)
    ]
    print(f"调度层: {num_workers} 个 worker，全局并发上限 {max_concurrent}" + (
        f"，每 expert 上限 {per_expert_limit}" if per_expert_limit else ""
    ))

    servicer = InternalChatServicerImpl(scheduler=scheduler, rate_limiter=rate_limiter)
    server = grpc.aio.server()
    chat_pb2_grpc.add_InternalChatServicer_to_server(servicer, server)

    server.add_insecure_port(listen_addr)
    print(f"Internal gRPC 服务监听: {listen_addr}")
    await server.start()

    # 优雅退出：监听 SIGTERM / SIGINT，收到后停止服务并取消 worker
    loop = asyncio.get_running_loop()
    stop_ev = asyncio.Event()

    def on_stop():
        stop_ev.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, on_stop)
        except NotImplementedError:
            pass

    await stop_ev.wait()
    print("正在关闭 gRPC 服务...")
    for t in worker_tasks:
        t.cancel()
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    await server.stop(grace=5)
    await server.wait_for_termination()
    print("已退出。")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="启动内网 gRPC 推理服务（StreamChat），含调度层与限流",
    )
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="监听 host，默认 0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="监听 port，默认 50051")
    parser.add_argument("--bind", type=str, default=None, help="直接指定监听地址，如 0.0.0.0:50051")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="基座模型 ID 或本地路径")
    parser.add_argument("--lora-config", type=str, default=None, help="lora_paths.json 路径")
    parser.add_argument("--max-lora-rank", type=int, default=DEFAULT_MAX_LORA_RANK, help="LoRA 最大 rank")
    parser.add_argument("--max-loras", type=int, default=DEFAULT_MAX_LORAS, help="最大同时 LoRA 数")
    parser.add_argument("--max-cpu-loras", type=int, default=2, help="CPU 侧 LoRA 缓存数")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help="全局最大并发 StreamChat 请求数（Rate Limiting）",
    )
    parser.add_argument(
        "--per-expert-limit",
        type=int,
        default=None,
        help="每个 expert_id 最大并发数，不设则不限制",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="调度层 worker 数量（从队列取 job 并调用 vLLM）",
    )
    args = parser.parse_args()

    asyncio.run(
        serve(
            host=args.host,
            port=args.port,
            bind=args.bind,
            base_model=args.model,
            lora_config_path=args.lora_config,
            max_lora_rank=args.max_lora_rank,
            max_loras=args.max_loras,
            max_cpu_loras=args.max_cpu_loras,
            max_concurrent=args.max_concurrent,
            per_expert_limit=args.per_expert_limit,
            num_workers=args.workers,
        )
    )


if __name__ == "__main__":
    main()

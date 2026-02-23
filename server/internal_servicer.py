#!/usr/bin/env python3
"""
gRPC Servicer 实现：继承生成的桩代码 InternalChatServicer，实现 StreamChat 逻辑。
接入业务感知调度层：异步优先级队列（expert 聚类）、请求拦截（流式截断 + 并发限制）。
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

# 项目根加入 path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 生成的桩代码
from grpc_gen.proto import chat_pb2, chat_pb2_grpc

# 调度层与拦截
from server.rate_limiter import (
    RateLimiter,
    consume_response_queue_until_sentinel,
)
from server.scheduler import ExpertClusterQueue, StreamChatJob, STREAM_SENTINEL
from vllm import SamplingParams


class InternalChatServicerImpl(chat_pb2_grpc.InternalChatServicer):
    """
    继承生成的 InternalChatServicer，实现 StreamChat。
    请求先经 RateLimiter 限制并发，再进入 ExpertClusterQueue 按 expert_id 聚类，
    worker 消费队列并调用 vLLM；RPC 断开时通过 cancelled_ev 触发流式截断、取消推理。
    """

    def __init__(
        self,
        scheduler: ExpertClusterQueue,
        rate_limiter: RateLimiter,
    ):
        self._scheduler = scheduler
        self._rate_limiter = rate_limiter

    async def StreamChat(self, request, context):
        """
        服务端流式 RPC：请求拦截 -> 入队 -> 从 response_queue 取 chunk 并 yield。
        - 并发限制：rate_limiter.acquire(release)，超过限制则阻塞。
        - 流式截断：context.add_done_callback(cancelled_ev.set)，worker 内检测后中止并放入 SENTINEL。
        """
        prompt = request.prompt or ""
        expert_id = (request.expert_id or "").strip() or None
        if expert_id and expert_id.lower() == "base":
            expert_id = None
        max_tokens = request.max_tokens if request.max_tokens > 0 else 512
        temperature = request.temperature if request.temperature >= 0 else 0.7

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )

        await self._rate_limiter.acquire(expert_id)
        response_queue: asyncio.Queue = asyncio.Queue()
        cancelled_ev = asyncio.Event()
        request_id = str(uuid.uuid4())

        def on_rpc_done():
            cancelled_ev.set()

        context.add_done_callback(on_rpc_done)

        job = StreamChatJob(
            expert_id=expert_id,
            prompt=prompt,
            sampling_params=sampling_params,
            response_queue=response_queue,
            cancelled_ev=cancelled_ev,
            request_id=request_id,
        )
        await self._scheduler.put(job)

        try:
            async for chunk in consume_response_queue_until_sentinel(
                response_queue,
                STREAM_SENTINEL,
                cancelled_ev,
            ):
                if not context.is_active():
                    break
                yield chunk
        finally:
            self._rate_limiter.release(expert_id)

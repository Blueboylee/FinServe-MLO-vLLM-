#!/usr/bin/env python3
"""
业务感知调度层 - 异步优先级队列：在 gRPC 进入 vLLM 前，根据 expert_id 实现请求聚类。
同一 expert 的请求尽量连续处理，减少 LoRA 切换，提升吞吐。
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# SamplingParams 仅作类型用，实际由 async_inference 传入
try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = Any  # type: ignore[misc, assignment]

# 流结束标记，放入 response_queue 表示推理结束
STREAM_SENTINEL = object()


def _expert_key(expert_id: str | None) -> str:
    """统一基座与专家的 key，用于队列分片。"""
    return expert_id if expert_id else "_base_"


@dataclass
class StreamChatJob:
    """一次 StreamChat 请求的调度单元：进入队列、由 worker 消费、结果写入 response_queue。"""
    expert_id: str | None
    prompt: str
    sampling_params: "SamplingParams"
    response_queue: asyncio.Queue
    cancelled_ev: asyncio.Event
    request_id: str = ""


class ExpertClusterQueue:
    """
    按 expert_id 聚类的异步优先级队列。
    - put(job)：按 job.expert_id 放入对应子队列，并 notify 等待中的 get()。
    - get()：优先返回当前正在服务的 expert 的请求（聚类），否则任意非空；无请求时阻塞直到 put()。
    """

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {}
        self._current_expert: str | None = None
        self._cond = asyncio.Condition()

    async def put(self, job: StreamChatJob) -> None:
        """放入队列并唤醒等待中的 get()。"""
        key = _expert_key(job.expert_id)
        async with self._cond:
            if key not in self._queues:
                self._queues[key] = asyncio.Queue()
            self._queues[key].put_nowait(job)
            self._cond.notify()

    def _pop_one(self) -> StreamChatJob | None:
        """在已持 _cond 时调用，优先当前 expert，否则任意非空；无则返回 None。"""
        if self._current_expert is not None and self._current_expert in self._queues:
            q = self._queues[self._current_expert]
            if not q.empty():
                return q.get_nowait()
        for key, q in self._queues.items():
            if not q.empty():
                self._current_expert = key
                return q.get_nowait()
        return None

    async def get(self) -> StreamChatJob:
        """阻塞取出；优先当前 expert（聚类），无请求时等待 put()。"""
        async with self._cond:
            while True:
                job = self._pop_one()
                if job is not None:
                    return job
                await self._cond.wait()


async def run_worker(
    scheduler: ExpertClusterQueue,
    engine: Any,
    lora_paths: dict[str, str] | None,
) -> None:
    """
    单 worker 循环：从调度队列取 job，调用 vLLM 流式推理，结果写入 job.response_queue；
    若 job.cancelled_ev 被置位则中止并仍放入 SENTINEL，保证调用方能退出。
    """
    from scripts.async_inference import generate_with_expert
    from server.stream_bridge import bridge_vllm_to_grpc_stream

    while True:
        job = await scheduler.get()
        if job.cancelled_ev.is_set():
            try:
                job.response_queue.put_nowait(STREAM_SENTINEL)
            except asyncio.QueueFull:
                pass
            continue
        try:
            vllm_stream = generate_with_expert(
                engine,
                job.prompt,
                job.sampling_params,
                expert_id=job.expert_id,
                lora_paths=lora_paths,
            )
            async for chunk in bridge_vllm_to_grpc_stream(vllm_stream):
                if job.cancelled_ev.is_set():
                    break
                try:
                    job.response_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception:
            pass  # 已在下文 finally 中放入 SENTINEL，调用方可正常结束
        finally:
            try:
                job.response_queue.put_nowait(STREAM_SENTINEL)
            except asyncio.QueueFull:
                pass

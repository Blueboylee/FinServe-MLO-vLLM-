#!/usr/bin/env python3
"""
请求拦截与预处理：并发限制（Rate Limiting）与流式截断支持。
- 并发限制：全局/按 expert 的 Semaphore，限制同时进入 vLLM 的请求数。
- 流式截断：RPC 断开时通过 cancelled_ev 取消推理任务（与 scheduler 中的 worker 配合）。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class RateLimiter:
    """
    并发限制（Rate Limiting）：限制同时进行中的 StreamChat 请求数。
    使用 asyncio.Semaphore 实现；可选按 expert_id 分桶限制。
    """

    def __init__(
        self,
        max_concurrent: int = 8,
        per_expert_limit: Optional[int] = None,
    ) -> None:
        """
        - max_concurrent: 全局最大并发请求数，超过则 acquire 阻塞直到有空位。
        - per_expert_limit: 若设置，则每个 expert_id 单独一个 semaphore，限制该 expert 的并发。
        """
        self._global_sem = asyncio.Semaphore(max_concurrent)
        self._per_expert_limit = per_expert_limit
        self._per_expert_sems: dict[str, asyncio.Semaphore] = {}
        self._per_expert_lock = asyncio.Lock()

    def _expert_key(self, expert_id: str | None) -> str:
        return expert_id if expert_id else "_base_"

    async def acquire(self, expert_id: str | None = None) -> None:
        """在进入推理前调用；阻塞直到允许通过。"""
        await self._global_sem.acquire()
        if self._per_expert_limit is not None:
            key = self._expert_key(expert_id)
            async with self._per_expert_lock:
                if key not in self._per_expert_sems:
                    self._per_expert_sems[key] = asyncio.Semaphore(self._per_expert_limit)
                sem = self._per_expert_sems[key]
            await sem.acquire()

    def release(self, expert_id: str | None = None) -> None:
        """请求结束（或取消）时调用，释放占位。"""
        self._global_sem.release()
        if self._per_expert_limit is not None:
            key = self._expert_key(expert_id)
            if key in self._per_expert_sems:
                self._per_expert_sems[key].release()


async def consume_response_queue_until_sentinel(
    response_queue: asyncio.Queue,
    sentinel: object,
    cancelled_ev: asyncio.Event,
) -> AsyncIterator[Any]:
    """
    从 response_queue 中读取并 yield，直到遇到 sentinel 或 cancelled_ev 被置位。
    用于 Servicer 侧：从 worker 写入的 queue 里取 chunk 并 yield 给 gRPC；若 RPC 断开（cancelled_ev）
    则不再阻塞等待，直接结束迭代。
    """
    while True:
        if cancelled_ev.is_set():
            return
        try:
            chunk = await asyncio.wait_for(response_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        if chunk is sentinel:
            return
        yield chunk
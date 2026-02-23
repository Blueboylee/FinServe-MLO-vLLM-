#!/usr/bin/env python3
"""
gRPC 客户端集成：在 FastAPI 内部维护一个异步 gRPC 连接池。
支持多 channel 轮询，与 InternalChat.StreamChat 的 unary_stream 调用配合。
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grpc
from grpc_gen.proto import chat_pb2, chat_pb2_grpc

# 默认内网 gRPC 地址
DEFAULT_GRPC_TARGET = os.environ.get("GRPC_TARGET", "localhost:50051")


class GrpcChatPool:
    """
    异步 gRPC 连接池：维护多个 grpc.aio.Channel，按轮询方式取 Stub 调用 StreamChat。
    """

    def __init__(
        self,
        target: str = DEFAULT_GRPC_TARGET,
        pool_size: int = 4,
    ) -> None:
        self._target = target
        self._pool_size = pool_size
        self._channels: list[grpc.aio.Channel] = []
        self._index = 0
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "GrpcChatPool":
        await self.connect()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def connect(self) -> None:
        """创建并缓存 pool_size 个 aio channel。"""
        for _ in range(self._pool_size):
            channel = grpc.aio.insecure_channel(self._target)
            self._channels.append(channel)
        return None

    async def close(self) -> None:
        """关闭所有 channel。"""
        for ch in self._channels:
            await ch.close()
        self._channels.clear()

    def _next_stub(self) -> chat_pb2_grpc.InternalChatStub:
        """轮询取下一个 channel 的 Stub（需在已 connect 后调用）。"""
        if not self._channels:
            raise RuntimeError("GrpcChatPool not connected")
        self._index = (self._index + 1) % len(self._channels)
        return chat_pb2_grpc.InternalChatStub(self._channels[self._index])

    async def stream_chat(
        self,
        prompt: str,
        expert_id: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """
        调用内网 StreamChat，返回异步迭代器，yield StreamChatChunk。
        """
        stub = self._next_stub()
        request = chat_pb2.StreamChatRequest(
            prompt=prompt,
            expert_id=expert_id or "",
            max_tokens=max_tokens,
            temperature=temperature,
        )
        async for chunk in stub.StreamChat(request):
            yield chunk

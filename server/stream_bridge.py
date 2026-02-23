#!/usr/bin/env python3
"""
流式响应桥接：将 vLLM 输出的异步迭代器（AsyncIterator[RequestOutput]）
与 gRPC 的 yield 流式输出（StreamChatChunk）打通。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import AsyncIterator

# 项目根
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grpc_gen.proto import chat_pb2

# 仅做类型注解时延迟导入 vLLM
try:
    from vllm.outputs import RequestOutput
except ImportError:
    RequestOutput = None  # type: ignore[misc, assignment]


async def bridge_vllm_to_grpc_stream(
    vllm_output_stream: "AsyncIterator[RequestOutput]",
) -> AsyncIterator[chat_pb2.StreamChatChunk]:
    """
    将 vLLM 的异步输出流转换为 gRPC 可 yield 的 StreamChatChunk 流。

    - 输入：vLLM engine.generate() 返回的 AsyncIterator[RequestOutput]
    - 输出：AsyncIterator[StreamChatChunk]，每个 Chunk 含 text_delta 与 finished

    这样 gRPC Servicer 中可直接：
        async for chunk in bridge_vllm_to_grpc_stream(generate_with_expert(...)):
            yield chunk
    """
    async for output in vllm_output_stream:
        if not output.outputs:
            continue
        text = output.outputs[0].text
        yield chat_pb2.StreamChatChunk(
            text_delta=text,
            finished=output.finished,
        )

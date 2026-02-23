#!/usr/bin/env python3
"""
调用内网 gRPC 服务 StreamChat 的示例客户端（流式接收 Chunk）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grpc
from grpc_gen.proto import chat_pb2, chat_pb2_grpc

DEFAULT_TARGET = os.environ.get("GRPC_TARGET", "localhost:50051")


def run_stream_chat(
    prompt: str,
    expert_id: str = "expert-a",
    max_tokens: int = 256,
    temperature: float = 0.7,
    target: str = DEFAULT_TARGET,
) -> None:
    """发起 StreamChat 请求并打印流式返回的 text_delta。"""
    channel = grpc.insecure_channel(target)
    stub = chat_pb2_grpc.InternalChatStub(channel)
    request = chat_pb2.StreamChatRequest(
        prompt=prompt,
        expert_id=expert_id,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    for chunk in stub.StreamChat(request):
        if chunk.text_delta:
            print(chunk.text_delta, end="", flush=True)
        if chunk.finished:
            break
    print()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="gRPC StreamChat 客户端示例")
    parser.add_argument("prompt", nargs="?", default="请用一句话介绍大语言模型。")
    parser.add_argument("--expert-id", type=str, default="expert-a")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    run_stream_chat(
        prompt=args.prompt,
        expert_id=args.expert_id,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        target=args.target,
    )


if __name__ == "__main__":
    main()

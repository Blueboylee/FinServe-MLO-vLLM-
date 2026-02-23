#!/usr/bin/env python3
"""
FastAPI 接口开发 + 协议转换与流映射：定义 /v1/chat/finance 的 POST 接口，
将外部 JSON 映射为 gRPC 请求，将 gRPC 流包装成 StreamingResponse，并集成关键词路由。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from gateway.grpc_pool import DEFAULT_GRPC_TARGET, GrpcChatPool
from gateway.router import route_expert_id

# 请求体模型（外部 JSON）
class FinanceChatRequest(BaseModel):
    prompt: str
    expert_id: Optional[str] = None  # 不传则按关键词路由
    max_tokens: int = 512
    temperature: float = 0.7


def create_app(grpc_target: str = DEFAULT_GRPC_TARGET, pool_size: int = 4) -> FastAPI:
    """创建 FastAPI 应用，并挂载 /v1/chat/finance 与 gRPC 连接池。"""
    app = FastAPI(title="FinServe Finance Chat Gateway", version="0.1.0")
    pool = GrpcChatPool(target=grpc_target, pool_size=pool_size)

    @app.on_event("startup")
    async def startup() -> None:
        await pool.connect()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await pool.close()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "gateway"}

    @app.post("/v1/chat/finance")
    async def chat_finance(body: FinanceChatRequest) -> StreamingResponse:
        """
        POST /v1/chat/finance：接收 JSON，按关键词路由 expert_id，调用内网 gRPC StreamChat，
        将 gRPC 返回的流包装成 SSE 流（每行一个 JSON：{ "text_delta": "...", "finished": bool }）。
        """
        prompt = (body.prompt or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt 不能为空")

        expert_id = body.expert_id
        if expert_id is None or (isinstance(expert_id, str) and not expert_id.strip()):
            expert_id = route_expert_id(prompt, default="base")
        else:
            expert_id = expert_id.strip() or "base"

        max_tokens = body.max_tokens if body.max_tokens > 0 else 512
        temperature = body.temperature if body.temperature >= 0 else 0.7

        async def stream() -> Any:
            try:
                async for chunk in pool.stream_chat(
                    prompt=prompt,
                    expert_id=expert_id if expert_id != "base" else None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    line = json.dumps(
                        {"text_delta": chunk.text_delta, "finished": chunk.finished},
                        ensure_ascii=False,
                    )
                    yield f"data: {line}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e), 'finished': True})}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


# 供 uvicorn 直接挂载: uvicorn gateway.app:app
app = create_app()

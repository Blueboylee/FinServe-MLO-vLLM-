#!/usr/bin/env python3
"""
对外 Web 网关入口：启动 FastAPI，对外提供 /v1/chat/finance，内部通过 gRPC 连接池调用内网服务。
需先启动内网 gRPC 服务：python -m server.run_grpc_server
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.app import create_app

DEFAULT_HOST = os.environ.get("GATEWAY_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("GATEWAY_PORT", "8000"))
DEFAULT_GRPC_TARGET = os.environ.get("GRPC_TARGET", "localhost:50051")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="启动 FinServe 对外 Web 网关")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="网关监听 host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="网关监听 port")
    parser.add_argument(
        "--grpc-target",
        type=str,
        default=DEFAULT_GRPC_TARGET,
        help="内网 gRPC 服务地址",
    )
    parser.add_argument("--pool-size", type=int, default=4, help="gRPC 连接池大小")
    args = parser.parse_args()

    app = create_app(grpc_target=args.grpc_target, pool_size=args.pool_size)

    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

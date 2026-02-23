#!/usr/bin/env python3
"""根据 grpc_gen/proto/chat.proto 重新生成 chat_pb2.py 与 chat_pb2_grpc.py。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = ROOT / "grpc_gen" / "proto"
PROTO_FILE = PROTO_DIR / "chat.proto"

if not PROTO_FILE.is_file():
    print(f"未找到 {PROTO_FILE}", file=sys.stderr)
    sys.exit(1)

# python -m grpc_tools.protoc -I grpc_gen --python_out=grpc_gen --grpc_python_out=grpc_gen grpc_gen/proto/chat.proto
# 输出目录设为 grpc_gen，这样生成的 import 为 from proto import chat_pb2
cmd = [
    sys.executable, "-m", "grpc_tools.protoc",
    f"-I{ROOT / 'grpc_gen'}",
    f"--python_out={ROOT / 'grpc_gen'}",
    f"--grpc_python_out={ROOT / 'grpc_gen'}",
    str(PROTO_FILE.relative_to(ROOT)),
]
print(" ".join(cmd))
subprocess.run(cmd, cwd=str(ROOT), check=True)
print("已生成 chat_pb2.py 与 chat_pb2_grpc.py")

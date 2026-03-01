#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 多专家服务：Python 启动方式，与 serve_experts.sh 等效。
共享 Qwen2.5 32B GPTQ 基座 + expert-a / expert-b 两个 QLoRA。
"""
import sys

if sys.version_info < (3, 10):
    print("错误: 需要 Python 3.10+")
    sys.exit(1)

import os
from pathlib import Path

# 路径
BASE_DIR = Path(os.environ.get("BASE_DIR", "./models/base")).resolve()
EXPERTS_DIR = Path(os.environ.get("EXPERTS_DIR", "./models/experts")).resolve()
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

EXPERT_A = EXPERTS_DIR / "expert-a"
EXPERT_B = EXPERTS_DIR / "expert-b"

if not BASE_DIR.exists():
    print(f"错误: 基座不存在: {BASE_DIR}")
    sys.exit(1)
if not EXPERT_A.exists() or not EXPERT_B.exists():
    print(f"错误: 专家目录不存在: {EXPERT_A} / {EXPERT_B}")
    sys.exit(1)

# 通过 vllm serve 子进程启动（兼容性最好）
import subprocess

cmd = [
    "vllm", "serve", str(BASE_DIR),
    "--quantization", "gptq",
    "--enable-lora",
    "--max-loras", "2",
    "--max-lora-rank", "64",
    "--lora-modules", f"expert-a={EXPERT_A}", f"expert-b={EXPERT_B}",
    "--host", HOST,
    "--port", str(PORT),
    "--gpu-memory-utilization", "0.9",
    "--max-num-seqs", "256",
]

print("启动 vLLM 多专家服务...")
print(f"  基座: {BASE_DIR}")
print(f"  专家: expert-a, expert-b")
print(f"  地址: http://{HOST}:{PORT}")
print("  请求时 model 参数: expert-a 或 expert-b")
print()

subprocess.run(cmd)

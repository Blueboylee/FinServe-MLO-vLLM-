#!/usr/bin/env python3
"""
调用已启动的 vLLM 服务，使用基座或专家 A / 专家 B 进行推理。
服务需先由 scripts/run_serve.py 启动。
"""
from __future__ import annotations

import os
import sys

# 可选：使用 OpenAI 兼容客户端
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct"  # 与 run_serve 中的基座一致
EXPERT_A = "expert-a"
EXPERT_B = "expert-b"


def chat(
    prompt: str,
    model: str = EXPERT_A,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """请求 vLLM 完成补全，返回生成文本。"""
    if OpenAI is None:
        print("请安装 openai: pip install openai", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(base_url=BASE_URL, api_key="dummy")
    resp = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].text


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="调用 vLLM 专家模型推理")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="请用一句话介绍什么是大语言模型。",
        help="输入 prompt",
    )
    parser.add_argument(
        "--model",
        choices=[BASE_MODEL, EXPERT_A, EXPERT_B],
        default=EXPERT_A,
        help="选择模型：基座 / expert-a / expert-b",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    text = chat(
        args.prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(text)


if __name__ == "__main__":
    main()

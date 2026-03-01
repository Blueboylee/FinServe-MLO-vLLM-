#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 vLLM 加载共享 Qwen2.5 32B GPTQ 基座 + 两个 QLoRA 专家进行推理。
适配 V100 GPU，Python 3.10。
"""
import sys

if sys.version_info < (3, 10):
    print("错误: 需要 Python 3.10 或更高版本")
    sys.exit(1)

import argparse
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def build_messages(system: str, user: str):
    """构建 Qwen 对话格式。"""
    return [
        {"role": "system", "content": system or "You are a helpful assistant."},
        {"role": "user", "content": user},
    ]


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5 32B 专家模型推理")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./models/base",
        help="基座模型目录（GPTQ 4bit）",
    )
    parser.add_argument(
        "--experts-dir",
        type=str,
        default="./models/experts",
        help="专家模型根目录",
    )
    parser.add_argument(
        "--expert",
        type=str,
        choices=["expert-a", "expert-b"],
        default="expert-a",
        help="使用的专家",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="你好，请介绍一下你自己。",
        help="用户输入",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="",
        help="系统提示（可选）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="张量并行数（多卡时使用）",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    experts_dir = Path(args.experts_dir).resolve()
    expert_a_dir = experts_dir / "expert-a"
    expert_b_dir = experts_dir / "expert-b"

    if not base_dir.exists():
        raise FileNotFoundError(f"基座模型目录不存在: {base_dir}，请先运行 download_models.py")

    expert_dir = expert_a_dir if args.expert == "expert-a" else expert_b_dir
    if not expert_dir.exists():
        raise FileNotFoundError(f"专家目录不存在: {expert_dir}，请先运行 download_models.py")

    # 初始化 vLLM：GPTQ + LoRA
    llm = LLM(
        model=str(base_dir),
        quantization="gptq",
        enable_lora=True,
        max_lora_rank=64,
        max_loras=2,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.9,  # V100 显存利用
    )

    # LoRA 请求
    lora_request = LoRARequest(
        lora_name=args.expert,
        lora_int_id=1,
        lora_path=str(expert_dir),
    )

    # 加载 tokenizer 并构建 prompt
    tokenizer = llm.get_tokenizer()
    messages = build_messages(args.system, args.prompt)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    outputs = llm.generate(
        [prompt],
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    for out in outputs:
        print(out.outputs[0].text)


if __name__ == "__main__":
    main()

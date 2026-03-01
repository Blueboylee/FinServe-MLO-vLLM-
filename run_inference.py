#!/usr/bin/env python3
"""
Qwen 2.5 32B 双专家 QLoRA 离线推理
共享 4bit GPTQ 基座，按需切换专家 A / 专家 B
适用于 V100 GPU
"""

import argparse
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def main():
    parser = argparse.ArgumentParser(description="双专家模型离线推理")
    parser.add_argument(
        "--base-model",
        type=str,
        default="./models/base",
        help="基座模型路径或 HuggingFace 模型 ID",
    )
    parser.add_argument(
        "--expert-dir",
        type=str,
        default="./models/experts",
        help="专家 LoRA 目录",
    )
    parser.add_argument(
        "--expert",
        choices=["expert-a", "expert-b", "base"],
        default="expert-a",
        help="使用的专家: expert-a, expert-b, 或 base(仅基座)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="你好，请介绍一下你自己。",
        help="输入提示",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU 显存利用率 (V100 32GB 建议 0.9)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="最大序列长度，降低可省显存",
    )
    args = parser.parse_args()

    base_path = Path(args.base_model)
    expert_dir = Path(args.expert_dir)

    # 若本地路径不存在，尝试作为 HF 模型 ID
    if not base_path.exists():
        model_id = args.base_model
        print(f"使用远程基座模型: {model_id}")
        # HuggingFace GPTQ 模型需显式指定
        quantization = "gptq" if "GPTQ" in model_id or "gptq" in model_id.lower() else None
    else:
        model_id = str(base_path.absolute())
        print(f"使用本地基座模型: {model_id}")
        # 本地 GPTQ: 检测 quantize_config.json
        quantization = "gptq" if (base_path / "quantize_config.json").exists() else None
    if quantization:
        print("使用 GPTQ 量化")

    # 构建 LoRA 请求
    lora_request = None
    if args.expert != "base":
        expert_path = expert_dir / args.expert
        if not expert_path.exists():
            raise FileNotFoundError(
                f"专家路径不存在: {expert_path}\n"
                "请先运行: python download_models.py"
            )
        lora_request = LoRARequest(
            lora_name=args.expert,
            lora_int_id=hash(args.expert) % (2**32),
            lora_path=str(expert_path.absolute()),
        )
        print(f"加载专家: {args.expert}")

    # 初始化 vLLM 引擎
    print("初始化 vLLM 引擎...")
    llm = LLM(
        model=model_id,
        quantization=quantization,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=2,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print(f"\n输入: {args.prompt}")
    print("-" * 50)
    outputs = llm.generate(
        [args.prompt],
        sampling_params,
        lora_request=lora_request,
    )
    text = outputs[0].outputs[0].text
    print(f"输出: {text}")
    print("-" * 50)


if __name__ == "__main__":
    main()

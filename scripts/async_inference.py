#!/usr/bin/env python3
"""
基于 AsyncLLMEngine 的异步推理封装：根据输入的 expert_id 调用对应的 LoRA 权重。
需先运行 scripts/download_experts.py 生成 lora_paths.json，并确保显存可加载 Qwen2.5-32B。
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from typing import AsyncIterator

# 保证项目根在 path 中，以便导入 lora_loader
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_loader import get_lora_request, load_lora_paths

# vLLM
from vllm import EngineArgs, SamplingParams
try:
    from vllm import AsyncLLMEngine
except ImportError:
    from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput

# 默认与 run_serve 一致
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_MAX_LORA_RANK = 64
DEFAULT_MAX_LORAS = 2


def create_engine(
    model: str = DEFAULT_BASE_MODEL,
    max_loras: int = DEFAULT_MAX_LORAS,
    max_lora_rank: int = DEFAULT_MAX_LORA_RANK,
    max_cpu_loras: int | None = 2,
    **engine_kwargs,
) -> AsyncLLMEngine:
    """创建已启用 LoRA 的 AsyncLLMEngine。"""
    engine_args = EngineArgs(
        model=model,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
        max_cpu_loras=max_cpu_loras or 2,
        **engine_kwargs,
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


async def generate_with_expert(
    engine: AsyncLLMEngine,
    prompt: str,
    sampling_params: SamplingParams,
    expert_id: str | None = None,
    request_id: str | None = None,
    lora_paths: dict[str, str] | None = None,
) -> AsyncIterator[RequestOutput]:
    """
    异步生成器：根据 expert_id 选择 LoRA（None 表示基座），逐次 yield RequestOutput。

    - expert_id 为 None 或 "base"：使用基座，不挂载 LoRA。
    - 否则使用 lora_loader 解析为 LoRARequest，并挂载对应专家权重。
    """
    request_id = request_id or str(uuid.uuid4())
    lora_request: LoRARequest | None = get_lora_request(
        expert_id, lora_paths=lora_paths, config_path=ROOT / "lora_paths.json"
    )
    async for output in engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
        lora_request=lora_request,
    ):
        yield output


async def generate_with_expert_text(
    engine: AsyncLLMEngine,
    prompt: str,
    expert_id: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    **sampling_kwargs,
) -> str:
    """
    根据 expert_id 做一次完整生成，返回拼接后的生成文本（非流式）。
    """
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        **sampling_kwargs,
    )
    full_text = ""
    async for output in generate_with_expert(
        engine, prompt, sampling_params, expert_id=expert_id
    ):
        if output.outputs:
            full_text = output.outputs[0].text
            if output.finished:
                break
    return full_text


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="AsyncLLMEngine + expert_id 推理示例"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="请用一句话介绍什么是大语言模型。",
        help="输入 prompt",
    )
    parser.add_argument(
        "--expert-id",
        type=str,
        default="expert-a",
        help="专家 ID：expert-a / expert-b，或 base 表示基座",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="基座模型",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="流式输出每个 RequestOutput",
    )
    args = parser.parse_args()

    expert_id = None if args.expert_id.strip().lower() == "base" else args.expert_id
    paths = load_lora_paths(ROOT / "lora_paths.json")
    if expert_id and expert_id not in paths:
        print(f"错误: 未找到 expert_id={expert_id!r}，请先运行 scripts/download_experts.py", file=sys.stderr)
        print(f"当前配置: {list(paths.keys())}", file=sys.stderr)
        sys.exit(1)

    print("正在创建 AsyncLLMEngine（首次会加载模型）...")
    engine = create_engine(model=args.model)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if args.stream:
        print(f"expert_id={expert_id or 'base'} 流式输出:\n")
        async for out in generate_with_expert(
            engine, args.prompt, sampling_params, expert_id=expert_id, lora_paths=paths
        ):
            if out.outputs:
                text = out.outputs[0].text
                if text:
                    print(text, end="", flush=True)
                if out.finished:
                    break
        print()
    else:
        text = await generate_with_expert_text(
            engine,
            args.prompt,
            expert_id=expert_id,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(text)


if __name__ == "__main__":
    asyncio.run(main())

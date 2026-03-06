"""
方案一（Python 版）：vLLM 多 LoRA 动态加载服务

通过 Python API 启动 vLLM 引擎，支持动态切换 Expert-A / Expert-B。
适用于需要更灵活控制推理流程的场景。
"""

import json
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL_DIR = Path("./models")
BASE_MODEL = str(MODEL_DIR / "Qwen3-VL-8B-Instruct-AWQ-4bit")
EXPERT_A_PATH = str(MODEL_DIR / "Qwen3-VL-Finance-expert-a")
EXPERT_B_PATH = str(MODEL_DIR / "Qwen3-VL-Finance-expert-b")

def detect_lora_rank(adapter_path: str) -> int:
    config_path = Path(adapter_path) / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)["r"]
    return 64

def main():
    lora_rank = detect_lora_rank(EXPERT_A_PATH)
    print(f"检测到 LoRA rank: {lora_rank}")

    llm = LLM(
        model=BASE_MODEL,
        quantization="awq",
        enable_lora=True,
        max_loras=2,
        max_lora_rank=lora_rank,
        max_cpu_loras=2,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 4, "video": 1},
    )

    expert_a = LoRARequest("finance-expert-a", 1, EXPERT_A_PATH)
    expert_b = LoRARequest("finance-expert-b", 2, EXPERT_B_PATH)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        max_tokens=1024,
    )

    # -------- 示例：纯文本推理 --------
    test_prompt = "请分析2024年A股市场的整体走势和主要驱动因素。"

    messages_a = [
        {"role": "system", "content": "你是金融分析专家A，擅长宏观经济分析。"},
        {"role": "user", "content": test_prompt},
    ]
    messages_b = [
        {"role": "system", "content": "你是金融分析专家B，擅长行业和个股分析。"},
        {"role": "user", "content": test_prompt},
    ]

    print("\n" + "=" * 60)
    print("Expert-A 回答：")
    print("=" * 60)
    outputs_a = llm.chat(messages_a, sampling_params=sampling_params, lora_request=expert_a)
    print(outputs_a[0].outputs[0].text)

    print("\n" + "=" * 60)
    print("Expert-B 回答：")
    print("=" * 60)
    outputs_b = llm.chat(messages_b, sampling_params=sampling_params, lora_request=expert_b)
    print(outputs_b[0].outputs[0].text)

    print("\n" + "=" * 60)
    print("基座模型（无 LoRA）回答：")
    print("=" * 60)
    messages_base = [
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": test_prompt},
    ]
    outputs_base = llm.chat(messages_base, sampling_params=sampling_params)
    print(outputs_base[0].outputs[0].text)


if __name__ == "__main__":
    main()

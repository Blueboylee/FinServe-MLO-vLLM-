"""
方案二（保底）：将 LoRA 合并到基座模型中，再用 vLLM 部署

当 vLLM 的 Qwen3-VL + LoRA 动态加载出现 AssertionError 时使用此方案。
会生成两个独立的合并模型目录，然后选择一个来服务。

如果需要同时服务两个专家，可以在不同端口启动两个 vLLM 实例，
或者在应用层实现路由。
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor


MODEL_DIR = Path("./models")
BASE_MODEL = str(MODEL_DIR / "Qwen3-VL-8B-Instruct-AWQ-4bit")
EXPERT_A_PATH = str(MODEL_DIR / "Qwen3-VL-Finance-expert-a")
EXPERT_B_PATH = str(MODEL_DIR / "Qwen3-VL-Finance-expert-b")
MERGED_A_PATH = str(MODEL_DIR / "Qwen3-VL-Finance-merged-expert-a")
MERGED_B_PATH = str(MODEL_DIR / "Qwen3-VL-Finance-merged-expert-b")


def merge_lora(base_model_path: str, lora_path: str, output_path: str):
    """将 LoRA adapter 合并到基座模型并保存"""
    print(f"\n合并 LoRA: {lora_path}")
    print(f"  基座模型: {base_model_path}")
    print(f"  输出路径: {output_path}")

    peft_config = PeftConfig.from_pretrained(lora_path)
    print(f"  LoRA rank: {peft_config.r}, alpha: {peft_config.lora_alpha}")
    print(f"  目标模块: {peft_config.target_modules}")

    print("  加载基座模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("  加载 LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_path)

    print("  合并权重...")
    merged_model = model.merge_and_unload()

    print("  保存合并模型...")
    merged_model.save_pretrained(output_path)

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    processor.save_pretrained(output_path)

    config_src = Path(base_model_path)
    for cfg_file in ["config.json", "generation_config.json",
                      "preprocessor_config.json", "chat_template.json"]:
        src = config_src / cfg_file
        if src.exists():
            import shutil
            shutil.copy2(src, Path(output_path) / cfg_file)

    print(f"  [✓] 合并完成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 到基座模型")
    parser.add_argument("--expert", choices=["a", "b", "both"], default="both",
                        help="要合并的专家: a, b, 或 both")
    args = parser.parse_args()

    if args.expert in ("a", "both"):
        merge_lora(BASE_MODEL, EXPERT_A_PATH, MERGED_A_PATH)

    if args.expert in ("b", "both"):
        merge_lora(BASE_MODEL, EXPERT_B_PATH, MERGED_B_PATH)

    print("\n" + "=" * 60)
    print("合并完成！使用以下命令启动 vLLM 服务：")
    print("=" * 60)

    if args.expert in ("a", "both"):
        print(f"\n# 启动 Expert-A（端口 8000）:")
        print(f"vllm serve {MERGED_A_PATH} \\")
        print(f"    --host 0.0.0.0 --port 8000 \\")
        print(f"    --max-model-len 4096 --trust-remote-code")

    if args.expert in ("b", "both"):
        port = 8001 if args.expert == "both" else 8000
        print(f"\n# 启动 Expert-B（端口 {port}）:")
        print(f"vllm serve {MERGED_B_PATH} \\")
        print(f"    --host 0.0.0.0 --port {port} \\")
        print(f"    --max-model-len 4096 --trust-remote-code")


if __name__ == "__main__":
    main()

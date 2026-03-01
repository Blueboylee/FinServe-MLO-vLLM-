#!/usr/bin/env python3
"""
下载 Qwen 2.5 32B 基座模型及双专家 LoRA 适配器
- 基座: Qwen2.5-32B-Instruct-GPTQ-Int4 (4bit GPTQ, 共享)
- 专家A: GaryLeenene/qwen25-32b-expert-a-qlora
- 专家B: GaryLeenene/qwen25-32b-expert-b-qlora
"""

import argparse
from pathlib import Path


def download_from_modelscope(model_id: str, local_dir: str) -> str:
    """从 ModelScope 下载模型"""
    from modelscope import snapshot_download
    path = snapshot_download(model_id, local_dir=local_dir)
    print(f"✓ ModelScope 下载完成: {model_id} -> {path}")
    return path


def download_from_huggingface(repo_id: str, local_dir: str) -> str:
    """从 HuggingFace 下载模型"""
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print(f"✓ HuggingFace 下载完成: {repo_id} -> {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="下载基座模型与专家 LoRA")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./models/base",
        help="基座模型保存目录",
    )
    parser.add_argument(
        "--expert-dir",
        type=str,
        default="./models/experts",
        help="专家 LoRA 保存目录",
    )
    parser.add_argument(
        "--base-source",
        choices=["hf", "modelscope"],
        default="hf",
        help="基座模型来源: hf=HuggingFace, modelscope=ModelScope",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="跳过基座下载（已存在时使用）",
    )
    parser.add_argument(
        "--skip-experts",
        action="store_true",
        help="跳过专家下载",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    expert_dir = Path(args.expert_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    expert_dir.mkdir(parents=True, exist_ok=True)

    # 基座模型: Qwen 2.5 32B 4bit GPTQ
    # HuggingFace 有官方 GPTQ 版本，ModelScope 可用 Instruct 版本（需注意量化格式）
    base_model_config = {
        "hf": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
        "modelscope": "Qwen/Qwen2.5-32B-Instruct",  # ModelScope 无 GPTQ 时用全量版
    }
    base_id = base_model_config[args.base_source]

    if not args.skip_base:
        print("\n========== 下载基座模型 ==========")
        if args.base_source == "hf":
            download_from_huggingface(base_id, str(base_dir))
        else:
            download_from_modelscope(base_id, str(base_dir))
    else:
        print("跳过基座模型下载")
        if not (base_dir / "config.json").exists() and not (base_dir / "quantize_config.json").exists():
            print(f"  警告: {base_dir} 下未找到 config.json，请确保已正确下载")

    if not args.skip_experts:
        print("\n========== 下载专家 LoRA ==========")
        experts = [
            ("expert-a", "GaryLeenene/qwen25-32b-expert-a-qlora"),
            ("expert-b", "GaryLeenene/qwen25-32b-expert-b-qlora"),
        ]
        for name, model_id in experts:
            out_path = expert_dir / name
            out_path.mkdir(parents=True, exist_ok=True)
            download_from_modelscope(model_id, str(out_path))

    print("\n========== 下载完成 ==========")
    print(f"基座模型: {base_dir.absolute()}")
    print(f"专家模型: {expert_dir.absolute()}")
    print("\n使用方式:")
    print("  离线推理: python run_inference.py")
    print("  启动服务: bash serve_experts.sh")


if __name__ == "__main__":
    main()

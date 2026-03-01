#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 ModelScope 下载 Qwen2.5 32B 基座模型（GPTQ 4bit）及两个 QLoRA 专家模型。
确保所有模型均从 ModelScope 下载，不使用 HuggingFace。
Python 3.10 兼容。
"""
import sys

if sys.version_info < (3, 10):
    print("错误: 需要 Python 3.10 或更高版本")
    sys.exit(1)

import argparse
import os
from pathlib import Path


def download_from_modelscope(model_id: str, local_dir: str) -> str:
    """使用 ModelScope 下载模型到指定目录。"""
    from modelscope import snapshot_download

    # snapshot_download 从 modelscope.cn 下载，不使用 HuggingFace
    model_dir = snapshot_download(
        model_id,
        local_dir=local_dir,
    )
    return model_dir


def main():
    parser = argparse.ArgumentParser(description="从 ModelScope 下载基座模型和专家 LoRA")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./models/base",
        help="基座模型保存目录",
    )
    parser.add_argument(
        "--experts-dir",
        type=str,
        default="./models/experts",
        help="专家模型保存根目录",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="仅下载基座模型",
    )
    parser.add_argument(
        "--experts-only",
        action="store_true",
        help="仅下载专家模型",
    )
    args = parser.parse_args()

    # 模型 ID（全部来自 ModelScope，不使用 HuggingFace）
    # 若基座下载失败，可尝试: qwen/Qwen2.5-32B-Instruct-GPTQ-Int4
    BASE_MODEL_ID = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    EXPERT_A_ID = "GaryLeenene/qwen25-32b-expert-a-qlora"
    EXPERT_B_ID = "GaryLeenene/qwen25-32b-expert-b-qlora"

    base_dir = Path(args.base_dir).resolve()
    experts_dir = Path(args.experts_dir).resolve()

    if not args.experts_only:
        print(f"[1/3] 下载基座模型: {BASE_MODEL_ID}")
        print(f"      保存到: {base_dir}")
        base_dir.mkdir(parents=True, exist_ok=True)
        download_from_modelscope(BASE_MODEL_ID, str(base_dir))
        print("      基座模型下载完成。\n")

    if not args.base_only:
        expert_a_dir = experts_dir / "expert-a"
        expert_b_dir = experts_dir / "expert-b"
        experts_dir.mkdir(parents=True, exist_ok=True)

        print(f"[2/3] 下载专家 A: {EXPERT_A_ID}")
        print(f"      保存到: {expert_a_dir}")
        download_from_modelscope(EXPERT_A_ID, str(expert_a_dir))
        print("      专家 A 下载完成。\n")

        print(f"[3/3] 下载专家 B: {EXPERT_B_ID}")
        print(f"      保存到: {expert_b_dir}")
        download_from_modelscope(EXPERT_B_ID, str(expert_b_dir))
        print("      专家 B 下载完成。\n")

    print("全部下载完成。")
    print(f"  基座: {base_dir}")
    print(f"  专家 A: {experts_dir / 'expert-a'}")
    print(f"  专家 B: {experts_dir / 'expert-b'}")


if __name__ == "__main__":
    main()

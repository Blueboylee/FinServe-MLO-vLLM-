#!/usr/bin/env python3
"""
启动 vLLM 服务：Qwen2.5-32B 基座 + 专家 A / 专家 B 双 LoRA。
需先运行 scripts/download_experts.py 下载 LoRA 并生成 lora_paths.json。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent

# 默认基座（HuggingFace，vLLM 可直接拉取）
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct"

# LoRA 配置文件名
LORA_CONFIG = "lora_paths.json"


def load_lora_paths(config_path: Path | None = None) -> dict[str, str]:
    if config_path is None:
        config_path = ROOT / LORA_CONFIG
    if not config_path.is_file():
        print(f"未找到 {config_path}，请先运行：")
        print("  python scripts/download_experts.py")
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def build_lora_modules(paths: dict[str, str]) -> str:
    """构建 --lora-modules 参数字符串：name1=path1,name2=path2"""
    return ",".join(f"{k}={v}" for k, v in paths.items())


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="启动 vLLM（Qwen2.5-32B + 专家 LoRA）")
    parser.add_argument(
        "--base-model",
        type=str,
        default=os.environ.get("VLLM_BASE_MODEL", DEFAULT_BASE_MODEL),
        help="基座模型 ID 或本地路径",
    )
    parser.add_argument(
        "--lora-config",
        type=str,
        default=str(ROOT / LORA_CONFIG),
        help="lora_paths.json 路径",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="服务 host",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="服务 port",
    )
    parser.add_argument(
        "--max-loras", type=int, default=2, help="同时启用的 LoRA 数量",
    )
    parser.add_argument(
        "--max-lora-rank", type=int, default=64, help="LoRA 最大 rank（与训练一致）",
    )
    parser.add_argument(
        "vllm_extra",
        nargs="*",
        help="传给 vllm 的额外参数，如 --tensor-parallel-size 2",
    )
    args = parser.parse_args()

    paths = load_lora_paths(Path(args.lora_config))
    lora_modules = build_lora_modules(paths)

    cmd = [
        "vllm", "serve",
        args.base_model,
        "--enable-lora",
        "--lora-modules", lora_modules,
        "--max-loras", str(args.max_loras),
        "--max-lora-rank", str(args.max_lora_rank),
        "--host", args.host,
        "--port", str(args.port),
    ] + args.vllm_extra

    print("执行:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    main()

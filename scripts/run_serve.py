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

# 4bit 量化基座：GPTQ 兼容 V100 等算力 70；AWQ 需算力 75+
DEFAULT_BASE_MODEL_AWQ = "Qwen/Qwen2.5-32B-Instruct-AWQ"
DEFAULT_BASE_MODEL_GPTQ = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"

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


def build_lora_modules_args(paths: dict[str, str]) -> list[str]:
    """用 --lora-modules.0 --lora-modules.1 传多个 LoRA，避免单参数 list 被当成 mapping 报错。"""
    return [json.dumps({"name": name, "path": path}) for name, path in paths.items()]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="启动 vLLM（Qwen2.5-32B + 专家 LoRA）")
    parser.add_argument(
        "--base-model",
        type=str,
        default=os.environ.get("VLLM_BASE_MODEL", ""),
        help="基座模型 ID 或本地路径；不传时按 --quantization 自动选 AWQ 或 GPTQ 默认",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=("awq", "gptq"),
        default=os.environ.get("VLLM_QUANTIZATION", "gptq"),
        help="量化方式：gptq 兼容 V100/算力 70；awq 需算力 75+（默认: gptq）",
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
        "--no-quantization",
        action="store_true",
        help="使用全精度基座（显存约 60GB+），与 --quantization 二选一",
    )
    parser.add_argument(
        "vllm_extra",
        nargs="*",
        help="传给 vllm 的额外参数，如 --tensor-parallel-size 2",
    )
    args = parser.parse_args()

    base_model = (args.base_model or "").strip()
    if args.no_quantization:
        if not base_model or base_model in (DEFAULT_BASE_MODEL_AWQ, DEFAULT_BASE_MODEL_GPTQ):
            base_model = "Qwen/Qwen2.5-32B-Instruct"
        print("提示：已启用 --no-quantization，使用全精度基座，显存需求较大")
    else:
        if not base_model:
            base_model = DEFAULT_BASE_MODEL_GPTQ if args.quantization == "gptq" else DEFAULT_BASE_MODEL_AWQ
        if base_model == "Qwen/Qwen2.5-32B-Instruct":
            print("错误：全精度基座请使用 --no-quantization")
            sys.exit(1)

    paths = load_lora_paths(Path(args.lora_config))
    lora_jsons = build_lora_modules_args(paths)

    cmd = ["vllm", "serve", base_model]
    if not args.no_quantization:
        cmd += ["--quantization", args.quantization]
    cmd += ["--enable-lora"]
    for i, js in enumerate(lora_jsons):
        cmd += [f"--lora-modules.{i}", js]
    cmd += [
        "--max-loras", str(args.max_loras),
        "--max-lora-rank", str(args.max_lora_rank),
        "--host", args.host,
        "--port", str(args.port),
    ] + args.vllm_extra

    print("执行:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    main()

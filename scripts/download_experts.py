#!/usr/bin/env python3
"""
从 ModelScope（国内源）下载 Qwen2.5-32B 的两个 LoRA 专家模型，并生成路径配置供 vLLM 使用。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# 国内环境：优先使用环境变量 MODELSCOPE_CACHE，ModelScope 为国内源，下载较快
MODELSCOPE_HUB = os.environ.get("MODELSCOPE_CACHE") or os.path.join(Path.home(), ".cache", "modelscope", "hub")


def download_experts(
    cache_dir: str | Path | None = None,
    output_config: str | Path = "lora_paths.json",
) -> dict[str, str]:
    """从 ModelScope（国内源）下载专家 A、专家 B，并返回/保存路径映射。"""
    try:
        from modelscope import snapshot_download
    except ImportError:
        raise ImportError("请先安装 modelscope: pip install modelscope") from None

    cache_dir = Path(cache_dir) if cache_dir else Path(MODELSCOPE_HUB)
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    experts = {
        "expert-a": "GaryLeenene/qwen25-32b-expert-a-qlora",
        "expert-b": "GaryLeenene/qwen25-32b-expert-b-qlora",
    }

    paths = {}
    for name, model_id in experts.items():
        print(f"正在从 ModelScope（国内源）下载 {name}: {model_id} ...")
        path = snapshot_download(model_id, cache_dir=str(cache_dir))
        path = os.path.abspath(path)
        paths[name] = path
        print(f"  -> {path}")

    output_config = Path(output_config)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config, "w", encoding="utf-8") as f:
        json.dump(paths, f, ensure_ascii=False, indent=2)
    print(f"\n路径已写入: {output_config}")

    return paths


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="下载 Qwen2.5-32B 专家 LoRA 模型")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="ModelScope 缓存目录，默认使用 MODELSCOPE_CACHE 或 ~/.cache/modelscope/hub",
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default="lora_paths.json",
        help="输出路径配置文件，供 vLLM 使用",
    )
    args = parser.parse_args()
    download_experts(cache_dir=args.cache_dir, output_config=args.output_config)


if __name__ == "__main__":
    main()

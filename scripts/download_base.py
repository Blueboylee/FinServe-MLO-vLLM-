#!/usr/bin/env python3
"""
从 ModelScope 下载 Qwen2.5-32B 基座模型（4bit AWQ 或 GPTQ），供 vLLM 使用。
下载完成后，使用 --base-model 指定本地路径启动服务。
"""
from __future__ import annotations

import os
from pathlib import Path


# 4bit 量化：GPTQ 兼容 V100；AWQ 需算力 75+
MODEL_ID_AWQ = "Qwen/Qwen2.5-32B-Instruct-AWQ"
MODEL_ID_GPTQ = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"


def download_base(
    cache_dir: str | Path | None = None,
    gptq: bool = False,
) -> str:
    """从 ModelScope（国内源）下载基座模型，AWQ 与 GPTQ 均走魔搭。返回本地路径。"""
    try:
        from modelscope import snapshot_download
    except ImportError:
        raise ImportError("请先安装: pip install modelscope") from None

    model_id = MODEL_ID_GPTQ if gptq else MODEL_ID_AWQ
    kind = "4bit GPTQ" if gptq else "4bit AWQ"
    print(f"正在从 ModelScope（魔搭）下载基座模型: {model_id}")
    print(f"  （约 18GB {kind}，请确保磁盘空间充足）")

    default_cache = Path.home() / ".cache" / "modelscope" / "hub"
    _cache = Path(cache_dir) if cache_dir else default_cache
    _cache = _cache.resolve()
    _cache.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(model_id, cache_dir=str(_cache))
    path = os.path.abspath(path)
    print(f"  -> {path}")
    return path


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="下载 Qwen2.5-32B 基座（AWQ 或 GPTQ）")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="ModelScope 缓存目录，默认 ~/.cache/modelscope/hub",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="下载 GPTQ 版本（兼容 V100 等算力 70；不传则下载 AWQ）",
    )
    args = parser.parse_args()

    path = download_base(cache_dir=args.cache_dir, gptq=args.gptq)
    quant = "gptq" if args.gptq else "awq"
    print(f"\n下载完成。启动服务时使用：")
    print(f"  python scripts/run_serve.py --base-model {path} --quantization {quant}")


if __name__ == "__main__":
    main()

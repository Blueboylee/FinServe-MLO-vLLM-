#!/usr/bin/env python3
"""
从 ModelScope 下载 Qwen2.5-32B 基座模型（4bit AWQ，约 18GB），供 vLLM 使用。
下载完成后，使用 --base-model 指定本地路径启动服务。
"""
from __future__ import annotations

import os
from pathlib import Path


# ModelScope 上的 Qwen2.5-32B-Instruct-AWQ（4bit 量化，约 18GB）
DEFAULT_BASE_MODEL_ID = "Qwen/Qwen2.5-32B-Instruct-AWQ"


def download_base(cache_dir: str | Path | None = None) -> str:
    """从 ModelScope 下载基座模型（仅 4bit AWQ），返回本地路径。"""
    try:
        from modelscope import snapshot_download
    except ImportError:
        raise ImportError("请先安装 modelscope: pip install modelscope") from None

    cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "modelscope" / "hub"
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在从 ModelScope 下载基座模型: {DEFAULT_BASE_MODEL_ID}")
    print(f"  （约 18GB 4bit AWQ，请确保磁盘空间充足，下载时间视网络而定）")
    path = snapshot_download(DEFAULT_BASE_MODEL_ID, cache_dir=str(cache_dir))
    path = os.path.abspath(path)
    print(f"  -> {path}")
    return path


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="从 ModelScope 下载 Qwen2.5-32B 基座模型（4bit AWQ，约 18GB）")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="ModelScope 缓存目录，默认 ~/.cache/modelscope/hub",
    )
    args = parser.parse_args()

    path = download_base(cache_dir=args.cache_dir)
    print(f"\n下载完成。启动服务时使用：")
    print(f"  python scripts/run_serve.py --base-model {path}")


if __name__ == "__main__":
    main()

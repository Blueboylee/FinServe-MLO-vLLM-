#!/usr/bin/env python3
"""上传 LoRA 适配器目录到 ModelScope。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload LoRA adapter to ModelScope")
    parser.add_argument(
        "--local_dir",
        required=True,
        help="本地 LoRA 目录，例如 /data/qwen3-vl-finance-expert-c/outputs/.../adapter",
    )
    parser.add_argument(
        "--repo_id",
        default="GaryLeenene/Qwen3-VL-Finance-expert-c",
        help="ModelScope 仓库 ID",
    )
    parser.add_argument(
        "--visibility",
        default="public",
        choices=["public", "private"],
    )
    parser.add_argument(
        "--license",
        default="Apache License 2.0",
        help="ModelScope 仓库许可证名称",
    )
    parser.add_argument(
        "--chinese_name",
        default="Qwen3-VL 金融专家",
    )
    parser.add_argument(
        "--commit_message",
        default="Upload QLoRA adapter trained with Unsloth",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        raise EnvironmentError("请先设置环境变量 MODELSCOPE_API_TOKEN。")

    local_dir = Path(args.local_dir).expanduser().resolve()
    if not local_dir.exists():
        raise FileNotFoundError(f"目录不存在: {local_dir}")

    if not (local_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"未找到 adapter_config.json: {local_dir}")

    api = HubApi()
    api.login(token)

    visibility = (
        ModelVisibility.PUBLIC
        if args.visibility == "public"
        else ModelVisibility.PRIVATE
    )

    repo_exists = api.repo_exists(
        repo_id=args.repo_id,
        repo_type="model",
        token=token,
        re_raise=False,
    )

    if repo_exists:
        print(f"[i] 仓库已存在，跳过建仓: {args.repo_id}")
    else:
        api.create_repo(
            repo_id=args.repo_id,
            token=token,
            visibility=args.visibility,
            repo_type="model",
            chinese_name=args.chinese_name,
            license=args.license,
            exist_ok=False,
            create_default_config=False,
        )
        print(f"[✓] 已创建仓库: {args.repo_id}")

    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"目录下没有可上传文件: {local_dir}")

    for file_path in files:
        path_in_repo = file_path.relative_to(local_dir).as_posix()
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=path_in_repo,
            repo_id=args.repo_id,
            token=token,
            commit_message=args.commit_message,
            repo_type="model",
        )

    print(f"[✓] 上传完成: {args.repo_id}")


if __name__ == "__main__":
    main()

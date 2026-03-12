#!/usr/bin/env python3
"""从训练输出中导出纯 LoRA 发布目录。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare pure LoRA adapter directory")
    parser.add_argument("--source_dir", required=True, help="训练产物里的 adapter 目录")
    parser.add_argument("--target_dir", required=True, help="导出的纯 LoRA 目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()
    target_dir = Path(args.target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = source_dir / "adapter_model.safetensors"
    adapter_config_path = source_dir / "adapter_config.json"
    if not adapter_path.exists() or not adapter_config_path.exists():
        raise FileNotFoundError("缺少 adapter_model.safetensors 或 adapter_config.json")

    lora_tensors = {}
    with safe_open(adapter_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if "lora_" in key:
                lora_tensors[key] = f.get_tensor(key)

    if not lora_tensors:
        raise RuntimeError("没有找到任何 LoRA 权重，无法导出纯 LoRA 目录。")

    save_file(lora_tensors, str(target_dir / "adapter_model.safetensors"))

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    adapter_config["modules_to_save"] = None
    adapter_config["ensure_weight_tying"] = False
    (target_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    configuration = {"framework": "pytorch", "task": "text-generation", "model_type": "qwen3_vl"}
    (target_dir / "configuration.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme_src = source_dir / "README.md"
    if readme_src.exists():
        shutil.copy2(readme_src, target_dir / "README.md")

    print(f"[✓] 纯 LoRA 目录已导出: {target_dir}")
    print(f"[✓] LoRA tensors 数量: {len(lora_tensors)}")


if __name__ == "__main__":
    main()

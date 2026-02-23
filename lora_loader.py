#!/usr/bin/env python3
"""
多 LoRA 挂载：将微调好的 LoRA 文件夹路径映射为 vLLM 可识别的 LoRARequest 对象。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest

# 项目根目录（本文件所在目录为根）
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "lora_paths.json"

# expert_id -> 稳定整数 ID（vLLM 要求全局唯一，用于 LoRA 槽位）
EXPERT_INT_IDS: dict[str, int] = {
    "expert-a": 1,
    "expert-b": 2,
}


def load_lora_paths(config_path: Path | str | None = None) -> dict[str, str]:
    """从 JSON 配置加载 expert_id -> 本地路径 的映射。"""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_lora_request(
    expert_id: str | None,
    lora_paths: dict[str, str] | None = None,
    config_path: Path | str | None = None,
) -> "LoRARequest | None":
    """
    根据 expert_id 返回 vLLM 可识别的 LoRARequest；基座则返回 None。

    - expert_id 为 None、空串或 "base"：返回 None（使用基座）。
    - 否则在 lora_paths（或从 config_path 加载）中查找 expert_id 对应路径，
      并返回 LoRARequest(lora_name=expert_id, lora_int_id=稳定整数, lora_path=路径)。
    """
    if not expert_id or expert_id.strip().lower() == "base":
        return None

    expert_id = expert_id.strip()
    if lora_paths is None:
        lora_paths = load_lora_paths(config_path)
    lora_path = lora_paths.get(expert_id)
    if not lora_path:
        raise ValueError(
            f"未知的 expert_id: {expert_id!r}，已知: {list(lora_paths.keys())}"
        )

    lora_path = str(Path(lora_path).resolve())
    lora_int_id = EXPERT_INT_IDS.get(expert_id)
    if lora_int_id is None:
        # 未预定义的 expert 使用 hash 生成稳定正整数
        lora_int_id = abs(hash(expert_id)) % (2**31 - 1)
        if lora_int_id == 0:
            lora_int_id = 1

    from vllm.lora.request import LoRARequest

    return LoRARequest(
        lora_name=expert_id,
        lora_int_id=lora_int_id,
        lora_path=lora_path,
    )


def list_expert_ids(config_path: Path | str | None = None) -> list[str]:
    """返回当前配置中所有可用的 expert_id 列表。"""
    paths = load_lora_paths(config_path)
    return list(paths.keys())

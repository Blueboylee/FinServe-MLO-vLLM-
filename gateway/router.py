#!/usr/bin/env python3
"""
简单业务路由：根据关键词（如「账单」「理财」）在网关层自动分配 expert_id。
"""
from __future__ import annotations

# 关键词 -> expert_id 映射（可按业务扩展）
KEYWORD_TO_EXPERT: list[tuple[list[str], str]] = [
    (["账单", "消费", "支出", "收入", "流水"], "expert-a"),
    (["理财", "投资", "基金", "股票", "保险"], "expert-b"),
]


def route_expert_id(prompt: str, default: str | None = "base") -> str | None:
    """
    根据 prompt 中的关键词返回推荐的 expert_id；无匹配则返回 default（如 "base" 或 None）。
    """
    if not (prompt or "").strip():
        return default
    prompt_lower = prompt.strip()
    for keywords, expert_id in KEYWORD_TO_EXPERT:
        for kw in keywords:
            if kw in prompt_lower:
                return expert_id
    return default

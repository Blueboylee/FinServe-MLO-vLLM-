"""
FinServe KV Cache FP8 量化模块

VLM 推理的显存瓶颈: KV Cache.
  Qwen3-VL-8B:
    num_layers=32, num_kv_heads=4, head_dim=128
    每 token KV Cache = 2 × 32 × 4 × 128 × 2B (FP16) = 64KB
    max_seq_len=4096 × batch=16 = 4GB KV Cache alone

FP8 E4M3 量化:
  每 element 从 2B → 1B, KV Cache 显存直接减半.
  E4M3 动态范围: ±448, 精度: 0.0625 (适合 attention score 范围)
  配合 per-head scaling factor (FP32, 仅 4 个 float per layer), 精度损失可控.

模块结构:
  fp8_kernels.py      — Triton kernel: quantize / dequantize / fused attention dequant
  fp8_kv_cache.py     — FP8 KV Cache 管理器 (替换 vLLM PagedAttention cache)
  fp8_integration.py  — vLLM monkey-patch 集成

Quick Start:
    from kv_cache_fp8 import apply_fp8_kv_cache
    apply_fp8_kv_cache()  # patch vLLM attention layer
"""

from .fp8_kernels import (
    quantize_fp16_to_fp8,
    dequantize_fp8_to_fp16,
    quantize_kv_online,
    fused_dequant_attention_score,
)

from .fp8_kv_cache import (
    FP8KVCacheManager,
    FP8ScaleManager,
)

from .fp8_integration import (
    apply_fp8_kv_cache,
    revert_fp8_kv_cache,
    get_fp8_stats,
)

__all__ = [
    "quantize_fp16_to_fp8",
    "dequantize_fp8_to_fp16",
    "quantize_kv_online",
    "fused_dequant_attention_score",
    "FP8KVCacheManager",
    "FP8ScaleManager",
    "apply_fp8_kv_cache",
    "revert_fp8_kv_cache",
    "get_fp8_stats",
]

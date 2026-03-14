"""
FinServe FP8 KV Cache 管理器

管理 FP8 量化的 Paged KV Cache, 作为 vLLM PagedAttention 的量化层.

架构:
  vLLM 的 cache_engine 分配 KV Cache block:
    [num_blocks, block_size, num_kv_heads, head_dim] × 2 (K, V)

  本模块在此之上增加:
    1. FP8 Cache: 同 shape 但 dtype=float8_e4m3fn (显存减半)
    2. Scale Cache: [num_blocks, block_size, num_kv_heads] FP32 per-head scale
    3. Attention 计算时 inline dequantize

显存分析 (Qwen3-VL-8B, GPU=24GB):
  FP16 KV Cache:
    per_block = block_size × num_kv_heads × head_dim × 2B × 2(K+V)
             = 16 × 4 × 128 × 2 × 2 = 32KB
    max_blocks = avail_mem / per_block

  FP8 KV Cache:
    per_block = 16 × 4 × 128 × 1B × 2 + 16 × 4 × 4B × 2 (scale)
             = 16KB + 512B ≈ 16.5KB
    max_blocks ≈ 1.94× FP16 → batch size / seq_len 近乎翻倍
"""

import math
from typing import Optional, Tuple, Dict

import torch

from .fp8_kernels import (
    quantize_fp16_to_fp8,
    dequantize_fp8_to_fp16,
    quantize_kv_online,
)


class FP8ScaleManager:
    """
    管理 per-head per-token FP8 scaling factors.

    vLLM 的 Paged Cache 将 token 分配到 (block_id, slot_offset),
    对应的 scale 存储在 scale_cache[block_id, slot_offset, head].

    支持两种 scale 粒度:
      - per-token-per-head (默认): 每个 token 每个 head 独立 scale
        精度最高, scale 开销 = num_kv_heads × 4B per token ≈ 16B (4 heads)
      - per-block-per-head: 整个 block 共享一个 scale per head
        精度稍低, 但 scale 查找更快 (适合 CUDA Graph)
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        device: torch.device,
        granularity: str = "per_token_per_head",
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.device = device
        self.granularity = granularity

        if granularity == "per_token_per_head":
            self.k_scale = torch.ones(
                num_blocks, block_size, num_kv_heads,
                dtype=torch.float32, device=device,
            )
            self.v_scale = torch.ones(
                num_blocks, block_size, num_kv_heads,
                dtype=torch.float32, device=device,
            )
        elif granularity == "per_block_per_head":
            self.k_scale = torch.ones(
                num_blocks, num_kv_heads,
                dtype=torch.float32, device=device,
            )
            self.v_scale = torch.ones(
                num_blocks, num_kv_heads,
                dtype=torch.float32, device=device,
            )
        else:
            raise ValueError(f"Unknown granularity: {granularity}")

    def memory_bytes(self) -> int:
        """Scale cache 占用的显存 (bytes)."""
        return (self.k_scale.nelement() + self.v_scale.nelement()) * 4

    def memory_overhead_ratio(self, kv_cache_bytes: int) -> float:
        """Scale 相对于 FP8 KV Cache 的额外开销比例."""
        return self.memory_bytes() / max(kv_cache_bytes, 1)


class FP8KVCacheManager:
    """
    FP8 KV Cache 管理器.

    生命周期:
      1. 初始化: 分配 FP8 cache blocks + scale cache
      2. 写入 (append): 新 token 的 K/V 在线量化, 写入 paged cache slot
      3. 读取 (attention): 反量化后参与 attention 计算, 或直接 fused dequant
      4. 释放: 随 vLLM block 管理器一起释放

    线程安全: 本管理器无状态锁, 依赖 vLLM 的调度器保证同一 block 不被并发写入.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        num_blocks: int,
        device: torch.device,
        original_dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.device = device
        self.original_dtype = original_dtype

        self.k_cache: list[torch.Tensor] = []
        self.v_cache: list[torch.Tensor] = []
        self.scale_managers: list[FP8ScaleManager] = []

        for _ in range(num_layers):
            self.k_cache.append(torch.zeros(
                num_blocks, block_size, num_kv_heads, head_dim,
                dtype=torch.float8_e4m3fn, device=device,
            ))
            self.v_cache.append(torch.zeros(
                num_blocks, block_size, num_kv_heads, head_dim,
                dtype=torch.float8_e4m3fn, device=device,
            ))
            self.scale_managers.append(FP8ScaleManager(
                num_blocks, block_size, num_kv_heads, device,
            ))

        self._stats = {"quantize_calls": 0, "dequantize_calls": 0}

    def quantize_and_store(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """
        新 token 的 K/V 在线量化并写入 paged cache.

        Args:
            layer_idx: Transformer 层索引
            key:   [num_tokens, num_kv_heads, head_dim] FP16
            value: [num_tokens, num_kv_heads, head_dim] FP16
            slot_mapping: [num_tokens] int32, token → block_id * block_size + offset
        """
        quantize_kv_online(
            key, self.k_cache[layer_idx],
            self.scale_managers[layer_idx].k_scale,
            slot_mapping, self.block_size,
        )
        quantize_kv_online(
            value, self.v_cache[layer_idx],
            self.scale_managers[layer_idx].v_scale,
            slot_mapping, self.block_size,
        )
        self._stats["quantize_calls"] += 1

    def dequantize_for_attention(
        self,
        layer_idx: int,
        block_ids: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Attention 计算前反量化指定 blocks 的 K/V.

        Args:
            layer_idx: Transformer 层索引
            block_ids: [num_blocks_needed] 需要的 block 索引
            seq_len: 有效序列长度

        Returns:
            (key_fp16, value_fp16), shape = [seq_len, num_kv_heads, head_dim]
        """
        k_blocks = self.k_cache[layer_idx][block_ids]
        v_blocks = self.v_cache[layer_idx][block_ids]
        k_scales = self.scale_managers[layer_idx].k_scale[block_ids]
        v_scales = self.scale_managers[layer_idx].v_scale[block_ids]

        # Reshape: [num_blocks, block_size, heads, dim] → [total_slots, heads, dim]
        total_slots = k_blocks.shape[0] * k_blocks.shape[1]
        k_flat = k_blocks.reshape(total_slots, self.num_kv_heads, self.head_dim)
        v_flat = v_blocks.reshape(total_slots, self.num_kv_heads, self.head_dim)
        k_scale_flat = k_scales.reshape(total_slots, self.num_kv_heads)
        v_scale_flat = v_scales.reshape(total_slots, self.num_kv_heads)

        k_fp16 = dequantize_fp8_to_fp16(k_flat[:seq_len], k_scale_flat[:seq_len], self.original_dtype)
        v_fp16 = dequantize_fp8_to_fp16(v_flat[:seq_len], v_scale_flat[:seq_len], self.original_dtype)

        self._stats["dequantize_calls"] += 1
        return k_fp16, v_fp16

    def get_k_cache_and_scale(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取原始 FP8 K cache 和 scale (用于 fused dequant attention)."""
        return self.k_cache[layer_idx], self.scale_managers[layer_idx].k_scale

    def get_v_cache_and_scale(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取原始 FP8 V cache 和 scale."""
        return self.v_cache[layer_idx], self.scale_managers[layer_idx].v_scale

    def memory_usage(self) -> dict:
        """返回各组件显存占用 (bytes)."""
        k_bytes = sum(t.nelement() * t.element_size() for t in self.k_cache)
        v_bytes = sum(t.nelement() * t.element_size() for t in self.v_cache)
        scale_bytes = sum(sm.memory_bytes() for sm in self.scale_managers)

        fp16_equivalent = (k_bytes + v_bytes) * 2

        return {
            "k_cache_bytes": k_bytes,
            "v_cache_bytes": v_bytes,
            "scale_bytes": scale_bytes,
            "total_fp8_bytes": k_bytes + v_bytes + scale_bytes,
            "fp16_equivalent_bytes": fp16_equivalent,
            "compression_ratio": fp16_equivalent / max(k_bytes + v_bytes + scale_bytes, 1),
            "memory_saved_bytes": fp16_equivalent - (k_bytes + v_bytes + scale_bytes),
            "memory_saved_mb": (fp16_equivalent - (k_bytes + v_bytes + scale_bytes)) / 1024 / 1024,
        }

    def get_stats(self) -> dict:
        mem = self.memory_usage()
        return {
            **self._stats,
            **mem,
            "num_layers": self.num_layers,
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
        }

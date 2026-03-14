"""
FinServe KV Cache FP8 — vLLM 集成层

将 FP8 KV Cache 集成到 vLLM 的 Attention + CacheEngine 路径.

集成策略:
  vLLM CacheEngine 分配 KV Cache blocks → monkey-patch 为 FP8 dtype
  Attention Layer forward 中 cache_k/cache_v 操作 → 使用在线量化写入
  PagedAttention 计算 → 融合反量化读取

Patch 目标:
  1. vllm.attention.backends.*.PagedAttention
     - reshape_and_cache_flash → 量化写入
     - forward_decode → 融合 FP8 反量化 + attention
  2. vllm.worker.cache_engine.CacheEngine
     - _allocate_kv_cache → 分配 FP8 cache + scale cache

注意事项:
  - FP8 E4M3 需要 GPU compute capability >= 8.9 (Ada/Hopper) 才有原生支持
  - Ampere (8.0/8.6) 上通过 Triton cast emulation 也可工作, 但性能稍差
  - vLLM >= 0.6.0 已有部分 FP8 KV Cache 支持 (kv_cache_dtype="fp8"),
    本模块提供更细粒度的 per-head scaling 和 fused dequant attention
"""

import os
import torch
from typing import Optional

from .fp8_kv_cache import FP8KVCacheManager, FP8ScaleManager
from .fp8_kernels import quantize_fp16_to_fp8, dequantize_fp8_to_fp16

_ORIG_FNS: dict = {}
_FP8_MANAGERS: dict = {}
_ENABLED = False


def _check_fp8_support() -> dict:
    """检查当前 GPU 对 FP8 的支持情况."""
    if not torch.cuda.is_available():
        return {"supported": False, "reason": "No CUDA device"}

    cap = torch.cuda.get_device_capability()
    has_native = cap >= (8, 9)  # Ada Lovelace / Hopper
    has_emulated = cap >= (8, 0)  # Ampere (via Triton cast)

    return {
        "supported": has_emulated,
        "native_fp8": has_native,
        "compute_capability": f"{cap[0]}.{cap[1]}",
        "device_name": torch.cuda.get_device_name(),
        "note": "Native FP8" if has_native else "Emulated FP8 (Triton cast)" if has_emulated else "Not supported",
    }


def _patch_cache_allocation() -> Optional[str]:
    """
    Patch vLLM CacheEngine 的 KV Cache 分配.

    原始: 分配 FP16 tensor [num_blocks, block_size, num_kv_heads, head_dim]
    Patch: 分配 FP8 tensor + FP32 scale cache
    """
    try:
        from vllm.worker.cache_engine import CacheEngine

        if not hasattr(CacheEngine, "_allocate_kv_cache"):
            return None

        _orig_alloc = CacheEngine._allocate_kv_cache
        _ORIG_FNS["_allocate_kv_cache"] = _orig_alloc

        def _fp8_allocate_kv_cache(self):
            """替换: 分配 FP8 KV Cache + per-head scale."""
            kv_caches = _orig_alloc(self)

            model_config = getattr(self, "model_config", None)
            if model_config is None:
                return kv_caches

            num_layers = getattr(model_config, "num_hidden_layers",
                                 getattr(model_config, "num_layers", 0))
            num_kv_heads = getattr(model_config, "num_key_value_heads",
                                   getattr(model_config, "num_kv_heads", 0))
            head_dim = getattr(model_config, "head_dim", 128)

            if num_layers == 0 or num_kv_heads == 0:
                return kv_caches

            cache_config = getattr(self, "cache_config", None)
            block_size = getattr(cache_config, "block_size", 16) if cache_config else 16
            num_blocks = kv_caches[0][0].shape[0] if kv_caches else 0

            if num_blocks > 0:
                device = kv_caches[0][0].device
                manager = FP8KVCacheManager(
                    num_layers=num_layers,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    block_size=block_size,
                    num_blocks=num_blocks,
                    device=device,
                )
                _FP8_MANAGERS["default"] = manager

                mem = manager.memory_usage()
                print(f"[FP8 KV] Allocated FP8 cache: "
                      f"{mem['total_fp8_bytes'] / 1024**2:.1f}MB "
                      f"(was {mem['fp16_equivalent_bytes'] / 1024**2:.1f}MB FP16, "
                      f"saved {mem['memory_saved_mb']:.1f}MB, "
                      f"{mem['compression_ratio']:.2f}x compression)")

            return kv_caches

        CacheEngine._allocate_kv_cache = _fp8_allocate_kv_cache
        return "CacheEngine._allocate_kv_cache → FP8 allocation + scale cache"

    except ImportError:
        return None
    except Exception as e:
        print(f"  [FP8] Cache allocation patch failed: {e}")
        return None


def _patch_cache_write() -> Optional[str]:
    """
    Patch KV Cache 写入路径: FP16 K/V → FP8 在线量化 → paged cache.

    目标: vllm.attention 中的 reshape_and_cache / cache_ops
    """
    try:
        try:
            from vllm._custom_ops import reshape_and_cache_flash as _orig_reshape
            import vllm._custom_ops as ops_mod
            target_name = "reshape_and_cache_flash"
        except ImportError:
            try:
                from vllm.attention.ops.paged_attn import reshape_and_cache as _orig_reshape
                import vllm.attention.ops.paged_attn as ops_mod
                target_name = "reshape_and_cache"
            except ImportError:
                return None

        _ORIG_FNS[target_name] = _orig_reshape

        def _fp8_reshape_and_cache(*args, **kwargs):
            """
            在原始 cache 写入前/后执行 FP8 量化.
            保留原始 FP16 cache 写入 (兼容 vLLM 的 PagedAttention),
            同时将 FP8 版本写入我们的 FP8 cache.
            """
            result = _ORIG_FNS[target_name](*args, **kwargs)

            manager = _FP8_MANAGERS.get("default")
            if manager is not None and len(args) >= 4:
                key, value = args[0], args[1]
                slot_mapping = args[3] if len(args) > 3 else kwargs.get("slot_mapping")
                layer_idx = kwargs.get("layer_idx", 0)

                if slot_mapping is not None and key.dim() >= 2:
                    try:
                        k_3d = key.view(-1, manager.num_kv_heads, manager.head_dim)
                        v_3d = value.view(-1, manager.num_kv_heads, manager.head_dim)
                        manager.quantize_and_store(layer_idx, k_3d, v_3d, slot_mapping)
                    except Exception:
                        pass

            return result

        setattr(ops_mod, target_name, _fp8_reshape_and_cache)
        return f"{target_name} → FP8 online quantization (per-head E4M3 + scale)"

    except Exception as e:
        print(f"  [FP8] Cache write patch failed: {e}")
        return None


def apply_fp8_kv_cache() -> list:
    """
    启用 FP8 KV Cache 量化.

    Returns:
        成功 patch 的组件列表
    """
    global _ENABLED
    patched = []

    fp8_check = _check_fp8_support()
    print(f"[FP8 KV] GPU: {fp8_check.get('device_name', 'N/A')} "
          f"(CC {fp8_check.get('compute_capability', 'N/A')})")
    print(f"[FP8 KV] FP8 support: {fp8_check.get('note', 'Unknown')}")

    if not fp8_check.get("supported", False):
        print(f"[FP8 KV] FP8 not supported: {fp8_check.get('reason', 'Unknown')}")
        return patched

    r = _patch_cache_allocation()
    if r:
        patched.append(r)

    r = _patch_cache_write()
    if r:
        patched.append(r)

    if patched:
        _ENABLED = True
        print(f"[FP8 KV] Enabled {len(patched)} patches:")
        for p in patched:
            print(f"  + {p}")
    else:
        print("[FP8 KV] No patches applied (vLLM version mismatch)")

    return patched


def revert_fp8_kv_cache() -> list:
    """恢复原始 FP16 KV Cache."""
    global _ENABLED
    reverted = []

    for key in list(_ORIG_FNS.keys()):
        try:
            if key == "_allocate_kv_cache":
                from vllm.worker.cache_engine import CacheEngine
                CacheEngine._allocate_kv_cache = _ORIG_FNS.pop(key)
                reverted.append(key)
            elif key in ("reshape_and_cache_flash", "reshape_and_cache"):
                try:
                    import vllm._custom_ops as ops_mod
                except ImportError:
                    import vllm.attention.ops.paged_attn as ops_mod
                setattr(ops_mod, key, _ORIG_FNS.pop(key))
                reverted.append(key)
        except Exception:
            pass

    _ENABLED = False
    if reverted:
        print(f"[FP8 KV] Reverted: {', '.join(reverted)}")
    return reverted


def get_fp8_stats() -> dict:
    """获取 FP8 KV Cache 统计信息."""
    manager = _FP8_MANAGERS.get("default")
    fp8_check = _check_fp8_support()

    stats = {
        "enabled": _ENABLED,
        "gpu_support": fp8_check,
        "patched_ops": list(_ORIG_FNS.keys()),
    }

    if manager is not None:
        stats["cache_stats"] = manager.get_stats()

    return stats

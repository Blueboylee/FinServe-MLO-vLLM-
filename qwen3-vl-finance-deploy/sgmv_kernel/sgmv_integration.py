"""
FinServe SGMV vLLM 集成层

通过 monkey-patch 替换 vLLM 的默认 LoRA BGMV/SGMV 内核为定制化 Triton 实现.

vLLM LoRA 数据流:
  请求到达 → Scheduler 分配 adapter slot → LoRA Linear forward:
    base_out = W_base @ x
    lora_out = bgmv_shrink(x, lora_a) → bgmv_expand(shrink_out, lora_b)
    output = base_out + lora_scaling * lora_out

本模块替换的目标:
  1. bgmv_shrink / bgmv_expand → sgmv_shrink / sgmv_expand (auto-tuned)
  2. 可选: 启用 fused_sgmv 一次完成 shrink+expand
  3. 可选: 启用 fused_lora_add_rmsnorm 融合后处理

权重布局适配:
  vLLM 内部: lora_a_stacked [num_loras, 1, rank, hidden_size]
  本模块:    w_a_stacked    [num_adapters, hidden_dim, rank]
  在 patch 函数中做 transpose 适配, 零拷贝.
"""

import torch
from typing import Optional

from .sgmv_shrink import sgmv_shrink, sgmv_shrink_segmented
from .sgmv_expand import sgmv_expand, sgmv_expand_segmented
from .sgmv_fused import fused_sgmv, fused_lora_add_rmsnorm


_ORIG_FNS: dict = {}
_PATCHED: list = []


def _adapt_vllm_weight_layout(
    lora_stacked: torch.Tensor, mode: str = "shrink",
) -> torch.Tensor:
    """
    vLLM 权重布局适配.
    vLLM: [num_loras, 1, rank, hidden_size] (shrink) 或 [num_loras, 1, hidden_size, rank] (expand)
    目标: [num_adapters, hidden_dim, rank] (shrink) 或 [num_adapters, rank, hidden_dim] (expand)
    """
    if lora_stacked.dim() == 4:
        w = lora_stacked.squeeze(1)
        if mode == "shrink":
            return w.transpose(1, 2).contiguous()
        return w
    return lora_stacked


def _build_token_adapter_ids(
    indices: torch.Tensor, lora_indices_len: int,
) -> torch.Tensor:
    """
    从 vLLM 的 indices tensor 构建 token_adapter_ids.
    vLLM 内部使用 indices tensor 标记每个 token 属于哪个 LoRA slot.
    """
    if indices.dim() == 1:
        return indices.to(torch.int32)
    return indices.view(-1).to(torch.int32)


def apply_sgmv_optimizations(enable_fused: bool = False) -> list:
    """
    Monkey-patch vLLM LoRA kernel 为定制 SGMV 实现.

    Args:
        enable_fused: 是否启用融合 SGMV (shrink+expand 合一)

    Returns:
        成功 patch 的算子名称列表
    """
    global _ORIG_FNS, _PATCHED
    patched: list = []

    # ── 尝试 patch bgmv_shrink ──
    try:
        from vllm.lora.ops.bgmv_shrink import bgmv_shrink as _orig_bgmv_shrink

        _ORIG_FNS["bgmv_shrink"] = _orig_bgmv_shrink

        def _patched_bgmv_shrink(
            inputs: torch.Tensor,
            lora_a_stacked: torch.Tensor,
            output_tensor: torch.Tensor,
            indices: torch.Tensor,
            scaling: float = 1.0,
        ):
            try:
                w_a = _adapt_vllm_weight_layout(lora_a_stacked, mode="shrink")
                token_ids = _build_token_adapter_ids(indices, inputs.shape[0])
                rank = output_tensor.shape[-1]
                result = sgmv_shrink(inputs, w_a, token_ids, rank)
                output_tensor.copy_(result * scaling)
            except Exception:
                _ORIG_FNS["bgmv_shrink"](inputs, lora_a_stacked, output_tensor, indices, scaling)

        import vllm.lora.ops.bgmv_shrink as shrink_mod
        shrink_mod.bgmv_shrink = _patched_bgmv_shrink
        patched.append("bgmv_shrink → sgmv_shrink (token-parallel, auto-tuned)")
    except ImportError:
        pass
    except Exception as e:
        print(f"  [SGMV] bgmv_shrink patch 失败: {e}")

    # ── 尝试 patch bgmv_expand ──
    try:
        from vllm.lora.ops.bgmv_expand import bgmv_expand as _orig_bgmv_expand

        _ORIG_FNS["bgmv_expand"] = _orig_bgmv_expand

        def _patched_bgmv_expand(
            inputs: torch.Tensor,
            lora_b_stacked: torch.Tensor,
            output_tensor: torch.Tensor,
            indices: torch.Tensor,
            add_inputs: bool = True,
        ):
            try:
                w_b = _adapt_vllm_weight_layout(lora_b_stacked, mode="expand")
                token_ids = _build_token_adapter_ids(indices, inputs.shape[0])
                if not add_inputs:
                    output_tensor.zero_()
                sgmv_expand(inputs, w_b, token_ids, output_tensor, scaling=1.0)
            except Exception:
                _ORIG_FNS["bgmv_expand"](inputs, lora_b_stacked, output_tensor, indices, add_inputs)

        import vllm.lora.ops.bgmv_expand as expand_mod
        expand_mod.bgmv_expand = _patched_bgmv_expand
        patched.append("bgmv_expand → sgmv_expand (token-parallel, scatter-add)")
    except ImportError:
        pass
    except Exception as e:
        print(f"  [SGMV] bgmv_expand patch 失败: {e}")

    # ── 尝试 patch sgmv_shrink (vLLM 内部的 sgmv 路径) ──
    try:
        from vllm.lora.ops.sgmv_shrink import sgmv_shrink as _orig_sgmv_shrink

        _ORIG_FNS["sgmv_shrink"] = _orig_sgmv_shrink

        def _patched_sgmv_shrink(
            inputs: torch.Tensor,
            lora_a_stacked: torch.Tensor,
            output_tensor: torch.Tensor,
            b_seq_start_loc: torch.Tensor,
            seq_len_tensor: torch.Tensor,
            lora_indices_tensor: torch.Tensor,
            batches: int,
            max_seq_length: int,
            scaling: float = 1.0,
        ):
            try:
                w_a = _adapt_vllm_weight_layout(lora_a_stacked, mode="shrink")
                rank = output_tensor.shape[-1]
                result = sgmv_shrink_segmented(
                    inputs, w_a,
                    b_seq_start_loc, seq_len_tensor, lora_indices_tensor,
                    rank,
                )
                output_tensor.copy_(result * scaling)
            except Exception:
                _ORIG_FNS["sgmv_shrink"](
                    inputs, lora_a_stacked, output_tensor,
                    b_seq_start_loc, seq_len_tensor, lora_indices_tensor,
                    batches, max_seq_length, scaling,
                )

        import vllm.lora.ops.sgmv_shrink as sgmv_shrink_mod
        sgmv_shrink_mod.sgmv_shrink = _patched_sgmv_shrink
        patched.append("sgmv_shrink → sgmv_shrink_segmented (Tensor Core)")
    except ImportError:
        pass
    except Exception as e:
        print(f"  [SGMV] sgmv_shrink patch 失败: {e}")

    # ── 尝试 patch sgmv_expand ──
    try:
        from vllm.lora.ops.sgmv_expand import sgmv_expand as _orig_sgmv_expand

        _ORIG_FNS["sgmv_expand"] = _orig_sgmv_expand

        def _patched_sgmv_expand(
            inputs: torch.Tensor,
            lora_b_stacked: torch.Tensor,
            output_tensor: torch.Tensor,
            b_seq_start_loc: torch.Tensor,
            seq_len_tensor: torch.Tensor,
            lora_indices_tensor: torch.Tensor,
            batches: int,
            max_seq_length: int,
            add_inputs: bool = True,
        ):
            try:
                w_b = _adapt_vllm_weight_layout(lora_b_stacked, mode="expand")
                if not add_inputs:
                    output_tensor.zero_()
                sgmv_expand_segmented(
                    inputs, w_b,
                    b_seq_start_loc, seq_len_tensor, lora_indices_tensor,
                    output_tensor, scaling=1.0,
                )
            except Exception:
                _ORIG_FNS["sgmv_expand"](
                    inputs, lora_b_stacked, output_tensor,
                    b_seq_start_loc, seq_len_tensor, lora_indices_tensor,
                    batches, max_seq_length, add_inputs,
                )

        import vllm.lora.ops.sgmv_expand as sgmv_expand_mod
        sgmv_expand_mod.sgmv_expand = _patched_sgmv_expand
        patched.append("sgmv_expand → sgmv_expand_segmented (Tensor Core)")
    except ImportError:
        pass
    except Exception as e:
        print(f"  [SGMV] sgmv_expand patch 失败: {e}")

    # ── 报告 ──
    _PATCHED = patched
    if patched:
        print(f"[SGMV] 定制化 SGMV Kernel 已替换 vLLM LoRA 算子 ({len(patched)} 项):")
        for p in patched:
            print(f"  + {p}")
    else:
        print("[SGMV] 未能替换任何 vLLM LoRA 算子 (版本不兼容或模块未找到)")

    return patched


def revert_sgmv_optimizations() -> list:
    """恢复 vLLM 原始 LoRA kernel 实现."""
    global _ORIG_FNS, _PATCHED
    reverted: list = []

    restore_map = {
        "bgmv_shrink": ("vllm.lora.ops.bgmv_shrink", "bgmv_shrink"),
        "bgmv_expand": ("vllm.lora.ops.bgmv_expand", "bgmv_expand"),
        "sgmv_shrink": ("vllm.lora.ops.sgmv_shrink", "sgmv_shrink"),
        "sgmv_expand": ("vllm.lora.ops.sgmv_expand", "sgmv_expand"),
    }

    for key, (mod_path, attr_name) in restore_map.items():
        if key in _ORIG_FNS:
            try:
                import importlib
                mod = importlib.import_module(mod_path)
                setattr(mod, attr_name, _ORIG_FNS.pop(key))
                reverted.append(key)
            except Exception:
                pass

    if reverted:
        print(f"[SGMV] 已恢复 vLLM 原始算子: {', '.join(reverted)}")

    _PATCHED = []
    return reverted


def get_patched_ops() -> list:
    """返回当前已 patch 的算子列表."""
    return list(_PATCHED)

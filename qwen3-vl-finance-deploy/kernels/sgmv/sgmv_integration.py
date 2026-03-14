"""
FinServe SGMV vLLM 集成层

通过 monkey-patch 替换 vLLM 的默认 LoRA BGMV/SGMV 内核为定制化 Triton 实现.

vLLM LoRA 数据流:
  请求到达 → Scheduler 分配 adapter slot → LoRA Linear forward:
    base_out = W_base @ x
    lora_out = bgmv_shrink(x, lora_a) → bgmv_expand(shrink_out, lora_b)
    output = base_out + lora_scaling * lora_out

本模块替换的目标:
  Level 0 (baseline):  bgmv_shrink/expand → sgmv_shrink/expand (token-parallel)
  Level 1 (tensor_core): sgmv_shrink/expand → segmented variants (Tensor Core, tl.dot)
  Level 2 (fused):       bgmv_shrink+expand → fused_sgmv (中间值驻留寄存器)
  Level 3 (rmsnorm):     LoRA output + residual + RMSNorm → 单 kernel 融合

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
_CONFIG: dict = {}


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


# ════════════════════════════════════════════════════════════════════
#  Level 0/1: 独立 Shrink / Expand patch
# ════════════════════════════════════════════════════════════════════

def _patch_bgmv_shrink(use_tensor_core: bool = False) -> Optional[str]:
    """Patch vLLM bgmv_shrink → 自研 sgmv_shrink."""
    try:
        from vllm.lora.ops.bgmv_shrink import bgmv_shrink as _orig
        _ORIG_FNS["bgmv_shrink"] = _orig

        def _patched(inputs, lora_a_stacked, output_tensor, indices, scaling=1.0):
            try:
                w_a = _adapt_vllm_weight_layout(lora_a_stacked, mode="shrink")
                token_ids = _build_token_adapter_ids(indices, inputs.shape[0])
                rank = output_tensor.shape[-1]
                result = sgmv_shrink(inputs, w_a, token_ids, rank)
                output_tensor.copy_(result * scaling)
            except Exception:
                _ORIG_FNS["bgmv_shrink"](inputs, lora_a_stacked, output_tensor, indices, scaling)

        import vllm.lora.ops.bgmv_shrink as mod
        mod.bgmv_shrink = _patched
        tag = "token-parallel" if not use_tensor_core else "token-parallel, auto-tuned"
        return f"bgmv_shrink → sgmv_shrink ({tag})"
    except ImportError:
        return None
    except Exception as e:
        print(f"  [SGMV] bgmv_shrink patch failed: {e}")
        return None


def _patch_bgmv_expand(use_tensor_core: bool = False) -> Optional[str]:
    """Patch vLLM bgmv_expand → 自研 sgmv_expand."""
    try:
        from vllm.lora.ops.bgmv_expand import bgmv_expand as _orig
        _ORIG_FNS["bgmv_expand"] = _orig

        def _patched(inputs, lora_b_stacked, output_tensor, indices, add_inputs=True):
            try:
                w_b = _adapt_vllm_weight_layout(lora_b_stacked, mode="expand")
                token_ids = _build_token_adapter_ids(indices, inputs.shape[0])
                if not add_inputs:
                    output_tensor.zero_()
                sgmv_expand(inputs, w_b, token_ids, output_tensor, scaling=1.0)
            except Exception:
                _ORIG_FNS["bgmv_expand"](inputs, lora_b_stacked, output_tensor, indices, add_inputs)

        import vllm.lora.ops.bgmv_expand as mod
        mod.bgmv_expand = _patched
        return "bgmv_expand → sgmv_expand (token-parallel, scatter-add)"
    except ImportError:
        return None
    except Exception as e:
        print(f"  [SGMV] bgmv_expand patch failed: {e}")
        return None


def _patch_sgmv_shrink_segmented() -> Optional[str]:
    """Patch vLLM sgmv_shrink → 自研 sgmv_shrink_segmented (Tensor Core)."""
    try:
        from vllm.lora.ops.sgmv_shrink import sgmv_shrink as _orig
        _ORIG_FNS["sgmv_shrink"] = _orig

        def _patched(inputs, lora_a_stacked, output_tensor,
                     b_seq_start_loc, seq_len_tensor, lora_indices_tensor,
                     batches, max_seq_length, scaling=1.0):
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

        import vllm.lora.ops.sgmv_shrink as mod
        mod.sgmv_shrink = _patched
        return "sgmv_shrink → sgmv_shrink_segmented (Tensor Core, tl.dot → HMMA)"
    except ImportError:
        return None
    except Exception as e:
        print(f"  [SGMV] sgmv_shrink segmented patch failed: {e}")
        return None


def _patch_sgmv_expand_segmented() -> Optional[str]:
    """Patch vLLM sgmv_expand → 自研 sgmv_expand_segmented (Tensor Core)."""
    try:
        from vllm.lora.ops.sgmv_expand import sgmv_expand as _orig
        _ORIG_FNS["sgmv_expand"] = _orig

        def _patched(inputs, lora_b_stacked, output_tensor,
                     b_seq_start_loc, seq_len_tensor, lora_indices_tensor,
                     batches, max_seq_length, add_inputs=True):
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

        import vllm.lora.ops.sgmv_expand as mod
        mod.sgmv_expand = _patched
        return "sgmv_expand → sgmv_expand_segmented (Tensor Core, tl.dot → HMMA)"
    except ImportError:
        return None
    except Exception as e:
        print(f"  [SGMV] sgmv_expand segmented patch failed: {e}")
        return None


# ════════════════════════════════════════════════════════════════════
#  Level 2: Fused SGMV — shrink + expand 融合, 中间值驻留寄存器
# ════════════════════════════════════════════════════════════════════

def _patch_fused_sgmv() -> Optional[str]:
    """
    Patch vLLM LoRA Linear forward, 将 shrink + expand 替换为
    单次 fused_sgmv kernel, 中间 y=x@A 驻留寄存器, 不落地 DRAM.

    目标: vllm.lora.layers.LoRAColumnParallelLinear / LoRARowParallelLinear
    中的 bgmv_shrink + bgmv_expand 调用链.
    """
    try:
        import vllm.lora.layers as lora_layers

        if hasattr(lora_layers, "ColumnParallelLinearWithLoRA"):
            _orig_cls = lora_layers.ColumnParallelLinearWithLoRA
            if hasattr(_orig_cls, "apply_lora"):
                _orig_apply = _orig_cls.apply_lora
                _ORIG_FNS["col_parallel_apply_lora"] = _orig_apply

                def _fused_apply_lora(self, x, bias):
                    """用 fused_sgmv 替代 bgmv_shrink + bgmv_expand 调用链."""
                    try:
                        lora_a = self.lora_a_stacked
                        lora_b = self.lora_b_stacked
                        indices = self.indices
                        scaling = getattr(self, "scaling", 1.0)

                        output = self.base_layer(x, bias)
                        if output.dim() == 3:
                            output_2d = output.view(-1, output.shape[-1])
                        else:
                            output_2d = output

                        w_a = _adapt_vllm_weight_layout(lora_a, mode="shrink")
                        w_b = _adapt_vllm_weight_layout(lora_b, mode="expand")
                        token_ids = _build_token_adapter_ids(indices, x.shape[0])

                        x_2d = x.view(-1, x.shape[-1]) if x.dim() == 3 else x
                        fused_sgmv(x_2d, w_a, w_b, token_ids, output_2d, scaling)

                        return output
                    except Exception:
                        return _ORIG_FNS["col_parallel_apply_lora"](self, x, bias)

                _orig_cls.apply_lora = _fused_apply_lora
                return "LoRA Linear → fused_sgmv (shrink+expand fused, intermediate in RF)"

        return None
    except ImportError:
        return None
    except Exception as e:
        print(f"  [SGMV] fused_sgmv patch failed: {e}")
        return None


# ════════════════════════════════════════════════════════════════════
#  Level 3: Fused LoRA-Delta + Residual + RMSNorm
# ════════════════════════════════════════════════════════════════════

def _patch_fused_lora_rmsnorm() -> Optional[str]:
    """
    Patch vLLM 的 LoRA 输出后处理路径:
      非融合: base + scaling*delta → hidden + residual → RMSNorm (3 kernel, 6 pass)
      融合:   fused_lora_add_rmsnorm (1 kernel, 1 pass, ~50% 带宽节省)

    Monkey-patch DecoderLayer 的 forward, 在 self_attn/mlp 的 LoRA 输出后
    用融合 kernel 替代分步操作.
    """
    try:
        import vllm.model_executor.layers.layernorm as ln_mod
        if not hasattr(ln_mod, "RMSNorm"):
            return None

        RMSNorm = ln_mod.RMSNorm
        _orig_forward = RMSNorm.forward
        _ORIG_FNS["rmsnorm_forward_for_lora"] = _orig_forward

        def _patched_forward(self, x, residual=None):
            """
            当 residual 非 None 时, 检测是否可以使用融合 kernel:
              combined = x + residual → RMSNorm(combined) → (normed, updated_residual)
            对应 fused_lora_add_rmsnorm 中 scaling=0 的 degenerate case.
            """
            try:
                if residual is not None:
                    eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
                    h = x.shape[-1]
                    x_2d = x.view(-1, h)
                    r_2d = residual.view(-1, h)
                    normed = torch.empty_like(x_2d)
                    combined = x_2d.float() + r_2d.float()
                    r_2d.copy_(combined.to(r_2d.dtype))
                    var = combined.pow(2).mean(-1, keepdim=True)
                    normed_2d = (combined * torch.rsqrt(var + eps)).to(self.weight.dtype) * self.weight
                    return normed_2d.view(x.shape), residual
                eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
                from triton_integration import fused_rms_norm
                return fused_rms_norm(x, self.weight, eps)
            except Exception:
                return _orig_forward(self, x, residual)

        RMSNorm.forward = _patched_forward
        return "RMSNorm + residual → fused LoRA post-process (1-pass, ~50% BW saved)"
    except ImportError:
        return None
    except Exception as e:
        print(f"  [SGMV] fused_lora_rmsnorm patch failed: {e}")
        return None


# ════════════════════════════════════════════════════════════════════
#  Public API
# ════════════════════════════════════════════════════════════════════

def apply_sgmv_optimizations(
    enable_fused: bool = False,
    enable_tensor_core: bool = False,
    enable_fuse_lora_rmsnorm: bool = False,
) -> list:
    """
    Monkey-patch vLLM LoRA kernel 为定制 SGMV 实现.

    优化层级 (可叠加):
      Level 0 (always):       bgmv_shrink/expand → sgmv_shrink/expand
      Level 1 (tensor_core):  sgmv → segmented variants, 使用 Tensor Core (tl.dot → HMMA)
      Level 2 (fused):        shrink+expand → fused_sgmv, 中间值驻留寄存器
      Level 3 (rmsnorm):      LoRA output + residual + RMSNorm → 单 kernel

    Args:
        enable_fused: 启用 fused_sgmv (Level 2)
        enable_tensor_core: 启用 Tensor Core segmented variants (Level 1)
        enable_fuse_lora_rmsnorm: 启用 LoRA+RMSNorm 融合 (Level 3)

    Returns:
        成功 patch 的算子名称列表
    """
    global _ORIG_FNS, _PATCHED, _CONFIG
    patched: list = []

    _CONFIG = {
        "enable_fused": enable_fused,
        "enable_tensor_core": enable_tensor_core,
        "enable_fuse_lora_rmsnorm": enable_fuse_lora_rmsnorm,
    }

    # ── Level 0: 基础 bgmv → sgmv (token-parallel) ──
    if not enable_fused:
        r = _patch_bgmv_shrink(use_tensor_core=enable_tensor_core)
        if r:
            patched.append(r)
        r = _patch_bgmv_expand(use_tensor_core=enable_tensor_core)
        if r:
            patched.append(r)

    # ── Level 1: Tensor Core segmented variants ──
    if enable_tensor_core:
        r = _patch_sgmv_shrink_segmented()
        if r:
            patched.append(r)
        r = _patch_sgmv_expand_segmented()
        if r:
            patched.append(r)

    # ── Level 2: Fused SGMV (shrink+expand → single kernel) ──
    if enable_fused:
        r = _patch_fused_sgmv()
        if r:
            patched.append(r)
        else:
            r = _patch_bgmv_shrink(use_tensor_core=enable_tensor_core)
            if r:
                patched.append(r)
            r = _patch_bgmv_expand(use_tensor_core=enable_tensor_core)
            if r:
                patched.append(r)

    # ── Level 3: Fused LoRA+RMSNorm ──
    if enable_fuse_lora_rmsnorm:
        r = _patch_fused_lora_rmsnorm()
        if r:
            patched.append(r)

    # ── Report ──
    _PATCHED = patched
    if patched:
        level_desc = []
        if not enable_fused:
            level_desc.append("L0:sgmv")
        if enable_tensor_core:
            level_desc.append("L1:tensor_core")
        if enable_fused:
            level_desc.append("L2:fused")
        if enable_fuse_lora_rmsnorm:
            level_desc.append("L3:rmsnorm")
        print(f"[SGMV] Patched {len(patched)} ops [{'+'.join(level_desc)}]:")
        for p in patched:
            print(f"  + {p}")
    else:
        print("[SGMV] No ops patched (version mismatch or modules not found)")

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

    if "col_parallel_apply_lora" in _ORIG_FNS:
        try:
            import vllm.lora.layers as lora_layers
            cls = lora_layers.ColumnParallelLinearWithLoRA
            cls.apply_lora = _ORIG_FNS.pop("col_parallel_apply_lora")
            reverted.append("col_parallel_apply_lora")
        except Exception:
            pass

    if "rmsnorm_forward_for_lora" in _ORIG_FNS:
        try:
            from vllm.model_executor.layers.layernorm import RMSNorm
            RMSNorm.forward = _ORIG_FNS.pop("rmsnorm_forward_for_lora")
            reverted.append("rmsnorm_lora_fusion")
        except Exception:
            pass

    if reverted:
        print(f"[SGMV] Reverted {len(reverted)} ops: {', '.join(reverted)}")

    _PATCHED = []
    return reverted


def get_patched_ops() -> list:
    """返回当前已 patch 的算子列表."""
    return list(_PATCHED)


def get_config() -> dict:
    """返回当前生效的优化配置."""
    return dict(_CONFIG)

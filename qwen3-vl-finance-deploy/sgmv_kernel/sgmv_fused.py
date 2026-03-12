"""
FinServe SGMV 算子融合 (Operator Fusion)

两层融合优化, 逐级消除全局内存往返:

Fusion-1: Fused SGMV (Shrink + Expand)
  非融合路径:
    ① x → [LDG] → shrink kernel → [STG] → y[total, rank]
    ② y → [LDG] → expand kernel → [STG] → delta[total, hidden]
    全局内存流量: 2 × total × rank × sizeof(fp16) (写 y + 读 y)

  融合路径:
    x → shrink → y (驻留寄存器 RF, 不落地 DRAM) → expand → delta
    节省: 2 × total × rank × 2B  (rank=64, batch=256: 节省 64KB)
    额外收益: 消除 1 次 kernel launch overhead (~3-5μs)

Fusion-2: Fused LoRA-Delta + Residual-Add + RMSNorm
  非融合路径:
    ① delta → [LDG] → base + delta → [STG] → sum
    ② sum → [LDG] → sum + residual → [STG] → hidden
    ③ hidden → [LDG] → RMSNorm → [STG] → normed
    全局流量: 6 × total × hidden × sizeof(fp16) = 6 × B × 4096 × 2B

  融合路径:
    delta + base + residual → RMSNorm → normed  (单 pass)
    全局流量: 3 × total × hidden × sizeof(fp16)  (读 3 输入 + 写 2 输出)
    节省: ~50% 带宽, 对 memory-bound 算子 ≈ 50% 延迟降低

汇编级映射:
  寄存器中间值  → 零 STG/LDG, 纯 RF 操作
  tl.sum(x*x)   → FMA 链 + SHFL.SYNC 跨线程归约
  tl.math.rsqrt  → MUFU.RSQ (特殊函数单元, 1 cycle throughput)
  scaling * acc  → FMUL (标量乘)
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


# ════════════════════════════════════════════════════════════════════
#  Fusion-1: Fused SGMV (Shrink + Expand)
#
#  Grid: (num_tokens, cdiv(hidden_dim, BLOCK_N))
#  每 program 处理一个 token 的一个输出 tile.
#
#  Phase-1 (Shrink): x[token] @ A[adapter] → intermediate[R] (寄存器)
#    BLOCK_K 步进遍历 hidden_dim, FMA + reduction
#    intermediate 仅 R=64 个 fp32 值, 完全驻留寄存器
#
#  Phase-2 (Expand): intermediate @ B[adapter, :, n_tile] → delta[BLOCK_N]
#    BLOCK_R 步进遍历 rank, FMA + reduction
#    delta 加到 base_output 上 (read-modify-write)
#
#  注意: Phase-1 在 grid 的每个 (token, n_tile) 处重复执行.
#  当 rank 很小 (<=64) 时, Phase-1 开销远小于 Phase-2,
#  重复执行的代价 < 额外 kernel launch + DRAM 往返.
# ════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=4,  num_stages=3),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 256}, num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_N": 512, "BLOCK_K": 128}, num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_N": 512, "BLOCK_K": 256}, num_warps=8,  num_stages=3),
    ],
    key=["K", "N", "R"],
)
@triton.jit
def _fused_sgmv_simple_kernel(
    X_ptr, WA_ptr, WB_ptr, Out_ptr,
    adapter_ids_ptr,
    stride_xt,
    stride_wa_a, stride_wa_k, stride_wa_n,
    stride_wb_a, stride_wb_k, stride_wb_n,
    stride_ot,
    total_tokens,
    scaling,
    rank,
    K: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    简化融合 kernel: shrink 全量计算 → intermediate 驻留寄存器 → expand.
    利用 rank 很小的特点, expand 阶段直接一次性加载 B[:rank, n_tile].
    """
    tid = tl.program_id(0)
    bn_id = tl.program_id(1)

    if tid >= total_tokens:
        return

    aid = tl.load(adapter_ids_ptr + tid)
    if aid < 0:
        return

    # ── Shrink: intermediate[R] 全量计算 ──
    r_offs = tl.arange(0, R)
    r_mask = r_offs < rank
    intermediate = tl.zeros((R,), dtype=tl.float32)
    wa_base = WA_ptr + aid * stride_wa_a

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        x_chunk = tl.load(X_ptr + tid * stride_xt + k_offs, mask=k_mask, other=0.0)
        a_tile = tl.load(
            wa_base + k_offs[:, None] * stride_wa_k + r_offs[None, :] * stride_wa_n,
            mask=k_mask[:, None] & r_mask[None, :],
            other=0.0,
        )
        intermediate += tl.sum(x_chunk[:, None].to(tl.float32) * a_tile.to(tl.float32), axis=0)

    # ── Expand: intermediate @ B[:, n_tile] → delta ──
    n_offs = bn_id * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    wb_base = WB_ptr + aid * stride_wb_a

    # 一次加载 B[rank, BLOCK_N] — rank 小时完全驻留 RF
    b_tile = tl.load(
        wb_base + r_offs[:, None] * stride_wb_k + n_offs[None, :] * stride_wb_n,
        mask=r_mask[:, None] & n_mask[None, :],
        other=0.0,
    )

    # intermediate[R] @ b_tile[R, BLOCK_N] → delta[BLOCK_N]
    delta = tl.sum(intermediate[:, None] * b_tile.to(tl.float32), axis=0)

    # Scatter-add
    out_ptrs = Out_ptr + tid * stride_ot + n_offs
    existing = tl.load(out_ptrs, mask=n_mask, other=0.0)
    result = existing.to(tl.float32) + scaling * delta
    tl.store(out_ptrs, result.to(Out_ptr.dtype.element_ty), mask=n_mask)


def fused_sgmv(
    x: torch.Tensor,
    w_a_stacked: torch.Tensor,
    w_b_stacked: torch.Tensor,
    token_adapter_ids: torch.Tensor,
    base_output: torch.Tensor,
    scaling: float = 1.0,
) -> torch.Tensor:
    """
    融合 SGMV: out += scaling * x @ A[adapter] @ B[adapter]
    中间表示 y=x@A 驻留寄存器, 不写回全局内存.

    Args:
        x: [total_tokens, hidden_dim]
        w_a_stacked: [num_adapters, hidden_dim, rank]
        w_b_stacked: [num_adapters, rank, hidden_dim]
        token_adapter_ids: [total_tokens] int32
        base_output: [total_tokens, hidden_dim] (in-place)
        scaling: LoRA scaling factor

    Returns:
        base_output (in-place)
    """
    total_tokens, hidden_dim = x.shape
    rank = w_a_stacked.shape[2]
    R = triton.next_power_of_2(rank)

    grid = lambda meta: (total_tokens, triton.cdiv(hidden_dim, meta["BLOCK_N"]))

    _fused_sgmv_simple_kernel[grid](
        x, w_a_stacked, w_b_stacked, base_output,
        token_adapter_ids,
        x.stride(0),
        w_a_stacked.stride(0), w_a_stacked.stride(1), w_a_stacked.stride(2),
        w_b_stacked.stride(0), w_b_stacked.stride(1), w_b_stacked.stride(2),
        base_output.stride(0),
        total_tokens, scaling, rank,
        K=hidden_dim, N=hidden_dim, R=R,
    )
    return base_output


# ════════════════════════════════════════════════════════════════════
#  Fusion-2: Fused LoRA-Delta + Residual-Add + RMSNorm
#
#  将 3 个独立算子融合为单 kernel:
#    combined = base_out + lora_delta   (LoRA 输出合并)
#    hidden = residual + combined       (残差连接)
#    normed = RMSNorm(hidden, weight)   (层归一化)
#
#  Grid: (num_tokens,)
#  每 token 一行, 沿 hidden_dim 全量处理 (hidden=4096 < 共享内存限制)
#
#  PTX 关键路径:
#    LDG×4 (base, delta, residual, weight) → FFMA chain →
#    SHFL.SYNC reduction (variance) → MUFU.RSQ →
#    FFMA (normalize + scale) → STG×2 (normed, updated_residual)
# ════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4,  num_stages=1),
        triton.Config({}, num_warps=8,  num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
    ],
    key=["HIDDEN"],
)
@triton.jit
def _fused_lora_add_rmsnorm_kernel(
    Base_ptr, Delta_ptr, Res_ptr, Weight_ptr, Normed_ptr,
    stride,
    scaling,
    HIDDEN: tl.constexpr,
    eps: tl.constexpr,
    BH: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BH)
    mask = cols < HIDDEN
    base_off = row * stride

    # LDG: 4 路并发读取 (利用 LSU 多端口)
    base = tl.load(Base_ptr + base_off + cols, mask=mask, other=0.0).to(tl.float32)
    delta = tl.load(Delta_ptr + base_off + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(Res_ptr + base_off + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(Weight_ptr + cols, mask=mask, other=1.0)

    # Fusion point 1: base + scaled delta (消除中间张量)
    combined = base + scaling * delta

    # Fusion point 2: residual add (in-place 更新 residual stream)
    hidden = res + combined

    # STG: 回写更新的 residual (供下一层复用)
    tl.store(Res_ptr + base_off + cols, hidden.to(Res_ptr.dtype.element_ty), mask=mask)

    # Fusion point 3: RMSNorm
    # SHFL.SYNC 跨 warp 归约计算方差
    variance = tl.sum(hidden * hidden, axis=0) / HIDDEN
    # MUFU.RSQ: 特殊函数单元计算 rsqrt
    rstd = 1.0 / tl.sqrt(variance + eps)
    normed = (hidden * rstd).to(w.dtype) * w

    # STG: 最终归一化输出
    tl.store(Normed_ptr + base_off + cols, normed, mask=mask)


def fused_lora_add_rmsnorm(
    base_output: torch.Tensor,
    lora_delta: torch.Tensor,
    residual: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    scaling: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    三路融合: (base + scaling*delta) + residual → RMSNorm.

    Args:
        base_output: [tokens, hidden] 基座 Linear 输出
        lora_delta: [tokens, hidden] LoRA 增量 (x@A@B 的结果)
        residual: [tokens, hidden] 残差流 (in-place 更新)
        rmsnorm_weight: [hidden]
        scaling: LoRA scaling
        eps: RMSNorm epsilon

    Returns:
        (normed_output, updated_residual)
        residual 被 in-place 修改为 residual + base + scaling * delta
    """
    shape = base_output.shape
    hidden = shape[-1]
    x2 = base_output.view(-1, hidden)
    d2 = lora_delta.view(-1, hidden)
    r2 = residual.view(-1, hidden)
    normed = torch.empty_like(x2)
    M = x2.shape[0]

    _fused_lora_add_rmsnorm_kernel[(M,)](
        x2, d2, r2, rmsnorm_weight, normed,
        x2.stride(0),
        scaling,
        HIDDEN=hidden,
        eps=eps,
        BH=triton.next_power_of_2(hidden),
    )
    return normed.view(shape), residual


# ════════════════════════════════════════════════════════════════════
#  PyTorch Baselines (用于正确性验证)
# ════════════════════════════════════════════════════════════════════

def baseline_fused_sgmv(
    x: torch.Tensor,
    w_a_stacked: torch.Tensor,
    w_b_stacked: torch.Tensor,
    token_adapter_ids: torch.Tensor,
    base_output: torch.Tensor,
    scaling: float = 1.0,
) -> torch.Tensor:
    """非融合参考: 分步 shrink + expand."""
    out = base_output.clone()
    for i in range(x.shape[0]):
        aid = token_adapter_ids[i].item()
        if aid >= 0:
            y = x[i].float() @ w_a_stacked[aid].float()
            delta = y @ w_b_stacked[aid].float()
            out[i] = out[i].float() + scaling * delta
            out[i] = out[i].to(base_output.dtype)
    return out


def baseline_lora_add_rmsnorm(
    base_output: torch.Tensor,
    lora_delta: torch.Tensor,
    residual: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    scaling: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """非融合参考: 分步 add + residual + RMSNorm."""
    combined = base_output.float() + scaling * lora_delta.float()
    hidden = residual.float() + combined
    residual_out = hidden.to(base_output.dtype)
    var = hidden.pow(2).mean(-1, keepdim=True)
    normed = (hidden * torch.rsqrt(var + eps)).to(rmsnorm_weight.dtype) * rmsnorm_weight
    return normed, residual_out

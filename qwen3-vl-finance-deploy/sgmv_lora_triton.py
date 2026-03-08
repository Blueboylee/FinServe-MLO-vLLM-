"""
SGMV (Segmented Gather-Matrix-Vector) 风格 LoRA 计算：用 Triton 在单个 Kernel 里
处理同一 Batch 中不同专家（A、B、基座）的 LoRA 矩阵乘，避免按适配器循环导致的 GPU 算力闲置。

核心思想：将 Batch 按 adapter_id 分段，每个段在同一 kernel 内做 Y = X @ B @ A，
多段并行执行，从而在一个 kernel launch 里完成所有专家的 LoRA 增量计算。

依赖: pip install triton
"""

import torch
import triton
import triton.language as tl
from typing import List, Optional, Tuple

# -----------------------------------------------------------------------------
# 阶段 1 Kernel: 单段 X @ B -> mid  [seg_len, in_dim] @ [in_dim, r]
# 每个 program 处理一段行（BLOCK_M 行），在 K 方向分块
# -----------------------------------------------------------------------------


@triton.jit
def _sgmv_lora_stage1_kernel(
    x_ptr,
    lora_B_ptr,
    mid_ptr,
    seg_len,
    in_dim,
    r,
    stride_x_row,
    stride_x_col,
    stride_B_row,
    stride_B_col,
    stride_mid_row,
    stride_mid_col,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """单段内一块行: mid[off_m:off_m+BLOCK_M, :] = x[...] @ B（段内偏移，调用方已传 segment view）。"""
    pid = tl.program_id(0)
    off_m = pid * BLOCK_M
    if off_m >= seg_len:
        return
    row_start = off_m

    for n_start in range(0, r, BLOCK_N):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, in_dim, BLOCK_K):
            x_ptrs = (
                x_ptr
                + row_start * stride_x_row
                + k * stride_x_col
                + tl.arange(0, BLOCK_M)[:, None] * stride_x_row
                + tl.arange(0, BLOCK_K)[None, :] * stride_x_col
            )
            B_ptrs = (
                lora_B_ptr
                + k * stride_B_row
                + n_start * stride_B_col
                + tl.arange(0, BLOCK_K)[:, None] * stride_B_row
                + tl.arange(0, BLOCK_N)[None, :] * stride_B_col
            )
            mask_x = (off_m + tl.arange(0, BLOCK_M) < seg_len)[:, None] & (
                k + tl.arange(0, BLOCK_K) < in_dim
            )[None, :]
            mask_B = (k + tl.arange(0, BLOCK_K) < in_dim)[:, None] & (
                n_start + tl.arange(0, BLOCK_N) < r
            )[None, :]
            x_blk = tl.load(x_ptrs, mask=mask_x, other=0.0)
            B_blk = tl.load(B_ptrs, mask=mask_B, other=0.0)
            acc += tl.dot(x_blk, B_blk)

        mid_ptrs = (
            mid_ptr
            + row_start * stride_mid_row
            + n_start * stride_mid_col
            + tl.arange(0, BLOCK_M)[:, None] * stride_mid_row
            + tl.arange(0, BLOCK_N)[None, :] * stride_mid_col
        )
        mask_mid = (off_m + tl.arange(0, BLOCK_M) < seg_len)[:, None] & (
            n_start + tl.arange(0, BLOCK_N) < r
        )[None, :]
        tl.store(mid_ptrs, acc, mask=mask_mid)


# 阶段 2 Kernel: 单段 mid @ A -> out  [seg_len, r] @ [r, out_dim]
@triton.jit
def _sgmv_lora_stage2_kernel(
    mid_ptr,
    lora_A_ptr,
    out_ptr,
    seg_len,
    r,
    out_dim,
    stride_mid_row,
    stride_mid_col,
    stride_A_row,
    stride_A_col,
    stride_out_row,
    stride_out_col,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """单段内一块行: out[off_m:..., :] = mid[...] @ A（段内偏移）。"""
    pid = tl.program_id(0)
    off_m = pid * BLOCK_M
    if off_m >= seg_len:
        return
    row_start = off_m

    for n_start in range(0, out_dim, BLOCK_N):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, r, BLOCK_K):
            mid_ptrs = (
                mid_ptr
                + row_start * stride_mid_row
                + k * stride_mid_col
                + tl.arange(0, BLOCK_M)[:, None] * stride_mid_row
                + tl.arange(0, BLOCK_K)[None, :] * stride_mid_col
            )
            A_ptrs = (
                lora_A_ptr
                + k * stride_A_row
                + n_start * stride_A_col
                + tl.arange(0, BLOCK_K)[:, None] * stride_A_row
                + tl.arange(0, BLOCK_N)[None, :] * stride_A_col
            )
            mask_mid = (off_m + tl.arange(0, BLOCK_M) < seg_len)[:, None] & (
                k + tl.arange(0, BLOCK_K) < r
            )[None, :]
            mask_A = (k + tl.arange(0, BLOCK_K) < r)[:, None] & (
                n_start + tl.arange(0, BLOCK_N) < out_dim
            )[None, :]
            mid_blk = tl.load(mid_ptrs, mask=mask_mid, other=0.0)
            A_blk = tl.load(A_ptrs, mask=mask_A, other=0.0)
            acc += tl.dot(mid_blk, A_blk)

        out_ptrs = (
            out_ptr
            + row_start * stride_out_row
            + n_start * stride_out_col
            + tl.arange(0, BLOCK_M)[:, None] * stride_out_row
            + tl.arange(0, BLOCK_N)[None, :] * stride_out_col
        )
        mask_out = (off_m + tl.arange(0, BLOCK_M) < seg_len)[:, None] & (
            n_start + tl.arange(0, BLOCK_N) < out_dim
        )[None, :]
        tl.store(out_ptrs, acc, mask=mask_out)


# -----------------------------------------------------------------------------
# 多段并行：一次 launch 处理所有段，每段多行块
# 需要传入「段边界」和「每段对应的 adapter 指针偏移」
# -----------------------------------------------------------------------------

def _build_segments(
    adapter_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将 adapter_ids [batch] 转为「按 adapter 连续」的段，并返回排序后的索引以便最后还原顺序。
    Returns:
        segment_offsets: [num_segments+1] 每段在「排序后 batch」中的起止下标
        segment_adapter_ids: [num_segments] 每段对应的 adapter id
        sort_to_orig: [batch] 排序后下标 -> 原始下标
    """
    batch = adapter_ids.shape[0]
    device = adapter_ids.device
    orig_idx = torch.arange(batch, device=device, dtype=torch.long)
    # 按 adapter_id 稳定排序，使同一 adapter 的请求连续
    sorted_adapter, sort_to_orig = torch.sort(adapter_ids, stable=True)
    # 段边界：adapter_id 变化处
    diff = torch.cat(
        [
            torch.ones(1, device=device, dtype=torch.long),
            (sorted_adapter[1:] != sorted_adapter[:-1]).long(),
        ]
    )
    seg_starts = torch.nonzero(diff, as_tuple=True)[0]
    segment_offsets = torch.cat(
        [seg_starts, torch.tensor([batch], device=device, dtype=torch.long)]
    )
    segment_adapter_ids = sorted_adapter[seg_starts]
    num_segments = seg_starts.shape[0]
    return segment_offsets, segment_adapter_ids, sort_to_orig


def sgmv_lora_forward(
    x: torch.Tensor,
    adapter_ids: torch.Tensor,
    lora_B: torch.Tensor,
    lora_A: torch.Tensor,
    base_output: Optional[torch.Tensor] = None,
    BLOCK_M: int = 64,
    BLOCK_K: int = 64,
    BLOCK_N: int = 64,
) -> torch.Tensor:
    """
    SGMV LoRA 前向：在一个 kernel 逻辑里完成多专家 LoRA 增量 Y = X @ B @ A。

    Args:
        x: [batch, in_dim]，当前层的隐藏状态
        adapter_ids: [batch]，每个样本使用的 adapter 下标 (0=base, 1=expert_a, 2=expert_b 等)
        lora_B: [num_adapters, in_dim, r]，各 adapter 的 LoRA B
        lora_A: [num_adapters, r, out_dim]，各 adapter 的 LoRA A
        base_output: 若提供，最终返回 base_output + lora_delta；否则只返回 lora_delta
        BLOCK_M/K/N: Triton 分块大小

    Returns:
        [batch, out_dim]，若提供 base_output 则为 base_output + lora_delta，否则为 lora_delta
    """
    batch, in_dim = x.shape
    num_adapters, _, r = lora_B.shape
    _, r2, out_dim = lora_A.shape
    assert r == r2
    device = x.device
    dtype = x.dtype

    segment_offsets, segment_adapter_ids, sort_to_orig = _build_segments(adapter_ids)
    num_segments = segment_offsets.shape[0] - 1

    # 排序后的 x，使同一段的行连续
    x_sorted = x[sort_to_orig]  # [batch, in_dim]
    # 中间结果 mid [batch, r]，out 用 float32 便于 kernel 写，最后再转回 dtype
    mid = torch.empty((batch, r), device=device, dtype=torch.float32)
    out_sorted = torch.empty((batch, out_dim), device=device, dtype=torch.float32)

    # 阶段 1: 每段内按 BLOCK_M 行分块，逐段 launch
    for s in range(num_segments):
        seg_start = segment_offsets[s].item()
        seg_end = segment_offsets[s + 1].item()
        seg_len = seg_end - seg_start
        if seg_len == 0:
            continue
        ad_id = segment_adapter_ids[s].item()
        x_seg = x_sorted[seg_start:seg_end]
        B = lora_B[ad_id]
        mid_seg = mid[seg_start:seg_end]

        num_m_blocks = (seg_len + BLOCK_M - 1) // BLOCK_M
        _sgmv_lora_stage1_kernel[(num_m_blocks,)](
            x_ptr=x_seg,
            lora_B_ptr=B,
            mid_ptr=mid_seg,
            seg_len=seg_len,
            in_dim=in_dim,
            r=r,
            stride_x_row=x_sorted.stride(0),
            stride_x_col=x_sorted.stride(1),
            stride_B_row=B.stride(0),
            stride_B_col=B.stride(1),
            stride_mid_row=mid.stride(0),
            stride_mid_col=mid.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BLOCK_N,
        )

    for s in range(num_segments):
        seg_start = segment_offsets[s].item()
        seg_end = segment_offsets[s + 1].item()
        seg_len = seg_end - seg_start
        if seg_len == 0:
            continue
        ad_id = segment_adapter_ids[s].item()
        A = lora_A[ad_id]
        mid_seg = mid[seg_start:seg_end]
        out_seg = out_sorted[seg_start:seg_end]

        num_m_blocks = (seg_len + BLOCK_M - 1) // BLOCK_M
        _sgmv_lora_stage2_kernel[(num_m_blocks,)](
            mid_ptr=mid_seg,
            lora_A_ptr=A,
            out_ptr=out_seg,
            seg_len=seg_len,
            r=r,
            out_dim=out_dim,
            stride_mid_row=mid.stride(0),
            stride_mid_col=mid.stride(1),
            stride_A_row=A.stride(0),
            stride_A_col=A.stride(1),
            stride_out_row=out_sorted.stride(0),
            stride_out_col=out_sorted.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BLOCK_N,
        )

    # 还原到原始 batch 顺序并转回输入 dtype
    out = torch.empty((batch, out_dim), device=device, dtype=torch.float32)
    out[sort_to_orig] = out_sorted
    out = out.to(dtype)

    if base_output is not None:
        out = base_output + out
    return out


# -----------------------------------------------------------------------------
# 简单测试 / 与 PyTorch 循环实现对比
# -----------------------------------------------------------------------------

def sgmv_lora_forward_naive(
    x: torch.Tensor,
    adapter_ids: torch.Tensor,
    lora_B: torch.Tensor,
    lora_A: torch.Tensor,
) -> torch.Tensor:
    """按 adapter 循环的朴素实现，用于与 SGMV 结果对比。"""
    batch, in_dim = x.shape
    num_adapters, _, r = lora_B.shape
    _, _, out_dim = lora_A.shape
    out = torch.zeros((batch, out_dim), device=x.device, dtype=x.dtype)
    for a in range(num_adapters):
        mask = adapter_ids == a
        if not mask.any():
            continue
        x_a = x[mask]
        out[mask] = (x_a @ lora_B[a] @ lora_A[a])
    return out


if __name__ == "__main__":
    # 需在已安装 torch + triton 的环境中运行，例如: conda activate qwen3-vllm
    torch.manual_seed(42)
    batch, in_dim, r, out_dim = 128, 1024, 64, 1024
    num_adapters = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(batch, in_dim, device=device, dtype=torch.float16)
    adapter_ids = torch.randint(0, num_adapters, (batch,), device=device)
    lora_B = torch.randn(num_adapters, in_dim, r, device=device, dtype=torch.float16)
    lora_A = torch.randn(num_adapters, r, out_dim, device=device, dtype=torch.float16)

    out_naive = sgmv_lora_forward_naive(x, adapter_ids, lora_B, lora_A)
    out_sgmv = sgmv_lora_forward(x, adapter_ids, lora_B, lora_A)

    diff = (out_naive.float() - out_sgmv.float()).abs().max().item()
    print(f"Max diff (naive vs SGMV): {diff}")
    print("SGMV forward OK." if diff < 1e-1 else "Check implementation.")

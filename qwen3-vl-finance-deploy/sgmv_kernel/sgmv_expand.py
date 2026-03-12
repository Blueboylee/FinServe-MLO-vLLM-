"""
FinServe SGMV Expand Kernel — out[i] += scaling * y[i] @ B[adapter[i]]

多 LoRA 服务的第二步算子: 将 rank 维度展开回 hidden_dim,
并以 scatter-add 方式累加到基座模型输出上.

提供两种并行策略:
  1. Token-Parallel (sgmv_expand):
     每 token 一个 program, 内循环沿 hidden_dim tiling.
  2. Segment-Parallel (sgmv_expand_segmented):
     使用 tl.dot 调用 Tensor Core, 适合长段.

汇编级映射:
  tl.load  → LDG.E.128            (全局 128b 向量化加载)
  tl.store → STG.E.128            (全局 128b 向量化存储)
  tl.dot   → HMMA.16816.F32       (Tensor Core FP16→FP32)
  FMA loop → FFMA / HFMA2         (FP32/FP16 融合乘加)
  scatter  → LDG + FMA + STG      (read-modify-write, 无需 atomic)

内存带宽分析 (Qwen3-VL-8B, rank=64):
  B 矩阵:   rank × hidden_dim = 64 × 4096 = 256K elements = 512KB (fp16)
  每 token:  读 y[rank] + 读 B[rank×hidden] + 读 base[hidden] + 写 out[hidden]
           = 64 + 256K + 4096 + 4096 ≈ 265K elements → memory-bound
  L2 复用:  B 矩阵跨同 adapter token 复用, 第二次命中 L2 (40MB)
"""

import torch
import triton
import triton.language as tl


# ════════════════════════════════════════════════════════════════════
#  1. Token-Parallel SGMV Expand
#     Grid: (num_tokens, cdiv(hidden_dim, BLOCK_N))
#
#     每个 program 处理一个 token 的一个输出 tile:
#       out[token, n_tile:n_tile+BLOCK_N] += scaling * y[token] @ B[adapter, :, n_tile:]
#
#     内循环沿 rank 维度归约 (rank 通常很小, 1-2 次迭代即可).
#     外循环沿 hidden_dim 通过 grid 第二维 tiling.
# ════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_R": 16},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_R": 32},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_R": 64},  num_warps=4,  num_stages=3),
        triton.Config({"BLOCK_N": 256, "BLOCK_R": 16},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_N": 256, "BLOCK_R": 32},  num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_N": 256, "BLOCK_R": 64},  num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_N": 512, "BLOCK_R": 32},  num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_N": 512, "BLOCK_R": 64},  num_warps=8,  num_stages=3),
    ],
    key=["N", "R"],
)
@triton.jit
def _sgmv_expand_token_kernel(
    Y_ptr, W_ptr, Out_ptr,
    adapter_ids_ptr,
    stride_yt, stride_yn,
    stride_wa, stride_wk, stride_wn,
    stride_ot, stride_on,
    total_tokens,
    scaling,
    rank,
    N: tl.constexpr,
    R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    tid = tl.program_id(0)
    bn_id = tl.program_id(1)

    if tid >= total_tokens:
        return

    aid = tl.load(adapter_ids_ptr + tid)
    if aid < 0:
        return

    # 输出列偏移
    n_offs = bn_id * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_base = W_ptr + aid * stride_wa

    # 内循环: 沿 rank 维度归约
    for r_start in range(0, rank, BLOCK_R):
        r_offs = r_start + tl.arange(0, BLOCK_R)
        r_mask = r_offs < rank

        # 加载 y[token, r_tile]: shrink 输出的中间向量
        y_vals = tl.load(
            Y_ptr + tid * stride_yt + r_offs * stride_yn,
            mask=r_mask, other=0.0,
        )

        # 加载 B[adapter, r_tile, n_tile]: [BLOCK_R, BLOCK_N]
        b_tile = tl.load(
            w_base + r_offs[:, None] * stride_wk + n_offs[None, :] * stride_wn,
            mask=r_mask[:, None] & n_mask[None, :],
            other=0.0,
        )

        # FMA + reduction: y_vals[r] * b_tile[r, n] 沿 r 归约
        acc += tl.sum(
            y_vals[:, None].to(tl.float32) * b_tile.to(tl.float32),
            axis=0,
        )

    # Scatter-add: out[token, n_tile] += scaling * acc
    out_ptrs = Out_ptr + tid * stride_ot + n_offs * stride_on
    existing = tl.load(out_ptrs, mask=n_mask, other=0.0)
    result = existing.to(tl.float32) + scaling * acc
    tl.store(out_ptrs, result.to(Out_ptr.dtype.element_ty), mask=n_mask)


def sgmv_expand(
    y: torch.Tensor,
    w_b_stacked: torch.Tensor,
    token_adapter_ids: torch.Tensor,
    base_output: torch.Tensor,
    scaling: float = 1.0,
) -> torch.Tensor:
    """
    Token-parallel SGMV Expand with scatter-add.

    Args:
        y: [total_tokens, rank] shrink 输出
        w_b_stacked: [num_adapters, rank, hidden_dim] 堆叠的 LoRA-B 权重
        token_adapter_ids: [total_tokens] int32
        base_output: [total_tokens, hidden_dim] 基座模型输出 (in-place 累加)
        scaling: LoRA scaling factor (alpha / rank)

    Returns:
        base_output (已 in-place 累加 LoRA delta)
    """
    total_tokens, rank = y.shape
    hidden_dim = base_output.shape[1]

    R = triton.next_power_of_2(rank)
    grid = lambda meta: (total_tokens, triton.cdiv(hidden_dim, meta["BLOCK_N"]))

    _sgmv_expand_token_kernel[grid](
        y, w_b_stacked, base_output,
        token_adapter_ids,
        y.stride(0), y.stride(1),
        w_b_stacked.stride(0), w_b_stacked.stride(1), w_b_stacked.stride(2),
        base_output.stride(0), base_output.stride(1),
        total_tokens, scaling, rank,
        N=hidden_dim, R=R,
    )
    return base_output


# ════════════════════════════════════════════════════════════════════
#  2. Segment-Parallel SGMV Expand (Tensor Core)
#     Grid: (num_segments, cdiv(hidden_dim, BLOCK_N))
#
#     tl.dot: y_tile[BLOCK_M, BLOCK_R] @ b_tile[BLOCK_R, BLOCK_N]
#     → HMMA Tensor Core 矩阵乘, 段长 >= 16 时最优
# ════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_R": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_R": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_R": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_R": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_R": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_R": 16}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_R": 16}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_R": 32}, num_stages=3, num_warps=8),
    ],
    key=["N", "R"],
)
@triton.jit
def _sgmv_expand_seg_kernel(
    Y_ptr, W_ptr, Out_ptr,
    seg_starts_ptr, seg_lens_ptr, seg_adapters_ptr,
    stride_yt, stride_yn,
    stride_wa, stride_wk, stride_wn,
    stride_ot, stride_on,
    num_segs,
    scaling,
    N: tl.constexpr,
    R: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    seg_id = tl.program_id(0)
    bn_id = tl.program_id(1)

    if seg_id >= num_segs:
        return

    seg_start = tl.load(seg_starts_ptr + seg_id)
    seg_len = tl.load(seg_lens_ptr + seg_id)
    aid = tl.load(seg_adapters_ptr + seg_id)

    n_offs = bn_id * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    w_base = W_ptr + aid * stride_wa

    m_start = 0
    while m_start < seg_len:
        m_offs = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offs < seg_len

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for r_start in range(0, R, BLOCK_R):
            r_offs = r_start + tl.arange(0, BLOCK_R)
            r_mask = r_offs < R

            # 加载 y tile: [BLOCK_M, BLOCK_R]
            y_tile = tl.load(
                Y_ptr + (seg_start + m_offs[:, None]) * stride_yt + r_offs[None, :] * stride_yn,
                mask=m_mask[:, None] & r_mask[None, :],
                other=0.0,
            )

            # 加载 B tile: [BLOCK_R, BLOCK_N]
            b_tile = tl.load(
                w_base + r_offs[:, None] * stride_wk + n_offs[None, :] * stride_wn,
                mask=r_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

            # HMMA: Tensor Core matmul
            acc += tl.dot(y_tile, b_tile)

        # Scatter-add to output
        out_ptrs = Out_ptr + (seg_start + m_offs[:, None]) * stride_ot + n_offs[None, :] * stride_on
        existing = tl.load(out_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
        result = existing.to(tl.float32) + scaling * acc
        tl.store(out_ptrs, result.to(Out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])

        m_start += BLOCK_M


def sgmv_expand_segmented(
    y: torch.Tensor,
    w_b_stacked: torch.Tensor,
    seg_starts: torch.Tensor,
    seg_lens: torch.Tensor,
    seg_adapter_ids: torch.Tensor,
    base_output: torch.Tensor,
    scaling: float = 1.0,
) -> torch.Tensor:
    """
    Segment-parallel SGMV Expand (Tensor Core 加速).

    Args:
        y: [total_tokens, rank]
        w_b_stacked: [num_adapters, rank, hidden_dim]
        seg_starts, seg_lens, seg_adapter_ids: [num_segments] int32
        base_output: [total_tokens, hidden_dim] (in-place)
        scaling: LoRA scaling

    Returns:
        base_output (in-place)
    """
    total_tokens, rank = y.shape
    hidden_dim = base_output.shape[1]
    num_segs = seg_starts.shape[0]

    grid = lambda meta: (num_segs, triton.cdiv(hidden_dim, meta["BLOCK_N"]))

    _sgmv_expand_seg_kernel[grid](
        y, w_b_stacked, base_output,
        seg_starts, seg_lens, seg_adapter_ids,
        y.stride(0), y.stride(1),
        w_b_stacked.stride(0), w_b_stacked.stride(1), w_b_stacked.stride(2),
        base_output.stride(0), base_output.stride(1),
        num_segs, scaling,
        N=hidden_dim, R=rank,
    )
    return base_output


# ════════════════════════════════════════════════════════════════════
#  3. PyTorch Baseline
# ════════════════════════════════════════════════════════════════════

def baseline_sgmv_expand(
    y: torch.Tensor,
    w_b_stacked: torch.Tensor,
    token_adapter_ids: torch.Tensor,
    base_output: torch.Tensor,
    scaling: float = 1.0,
) -> torch.Tensor:
    """PyTorch 参考实现."""
    out = base_output.clone()
    for i in range(y.shape[0]):
        aid = token_adapter_ids[i].item()
        if aid >= 0:
            delta = y[i].float() @ w_b_stacked[aid].float()
            out[i] = out[i].float() + scaling * delta
            out[i] = out[i].to(base_output.dtype)
    return out

"""
FinServe SGMV Shrink Kernel — y[i] = x[i] @ A[adapter[i]]

多 LoRA 服务的核心 memory-bound 算子: 将 hidden_dim 维度投影到 rank 维度.
每个 token 根据 adapter_id 索引对应的 LoRA-A 权重矩阵, 执行向量-矩阵乘.

提供两种并行策略:
  1. Token-Parallel (sgmv_shrink):
     每 token 一个 program, 适合 token 未排序 / 段长度不均的场景.
  2. Segment-Parallel (sgmv_shrink_segmented):
     每段 (同 adapter 连续 token) 一组 tile, 使用 tl.dot 调用 Tensor Core.
     适合 LoRA-Aware Scheduler 已按 adapter 亲和排序的场景.

汇编级映射 (Triton → PTX/SASS):
  tl.load          → LDG.E.128 / LDG.E.64 (向量化全局加载, 128b/64b 事务)
  tl.store         → STG.E.128               (向量化全局存储)
  tl.dot           → HMMA.16816.F32          (FP16 Tensor Core 矩阵乘)
  tl.sum(a*b, 0)   → FMA + SHFL.SYNC.BFLY   (寄存器乘累加 + warp 蝶形归约)
  num_stages=N     → CP.ASYNC.CG.SHARED.GLOBAL + COMMIT_GROUP + WAIT_GROUP
                     (全局→共享内存异步流水线, 隐藏访存延迟)
  eviction_policy  → LDG.E.128.LTC64B        (L2 缓存驻留提示)

Qwen3-VL-8B 参数: hidden=4096, rank∈{4,8,16,32,64}
"""

import torch
import triton
import triton.language as tl


# ════════════════════════════════════════════════════════════════════
#  1. Token-Parallel SGMV Shrink
#     Grid: (num_tokens,)  每 token 独立 GEMV
#
#     寄存器布局:
#       acc[BN]  — fp32 累加器, 驻留 RF 零溢出
#       x_chunk[BK] — 输入 activation 向量片段
#       w_tile[BK, BN] — A 矩阵 tile
#
#     内存访问模式:
#       x: 行连续, stride=1 → 合并加载 (coalesced)
#       W: adapter 偏移 + k*stride_k + n → 连续 n 维合并
#       每个 k 迭代: 读 BK scalars(x) + BK×BN elements(W)
#       写: BN elements(y) — 单次突发写
# ════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 64},  num_warps=2,  num_stages=2),
        triton.Config({"BLOCK_K": 64},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_K": 128}, num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_K": 128}, num_warps=4,  num_stages=3),
        triton.Config({"BLOCK_K": 256}, num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_K": 256}, num_warps=8,  num_stages=3),
    ],
    key=["K", "BN"],
)
@triton.jit
def _sgmv_shrink_token_kernel(
    X_ptr, W_ptr, Y_ptr,
    adapter_ids_ptr,
    stride_xt: tl.constexpr,
    stride_wa, stride_wk, stride_wn,
    stride_yt: tl.constexpr,
    total_tokens,
    rank,
    K: tl.constexpr,
    BN: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tid = tl.program_id(0)
    if tid >= total_tokens:
        return

    aid = tl.load(adapter_ids_ptr + tid)
    if aid < 0:
        return

    n_offs = tl.arange(0, BN)
    n_mask = n_offs < rank
    acc = tl.zeros((BN,), dtype=tl.float32)

    w_base = W_ptr + aid * stride_wa

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # LDG: 向量化加载 x 的 BLOCK_K 个元素 → 寄存器
        x_chunk = tl.load(
            X_ptr + tid * stride_xt + k_offs,
            mask=k_mask, other=0.0,
        )

        # LDG: 加载 W[adapter, k_tile, :rank] → 寄存器 tile
        # 内存布局: W[a, k, n] 按 n 连续, 确保 warp 内线程合并访问 n 维
        w_tile = tl.load(
            w_base + k_offs[:, None] * stride_wk + n_offs[None, :] * stride_wn,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )

        # FMA + SHFL reduction: x_chunk[k] * w_tile[k, n] 沿 k 归约
        acc += tl.sum(
            x_chunk[:, None].to(tl.float32) * w_tile.to(tl.float32),
            axis=0,
        )

    # STG: 写回 y[token, :rank]
    tl.store(
        Y_ptr + tid * stride_yt + n_offs,
        acc.to(Y_ptr.dtype.element_ty),
        mask=n_mask,
    )


def sgmv_shrink(
    x: torch.Tensor,
    w_a_stacked: torch.Tensor,
    token_adapter_ids: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """
    Token-parallel SGMV Shrink.

    Args:
        x: [total_tokens, hidden_dim] 输入 activation (fp16/bf16)
        w_a_stacked: [num_adapters, hidden_dim, rank] 堆叠的 LoRA-A 权重
        token_adapter_ids: [total_tokens] int32, 每 token 的 adapter 索引, -1 表示无 adapter
        rank: LoRA rank

    Returns:
        y: [total_tokens, rank] 收缩后的中间表示
    """
    total_tokens, hidden_dim = x.shape
    y = torch.zeros(total_tokens, rank, dtype=x.dtype, device=x.device)

    BN = triton.next_power_of_2(rank)
    grid = (total_tokens,)

    _sgmv_shrink_token_kernel[grid](
        x, w_a_stacked, y,
        token_adapter_ids,
        x.stride(0),
        w_a_stacked.stride(0), w_a_stacked.stride(1), w_a_stacked.stride(2),
        y.stride(0),
        total_tokens, rank,
        K=hidden_dim, BN=BN,
    )
    return y


# ════════════════════════════════════════════════════════════════════
#  2. Segment-Parallel SGMV Shrink (Tensor Core)
#     Grid: (num_segments, cdiv(rank, BLOCK_N))
#
#     使用 tl.dot 映射到 HMMA 指令:
#       x_tile[BLOCK_M, BLOCK_K] @ w_tile[BLOCK_K, BLOCK_N]
#       → HMMA.16816.F32 (BLOCK_M >= 16, BLOCK_K >= 16, BLOCK_N >= 16)
#
#     num_stages 控制 cp.async 流水线深度:
#       stage=2: 双缓冲, 一组数据在计算, 另一组异步加载
#       stage=3: 三缓冲, 进一步隐藏高延迟全局访存
#
#     适合 LoRA-Aware Scheduler 已将同 adapter 请求聚合的场景,
#     段长度 >= 16 时 Tensor Core 利用率最高.
# ════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 64},  num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 64},  num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},  num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 64},  num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64},  num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},  num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},  num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},  num_stages=3, num_warps=8),
    ],
    key=["K", "N"],
)
@triton.jit
def _sgmv_shrink_seg_kernel(
    X_ptr, W_ptr, Y_ptr,
    seg_starts_ptr, seg_lens_ptr, seg_adapters_ptr,
    stride_xt, stride_xk,
    stride_wa, stride_wk, stride_wn,
    stride_yt, stride_yn,
    num_segs,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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

    # 外层循环: 沿 M (token) 维度 tiling
    m_start = 0
    while m_start < seg_len:
        m_offs = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offs < seg_len

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # 内层循环: 沿 K (hidden_dim) 维度 tiling
        # 每迭代: CP.ASYNC 预取下一 tile, HMMA 计算当前 tile
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K

            # 加载 x tile: [BLOCK_M, BLOCK_K]
            x_tile = tl.load(
                X_ptr + (seg_start + m_offs[:, None]) * stride_xt + k_offs[None, :] * stride_xk,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            # 加载 W tile: [BLOCK_K, BLOCK_N]
            # 权重跨段内所有 token 复用 → 自然驻留 L2/L1
            w_tile = tl.load(
                w_base + k_offs[:, None] * stride_wk + n_offs[None, :] * stride_wn,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

            # HMMA: Tensor Core 矩阵乘累加
            acc += tl.dot(x_tile, w_tile)

        # STG: 写回 y[seg_start+m:, n_tile]
        y_ptrs = Y_ptr + (seg_start + m_offs[:, None]) * stride_yt + n_offs[None, :] * stride_yn
        tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])

        m_start += BLOCK_M


def sgmv_shrink_segmented(
    x: torch.Tensor,
    w_a_stacked: torch.Tensor,
    seg_starts: torch.Tensor,
    seg_lens: torch.Tensor,
    seg_adapter_ids: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """
    Segment-parallel SGMV Shrink (Tensor Core 加速).

    Args:
        x: [total_tokens, hidden_dim]
        w_a_stacked: [num_adapters, hidden_dim, rank]
        seg_starts: [num_segments] int32, 每段起始 token 索引
        seg_lens: [num_segments] int32, 每段长度
        seg_adapter_ids: [num_segments] int32, 每段的 adapter 索引
        rank: LoRA rank

    Returns:
        y: [total_tokens, rank]
    """
    total_tokens, hidden_dim = x.shape
    y = torch.zeros(total_tokens, rank, dtype=x.dtype, device=x.device)
    num_segs = seg_starts.shape[0]

    padded_n = max(triton.next_power_of_2(rank), 16)
    grid = (num_segs, triton.cdiv(rank, padded_n))

    _sgmv_shrink_seg_kernel[grid](
        x, w_a_stacked, y,
        seg_starts, seg_lens, seg_adapter_ids,
        x.stride(0), x.stride(1),
        w_a_stacked.stride(0), w_a_stacked.stride(1), w_a_stacked.stride(2),
        y.stride(0), y.stride(1),
        num_segs,
        K=hidden_dim, N=rank,
    )
    return y


# ════════════════════════════════════════════════════════════════════
#  3. PyTorch Baseline
# ════════════════════════════════════════════════════════════════════

def baseline_sgmv_shrink(
    x: torch.Tensor,
    w_a_stacked: torch.Tensor,
    token_adapter_ids: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """PyTorch 参考实现, 用于正确性验证."""
    total_tokens = x.shape[0]
    y = torch.zeros(total_tokens, rank, dtype=x.dtype, device=x.device)
    for i in range(total_tokens):
        aid = token_adapter_ids[i].item()
        if aid >= 0:
            y[i] = x[i].float() @ w_a_stacked[aid].float()
            y[i] = y[i].to(x.dtype)
    return y

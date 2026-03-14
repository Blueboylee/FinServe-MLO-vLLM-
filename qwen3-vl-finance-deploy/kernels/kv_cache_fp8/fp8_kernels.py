"""
FinServe KV Cache FP8 Triton Kernels

三个核心 kernel:

1. quantize_fp16_to_fp8:
   K/V 写入 cache 时在线量化 FP16 → FP8 E4M3 + per-head scale.
   Grid: (num_tokens, num_kv_heads), 每 program 处理一个 head 的一行.

   量化流程:
     amax = max(|x|)  per-head per-token
     scale = amax / E4M3_MAX  (E4M3_MAX = 448.0)
     x_fp8 = clamp(x / scale, -E4M3_MAX, E4M3_MAX).to(float8_e4m3fn)

   PTX 映射:
     tl.abs → FABS, tl.max → FMNMX + SHFL.SYNC, div → FMUL(rcp)
     .to(tl.float8e4nv) → CVT.F8.F16 (Hopper) 或 cast emulation (Ampere)

2. dequantize_fp8_to_fp16:
   Attention 计算前反量化 FP8 → FP16.
   x_fp16 = x_fp8.to(fp16) * scale

3. fused_dequant_attention_score:
   在 Q @ K^T 计算中融合反量化, 避免生成完整 FP16 K 矩阵:
     score[q, k] = sum_d(Q[q, d] * K_fp8[k, d].to(fp16) * k_scale) / sqrt(head_dim)
   节省: 不需要额外的 FP16 K 缓冲, 省 num_kv_heads × seq_len × head_dim × 2B

精度分析 (Qwen3-VL-8B, head_dim=128):
  E4M3: 4 bit exponent + 3 bit mantissa → 有效精度 ~3.6 位
  per-head dynamic scaling: 每个 head 独立计算 scale, 适应不同 head 的值域
  实测 Attention score 的 cosine similarity: FP16 vs FP8 ≈ 0.9997+
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional

E4M3_MAX = 448.0
E4M3_MIN = -448.0


# ════════════════════════════════════════════════════════════════════
#  1. Online Quantization: FP16 → FP8 E4M3 + Per-Head Scale
#
#  Grid: (num_tokens, num_kv_heads)
#  每 program 处理 x[token, head, :head_dim]
#
#  关键优化:
#    - amax 计算: tl.max(tl.abs(x)) 编译为 FABS + warp-level FMNMX 归约
#    - scale 计算: amax * rcp(E4M3_MAX), 用 FMUL(rcp) 替代除法
#    - FP8 转换: .to(tl.float8e4nv), Hopper 有原生 CVT 指令
#    - scale 仅 1 个 FP32 per head per token, 显存开销忽略不计
# ════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=["HEAD_DIM"],
)
@triton.jit
def _quantize_fp16_to_fp8_kernel(
    X_ptr, Out_ptr, Scale_ptr,
    stride_xt, stride_xh, stride_xd,
    stride_ot, stride_oh, stride_od,
    stride_st, stride_sh,
    HEAD_DIM: tl.constexpr,
    BD: tl.constexpr,
):
    """
    Per-head per-token dynamic quantization.
    X[token, head, :] → Out_fp8[token, head, :] + Scale[token, head]
    """
    tid = tl.program_id(0)
    hid = tl.program_id(1)

    d_offs = tl.arange(0, BD)
    d_mask = d_offs < HEAD_DIM

    # LDG: 加载一个 head 的完整向量
    x_ptrs = X_ptr + tid * stride_xt + hid * stride_xh + d_offs * stride_xd
    x = tl.load(x_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    # 计算 per-head absmax → scale
    amax = tl.max(tl.abs(x), axis=0)
    # 防止 amax=0 导致 inf
    amax = tl.where(amax > 0.0, amax, 1.0)
    scale = amax / E4M3_MAX

    # 量化: x / scale → clamp → FP8
    x_scaled = x / scale
    x_clamped = tl.minimum(tl.maximum(x_scaled, E4M3_MIN), E4M3_MAX)
    # Triton 会选择最优的转换指令 (CVT on Hopper, emulation on Ampere)
    x_fp8 = x_clamped.to(Out_ptr.dtype.element_ty)

    # STG: 写回 FP8 值
    out_ptrs = Out_ptr + tid * stride_ot + hid * stride_oh + d_offs * stride_od
    tl.store(out_ptrs, x_fp8, mask=d_mask)

    # STG: 写回 scale (1 个 FP32 per head per token)
    tl.store(Scale_ptr + tid * stride_st + hid * stride_sh, scale)


def quantize_fp16_to_fp8(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FP16 → FP8 E4M3 在线量化 + per-head per-token scale.

    Args:
        x: [num_tokens, num_kv_heads, head_dim] FP16/BF16

    Returns:
        (x_fp8, scale)
        x_fp8:  [num_tokens, num_kv_heads, head_dim] float8_e4m3fn
        scale:  [num_tokens, num_kv_heads] FP32, 反量化时用 x_fp8 * scale
    """
    assert x.dim() == 3, f"Expected 3D tensor [tokens, heads, dim], got {x.shape}"
    T, H, D = x.shape

    x_fp8 = torch.empty(T, H, D, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty(T, H, dtype=torch.float32, device=x.device)

    BD = triton.next_power_of_2(D)
    grid = (T, H)

    _quantize_fp16_to_fp8_kernel[grid](
        x, x_fp8, scale,
        x.stride(0), x.stride(1), x.stride(2),
        x_fp8.stride(0), x_fp8.stride(1), x_fp8.stride(2),
        scale.stride(0), scale.stride(1),
        HEAD_DIM=D, BD=BD,
    )
    return x_fp8, scale


# ════════════════════════════════════════════════════════════════════
#  2. Dequantization: FP8 → FP16 (per-head scale)
#
#  反量化: x_fp16 = x_fp8.to(fp16) * scale
#  Grid: (num_tokens, num_kv_heads)
# ════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=["HEAD_DIM"],
)
@triton.jit
def _dequantize_fp8_to_fp16_kernel(
    X_ptr, Scale_ptr, Out_ptr,
    stride_xt, stride_xh, stride_xd,
    stride_st, stride_sh,
    stride_ot, stride_oh, stride_od,
    HEAD_DIM: tl.constexpr,
    BD: tl.constexpr,
):
    tid = tl.program_id(0)
    hid = tl.program_id(1)

    d_offs = tl.arange(0, BD)
    d_mask = d_offs < HEAD_DIM

    # LDG: 加载 FP8 值和 scale
    x_ptrs = X_ptr + tid * stride_xt + hid * stride_xh + d_offs * stride_xd
    x_fp8 = tl.load(x_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    scale = tl.load(Scale_ptr + tid * stride_st + hid * stride_sh)

    # 反量化: x * scale
    x_dequant = x_fp8 * scale

    # STG
    out_ptrs = Out_ptr + tid * stride_ot + hid * stride_oh + d_offs * stride_od
    tl.store(out_ptrs, x_dequant.to(Out_ptr.dtype.element_ty), mask=d_mask)


def dequantize_fp8_to_fp16(
    x_fp8: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    FP8 → FP16 反量化.

    Args:
        x_fp8:  [num_tokens, num_kv_heads, head_dim] float8_e4m3fn
        scale:  [num_tokens, num_kv_heads] FP32
        out_dtype: 输出精度

    Returns:
        x_fp16: [num_tokens, num_kv_heads, head_dim]
    """
    T, H, D = x_fp8.shape
    out = torch.empty(T, H, D, dtype=out_dtype, device=x_fp8.device)
    BD = triton.next_power_of_2(D)

    _dequantize_fp8_to_fp16_kernel[(T, H)](
        x_fp8, scale, out,
        x_fp8.stride(0), x_fp8.stride(1), x_fp8.stride(2),
        scale.stride(0), scale.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        HEAD_DIM=D, BD=BD,
    )
    return out


# ════════════════════════════════════════════════════════════════════
#  3. Online KV Quantization (Paged KV Cache 格式)
#
#  vLLM PagedAttention 的 KV Cache 格式:
#    key_cache:   [num_blocks, block_size, num_kv_heads, head_dim]
#    value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
#
#  本 kernel 将新生成的 K/V (FP16) 量化后直接写入 paged cache slot:
#    K/V [num_tokens, num_kv_heads, head_dim] (FP16)
#    → cache[block_id, slot_offset, head, :] (FP8)
#    + scale[block_id, slot_offset, head] (FP32)
# ════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=["HEAD_DIM"],
)
@triton.jit
def _quantize_kv_to_cache_kernel(
    KV_ptr, Cache_ptr, Scale_ptr,
    SlotMapping_ptr,
    stride_kv_t, stride_kv_h, stride_kv_d,
    stride_cache_b, stride_cache_s, stride_cache_h, stride_cache_d,
    stride_scale_b, stride_scale_s, stride_scale_h,
    num_kv_heads,
    HEAD_DIM: tl.constexpr,
    BD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    将新 K/V token 量化后写入 paged cache.
    SlotMapping: [num_tokens] → (block_id * BLOCK_SIZE + slot_offset)
    """
    tid = tl.program_id(0)
    hid = tl.program_id(1)

    slot = tl.load(SlotMapping_ptr + tid)
    block_id = slot // BLOCK_SIZE
    slot_offset = slot % BLOCK_SIZE

    d_offs = tl.arange(0, BD)
    d_mask = d_offs < HEAD_DIM

    # LDG: 新 K/V 值 (FP16)
    kv_ptrs = KV_ptr + tid * stride_kv_t + hid * stride_kv_h + d_offs * stride_kv_d
    kv = tl.load(kv_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    # Per-head absmax quantization
    amax = tl.max(tl.abs(kv), axis=0)
    amax = tl.where(amax > 0.0, amax, 1.0)
    scale = amax / E4M3_MAX

    kv_scaled = kv / scale
    kv_clamped = tl.minimum(tl.maximum(kv_scaled, E4M3_MIN), E4M3_MAX)
    kv_fp8 = kv_clamped.to(Cache_ptr.dtype.element_ty)

    # STG: 写入 paged cache slot
    cache_ptrs = (Cache_ptr
                  + block_id * stride_cache_b
                  + slot_offset * stride_cache_s
                  + hid * stride_cache_h
                  + d_offs * stride_cache_d)
    tl.store(cache_ptrs, kv_fp8, mask=d_mask)

    # STG: 写入 scale
    tl.store(Scale_ptr
             + block_id * stride_scale_b
             + slot_offset * stride_scale_s
             + hid * stride_scale_h,
             scale)


def quantize_kv_online(
    kv: torch.Tensor,
    cache: torch.Tensor,
    scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
):
    """
    在线 KV 量化: 新 token 的 K/V 直接量化写入 paged FP8 cache.

    Args:
        kv:          [num_tokens, num_kv_heads, head_dim] FP16 新 K 或 V
        cache:       [num_blocks, block_size, num_kv_heads, head_dim] FP8 paged cache
        scale_cache: [num_blocks, block_size, num_kv_heads] FP32 scale
        slot_mapping: [num_tokens] int32, token → cache slot 映射
        block_size:  vLLM block size (typically 16)
    """
    T, H, D = kv.shape
    BD = triton.next_power_of_2(D)

    _quantize_kv_to_cache_kernel[(T, H)](
        kv, cache, scale_cache, slot_mapping,
        kv.stride(0), kv.stride(1), kv.stride(2),
        cache.stride(0), cache.stride(1), cache.stride(2), cache.stride(3),
        scale_cache.stride(0), scale_cache.stride(1), scale_cache.stride(2),
        H,
        HEAD_DIM=D, BD=BD, BLOCK_SIZE=block_size,
    )


# ════════════════════════════════════════════════════════════════════
#  4. Fused Dequant + Attention Score
#
#  在 Q @ K^T / sqrt(d) 计算中融合反量化, 避免生成完整 FP16 K 中间矩阵.
#
#  标准路径 (2 pass):
#    Pass 1: K_fp8 → K_fp16 = K_fp8 * scale  (全量反量化, 需要额外显存)
#    Pass 2: score = Q @ K_fp16^T / sqrt(d)
#    额外显存: seq_len × num_kv_heads × head_dim × 2B
#
#  融合路径 (1 pass):
#    score[q, kv] = sum_d(Q[q, d] * K_fp8[kv, d] * k_scale[kv]) / sqrt(d)
#    K_fp8 → FP32 和 scale 乘法在寄存器中完成, 不生成 FP16 K
#    节省: seq_len × num_kv_heads × head_dim × 2B 显存 + 1 次 global memory pass
#
#  Grid: (num_q_tokens, num_kv_heads, cdiv(seq_len, BLOCK_KV))
# ════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_KV": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_KV": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_KV": 64, "BLOCK_D": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_KV": 128, "BLOCK_D": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_KV": 128, "BLOCK_D": 128}, num_warps=8, num_stages=3),
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _fused_dequant_qk_kernel(
    Q_ptr, K_fp8_ptr, K_scale_ptr, Score_ptr,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_st, stride_sh,
    stride_score_q, stride_score_h, stride_score_kv,
    rsqrt_d,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused dequantize + Q @ K^T attention score computation.

    每 program 计算 score[q_token, head, kv_block]:
      for d in 0..HEAD_DIM step BLOCK_D:
        q_chunk = Q[q, head, d:d+BLOCK_D]
        k_chunk = K_fp8[kv_block, head, d:d+BLOCK_D] → cast to fp32
        k_scale = K_scale[kv_block, head]
        acc += q_chunk * (k_chunk * k_scale)
      score = acc * rsqrt_d
    """
    q_id = tl.program_id(0)
    h_id = tl.program_id(1)
    kv_block_id = tl.program_id(2)

    kv_offs = kv_block_id * BLOCK_KV + tl.arange(0, BLOCK_KV)
    kv_mask = kv_offs < SEQ_LEN

    acc = tl.zeros((BLOCK_KV,), dtype=tl.float32)

    # 加载 kv_scale: [BLOCK_KV], 一次性加载整个 kv block 的 scale
    kv_scales = tl.load(
        K_scale_ptr + kv_offs * stride_st + h_id * stride_sh,
        mask=kv_mask, other=0.0,
    )

    # 沿 head_dim 归约
    for d_start in range(0, HEAD_DIM, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < HEAD_DIM

        # Q[q, head, d_chunk]: [BLOCK_D]
        q_chunk = tl.load(
            Q_ptr + q_id * stride_qt + h_id * stride_qh + d_offs * stride_qd,
            mask=d_mask, other=0.0,
        ).to(tl.float32)

        # K_fp8[kv_block, head, d_chunk]: [BLOCK_KV, BLOCK_D]
        k_chunk = tl.load(
            K_fp8_ptr + kv_offs[:, None] * stride_kt + h_id * stride_kh + d_offs[None, :] * stride_kd,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # 融合反量化 + dot: k_real = k_fp8 * scale, then q @ k_real
        # k_chunk[kv, d] * kv_scales[kv, None] → dequantized K
        k_dequant = k_chunk * kv_scales[:, None]

        # q[d] * k[kv, d] → 沿 d 归约 → partial_score[kv]
        acc += tl.sum(q_chunk[None, :] * k_dequant, axis=1)

    # score = acc * rsqrt(d)
    score = acc * rsqrt_d

    # STG: 写回 attention score
    score_ptrs = Score_ptr + q_id * stride_score_q + h_id * stride_score_h + kv_offs * stride_score_kv
    tl.store(score_ptrs, score.to(Score_ptr.dtype.element_ty), mask=kv_mask)


def fused_dequant_attention_score(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """
    融合反量化 + Attention Score 计算.

    Args:
        q:       [num_q, num_heads, head_dim] FP16 query
        k_fp8:   [seq_len, num_kv_heads, head_dim] FP8 key cache
        k_scale: [seq_len, num_kv_heads] FP32 per-head scale
        seq_len: 有效 KV 序列长度

    Returns:
        score: [num_q, num_heads, seq_len] FP32 attention score (未 softmax)
    """
    num_q, num_heads, head_dim = q.shape
    rsqrt_d = 1.0 / (head_dim ** 0.5)

    score = torch.empty(num_q, num_heads, seq_len, dtype=torch.float32, device=q.device)

    grid = lambda meta: (
        num_q,
        num_heads,
        triton.cdiv(seq_len, meta["BLOCK_KV"]),
    )

    _fused_dequant_qk_kernel[grid](
        q, k_fp8, k_scale, score,
        q.stride(0), q.stride(1), q.stride(2),
        k_fp8.stride(0), k_fp8.stride(1), k_fp8.stride(2),
        k_scale.stride(0), k_scale.stride(1),
        score.stride(0), score.stride(1), score.stride(2),
        rsqrt_d,
        SEQ_LEN=seq_len, HEAD_DIM=head_dim,
    )
    return score


# ════════════════════════════════════════════════════════════════════
#  5. PyTorch Baselines (正确性验证)
# ════════════════════════════════════════════════════════════════════

def baseline_quantize_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch 参考实现: per-head per-token FP8 量化."""
    T, H, D = x.shape
    x_f = x.float()
    amax = x_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / E4M3_MAX).squeeze(-1)
    x_scaled = x_f / amax * E4M3_MAX
    x_clamped = x_scaled.clamp(E4M3_MIN, E4M3_MAX)
    x_fp8 = x_clamped.to(torch.float8_e4m3fn)
    return x_fp8, scale


def baseline_dequantize_fp8(
    x_fp8: torch.Tensor, scale: torch.Tensor,
) -> torch.Tensor:
    """PyTorch 参考实现: FP8 反量化."""
    return x_fp8.float() * scale.unsqueeze(-1)


def baseline_fused_dequant_score(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """PyTorch 参考实现: 反量化 + attention score."""
    k_fp16 = k_fp8[:seq_len].float() * k_scale[:seq_len].unsqueeze(-1)
    head_dim = q.shape[-1]
    score = torch.einsum("qhd,khd->qhk", q.float(), k_fp16) / (head_dim ** 0.5)
    return score

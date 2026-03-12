"""
FinServe Qwen3-VL-8B · Triton Kernel Integration for vLLM

自定义 Triton kernel 替换 vLLM 内置算子:
  - RMSNorm          → fused_rms_norm         (autotuned)
  - SiluAndMul       → fused_silu_mul_concat  (零拷贝, autotuned)
  - Residual+RMSNorm → fused_add_rms_norm     (单 pass 融合)
  - RoPE             → fused_rotary_emb       (可选, 不 monkey-patch)

使用方式:
    from triton_integration import apply_triton_optimizations
    apply_triton_optimizations()
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


# ════════════════════════════════════════════════════════════════
#  1 · Fused RMSNorm  (autotuned num_warps)
# ════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4,  num_stages=1),
        triton.Config({}, num_warps=8,  num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
    ],
    key=["N"],
)
@triton.jit
def _rms_norm_fwd(
    X, W, Y,
    stride,
    N: tl.constexpr,
    eps: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BN)
    mask = cols < N
    off = row * stride + cols
    x = tl.load(X + off, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=1.0)
    xf = x.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(xf * xf, axis=0) / N + eps)
    tl.store(Y + off, (xf * rstd).to(x.dtype) * w, mask=mask)


def fused_rms_norm(
    x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6,
) -> torch.Tensor:
    sh = x.shape
    x2 = x.view(-1, sh[-1])
    y = torch.empty_like(x2)
    M, N = x2.shape
    _rms_norm_fwd[(M,)](
        x2, w, y, x2.stride(0),
        N=N, eps=eps, BN=triton.next_power_of_2(N),
    )
    return y.view(sh)


# ════════════════════════════════════════════════════════════════
#  2 · Fused SiLU × Mul — 拼接输入, 零拷贝  (vLLM monkey-patch 用)
#
#  vLLM 的 SiluAndMul 接收 [gate, up] 拼接张量 shape=[..., 2D]
#  直接从拼接输入读取, 不需要 slice + .contiguous()
# ════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 512},  num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=8),
        triton.Config({"BLOCK": 2048}, num_warps=8),
        triton.Config({"BLOCK": 4096}, num_warps=8),
        triton.Config({"BLOCK": 4096}, num_warps=16),
    ],
    key=["D"],
)
@triton.jit
def _silu_mul_concat_fwd(
    X, O,
    stride_x, stride_o,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """读 gate=X[row, :D] 和 up=X[row, D:2D], 写 silu(gate)*up 到 O[row, :D]"""
    row = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < D
    base = row * stride_x

    g = tl.load(X + base + cols, mask=mask, other=0.0)
    u = tl.load(X + base + D + cols, mask=mask, other=0.0)
    gf = g.to(tl.float32)
    # SiLU(x) = x * sigmoid(x)
    result = (gf * tl.sigmoid(gf)) * u.to(tl.float32)
    tl.store(O + row * stride_o + cols, result.to(g.dtype), mask=mask)


def fused_silu_mul_concat(x: torch.Tensor) -> torch.Tensor:
    """SiLU×Mul on concatenated [gate, up] tensor — zero copy."""
    shape = x.shape
    d = shape[-1] // 2
    x2 = x.view(-1, shape[-1])
    M = x2.shape[0]
    out = torch.empty(M, d, dtype=x.dtype, device=x.device)
    grid = lambda meta: (M, triton.cdiv(d, meta["BLOCK"]))
    _silu_mul_concat_fwd[grid](x2, out, x2.stride(0), d, D=d)
    return out.view(*shape[:-1], d)


# ════════════════════════════════════════════════════════════════
#  2b · Fused SiLU × Mul — 分离输入 (legacy, 供外部直接调用)
# ════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 2048}, num_warps=8),
        triton.Config({"BLOCK": 4096}, num_warps=8),
    ],
    key=["n"],
)
@triton.jit
def _silu_mul_sep_fwd(G, U, O, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(G + offs, mask=mask, other=0.0)
    u = tl.load(U + offs, mask=mask, other=0.0)
    gf = g.to(tl.float32)
    result = (gf * tl.sigmoid(gf)) * u.to(tl.float32)
    tl.store(O + offs, result.to(g.dtype), mask=mask)


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SiLU×Mul with separate gate/up tensors."""
    assert gate.shape == up.shape
    gate_c, up_c = gate.contiguous(), up.contiguous()
    o = torch.empty_like(gate_c)
    n = gate_c.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _silu_mul_sep_fwd[grid](gate_c, up_c, o, n=n)
    return o


# ════════════════════════════════════════════════════════════════
#  3 · Fused Residual-Add + RMSNorm  (autotuned)
# ════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4,  num_stages=1),
        triton.Config({}, num_warps=8,  num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
    ],
    key=["N"],
)
@triton.jit
def _add_rms_norm_fwd(
    X, Res, W, Y,
    stride,
    N: tl.constexpr,
    eps: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BN)
    mask = cols < N
    base = row * stride
    x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(Res + base + cols, mask=mask, other=0.0).to(tl.float32)
    h = x + res
    w = tl.load(W + cols, mask=mask, other=1.0)
    tl.store(Res + base + cols, h.to(w.dtype), mask=mask)
    rstd = 1.0 / tl.sqrt(tl.sum(h * h, axis=0) / N + eps)
    tl.store(Y + base + cols, (h * rstd).to(w.dtype) * w, mask=mask)


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (normed_output, updated_residual). residual is modified in-place."""
    sh = x.shape
    x2 = x.view(-1, sh[-1])
    r2 = residual.view(-1, sh[-1])
    y = torch.empty_like(x2)
    M, N = x2.shape
    _add_rms_norm_fwd[(M,)](
        x2, r2, w, y, x2.stride(0),
        N=N, eps=eps, BN=triton.next_power_of_2(N),
    )
    return y.view(sh), residual


# ════════════════════════════════════════════════════════════════
#  4 · Fused Rotary Position Embedding (不 monkey-patch, 可独立调用)
# ════════════════════════════════════════════════════════════════

@triton.jit
def _rotary_fwd(
    Q, Cos, Sin, O,
    stride_t, stride_h,
    HALF: tl.constexpr,
):
    tok = tl.program_id(0)
    hd = tl.program_id(1)
    base = tok * stride_t + hd * stride_h
    d = tl.arange(0, HALF)
    cos_off = tok * HALF

    x0 = tl.load(Q + base + d * 2)
    x1 = tl.load(Q + base + d * 2 + 1)
    co = tl.load(Cos + cos_off + d).to(tl.float32)
    si = tl.load(Sin + cos_off + d).to(tl.float32)

    r0 = x0.to(tl.float32) * co - x1.to(tl.float32) * si
    r1 = x0.to(tl.float32) * si + x1.to(tl.float32) * co
    tl.store(O + base + d * 2, r0.to(x0.dtype))
    tl.store(O + base + d * 2 + 1, r1.to(x0.dtype))


def fused_rotary_emb(
    q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> torch.Tensor:
    T, H, D = q.shape
    out = torch.empty_like(q)
    _rotary_fwd[(T, H)](
        q, cos, sin, out,
        q.stride(0), q.stride(1),
        HALF=D // 2,
    )
    return out


# ════════════════════════════════════════════════════════════════
#  5 · vLLM Monkey-Patch
# ════════════════════════════════════════════════════════════════

_ORIG_FNS: dict = {}


def apply_triton_optimizations() -> list:
    """
    Monkey-patch vLLM 的 RMSNorm / SiluAndMul 为 Triton kernel.

    要求 enforce_eager=True 以禁用 CUDA Graph, 否则 patch 不生效.
    返回成功 patch 的算子名称列表.
    """
    global _ORIG_FNS
    patched: list = []

    # ── RMSNorm ──
    try:
        from vllm.model_executor.layers.layernorm import RMSNorm

        _ORIG_FNS["rms_norm_forward"] = RMSNorm.forward

        def _patched_rms(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
            try:
                eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
                if residual is not None:
                    return fused_add_rms_norm(x, residual, self.weight, eps)
                return fused_rms_norm(x, self.weight, eps)
            except Exception:
                return _ORIG_FNS["rms_norm_forward"](self, x, residual)

        RMSNorm.forward = _patched_rms
        patched.append("RMSNorm → fused_rms_norm / fused_add_rms_norm")
    except Exception as e:
        print(f"  ⚠ RMSNorm patch 失败: {e}")

    # ── SiluAndMul (零拷贝: 直接读拼接输入) ──
    try:
        from vllm.model_executor.layers.activation import SiluAndMul

        _ORIG_FNS["silu_forward"] = SiluAndMul.forward

        def _patched_silu(self, x: torch.Tensor):
            try:
                return fused_silu_mul_concat(x)
            except Exception:
                return _ORIG_FNS["silu_forward"](self, x)

        SiluAndMul.forward = _patched_silu
        patched.append("SiluAndMul → fused_silu_mul_concat (zero-copy)")
    except Exception as e:
        print(f"  ⚠ SiluAndMul patch 失败: {e}")

    if patched:
        print(f"✅ Triton kernel 已替换 vLLM 算子 ({len(patched)} 项):")
        for p in patched:
            print(f"   ✓ {p}")
    else:
        print("⚠ 未能替换任何 vLLM 算子（vLLM 版本可能不兼容）")

    return patched


def revert_triton_optimizations() -> list:
    """恢复 vLLM 原始算子实现."""
    global _ORIG_FNS
    reverted: list = []

    if "rms_norm_forward" in _ORIG_FNS:
        try:
            from vllm.model_executor.layers.layernorm import RMSNorm
            RMSNorm.forward = _ORIG_FNS.pop("rms_norm_forward")
            reverted.append("RMSNorm")
        except Exception:
            pass

    if "silu_forward" in _ORIG_FNS:
        try:
            from vllm.model_executor.layers.activation import SiluAndMul
            SiluAndMul.forward = _ORIG_FNS.pop("silu_forward")
            reverted.append("SiluAndMul")
        except Exception:
            pass

    if reverted:
        print(f"↩ 已恢复 vLLM 原始算子: {', '.join(reverted)}")
    return reverted


# ── 便捷包装 (可独立调用, 不依赖 monkey-patch) ──

def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    return fused_rms_norm(x, weight, eps)


def triton_silu_mul(gate: torch.Tensor, up: torch.Tensor):
    return fused_silu_mul(gate, up)


def triton_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
):
    return fused_add_rms_norm(x, residual, weight, eps)


def triton_rotary_emb(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return fused_rotary_emb(q, cos, sin)

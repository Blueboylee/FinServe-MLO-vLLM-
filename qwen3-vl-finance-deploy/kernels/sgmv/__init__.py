"""
FinServe 定制化 SGMV Kernel — 汇编级优化与算子融合

多 LoRA 服务 (Multi-LoRA Serving) 的核心计算算子.
SGMV = Segmented Gather Matrix-Vector multiply,
用于在同一 batch 内为不同请求动态索引不同 LoRA adapter 权重并执行矩阵乘.

模块结构:
  sgmv_shrink      — x @ A[adapter]  (hidden_dim → rank)
  sgmv_expand      — y @ B[adapter]  (rank → hidden_dim, scatter-add)
  sgmv_fused       — 融合 shrink+expand + LoRA+RMSNorm
  sgmv_integration — vLLM monkey-patch 集成
  sgmv_cuda_graph  — CUDA Graph 兼容 (torch.library CustomOp + static shape padding)

Quick Start:
    from sgmv_kernel import sgmv_shrink, sgmv_expand, fused_sgmv
    from sgmv_kernel import apply_sgmv_optimizations

    # 独立使用
    y = sgmv_shrink(x, w_a_stacked, token_adapter_ids, rank)
    sgmv_expand(y, w_b_stacked, token_adapter_ids, base_output)

    # 融合使用
    fused_sgmv(x, w_a, w_b, adapter_ids, base_output, scaling)

    # vLLM 集成
    apply_sgmv_optimizations(enable_fused=True, enable_tensor_core=True)

    # CUDA Graph 兼容
    from sgmv_kernel.sgmv_cuda_graph import apply_cuda_graph_sgmv
    apply_cuda_graph_sgmv(hidden_dim=4096, rank=64)
"""

from .sgmv_shrink import (
    sgmv_shrink,
    sgmv_shrink_segmented,
    baseline_sgmv_shrink,
)

from .sgmv_expand import (
    sgmv_expand,
    sgmv_expand_segmented,
    baseline_sgmv_expand,
)

from .sgmv_fused import (
    fused_sgmv,
    fused_lora_add_rmsnorm,
    baseline_fused_sgmv,
    baseline_lora_add_rmsnorm,
)

from .sgmv_integration import (
    apply_sgmv_optimizations,
    revert_sgmv_optimizations,
    get_patched_ops,
    get_config,
)

from .sgmv_cuda_graph import (
    apply_cuda_graph_sgmv,
    get_graph_runner,
    SGMVGraphRunner,
    get_padded_size,
    pad_to_bucket,
    SHAPE_BUCKETS,
)

__all__ = [
    "sgmv_shrink",
    "sgmv_shrink_segmented",
    "sgmv_expand",
    "sgmv_expand_segmented",
    "fused_sgmv",
    "fused_lora_add_rmsnorm",
    "apply_sgmv_optimizations",
    "revert_sgmv_optimizations",
    "get_patched_ops",
    "get_config",
    "baseline_sgmv_shrink",
    "baseline_sgmv_expand",
    "baseline_fused_sgmv",
    "baseline_lora_add_rmsnorm",
    "apply_cuda_graph_sgmv",
    "get_graph_runner",
    "SGMVGraphRunner",
    "get_padded_size",
    "pad_to_bucket",
    "SHAPE_BUCKETS",
]

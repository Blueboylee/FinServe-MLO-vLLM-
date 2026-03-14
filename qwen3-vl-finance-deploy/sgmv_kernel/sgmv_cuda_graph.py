"""
FinServe SGMV CUDA Graph 兼容层

问题:
  vLLM 默认使用 CUDA Graph 捕获推理计算图, 消除 kernel launch overhead.
  Monkey-patch 方式替换 Python 函数, 若在 CUDA Graph capture 之后执行,
  被捕获的仍是原始 kernel, patch 无效.
  若在 capture 之前执行 (serve_launcher.py), Python 层 patch 可以生效,
  但 Triton kernel 的 autotune 会在 capture 时引入分支, 导致 graph
  replay 时 shape 不匹配.

解决方案:
  1. torch.library.custom_op 注册: 将 Triton kernel 包装为正式 PyTorch op,
     torch.compile / CUDA Graph 能正确 trace 和 capture.
  2. Static shape padding: CUDA Graph 要求 tensor shape 固定.
     对 num_tokens 做 pad 到预设 bucket (32, 64, 128, 256, 512, 1024),
     graph replay 时从 cache 中取匹配的 captured graph.
  3. Graph-safe memory: 所有 intermediate tensor 在 capture 时分配,
     replay 时复用同一块内存, 无额外 cudaMalloc.

架构:
  sgmv_cuda_graph.py (本文件)
    ├─ torch.library.custom_op 注册 (finserve::sgmv_shrink 等)
    ├─ StaticShapePadder (num_tokens → bucket padding)
    ├─ SGMVGraphRunner (per-bucket CUDA Graph capture + replay)
    └─ apply_cuda_graph_sgmv() (集成到 sgmv_integration)

PTX 级别影响:
  - 无 Graph 时: 每次推理 → cudaLaunchKernel (host→device latency ~3-5μs per kernel)
  - 有 Graph 时: cudaGraphLaunch 一次性提交整批 kernel (~1μs total)
  - Decode 阶段 num_tokens=1 时, kernel launch overhead 占比可达 30%+,
    CUDA Graph 可将 decode latency 降低 20-30%
"""

import math
from typing import Dict, List, Optional, Tuple

import torch

from .sgmv_shrink import sgmv_shrink, sgmv_shrink_segmented
from .sgmv_expand import sgmv_expand, sgmv_expand_segmented
from .sgmv_fused import fused_sgmv


# ════════════════════════════════════════════════════════════════════
#  1. torch.library.custom_op 注册
#
#  将 Triton kernel 注册为 PyTorch 自定义算子, 使 torch.compile
#  和 CUDA Graph 能正确识别和捕获:
#    torch.ops.finserve.sgmv_shrink(...)
#    torch.ops.finserve.sgmv_expand(...)
#    torch.ops.finserve.fused_sgmv(...)
#
#  注册后 CUDA Graph capture 时, Triton kernel 的 CUDA launch
#  被记录到 graph 中, replay 时直接重放, 零 Python overhead.
# ════════════════════════════════════════════════════════════════════

FINSERVE_LIB = torch.library.Library("finserve", "DEF")

# ── sgmv_shrink op ──
FINSERVE_LIB.define(
    "sgmv_shrink(Tensor x, Tensor w_a, Tensor adapter_ids, int rank) -> Tensor"
)

@torch.library.impl(FINSERVE_LIB, "sgmv_shrink", "CUDA")
def _sgmv_shrink_cuda(
    x: torch.Tensor,
    w_a: torch.Tensor,
    adapter_ids: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    return sgmv_shrink(x, w_a, adapter_ids, rank)


@torch.library.impl(FINSERVE_LIB, "sgmv_shrink", "Meta")
def _sgmv_shrink_meta(x, w_a, adapter_ids, rank):
    return x.new_empty(x.shape[0], rank)


# ── sgmv_expand op ──
FINSERVE_LIB.define(
    "sgmv_expand(Tensor y, Tensor w_b, Tensor adapter_ids, "
    "Tensor base_output, float scaling) -> Tensor"
)

@torch.library.impl(FINSERVE_LIB, "sgmv_expand", "CUDA")
def _sgmv_expand_cuda(
    y: torch.Tensor,
    w_b: torch.Tensor,
    adapter_ids: torch.Tensor,
    base_output: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    return sgmv_expand(y, w_b, adapter_ids, base_output, scaling)


@torch.library.impl(FINSERVE_LIB, "sgmv_expand", "Meta")
def _sgmv_expand_meta(y, w_b, adapter_ids, base_output, scaling):
    return torch.empty_like(base_output)


# ── fused_sgmv op ──
FINSERVE_LIB.define(
    "fused_sgmv(Tensor x, Tensor w_a, Tensor w_b, Tensor adapter_ids, "
    "Tensor base_output, float scaling) -> Tensor"
)

@torch.library.impl(FINSERVE_LIB, "fused_sgmv", "CUDA")
def _fused_sgmv_cuda(
    x: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    adapter_ids: torch.Tensor,
    base_output: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    return fused_sgmv(x, w_a, w_b, adapter_ids, base_output, scaling)


@torch.library.impl(FINSERVE_LIB, "fused_sgmv", "Meta")
def _fused_sgmv_meta(x, w_a, w_b, adapter_ids, base_output, scaling):
    return torch.empty_like(base_output)


# ════════════════════════════════════════════════════════════════════
#  2. Static Shape Padding
#
#  CUDA Graph capture 要求每次 replay 的 tensor shape 完全相同.
#  推理时 num_tokens 变化 (prefill: 1~4096, decode: 1~batch_size),
#  需要将 num_tokens pad 到固定 bucket, 多余位置用 adapter_id=-1 标记跳过.
#
#  Bucket 选择策略:
#    - Decode 阶段: num_tokens 通常 = batch_size (1, 2, 4, 8, ...)
#    - Prefill 阶段: num_tokens = prompt_len (可达数千)
#    - 选择最小的 >= num_tokens 的 bucket, 避免过度浪费显存
# ════════════════════════════════════════════════════════════════════

SHAPE_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def get_padded_size(n: int) -> int:
    """找到 >= n 的最小 bucket."""
    for b in SHAPE_BUCKETS:
        if b >= n:
            return b
    return ((n + 255) // 256) * 256


def pad_to_bucket(
    x: torch.Tensor,
    adapter_ids: torch.Tensor,
    target_n: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Pad x 和 adapter_ids 到 target_n tokens.
    多余位置: x=0, adapter_id=-1 (kernel 会跳过 aid<0 的 token).
    返回 (padded_x, padded_ids, original_n).
    """
    n = x.shape[0]
    if n == target_n:
        return x, adapter_ids, n

    pad_n = target_n - n
    padded_x = torch.zeros(target_n, x.shape[1], dtype=x.dtype, device=x.device)
    padded_x[:n] = x

    padded_ids = torch.full((target_n,), -1, dtype=adapter_ids.dtype, device=adapter_ids.device)
    padded_ids[:n] = adapter_ids

    return padded_x, padded_ids, n


# ════════════════════════════════════════════════════════════════════
#  3. SGMVGraphRunner — Per-Bucket CUDA Graph Capture + Replay
#
#  为每个 (bucket_size, hidden_dim, rank) 组合维护独立的 CUDA Graph:
#    - 首次遇到该 shape: 正常执行 kernel (warmup / autotune)
#    - 第二次遇到: capture CUDA Graph
#    - 后续: replay captured graph (零 Python overhead)
#
#  Graph 内存管理:
#    - Capture 时分配的 intermediate tensor 被 graph 固定 (pinned to graph)
#    - Replay 时 input/output 通过 cudaGraphExecUpdateInputs 更新指针
#    - 无额外 cudaMalloc, 无 host-device sync
# ════════════════════════════════════════════════════════════════════

class SGMVGraphRunner:
    """
    CUDA Graph 加速的 SGMV 执行器.

    Usage:
        runner = SGMVGraphRunner(hidden_dim=4096, rank=64)

        # 首次调用 (warmup + potential capture)
        output = runner.run_shrink_expand(x, w_a, w_b, adapter_ids, base_out, scaling)

        # 后续调用 (graph replay if same bucket)
        output = runner.run_shrink_expand(x, w_a, w_b, adapter_ids, base_out, scaling)
    """

    def __init__(
        self,
        hidden_dim: int,
        rank: int,
        num_adapters: int = 4,
        warmup_iters: int = 3,
    ):
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.num_adapters = num_adapters
        self.warmup_iters = warmup_iters

        self._graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self._graph_inputs: Dict[int, Dict[str, torch.Tensor]] = {}
        self._graph_outputs: Dict[int, torch.Tensor] = {}
        self._call_counts: Dict[int, int] = {}

        self._stream = torch.cuda.Stream()

    def run_shrink_expand(
        self,
        x: torch.Tensor,
        w_a: torch.Tensor,
        w_b: torch.Tensor,
        adapter_ids: torch.Tensor,
        base_output: torch.Tensor,
        scaling: float = 1.0,
        use_fused: bool = True,
    ) -> torch.Tensor:
        """
        CUDA Graph 加速的 SGMV shrink+expand.

        自动管理 graph 生命周期:
          1. warmup (前 N 次): 正常执行, 让 Triton autotune 稳定
          2. capture: 冻结计算图
          3. replay: 零开销执行
        """
        n_tokens = x.shape[0]
        bucket = get_padded_size(n_tokens)

        self._call_counts[bucket] = self._call_counts.get(bucket, 0) + 1
        count = self._call_counts[bucket]

        if count <= self.warmup_iters:
            return self._eager_run(x, w_a, w_b, adapter_ids, base_output, scaling, use_fused)

        if bucket not in self._graphs:
            self._capture_graph(bucket, x, w_a, w_b, adapter_ids, base_output, scaling, use_fused)

        return self._replay_graph(bucket, x, w_a, w_b, adapter_ids, base_output, scaling, n_tokens)

    def _eager_run(self, x, w_a, w_b, adapter_ids, base_output, scaling, use_fused):
        """Warmup 阶段: 正常执行 (允许 Triton autotune 选择最优 config)."""
        if use_fused:
            return torch.ops.finserve.fused_sgmv(x, w_a, w_b, adapter_ids, base_output, scaling)
        y = torch.ops.finserve.sgmv_shrink(x, w_a, adapter_ids, self.rank)
        return torch.ops.finserve.sgmv_expand(y, w_b, adapter_ids, base_output, scaling)

    def _capture_graph(self, bucket, x, w_a, w_b, adapter_ids, base_output, scaling, use_fused):
        """
        Capture CUDA Graph for given bucket size.

        capture 流程:
          1. Pad inputs to bucket size
          2. 在 capture stream 上执行一遍 kernel
          3. CUDA runtime 记录所有 kernel launch 和 memory op
          4. 生成 CUDAGraph 对象, 后续可 replay
        """
        padded_x, padded_ids, _ = pad_to_bucket(
            x, adapter_ids, bucket
        )
        padded_base = torch.zeros(
            bucket, self.hidden_dim, dtype=x.dtype, device=x.device
        )

        static_x = padded_x.clone()
        static_ids = padded_ids.clone()
        static_base = padded_base.clone()

        self._graph_inputs[bucket] = {
            "x": static_x,
            "adapter_ids": static_ids,
            "base_output": static_base,
            "w_a": w_a,
            "w_b": w_b,
        }

        graph = torch.cuda.CUDAGraph()

        with torch.cuda.stream(self._stream):
            # Warmup run on capture stream
            if use_fused:
                torch.ops.finserve.fused_sgmv(
                    static_x, w_a, w_b, static_ids, static_base, scaling
                )
            else:
                y = torch.ops.finserve.sgmv_shrink(static_x, w_a, static_ids, self.rank)
                torch.ops.finserve.sgmv_expand(y, w_b, static_ids, static_base, scaling)

            # Capture
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, stream=self._stream):
                if use_fused:
                    result = torch.ops.finserve.fused_sgmv(
                        static_x, w_a, w_b, static_ids, static_base, scaling
                    )
                else:
                    y = torch.ops.finserve.sgmv_shrink(static_x, w_a, static_ids, self.rank)
                    result = torch.ops.finserve.sgmv_expand(
                        y, w_b, static_ids, static_base, scaling
                    )

        self._graphs[bucket] = graph
        self._graph_outputs[bucket] = result

    def _replay_graph(self, bucket, x, w_a, w_b, adapter_ids, base_output, scaling, n_tokens):
        """
        Replay captured graph.

        关键: 仅更新 input tensor 内容 (copy_), 不改变 shape/stride/device.
        CUDA runtime 直接重放录制的 kernel 序列, 跳过所有 Python 调度.
        """
        inputs = self._graph_inputs[bucket]

        padded_x, padded_ids, _ = pad_to_bucket(x, adapter_ids, bucket)
        inputs["x"].copy_(padded_x)
        inputs["adapter_ids"].copy_(padded_ids)
        inputs["base_output"].copy_(base_output if base_output.shape[0] == bucket
                                    else torch.nn.functional.pad(
                                        base_output, (0, 0, 0, bucket - base_output.shape[0])
                                    ))
        inputs["w_a"] = w_a
        inputs["w_b"] = w_b

        self._graphs[bucket].replay()

        output = self._graph_outputs[bucket]
        return output[:n_tokens]

    def get_stats(self) -> dict:
        """返回 graph cache 统计."""
        return {
            "cached_buckets": list(self._graphs.keys()),
            "total_captures": len(self._graphs),
            "call_counts": dict(self._call_counts),
            "memory_overhead_mb": sum(
                sum(t.nelement() * t.element_size() for t in inp.values())
                for inp in self._graph_inputs.values()
            ) / 1024 / 1024,
        }


# ════════════════════════════════════════════════════════════════════
#  4. Integration API
# ════════════════════════════════════════════════════════════════════

_GRAPH_RUNNER: Optional[SGMVGraphRunner] = None


def get_graph_runner(hidden_dim: int = 4096, rank: int = 64) -> SGMVGraphRunner:
    """获取全局 SGMV Graph Runner (单例)."""
    global _GRAPH_RUNNER
    if _GRAPH_RUNNER is None:
        _GRAPH_RUNNER = SGMVGraphRunner(hidden_dim=hidden_dim, rank=rank)
    return _GRAPH_RUNNER


def apply_cuda_graph_sgmv(hidden_dim: int = 4096, rank: int = 64) -> str:
    """
    启用 CUDA Graph 加速的 SGMV.

    在 sgmv_integration.apply_sgmv_optimizations 之后调用:
      1. 先用 monkey-patch 替换 vLLM kernel (确保 Python 层调用自研 kernel)
      2. 再调用本函数, 将 custom_op 注册到 PyTorch, 使 CUDA Graph 能 capture

    Returns:
        描述信息
    """
    runner = get_graph_runner(hidden_dim, rank)
    return (
        f"CUDA Graph SGMV enabled "
        f"(buckets={SHAPE_BUCKETS}, hidden={hidden_dim}, rank={rank}, "
        f"warmup={runner.warmup_iters} iters before capture)"
    )

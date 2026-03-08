#!/usr/bin/env python
"""
SGMV vs 朴素 LoRA 压测对比

直接调用 sgmv_lora_forward（SGMV）与 sgmv_lora_forward_naive（按 adapter 循环），
在同一负载下测延迟与吞吐，对比「使用 SGMV 前 / 使用 SGMV 后」。

不依赖 vLLM，纯 LoRA 矩阵乘层面的 benchmark。
依赖: torch, triton（需 GPU）
"""

import argparse
import statistics
import time
from typing import List

import torch

from sgmv_lora_triton import sgmv_lora_forward, sgmv_lora_forward_naive


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    idx = int(round((len(vs) - 1) * q))
    return vs[idx]


def run_benchmark(
    x: torch.Tensor,
    adapter_ids: torch.Tensor,
    lora_B: torch.Tensor,
    lora_A: torch.Tensor,
    iterations: int,
    warmup: int,
    use_sgmv: bool,
) -> List[float]:
    """跑 iterations 次，返回每次耗时（秒）列表。"""
    fn = sgmv_lora_forward if use_sgmv else sgmv_lora_forward_naive
    for _ in range(warmup):
        fn(x, adapter_ids, lora_B, lora_A)
    if x.is_cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn(x, adapter_ids, lora_B, lora_A)
        if x.is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def main():
    p = argparse.ArgumentParser(description="SGMV vs 朴素 LoRA 压测对比")
    p.add_argument("--batch", type=int, default=256, help="batch size")
    p.add_argument("--in-dim", type=int, default=4096, help="hidden dim (in)")
    p.add_argument("--r", type=int, default=64, help="LoRA rank")
    p.add_argument("--out-dim", type=int, default=4096, help="hidden dim (out)")
    p.add_argument("--num-adapters", type=int, default=3, help="adapter 数量")
    p.add_argument("--iterations", type=int, default=500, help="每侧跑次数")
    p.add_argument("--warmup", type=int, default=20, help="预热次数")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("警告: 无 GPU，将用 CPU 跑（Triton 可能不可用或很慢）")

    torch.manual_seed(42)
    x = torch.randn(args.batch, args.in_dim, device=device, dtype=torch.float16)
    adapter_ids = torch.randint(0, args.num_adapters, (args.batch,), device=device)
    lora_B = torch.randn(args.num_adapters, args.in_dim, args.r, device=device, dtype=torch.float16)
    lora_A = torch.randn(args.num_adapters, args.r, args.out_dim, device=device, dtype=torch.float16)

    out_naive = sgmv_lora_forward_naive(x, adapter_ids, lora_B, lora_A)
    out_sgmv = sgmv_lora_forward(x, adapter_ids, lora_B, lora_A)
    diff = (out_naive.float() - out_sgmv.float()).abs().max().item()
    if diff > 0.5:
        print("错误: SGMV 与 naive 结果差异过大，请检查实现")
        return
    print("正确性校验通过 (max diff=%.4f)" % diff)

    print("\n压测配置: batch=%d, in=%d, r=%d, out=%d, adapters=%d, iterations=%d" % (
        args.batch, args.in_dim, args.r, args.out_dim, args.num_adapters, args.iterations))

    print("\n[1] 朴素 LoRA（按 adapter 循环）...")
    times_naive = run_benchmark(x, adapter_ids, lora_B, lora_A, args.iterations, args.warmup, use_sgmv=False)
    print("[2] SGMV（单 kernel 分段 LoRA）...")
    times_sgmv = run_benchmark(x, adapter_ids, lora_B, lora_A, args.iterations, args.warmup, use_sgmv=True)

    def stats(times: List[float], name: str):
        n = len(times)
        total = sum(times)
        avg_ms = statistics.fmean(times) * 1000
        p95_ms = percentile(times, 0.95) * 1000
        p99_ms = percentile(times, 0.99) * 1000
        throughput = n / total if total > 0 else 0
        return {"avg_ms": avg_ms, "p95_ms": p95_ms, "p99_ms": p99_ms, "throughput": throughput, "total_ms": total * 1000}

    s_naive = stats(times_naive, "naive")
    s_sgmv = stats(times_sgmv, "SGMV")

    print("\n" + "=" * 60)
    print("  【SGMV 压测对比】使用前 vs 使用后")
    print("=" * 60)
    print(f"{'指标':<22} {'使用前(朴素)':>14} {'使用后(SGMV)':>14} {'变化':>12}")
    print("-" * 64)
    for key, label in [
        ("avg_ms", "Latency Avg (ms)"),
        ("p95_ms", "Latency P95 (ms)"),
        ("p99_ms", "Latency P99 (ms)"),
        ("throughput", "Throughput (req/s)"),
    ]:
        nv, sv = s_naive[key], s_sgmv[key]
        chg = ((sv - nv) / nv * 100) if nv else 0
        if key == "throughput":
            print(f"{label:<22} {nv:>14.2f} {sv:>14.2f} {chg:>+11.1f}%")
        else:
            print(f"{label:<22} {nv:>14.2f} {sv:>14.2f} {chg:>+11.1f}%")
    print("=" * 60)
    print("说明: Latency 负变化 = SGMV 更快；Throughput 正变化 = SGMV 吞吐更高")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FP8 KV Cache 性能与精度评测

评测内容:
  1. 量化/反量化 kernel 吞吐 (GB/s)
  2. Fused dequant + attention score 吞吐
  3. 精度: FP8 vs FP16 cosine similarity
  4. 端到端: FP8 vs FP16 KV Cache 下的推理延迟与最大 batch size

配置参考 (Qwen3-VL-8B):
  num_layers=32, num_kv_heads=4, head_dim=128, block_size=16
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F


def bench_quantize_kernel(num_tokens, num_heads, head_dim, num_iters=100, warmup=10):
    """测试 FP16 → FP8 量化 kernel 吞吐."""
    from kv_cache_fp8.fp8_kernels import quantize_fp16_to_fp8

    x = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda")

    for _ in range(warmup):
        quantize_fp16_to_fp8(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        quantize_fp16_to_fp8(x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters

    bytes_read = x.nelement() * x.element_size()
    bytes_written = x.nelement() * 1 + num_tokens * num_heads * 4  # fp8 + scale
    bandwidth = (bytes_read + bytes_written) / elapsed / 1e9

    return {
        "op": "quantize_fp16_to_fp8",
        "shape": f"[{num_tokens}, {num_heads}, {head_dim}]",
        "latency_us": elapsed * 1e6,
        "bandwidth_gb_s": bandwidth,
        "throughput_mtokens_s": num_tokens / elapsed / 1e6,
    }


def bench_dequantize_kernel(num_tokens, num_heads, head_dim, num_iters=100, warmup=10):
    """测试 FP8 → FP16 反量化 kernel 吞吐."""
    from kv_cache_fp8.fp8_kernels import quantize_fp16_to_fp8, dequantize_fp8_to_fp16

    x = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda")
    x_fp8, scale = quantize_fp16_to_fp8(x)

    for _ in range(warmup):
        dequantize_fp8_to_fp16(x_fp8, scale)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        dequantize_fp8_to_fp16(x_fp8, scale)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters

    bytes_read = x_fp8.nelement() * 1 + scale.nelement() * 4
    bytes_written = num_tokens * num_heads * head_dim * 2
    bandwidth = (bytes_read + bytes_written) / elapsed / 1e9

    return {
        "op": "dequantize_fp8_to_fp16",
        "shape": f"[{num_tokens}, {num_heads}, {head_dim}]",
        "latency_us": elapsed * 1e6,
        "bandwidth_gb_s": bandwidth,
    }


def bench_fused_dequant_attention(num_q, seq_len, num_heads, head_dim, num_iters=50, warmup=5):
    """测试融合反量化 + attention score 吞吐."""
    from kv_cache_fp8.fp8_kernels import quantize_fp16_to_fp8, fused_dequant_attention_score

    q = torch.randn(num_q, num_heads, head_dim, dtype=torch.float16, device="cuda")
    k = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float16, device="cuda")
    k_fp8, k_scale = quantize_fp16_to_fp8(k)

    for _ in range(warmup):
        fused_dequant_attention_score(q, k_fp8, k_scale, seq_len)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        fused_dequant_attention_score(q, k_fp8, k_scale, seq_len)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters

    # Baseline: FP16 attention score
    for _ in range(warmup):
        _ = torch.einsum("qhd,khd->qhk", q.float(), k.float()) / (head_dim ** 0.5)
    torch.cuda.synchronize()

    start2 = time.perf_counter()
    for _ in range(num_iters):
        _ = torch.einsum("qhd,khd->qhk", q.float(), k.float()) / (head_dim ** 0.5)
    torch.cuda.synchronize()
    baseline_elapsed = (time.perf_counter() - start2) / num_iters

    return {
        "op": "fused_dequant_attention_score",
        "shape": f"q=[{num_q},{num_heads},{head_dim}] k=[{seq_len},{num_heads},{head_dim}]",
        "fused_latency_us": elapsed * 1e6,
        "baseline_latency_us": baseline_elapsed * 1e6,
        "speedup": baseline_elapsed / elapsed if elapsed > 0 else 0,
    }


def bench_precision(num_tokens, num_heads, head_dim):
    """FP8 量化精度分析."""
    from kv_cache_fp8.fp8_kernels import quantize_fp16_to_fp8, dequantize_fp8_to_fp16

    x = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda")
    x_fp8, scale = quantize_fp16_to_fp8(x)
    x_recovered = dequantize_fp8_to_fp16(x_fp8, scale)

    x_f = x.float().view(-1)
    r_f = x_recovered.float().view(-1)

    cos_sim = F.cosine_similarity(x_f.unsqueeze(0), r_f.unsqueeze(0)).item()
    mse = F.mse_loss(r_f, x_f).item()
    max_err = (x_f - r_f).abs().max().item()
    rel_err = ((x_f - r_f).abs() / (x_f.abs() + 1e-8)).mean().item()

    return {
        "shape": f"[{num_tokens}, {num_heads}, {head_dim}]",
        "cosine_similarity": cos_sim,
        "mse": mse,
        "max_absolute_error": max_err,
        "mean_relative_error": rel_err,
    }


def bench_memory_savings(num_layers, num_kv_heads, head_dim, block_size, num_blocks):
    """显存节省分析."""
    from kv_cache_fp8.fp8_kv_cache import FP8KVCacheManager

    manager = FP8KVCacheManager(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_blocks=num_blocks,
        device=torch.device("cuda"),
    )

    mem = manager.memory_usage()
    total_tokens = num_blocks * block_size

    return {
        "config": {
            "num_layers": num_layers, "num_kv_heads": num_kv_heads,
            "head_dim": head_dim, "block_size": block_size, "num_blocks": num_blocks,
        },
        "max_tokens": total_tokens,
        "fp16_kv_cache_mb": mem["fp16_equivalent_bytes"] / 1024**2,
        "fp8_kv_cache_mb": mem["total_fp8_bytes"] / 1024**2,
        "memory_saved_mb": mem["memory_saved_mb"],
        "compression_ratio": mem["compression_ratio"],
        "scale_overhead_ratio": mem["scale_bytes"] / max(mem["total_fp8_bytes"], 1),
    }


def main():
    parser = argparse.ArgumentParser(description="FP8 KV Cache Benchmark")
    parser.add_argument("--mode", choices=["kernel", "precision", "memory", "all"], default="all")
    parser.add_argument("--output", type=str, default="bench_results/fp8_kv_cache.json")
    args = parser.parse_args()

    results = {}

    if args.mode in ("kernel", "all"):
        print("=" * 60)
        print("Kernel Throughput Benchmark")
        print("=" * 60)

        configs = [
            (1, 4, 128),     # decode: single token
            (16, 4, 128),    # decode: batch=16
            (64, 4, 128),    # decode: batch=64
            (256, 4, 128),   # prefill: 256 tokens
            (1024, 4, 128),  # prefill: 1024 tokens
            (4096, 4, 128),  # prefill: full context
        ]

        quant_results = []
        dequant_results = []
        for T, H, D in configs:
            r = bench_quantize_kernel(T, H, D)
            print(f"  Quant   {r['shape']}: {r['latency_us']:.1f}us, {r['bandwidth_gb_s']:.1f} GB/s")
            quant_results.append(r)

            r = bench_dequantize_kernel(T, H, D)
            print(f"  Dequant {r['shape']}: {r['latency_us']:.1f}us, {r['bandwidth_gb_s']:.1f} GB/s")
            dequant_results.append(r)

        results["quantize"] = quant_results
        results["dequantize"] = dequant_results

        print("\nFused Dequant + Attention Score:")
        attn_results = []
        for num_q, seq_len in [(1, 512), (1, 2048), (16, 512), (16, 2048)]:
            r = bench_fused_dequant_attention(num_q, seq_len, 4, 128)
            print(f"  q={num_q} kv={seq_len}: fused={r['fused_latency_us']:.1f}us "
                  f"baseline={r['baseline_latency_us']:.1f}us "
                  f"speedup={r['speedup']:.2f}x")
            attn_results.append(r)
        results["fused_attention"] = attn_results

    if args.mode in ("precision", "all"):
        print("\n" + "=" * 60)
        print("Precision Analysis")
        print("=" * 60)

        prec_results = []
        for T in [64, 256, 1024, 4096]:
            r = bench_precision(T, 4, 128)
            print(f"  {r['shape']}: cos_sim={r['cosine_similarity']:.6f}, "
                  f"mse={r['mse']:.2e}, max_err={r['max_absolute_error']:.4f}")
            prec_results.append(r)
        results["precision"] = prec_results

    if args.mode in ("memory", "all"):
        print("\n" + "=" * 60)
        print("Memory Savings Analysis (Qwen3-VL-8B)")
        print("=" * 60)

        mem_results = []
        for num_blocks in [256, 512, 1024, 2048]:
            r = bench_memory_savings(32, 4, 128, 16, num_blocks)
            print(f"  {num_blocks} blocks ({r['max_tokens']} tokens): "
                  f"FP16={r['fp16_kv_cache_mb']:.1f}MB → FP8={r['fp8_kv_cache_mb']:.1f}MB "
                  f"(saved {r['memory_saved_mb']:.1f}MB, {r['compression_ratio']:.2f}x)")
            mem_results.append(r)
        results["memory"] = mem_results

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

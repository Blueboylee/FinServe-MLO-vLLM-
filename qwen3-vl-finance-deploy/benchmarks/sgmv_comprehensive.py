#!/usr/bin/env python3
"""
SGMV 多层级优化性能对比测试

覆盖 sgmv_integration.apply_sgmv_optimizations 的全部优化层级组合:
  L0 — SGMV token-parallel (sgmv_shrink/expand 替换 bgmv)
  L1 — Tensor Core segmented variants (tl.dot → HMMA)
  L2 — Fused SGMV (shrink+expand 融合, 中间值驻留寄存器)
  L3 — Fused LoRA+RMSNorm (LoRA delta + residual + RMSNorm 单 pass)

每种配置独立创建 LLM 实例, 确保 patch 状态隔离.
输出对比报告: 延迟、吞吐、加速比.
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vllm import LLM, SamplingParams
from sgmv_kernel.sgmv_integration import (
    apply_sgmv_optimizations,
    revert_sgmv_optimizations,
    get_config,
)


MODEL_DIR = Path("./models")
BASE_MODEL = str(MODEL_DIR / "Qwen3-VL-8B-Instruct-AWQ-4bit")
LORA_A = str(MODEL_DIR / "Qwen3-VL-Finance-expert-a")
LORA_B = str(MODEL_DIR / "Qwen3-VL-Finance-expert-b")

TEST_PROMPTS = [
    {"text": "请分析2024年A股市场的整体走势", "lora": LORA_A, "name": "Expert-A"},
    {"text": "请预测2024年科技股的投资机会", "lora": LORA_B, "name": "Expert-B"},
    {"text": "请分析2024年A股市场的整体走势", "lora": LORA_A, "name": "Expert-A-repeat"},
    {"text": "请预测2024年科技股的投资机会", "lora": LORA_B, "name": "Expert-B-repeat"},
]


OPTIMIZATION_CONFIGS = [
    {
        "name": "Baseline (vLLM native LoRA)",
        "desc": "No patches — original vLLM BGMV kernels",
        "args": None,
    },
    {
        "name": "L0: SGMV token-parallel",
        "desc": "bgmv_shrink/expand → sgmv_shrink/expand (auto-tuned Triton)",
        "args": {"enable_fused": False, "enable_tensor_core": False, "enable_fuse_lora_rmsnorm": False},
    },
    {
        "name": "L1: SGMV + Tensor Core",
        "desc": "sgmv → segmented variants (tl.dot → HMMA Tensor Core)",
        "args": {"enable_fused": False, "enable_tensor_core": True, "enable_fuse_lora_rmsnorm": False},
    },
    {
        "name": "L2: Fused SGMV",
        "desc": "shrink+expand fused, intermediate stays in register file",
        "args": {"enable_fused": True, "enable_tensor_core": False, "enable_fuse_lora_rmsnorm": False},
    },
    {
        "name": "L2+L1: Fused SGMV + Tensor Core",
        "desc": "Fused SGMV + Tensor Core segmented paths",
        "args": {"enable_fused": True, "enable_tensor_core": True, "enable_fuse_lora_rmsnorm": False},
    },
    {
        "name": "L3: Fused LoRA+RMSNorm",
        "desc": "LoRA delta + residual + RMSNorm → single kernel, ~50% BW saved",
        "args": {"enable_fused": False, "enable_tensor_core": False, "enable_fuse_lora_rmsnorm": True},
    },
    {
        "name": "Full: L0+L1+L2+L3",
        "desc": "All optimizations combined",
        "args": {"enable_fused": True, "enable_tensor_core": True, "enable_fuse_lora_rmsnorm": True},
    },
]


def create_llm() -> LLM:
    """创建 vLLM 实例 (无优化 patch, patch 在外部管理)."""
    return LLM(
        model=BASE_MODEL,
        quantization="compressed-tensors",
        enable_lora=True,
        max_loras=2,
        max_lora_rank=64,
        max_cpu_loras=2,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=512,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 4, "video": 1},
    )


def warmup(llm: LLM, num_runs: int = 2):
    sampling_params = SamplingParams(temperature=0.7, max_tokens=64, top_p=0.9)
    for i in range(num_runs):
        t0 = time.perf_counter()
        llm.generate([TEST_PROMPTS[0]["text"]], sampling_params,
                     lora_path=TEST_PROMPTS[0]["lora"])
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  warmup {i+1}/{num_runs}: {elapsed:.1f}ms")


def benchmark(llm: LLM, num_runs: int = 10) -> dict:
    """返回 {latencies_ms, avg_ms, p50_ms, p99_ms, min_ms, max_ms, std_ms}."""
    sampling_params = SamplingParams(temperature=0.7, max_tokens=128, top_p=0.9)
    latencies = []

    for i in range(num_runs):
        pd = TEST_PROMPTS[i % len(TEST_PROMPTS)]
        t0 = time.perf_counter()
        outputs = llm.generate([pd["text"]], sampling_params, lora_path=pd["lora"])
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)
        tokens = len(outputs[0].outputs[0].token_ids)
        print(f"  run {i+1}/{num_runs} ({pd['name']}): {elapsed:.1f}ms, {tokens} tokens")

    latencies_sorted = sorted(latencies)
    n = len(latencies)
    avg = sum(latencies) / n
    std = (sum((x - avg) ** 2 for x in latencies) / n) ** 0.5

    return {
        "latencies_ms": latencies,
        "avg_ms": avg,
        "p50_ms": latencies_sorted[n // 2],
        "p99_ms": latencies_sorted[min(n - 1, int(n * 0.99))],
        "min_ms": latencies_sorted[0],
        "max_ms": latencies_sorted[-1],
        "std_ms": std,
    }


def run_single_config(config: dict, num_runs: int, warmup_runs: int) -> dict:
    """运行单个优化配置的完整测试."""
    name = config["name"]
    print(f"\n{'='*70}")
    print(f"CONFIG: {name}")
    print(f"  {config['desc']}")
    print(f"{'='*70}")

    try:
        revert_sgmv_optimizations()

        if config["args"] is not None:
            print(f"  Applying: {config['args']}")
            patched = apply_sgmv_optimizations(**config["args"])
            print(f"  Patched: {patched}")
        else:
            print("  No patches (baseline)")

        llm = create_llm()
        warmup(llm, warmup_runs)
        stats = benchmark(llm, num_runs)

        result = {
            "name": name,
            "desc": config["desc"],
            "config_args": config["args"],
            "status": "success",
            **stats,
        }

        print(f"\n  Avg: {stats['avg_ms']:.1f}ms  P50: {stats['p50_ms']:.1f}ms  "
              f"P99: {stats['p99_ms']:.1f}ms  Std: {stats['std_ms']:.1f}ms")
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"name": name, "status": "failed", "error": str(e)}


def print_comparison(results: list):
    """打印对比表格."""
    ok = [r for r in results if r["status"] == "success"]
    if not ok:
        print("\nAll configs failed.")
        return

    baseline = next((r for r in ok if "Baseline" in r["name"]), ok[0])
    bla = baseline["avg_ms"]

    print(f"\n{'='*100}")
    print("Performance Comparison")
    print(f"{'='*100}")
    header = f"{'Config':<40} {'Avg(ms)':>9} {'P50':>9} {'P99':>9} {'Std':>8} {'Speedup':>9}"
    print(header)
    print("-" * 100)

    for r in sorted(ok, key=lambda x: x["avg_ms"]):
        sp = bla / r["avg_ms"]
        pct = (bla - r["avg_ms"]) / bla * 100
        sp_str = f"{sp:.3f}x" if r["name"] != baseline["name"] else "baseline"
        print(f"{r['name']:<40} {r['avg_ms']:>9.1f} {r['p50_ms']:>9.1f} "
              f"{r['p99_ms']:>9.1f} {r['std_ms']:>8.1f} {sp_str:>9}"
              + (f" ({pct:+.1f}%)" if r["name"] != baseline["name"] else ""))

    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser(description="SGMV Multi-Level Optimization Benchmark")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                        help="quick: baseline + L0 + Full | full: all 7 configs")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="bench_results")
    args = parser.parse_args()

    if args.mode == "quick":
        configs = [OPTIMIZATION_CONFIGS[i] for i in [0, 1, 3, 6]]
    else:
        configs = OPTIMIZATION_CONFIGS

    print(f"SGMV Comprehensive Benchmark — {args.mode} mode, {len(configs)} configs, "
          f"{args.num_runs} runs each\n")

    results = []
    t_start = time.time()

    for i, cfg in enumerate(configs, 1):
        print(f"\n{'#'*70}")
        print(f"# Test {i}/{len(configs)}")
        print(f"{'#'*70}")
        result = run_single_config(cfg, args.num_runs, args.warmup_runs)
        results.append(result)

        os.makedirs(args.output_dir, exist_ok=True)
        with open(f"{args.output_dir}/sgmv_comprehensive_partial.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    total_min = (time.time() - t_start) / 60
    print_comparison(results)

    output_file = f"{args.output_dir}/sgmv_comprehensive_{int(time.time())}.json"
    with open(output_file, "w") as f:
        json.dump({
            "summary": {"total_tests": len(results), "total_minutes": total_min,
                        "runs_per_test": args.num_runs},
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")
    print(f"Total time: {total_min:.1f} min")


if __name__ == "__main__":
    main()

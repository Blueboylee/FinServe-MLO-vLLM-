#!/usr/bin/env python3
"""
全面的 SGMV 优化性能测试脚本
覆盖多种优化场景，用于面试展示

优化场景:
1. Baseline - 原生 vLLM LoRA
2. SGMV Shrink - 仅替换 shrink kernel
3. SGMV Expand - 仅替换 expand kernel  
4. SGMV Full - shrink + expand
5. Fused SGMV - shrink+expand 融合
6. Fused LoRA+RMSNorm - 算子融合
7. All Optimizations - 全部优化组合
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from vllm import LLM, SamplingParams
from sgmv_kernel.sgmv_integration import apply_sgmv_optimizations


# 模型配置
BASE_MODEL = "/root/下载/FinServe-MLO-vLLM-/qwen3-vl-finance-deploy/models/Qwen3-VL-8B-Instruct-AWQ-4bit"
LORA_A = "/root/下载/FinServe-MLO-vLLM-/qwen3-vl-finance-deploy/loras/Expert-A"
LORA_B = "/root/下载/FinServe-MLO-vLLM-/qwen3-vl-finance-deploy/loras/Expert-B"

# 测试提示
TEST_PROMPTS = [
    {
        "text": "请分析2024年A股市场的整体走势",
        "lora": LORA_A,
        "name": "Expert-A"
    },
    {
        "text": "请预测2024年科技股的投资机会",
        "lora": LORA_B,
        "name": "Expert-B"
    },
    {
        "text": "请分析2024年A股市场的整体走势",
        "lora": LORA_A,
        "name": "Expert-A-repeat"
    },
    {
        "text": "请预测2024年科技股的投资机会",
        "lora": LORA_B,
        "name": "Expert-B-repeat"
    },
]


def create_llm(config: dict) -> LLM:
    """创建 vLLM 实例，根据配置应用不同的优化"""
    
    # 基础配置
    llm = LLM(
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
    
    # 应用优化
    if config.get("sgmv_shrink", False):
        print("\n✅ 应用 SGMV Shrink Kernel (Token/Segment 并行)")
    
    if config.get("sgmv_expand", False):
        print("\n✅ 应用 SGMV Expand Kernel (Token/Segment 并行 + Tensor Core)")
    
    if config.get("fused_sgmv", False):
        print("\n✅ 应用 Fused SGMV (shrink+expand 融合)")
    
    if config.get("fused_lora_rmsnorm", False):
        print("\n✅ 应用 Fused LoRA+RMSNorm (base+delta+residual+norm 三路融合)")
    
    if config.get("tensor_core", False):
        print("\n✅ 应用 Tensor Core 优化 (tl.dot)")
    
    if config.get("segment_parallel", False):
        print("\n✅ 应用 Segment Parallel 策略")
    
    if config.get("token_parallel", False):
        print("\n✅ 应用 Token Parallel 策略")
    
    if config.get("all_optimizations", False):
        print("\n✅ 应用全部优化组合")
        patched = apply_sgmv_optimizations(
            enable_fused=True,
            enable_tensor_core=True,
            enable_fuse_lora_rmsnorm=True
        )
        print(f"   已 patch 的算子: {patched}")
    
    if config.get("sgmv_full", False):
        print("\n✅ 应用 SGMV Full (shrink + expand)")
        patched = apply_sgmv_optimizations(
            enable_fused=False,
            enable_tensor_core=True,
            enable_fuse_lora_rmsnorm=False
        )
        print(f"   已 patch 的算子: {patched}")
    
    if config.get("sgmv_fused_only", False):
        print("\n✅ 应用 SGMV Fused Only")
        patched = apply_sgmv_optimizations(
            enable_fused=True,
            enable_tensor_core=False,
            enable_fuse_lora_rmsnorm=False
        )
        print(f"   已 patch 的算子: {patched}")
    
    if config.get("sgmv_baseline", False):
        print("\n✅ 应用 SGMV Baseline (无融合)")
        patched = apply_sgmv_optimizations(
            enable_fused=False,
            enable_tensor_core=False,
            enable_fuse_lora_rmsnorm=False
        )
        print(f"   已 patch 的算子: {patched}")
    
    return llm


def warmup(llm: LLM, num_runs: int = 2):
    """预热"""
    print(f"\n{'='*60}")
    print(f"Warmup: {num_runs} 次预热运行")
    print(f"{'='*60}")
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=128,
        top_p=0.9,
    )
    
    for i in range(num_runs):
        start = time.time()
        outputs = llm.generate(
            [TEST_PROMPTS[0]["text"]],
            sampling_params,
            lora_path=TEST_PROMPTS[0]["lora"]
        )
        elapsed = (time.time() - start) * 1000
        print(f"  Warmup {i+1}/{num_runs}: {elapsed:.2f}ms")


def benchmark(llm: LLM, num_runs: int = 10) -> list:
    """性能测试"""
    latencies = []
    outputs_text = []
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=128,
        top_p=0.9,
    )
    
    print(f"\n{'='*60}")
    print(f"Performance Benchmark: {num_runs} 次测试运行")
    print(f"{'='*60}")
    
    for i in range(num_runs):
        prompt_idx = i % len(TEST_PROMPTS)
        prompt_data = TEST_PROMPTS[prompt_idx]
        
        start = time.time()
        outputs = llm.generate(
            [prompt_data["text"]],
            sampling_params,
            lora_path=prompt_data["lora"]
        )
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)
        
        output_text = outputs[0].outputs[0].text.strip()
        outputs_text.append(output_text)
        
        print(f"  Run {i+1}/{num_runs} ({prompt_data['name']}): {elapsed:.2f}ms")
    
    return latencies, outputs_text


def run_test(config: dict, name: str, num_runs: int = 10, warmup_runs: int = 2) -> dict:
    """运行单个测试配置"""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    print(f"配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    try:
        llm = create_llm(config)
        warmup(llm, warmup_runs)
        latencies, outputs = benchmark(llm, num_runs)
        
        result = {
            "name": name,
            "config": config,
            "latencies_ms": latencies,
            "avg_ms": sum(latencies) / len(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "std_ms": (sum((x - sum(latencies)/len(latencies))**2 for x in latencies) / len(latencies))**0.5,
            "outputs": outputs[0],  # 只保存第一个输出作为示例
            "num_runs": num_runs,
            "warmup_runs": warmup_runs,
            "status": "success"
        }
        
        print(f"\n📊 统计结果:")
        print(f"  Avg: {result['avg_ms']:.2f}ms")
        print(f"  Min: {result['min_ms']:.2f}ms")
        print(f"  Max: {result['max_ms']:.2f}ms")
        print(f"  Std: {result['std_ms']:.2f}ms")
        
        return result
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            "name": name,
            "config": config,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def get_all_test_configs():
    """获取所有测试配置"""
    configs = [
        # 1. Baseline - 原生 vLLM LoRA
        {
            "name": "Baseline (vLLM Native LoRA)",
            "config": {}
        },
        
        # 2. SGMV Shrink only
        {
            "name": "SGMV Shrink Only",
            "config": {"sgmv_shrink": True}
        },
        
        # 3. SGMV Expand only
        {
            "name": "SGMV Expand Only",
            "config": {"sgmv_expand": True}
        },
        
        # 4. SGMV Full (shrink + expand)
        {
            "name": "SGMV Full",
            "config": {"sgmv_full": True}
        },
        
        # 5. Fused SGMV (shrink+expand 融合)
        {
            "name": "SGMV Fused (shrink+expand)",
            "config": {"sgmv_fused_only": True}
        },
        
        # 6. Fused LoRA+RMSNorm
        {
            "name": "Fused LoRA+RMSNorm",
            "config": {"fused_lora_rmsnorm": True}
        },
        
        # 7. Tensor Core 优化
        {
            "name": "Tensor Core Optimization",
            "config": {"tensor_core": True}
        },
        
        # 8. Segment Parallel
        {
            "name": "Segment Parallel Strategy",
            "config": {"segment_parallel": True}
        },
        
        # 9. Token Parallel
        {
            "name": "Token Parallel Strategy",
            "config": {"token_parallel": True}
        },
        
        # 10. All Optimizations
        {
            "name": "All Optimizations Combined",
            "config": {"all_optimizations": True}
        },
        
        # 11. SGMV + Fused
        {
            "name": "SGMV Full + Fused",
            "config": {"sgmv_full": True, "fused_sgmv": True}
        },
        
        # 12. SGMV + Tensor Core
        {
            "name": "SGMV Full + Tensor Core",
            "config": {"sgmv_full": True, "tensor_core": True}
        },
        
        # 13. SGMV + Fused + TensorCore
        {
            "name": "SGMV Full + Fused + TensorCore",
            "config": {"sgmv_full": True, "fused_sgmv": True, "tensor_core": True}
        },
        
        # 14. SGMV Baseline (无融合)
        {
            "name": "SGMV Baseline (no fusion)",
            "config": {"sgmv_baseline": True}
        },
        
        # 15. Full Pipeline
        {
            "name": "Full Pipeline (SGMV+Fused+TensorCore+RMSNorm)",
            "config": {
                "sgmv_full": True,
                "fused_sgmv": True,
                "fused_lora_rmsnorm": True,
                "tensor_core": True
            }
        },
    ]
    
    return configs


def main():
    parser = argparse.ArgumentParser(description="SGMV Comprehensive Performance Benchmark")
    parser.add_argument("--mode", type=str, choices=["quick", "full", "custom"], default="quick",
                       help="测试模式: quick (快速), full (完整), custom (自定义)")
    parser.add_argument("--num-runs", type=int, default=10,
                       help="每个测试的运行次数")
    parser.add_argument("--duration", type=str, default="30m",
                       help="测试持续时间 (如: 30m, 2h)")
    parser.add_argument("--output-dir", type=str, default="bench_results",
                       help="输出目录")
    args = parser.parse_args()
    
    # 解析持续时间
    if args.duration.endswith("m"):
        duration_minutes = int(args.duration[:-1])
    elif args.duration.endswith("h"):
        duration_minutes = int(args.duration[:-1]) * 60
    else:
        duration_minutes = 30
    
    print(f"\n{'='*70}")
    print("SGMV Comprehensive Performance Benchmark")
    print(f"{'='*70}")
    print(f"测试模式: {args.mode}")
    print(f"测试时长: {args.duration} (~{duration_minutes} 分钟)")
    print(f"每个测试运行次数: {args.num_runs}")
    print(f"{'='*70}\n")
    
    # 选择测试配置
    if args.mode == "quick":
        # 快速测试：只测试关键配置
        configs = [
            get_all_test_configs()[0],  # Baseline
            get_all_test_configs()[3],  # SGMV Full
            get_all_test_configs()[4],  # Fused SGMV
            get_all_test_configs()[9],  # All Optimizations
            get_all_test_configs()[14], # Full Pipeline
        ]
        expected_time = "15-20 分钟"
    elif args.mode == "full":
        # 完整测试：所有配置
        configs = get_all_test_configs()
        expected_time = f"{len(configs) * 3-4} 分钟"
    else:
        # 自定义：可以指定特定配置
        configs = get_all_test_configs()
        expected_time = f"{len(configs) * 3-4} 分钟"
    
    print(f"预计测试时间: {expected_time}")
    print(f"测试配置数量: {len(configs)}")
    print()
    
    # 运行所有测试
    results = []
    start_time = time.time()
    
    for i, config_info in enumerate(configs, 1):
        print(f"\n{'#'*70}")
        print(f"# 测试 {i}/{len(configs)}")
        print(f"{'#'*70}\n")
        
        result = run_test(
            config_info["config"],
            config_info["name"],
            num_runs=args.num_runs
        )
        results.append(result)
        
        # 保存中间结果
        os.makedirs(args.output_dir, exist_ok=True)
        with open(f"{args.output_dir}/sgmv_comprehensive_partial.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 检查是否超时
        elapsed = (time.time() - start_time) / 60
        if elapsed > duration_minutes:
            print(f"\n⚠️  超时 (已运行 {elapsed:.1f} 分钟)，停止测试")
            break
    
    total_time = (time.time() - start_time) / 60
    
    # 生成最终报告
    print(f"\n{'='*70}")
    print("生成最终报告")
    print(f"{'='*70}")
    
    # 找出最佳配置
    successful_results = [r for r in results if r["status"] == "success"]
    
    if successful_results:
        best_avg = min(successful_results, key=lambda x: x["avg_ms"])
        best_min = min(successful_results, key=lambda x: x["min_ms"])
        
        # 计算加速比
        baseline = next((r for r in successful_results if "Baseline" in r["name"]), None)
        
        print(f"\n🏆 最佳配置:")
        print(f"  最佳 Avg: {best_avg['name']} ({best_avg['avg_ms']:.2f}ms)")
        print(f"  最佳 Min: {best_min['name']} ({best_min['min_ms']:.2f}ms)")
        
        if baseline:
            print(f"\n📈 相对于 Baseline 的加速比:")
            for r in successful_results:
                if r["name"] != baseline["name"]:
                    speedup = baseline["avg_ms"] / r["avg_ms"]
                    improvement = (baseline["avg_ms"] - r["avg_ms"]) / baseline["avg_ms"] * 100
                    print(f"  {r['name']}: {speedup:.3f}x ({improvement:.1f}%)")
        
        # 保存完整结果
        final_results = {
            "summary": {
                "total_tests": len(results),
                "successful_tests": len(successful_results),
                "total_time_minutes": total_time,
                "num_runs_per_test": args.num_runs
            },
            "baseline": baseline,
            "best_avg": best_avg,
            "best_min": best_min,
            "all_results": successful_results,
            "speedup_comparison": {}
        }
        
        if baseline:
            for r in successful_results:
                if r["name"] != baseline["name"]:
                    speedup = baseline["avg_ms"] / r["avg_ms"]
                    improvement = (baseline["avg_ms"] - r["avg_ms"]) / baseline["avg_ms"] * 100
                    final_results["speedup_comparison"][r["name"]] = {
                        "speedup": speedup,
                        "improvement_pct": improvement
                    }
        
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = f"{args.output_dir}/sgmv_comprehensive_{int(time.time())}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 完整结果已保存到: {output_file}")
        
        # 打印详细对比
        print(f"\n{'='*70}")
        print("详细性能对比")
        print(f"{'='*70}")
        print(f"{'配置':<45} {'Avg(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12} {'Std':<10} {'Speedup':<10}")
        print("-" * 105)
        
        if baseline:
            print(f"{'Baseline':<45} {baseline['avg_ms']:<12.2f} {baseline['min_ms']:<12.2f} {baseline['max_ms']:<12.2f} {baseline['std_ms']:<10.2f} {'1.000x':<10}")
        
        for r in sorted(successful_results, key=lambda x: x["avg_ms"]):
            if r["name"] != baseline["name"] if baseline else True:
                speedup = baseline["avg_ms"] / r["avg_ms"] if baseline else 1.0
                print(f"{r['name']:<45} {r['avg_ms']:<12.2f} {r['min_ms']:<12.2f} {r['max_ms']:<12.2f} {r['std_ms']:<10.2f} {speedup:<10.3f}x")
        
        print(f"\n{'='*70}")
        print(f"总耗时: {total_time:.1f} 分钟")
        print(f"{'='*70}")
        
    else:
        print("\n❌ 所有测试都失败了")
        for r in results:
            print(f"  {r['name']}: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()

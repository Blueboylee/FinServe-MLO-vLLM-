#!/usr/bin/env python3
"""
Triton Kernel 性能对比压测

直接对比使用和不使用 Triton kernel 的性能差异
运行一次，直接打印结果到终端

使用独立进程运行每个测试，避免显存累积
"""

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# 导入 Triton kernel 优化模块
from triton_integration import (
    apply_triton_optimizations,
    fused_rms_norm,
    fused_silu_mul,
    fused_add_rms_norm,
    fused_rotary_emb,
)

MODEL_DIR = Path("./models")
BASE_MODEL = str(MODEL_DIR / "Qwen3-VL-8B-Instruct-AWQ-4bit")
EXPERT_A_PATH = str(MODEL_DIR / "Qwen3-VL-Finance-expert-a")
EXPERT_B_PATH = str(MODEL_DIR / "Qwen3-VL-Finance-expert-b")


def detect_lora_rank(adapter_path: str) -> int:
    config_path = Path(adapter_path) / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)["r"]
    return 64


def create_llm(enable_triton: bool = False):
    """创建 vLLM 实例 - 使用更小的显存"""
    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=64,
        max_cpu_loras=2,
        max_model_len=1024,
        gpu_memory_utilization=0.50,
        trust_remote_code=True,
        enable_chunked_prefill=False,
        max_num_batched_tokens=1024,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1, "video": 0},
        enforce_eager=True,
    )
    
    if enable_triton:
        apply_triton_optimizations()
    
    return llm


def warmup_and_benchmark(llm, sampling_params, lora_request, messages, warmup_runs=5, num_runs=30):
    """Warmup + Benchmark 推理"""
    # Warmup
    warmup_messages = [
        {"role": "system", "content": "你是金融分析专家A，擅长宏观经济分析。"},
        {"role": "user", "content": "请分析2024年A股市场的整体走势和主要驱动因素。"},
    ]
    
    # Warmup runs
    for i in range(warmup_runs):
        llm.chat(warmup_messages, sampling_params=sampling_params, lora_request=lora_request)
        if (i + 1) % 5 == 0:
            print(f"  Warmup {i+1}/{warmup_runs}...")
    
    # Benchmark
    latencies = []
    ttfts = []
    output_lengths = []
    
    for i in range(num_runs):
        start_time = time.time()
        outputs = llm.chat(messages, sampling_params=sampling_params, lora_request=lora_request)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        output_len = len(outputs[0].outputs[0].token_ids)
        
        # TTFT: Time To First Token (假设 warmup 后第一次推理的前 token 时间)
        # 这里简化处理，直接用总时间除以输出 token 数得到平均 TPOT
        ttft = latency_ms / (output_len + 1)  # 粗略估计 TTFT
        ttfts.append(ttft)
        output_lengths.append(output_len)
        
        latencies.append(latency_ms)
        if (i + 1) % 10 == 0:
            print(f"  Run {i+1}/{num_runs}: {latency_ms:.2f} ms, output_len={output_len}")
    
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    std_latency = (sum((x - avg_latency) ** 2 for x in latencies) / len(latencies)) ** 0.5
    
    # TPS: tokens per second
    total_output_tokens = sum(output_lengths)
    total_latency_sec = sum(latencies) / 1000
    tps = total_output_tokens / total_latency_sec
    
    # TPOT: Time Per Output Token
    avg_ttft = sum(ttfts) / len(ttfts)
    tpot_values = [lat / length for lat, length in zip(latencies, output_lengths) if length > 0]
    avg_tpots = sum(tpot_values) / len(tpot_values) if tpot_values else 0
    
    # TPOT P50 and P99
    sorted_tpots = sorted(tpot_values) if tpot_values else []
    p50_idx = int(len(sorted_tpots) * 0.50)
    p99_idx = int(len(sorted_tpots) * 0.99)
    p50_tpots = sorted_tpots[p50_idx] if sorted_tpots else 0
    p99_tpots = sorted_tpots[p99_idx] if sorted_tpots else 0
    
    return {
        "avg": avg_latency,
        "min": min_latency,
        "max": max_latency,
        "std": std_latency,
        "latencies": latencies,
        "ttfts": ttfts,
        "output_lengths": output_lengths,
        "avg_ttft": avg_ttft,
        "avg_tpots": avg_tpots,
        "p50_tpots": p50_tpots,
        "p99_tpots": p99_tpots,
        "tps": tps,
        "output": outputs[0].outputs[0].text[:100],
    }


def run_test(enable_triton: bool, result_queue: mp.Queue):
    """运行单个测试"""
    try:
        lora_rank = detect_lora_rank(EXPERT_A_PATH)
        expert_a = LoRARequest("finance-expert-a", 1, EXPERT_A_PATH)
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            max_tokens=256,
        )
        
        test_prompt = "请分析2024年A股市场的整体走势和主要驱动因素。"
        messages_a = [
            {"role": "system", "content": "你是金融分析专家A，擅长宏观经济分析。"},
            {"role": "user", "content": test_prompt},
        ]
        
        llm = create_llm(enable_triton=enable_triton)
        
        results = warmup_and_benchmark(
            llm, sampling_params, expert_a, messages_a,
            warmup_runs=5, num_runs=30
        )
        
        result_queue.put({
            "enable_triton": enable_triton,
            "results": results,
        })
    except Exception as e:
        result_queue.put({
            "enable_triton": enable_triton,
            "error": str(e),
        })
        raise


def main():
    print("=" * 80)
    print("  Triton Kernel 性能对比压测 (30个测试用例)")
    print("=" * 80)
    print()
    
    result_queue = mp.Queue()
    
    # ========== 测试 1: 不使用 Triton (Baseline) ==========
    print("=" * 80)
    print("  测试 1: 不使用 Triton Kernel (Baseline)")
    print("=" * 80)
    print("  运行 30 个测试用例...")
    print()
    
    test1 = mp.Process(target=run_test, args=(False, result_queue))
    test1.start()
    test1.join()
    
    result1 = result_queue.get()
    if "error" in result1:
        print(f"  错误: {result1['error']}")
        return
    
    baseline_results = result1["results"]
    
    print(f"\n  Baseline 结果:")
    print(f"    Avg: {baseline_results['avg']:.2f} ms")
    print(f"    Min: {baseline_results['min']:.2f} ms")
    print(f"    Max: {baseline_results['max']:.2f} ms")
    print(f"    Std: {baseline_results['std']:.2f} ms")
    print(f"    Avg TTFT: {baseline_results['avg_ttft']:.2f} ms")
    print(f"    Avg TPOT: {baseline_results['avg_tpots']:.2f} ms")
    print(f"    TPOT P50: {baseline_results['p50_tpots']:.2f} ms")
    print(f"    TPOT P99: {baseline_results['p99_tpots']:.2f} ms")
    print(f"    TPS: {baseline_results['tps']:.2f} tokens/s")
    print(f"    Output: {baseline_results['output']}...")
    print()
    
    # ========== 测试 2: 使用 Triton ==========
    print("=" * 80)
    print("  测试 2: 使用 Triton Kernel 优化")
    print("=" * 80)
    print("  运行 30 个测试用例...")
    print()
    
    test2 = mp.Process(target=run_test, args=(True, result_queue))
    test2.start()
    test2.join()
    
    result2 = result_queue.get()
    if "error" in result2:
        print(f"  错误: {result2['error']}")
        return
    
    triton_results = result2["results"]
    
    print(f"\n  Triton 结果:")
    print(f"    Avg: {triton_results['avg']:.2f} ms")
    print(f"    Min: {triton_results['min']:.2f} ms")
    print(f"    Max: {triton_results['max']:.2f} ms")
    print(f"    Std: {triton_results['std']:.2f} ms")
    print(f"    Avg TTFT: {triton_results['avg_ttft']:.2f} ms")
    print(f"    Avg TPOT: {triton_results['avg_tpots']:.2f} ms")
    print(f"    TPOT P50: {triton_results['p50_tpots']:.2f} ms")
    print(f"    TPOT P99: {triton_results['p99_tpots']:.2f} ms")
    print(f"    TPS: {triton_results['tps']:.2f} tokens/s")
    print(f"    Output: {triton_results['output']}...")
    print()
    
    # ========== 性能对比 ==========
    print("=" * 80)
    print("  性能对比总结 (30个测试用例)")
    print("=" * 80)
    print()
    
    baseline_avg = baseline_results["avg"]
    triton_avg = triton_results["avg"]
    speedup = baseline_avg / triton_avg
    improvement = ((baseline_avg - triton_avg) / baseline_avg) * 100
    
    baseline_ttft = baseline_results["avg_ttft"]
    triton_ttft = triton_results["avg_ttft"]
    ttft_improvement = ((baseline_ttft - triton_ttft) / baseline_ttft) * 100 if baseline_ttft > 0 else 0
    
    baseline_tpots = baseline_results["avg_tpots"]
    triton_tpots = triton_results["avg_tpots"]
    tpots_improvement = ((baseline_tpots - triton_tpots) / baseline_tpots) * 100 if baseline_tpots > 0 else 0
    
    baseline_tps = baseline_results["tps"]
    triton_tps = triton_results["tps"]
    tps_improvement = ((triton_tps - baseline_tps) / baseline_tps) * 100 if baseline_tps > 0 else 0
    
    print(f"  {'Metric':<20} {'Baseline':>12} {'Triton':>12} {'Improvement':>15}")
    print("  " + "-" * 60)
    print(f"  {'Avg Latency (ms)':<20} {baseline_avg:>12.2f} {triton_avg:>12.2f} {improvement:>14.2f}%")
    print(f"  {'Avg TTFT (ms)':<20} {baseline_ttft:>12.2f} {triton_ttft:>12.2f} {ttft_improvement:>14.2f}%")
    print(f"  {'Avg TPOT (ms)':<20} {baseline_tpots:>12.2f} {triton_tpots:>12.2f} {tpots_improvement:>14.2f}%")
    print(f"  {'TPOT P50 (ms)':<20} {baseline_results['p50_tpots']:>12.2f} {triton_results['p50_tpots']:>12.2f} {((baseline_results['p50_tpots']-triton_results['p50_tpots'])/baseline_results['p50_tpots'])*100 if baseline_results['p50_tpots'] > 0 else 0:>14.2f}%")
    print(f"  {'TPOT P99 (ms)':<20} {baseline_results['p99_tpots']:>12.2f} {triton_results['p99_tpots']:>12.2f} {((baseline_results['p99_tpots']-triton_results['p99_tpots'])/baseline_results['p99_tpots'])*100 if baseline_results['p99_tpots'] > 0 else 0:>14.2f}%")
    print(f"  {'TPS (tokens/s)':<20} {baseline_tps:>12.2f} {triton_tps:>12.2f} {tps_improvement:>14.2f}%")
    print(f"  {'Std Latency (ms)':<20} {baseline_results['std']:>12.2f} {triton_results['std']:>12.2f} {((baseline_results['std']-triton_results['std'])/baseline_results['std'])*100 if baseline_results['std'] > 0 else 0:>14.2f}%")
    print()
    
    print(f"  加速比: {speedup:.3f}x")
    print()
    
    if speedup > 1.05:
        print(f"  ✅ Triton 版本显著更快 ({speedup:.3f}x, 提升 {improvement:.2f}%)")
    elif speedup > 1.02:
        print(f"  ⚠️  Triton 版本略快 ({speedup:.3f}x, 提升 {improvement:.2f}%)")
    elif speedup > 0.98:
        print(f"  - Triton 版本提升不明显 ({speedup:.3f}x, 提升 {improvement:.2f}%)")
    else:
        print(f"  ❌ Triton 版本反而变慢 ({speedup:.3f}x, 降低 {abs(improvement):.2f}%)")
    
    print()
    print("=" * 80)
    print("  压测完成！")
    print("=" * 80)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

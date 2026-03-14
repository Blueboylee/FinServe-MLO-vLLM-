#!/usr/bin/env python3
"""
SGMV 性能对比测试脚本

"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# 导入 SGMV 优化模块
from sgmv_kernel import apply_sgmv_optimizations, revert_sgmv_optimizations

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


def create_llm(enable_sgmv: bool = False):
    """创建 vLLM 实例"""
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
    
    if enable_sgmv:
        print("\n" + "=" * 60)
        print("启用 SGMV Kernel 优化")
        print("=" * 60)
        patched = apply_sgmv_optimizations(enable_fused=True)
        print(f"已 patch 的算子: {patched}")
    
    return llm


def warmup(llm, expert_a, sampling_params, messages_a, label="Warmup"):
    """Warmup 推理"""
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print("=" * 60)
    start = time.time()
    outputs = llm.chat(messages_a, sampling_params=sampling_params, lora_request=expert_a)
    elapsed = time.time() - start
    print(f"完成时间: {elapsed:.3f}s")
    return outputs[0].outputs[0].text[:100]


def benchmark_inference(llm, expert, sampling_params, messages, label, num_runs=3):
    """Benchmark 推理，收集更多性能指标"""
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print("=" * 60)
    
    latencies = []
    ttft_list = []
    tpot_list = []
    output_lengths = []
    
    for i in range(num_runs):
        start = time.perf_counter()
        outputs = llm.chat(messages, sampling_params=sampling_params, lora_request=expert)
        elapsed = time.perf_counter() - start
        
        output_text = outputs[0].outputs[0].text
        output_length = len(output_text)
        output_lengths.append(output_length)
        
        # 使用 vLLM 真实 metrics
        if hasattr(outputs[0], 'metrics') and outputs[0].metrics is not None:
            metrics = outputs[0].metrics
            # 真实的 TTFT：第一颗 Token 出来的时刻 - 请求到达时刻
            actual_ttft = (metrics.first_token_time - metrics.arrival_time) * 1000  # ms
            # 真实的 TPOT (Inter-Token Latency)
            token_ids = outputs[0].outputs[0].token_ids
            if len(token_ids) > 1:
                actual_tpot = (metrics.finished_time - metrics.first_token_time) / (len(token_ids) - 1) * 1000  # ms
            else:
                actual_tpot = 0.0
        else:
            # fallback: 使用估算值
            actual_ttft = elapsed * 1000 * 0.3
            actual_tpot = (elapsed * 1000) / max(output_length, 1)
        
        latencies.append(elapsed * 1000)
        ttft_list.append(actual_ttft)
        tpot_list.append(actual_tpot)
        
        print(f"  Run {i+1}: {elapsed:.3f}s ({elapsed*1000:.2f}ms)")
        print(f"    Output length: {output_length} tokens")
        print(f"    TTFT: {actual_ttft:.2f}ms, TPOT: {actual_tpot:.2f}ms")
    
    avg_ms = sum(latencies) / len(latencies)
    min_ms = min(latencies)
    max_ms = max(latencies)
    
    sorted_latencies = sorted(latencies)
    p50_idx = int(len(sorted_latencies) * 0.5)
    p99_idx = int(len(sorted_latencies) * 0.99)
    p50_ms = sorted_latencies[min(p50_idx, len(sorted_latencies)-1)]
    p99_ms = sorted_latencies[min(p99_idx, len(sorted_latencies)-1)]
    
    avg_ttft = sum(ttft_list) / len(ttft_list)
    avg_tpot = sum(tpot_list) / len(tpot_list)
    
    avg_output_length = sum(output_lengths) / len(output_lengths)
    
    roof = avg_output_length / (avg_ms / 1000)
    
    print(f"\n  统计结果:")
    print(f"    Avg Latency:  {avg_ms:.2f}ms")
    print(f"    Min Latency:  {min_ms:.2f}ms")
    print(f"    Max Latency:  {max_ms:.2f}ms")
    print(f"    P50 Latency:  {p50_ms:.2f}ms")
    print(f"    P99 Latency:  {p99_ms:.2f}ms")
    print(f"    Avg TTFT:     {avg_ttft:.2f}ms")
    print(f"    Avg TPOT:     {avg_tpot:.2f}ms")
    print(f"    Avg Output:   {avg_output_length:.1f} tokens")
    print(f"    ROOF:         {roof:.1f} tokens/s")
    
    return {
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "p50_ms": p50_ms,
        "p99_ms": p99_ms,
        "avg_ttft_ms": avg_ttft,
        "avg_tpot_ms": avg_tpot,
        "avg_output_length": avg_output_length,
        "roof_tokens_per_s": roof,
        "ttft_list_ms": ttft_list,
        "tpot_list_ms": tpot_list,
    }


def benchmark_multi_requests(llm, expert_a, expert_b, sampling_params, label, num_runs=3):
    """Benchmark 多请求测试，使用不同长度和 expert"""
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print("=" * 60)
    
    # 构造 8 个不同长度的请求
    prompts = [
        ("简短提问", expert_a, "请简单分析一下当前股市走势。"),
        ("中等长度报告", expert_b, "请写一篇1000字的2024年金融行业分析报告。"),
        ("详细分析", expert_a, "请分析2024年A股市场的整体走势和主要驱动因素。"),
        ("长篇报告", expert_b, "请写一篇3000字的金融行业深度研究报告，包括宏观经济、行业趋势和投资建议。"),
        ("简短预测", expert_a, "预测下季度股市表现。"),
        ("详细策略", expert_b, "请制定一份详细的资产配置策略，包括股票、债券和基金的比例建议。"),
        ("市场复盘", expert_a, "请复盘2024年全年A股市场表现，分析主要指数和行业轮动。"),
        ("综合分析", expert_b, "请写一篇5000字的2024年金融市场年度总结，涵盖股市、债市、汇率和大宗商品。"),
    ]
    
    messages_list = [
        [
            {"role": "system", "content": "你是金融分析专家，擅长宏观经济分析。"},
            {"role": "user", "content": prompt},
        ]
        for _, _, prompt in prompts
    ]
    
    latencies = []
    ttft_list = []
    tpot_list = []
    output_lengths = []
    expert_counts = {expert_a.lora_name: 0, expert_b.lora_name: 0}
    
    for run in range(num_runs):
        run_latencies = []
        run_ttft = []
        run_tpot = []
        run_outputs = []
        
        for i, (expert, messages) in enumerate(zip([p[1] for p in prompts], messages_list)):
            start = time.perf_counter()
            outputs = llm.chat(messages, sampling_params=sampling_params, lora_request=expert)
            elapsed = time.perf_counter() - start
            
            output_text = outputs[0].outputs[0].text
            output_length = len(output_text)
            
            if hasattr(outputs[0], 'metrics') and outputs[0].metrics is not None:
                metrics = outputs[0].metrics
                actual_ttft = (metrics.first_token_time - metrics.arrival_time) * 1000
                token_ids = outputs[0].outputs[0].token_ids
                if len(token_ids) > 1:
                    actual_tpot = (metrics.finished_time - metrics.first_token_time) / (len(token_ids) - 1) * 1000
                else:
                    actual_tpot = 0.0
            else:
                actual_ttft = elapsed * 1000 * 0.3
                actual_tpot = (elapsed * 1000) / max(output_length, 1)
            
            run_latencies.append(elapsed * 1000)
            run_ttft.append(actual_ttft)
            run_tpot.append(actual_tpot)
            run_outputs.append(output_length)
            expert_counts[expert.lora_name] += 1
        
        total_time = sum(run_latencies)
        latencies.append(total_time)
        ttft_list.append(sum(run_ttft) / len(run_ttft))
        tpot_list.append(sum(run_tpot) / len(run_tpot))
        output_lengths.append(sum(run_outputs))
        
        print(f"  Run {run+1}: {total_time:.2f}ms")
        print(f"    Total tokens: {run_outputs}")
        print(f"    Avg TTFT: {sum(run_ttft)/len(run_ttft):.2f}ms, Avg TPOT: {sum(run_tpot)/len(run_tpot):.2f}ms")
    
    avg_ms = sum(latencies) / len(latencies)
    min_ms = min(latencies)
    max_ms = max(latencies)
    
    sorted_latencies = sorted(latencies)
    p50_idx = int(len(sorted_latencies) * 0.5)
    p99_idx = int(len(sorted_latencies) * 0.99)
    p50_ms = sorted_latencies[min(p50_idx, len(sorted_latencies)-1)]
    p99_ms = sorted_latencies[min(p99_idx, len(sorted_latencies)-1)]
    
    avg_ttft = sum(ttft_list) / len(ttft_list)
    avg_tpot = sum(tpot_list) / len(tpot_list)
    avg_output_length = sum(output_lengths) / len(output_lengths)
    roof = avg_output_length / (avg_ms / 1000)
    
    print(f"\n  统计结果:")
    print(f"    Avg Latency:  {avg_ms:.2f}ms")
    print(f"    Min Latency:  {min_ms:.2f}ms")
    print(f"    Max Latency:  {max_ms:.2f}ms")
    print(f"    P50 Latency:  {p50_ms:.2f}ms")
    print(f"    P99 Latency:  {p99_ms:.2f}ms")
    print(f"    Avg TTFT:     {avg_ttft:.2f}ms")
    print(f"    Avg TPOT:     {avg_tpot:.2f}ms")
    print(f"    Avg Output:   {avg_output_length:.1f} tokens")
    print(f"    ROOF:         {roof:.1f} tokens/s")
    print(f"    Expert usage: {expert_counts}")
    
    return {
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "p50_ms": p50_ms,
        "p99_ms": p99_ms,
        "avg_ttft_ms": avg_ttft,
        "avg_tpot_ms": avg_tpot,
        "avg_output_length": avg_output_length,
        "roof_tokens_per_s": roof,
        "ttft_list_ms": ttft_list,
        "tpot_list_ms": tpot_list,
    }


def benchmark_concurrent(
    llm,
    expert_a,
    expert_b,
    sampling_params,
    total_requests=8,
    concurrent_workers=2,
    label="Concurrent Benchmark",
):
    """并发 Benchmark：多线程同时发起请求，测整体吞吐和单请求延迟分布"""
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print("=" * 60)

    # 构造与 multi_requests 一致的 8 个典型金融场景 prompt
    prompts = [
        ("简短提问", expert_a, "请简单分析一下当前股市走势。"),
        ("中等长度报告", expert_b, "请写一篇1000字的2024年金融行业分析报告。"),
        ("详细分析", expert_a, "请分析2024年A股市场的整体走势和主要驱动因素。"),
        ("长篇报告", expert_b, "请写一篇3000字的金融行业深度研究报告，包括宏观经济、行业趋势和投资建议。"),
        ("简短预测", expert_a, "预测下季度股市表现。"),
        ("详细策略", expert_b, "请制定一份详细的资产配置策略，包括股票、债券和基金的比例建议。"),
        ("市场复盘", expert_a, "请复盘2024年全年A股市场表现，分析主要指数和行业轮动。"),
        ("综合分析", expert_b, "请写一篇5000字的2024年金融市场年度总结，涵盖股市、债市、汇率和大宗商品。"),
    ]

    def build_messages(prompt_text: str):
        return [
            {"role": "system", "content": "你是金融分析专家，擅长宏观经济分析。"},
            {"role": "user", "content": prompt_text},
        ]

    latencies = []
    ttft_list = []
    tpot_list = []
    output_lengths = []
    expert_counts = {expert_a.lora_name: 0, expert_b.lora_name: 0}

    def run_one(idx: int):
        # 轮询 8 个场景，构造第 idx 个请求
        _, expert, prompt_text = prompts[idx % len(prompts)]
        messages = build_messages(prompt_text)

        start = time.perf_counter()
        outputs = llm.chat(messages, sampling_params=sampling_params, lora_request=expert)
        elapsed = time.perf_counter() - start

        output_text = outputs[0].outputs[0].text
        output_length = len(output_text)

        if hasattr(outputs[0], "metrics") and outputs[0].metrics is not None:
            metrics = outputs[0].metrics
            actual_ttft = (metrics.first_token_time - metrics.arrival_time) * 1000
            token_ids = outputs[0].outputs[0].token_ids
            if len(token_ids) > 1:
                actual_tpot = (
                    metrics.finished_time - metrics.first_token_time
                ) / (len(token_ids) - 1) * 1000
            else:
                actual_tpot = 0.0
        else:
            actual_ttft = elapsed * 1000 * 0.3
            actual_tpot = (elapsed * 1000) / max(output_length, 1)

        return {
            "latency_ms": elapsed * 1000,
            "ttft_ms": actual_ttft,
            "tpot_ms": actual_tpot,
            "output_length": output_length,
            "expert_name": expert.lora_name,
        }

    print(f"  Total requests: {total_requests}, concurrent workers: {concurrent_workers}")

    start_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
        futures = [executor.submit(run_one, i) for i in range(total_requests)]
        for fut in as_completed(futures):
            res = fut.result()
            latencies.append(res["latency_ms"])
            ttft_list.append(res["ttft_ms"])
            tpot_list.append(res["tpot_ms"])
            output_lengths.append(res["output_length"])
            expert_counts[res["expert_name"]] += 1
    elapsed_all = time.perf_counter() - start_all

    if not latencies:
        print("  没有成功的请求，无法统计。")
        return {}

    total_tokens = sum(output_lengths)
    avg_ms = sum(latencies) / len(latencies)
    min_ms = min(latencies)
    max_ms = max(latencies)

    sorted_latencies = sorted(latencies)
    p50_idx = int(len(sorted_latencies) * 0.5)
    p99_idx = int(len(sorted_latencies) * 0.99)
    p50_ms = sorted_latencies[min(p50_idx, len(sorted_latencies) - 1)]
    p99_ms = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]

    avg_ttft = sum(ttft_list) / len(ttft_list)
    avg_tpot = sum(tpot_list) / len(tpot_list)
    avg_output_length = sum(output_lengths) / len(output_lengths)

    # 单请求视角的 "roof"（与前面保持一致）
    roof = avg_output_length / (avg_ms / 1000)
    # 并发整体视角：整体吞吐 & QPS
    total_wall_ms = elapsed_all * 1000
    throughput_tokens_per_s = total_tokens / elapsed_all if elapsed_all > 0 else 0.0
    qps = total_requests / elapsed_all if elapsed_all > 0 else 0.0

    print(f"\n  统计结果（单请求视角）:")
    print(f"    Avg Latency:      {avg_ms:.2f}ms")
    print(f"    Min Latency:      {min_ms:.2f}ms")
    print(f"    Max Latency:      {max_ms:.2f}ms")
    print(f"    P50 Latency:      {p50_ms:.2f}ms")
    print(f"    P99 Latency:      {p99_ms:.2f}ms")
    print(f"    Avg TTFT:         {avg_ttft:.2f}ms")
    print(f"    Avg TPOT:         {avg_tpot:.2f}ms")
    print(f"    Avg Output:       {avg_output_length:.1f} tokens (approx by text length)")
    print(f"    ROOF:             {roof:.1f} tokens/s")

    print(f"\n  统计结果（整体并发视角）:")
    print(f"    Total wall time:  {total_wall_ms:.2f}ms")
    print(f"    Total tokens:     {total_tokens} (approx by text length)")
    print(f"    Throughput:       {throughput_tokens_per_s:.1f} tokens/s")
    print(f"    QPS:              {qps:.2f} req/s")
    print(f"    Expert usage:     {expert_counts}")

    return {
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "p50_ms": p50_ms,
        "p99_ms": p99_ms,
        "avg_ttft_ms": avg_ttft,
        "avg_tpot_ms": avg_tpot,
        "avg_output_length": avg_output_length,
        "roof_tokens_per_s": roof,
        "total_wall_ms": total_wall_ms,
        "total_tokens": total_tokens,
        "throughput_tokens_per_s": throughput_tokens_per_s,
        "qps": qps,
        "ttft_list_ms": ttft_list,
        "tpot_list_ms": tpot_list,
        "expert_counts": expert_counts,
    }


def run_compare_mode():
    """对比模式：先 baseline，再 optimized"""
    print("=" * 80)
    print("  SGMV 性能对比测试")
    print("=" * 80)
    print()
    
    lora_rank = detect_lora_rank(EXPERT_A_PATH)
    print(f"检测到 LoRA rank: {lora_rank}")
    
    expert_a = LoRARequest("finance-expert-a", 1, EXPERT_A_PATH)
    expert_b = LoRARequest("finance-expert-b", 2, EXPERT_B_PATH)
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
    )
    
    test_prompt = "请分析2024年A股市场的整体走势和主要驱动因素。"
    
    messages_a = [
        {"role": "system", "content": "你是金融分析专家A，擅长宏观经济分析。"},
        {"role": "user", "content": test_prompt},
    ]
    
    # ========== 测试 1: Baseline ==========
    print("=" * 80)
    print("  测试 1: Baseline (无 SGMV 优化)")
    print("=" * 80)
    
    llm_baseline = create_llm(enable_sgmv=False)
    
    # Warmup with multiple experts
    warmup(llm_baseline, expert_a, sampling_params, messages_a, "Baseline Warmup Expert-A")
    warmup(llm_baseline, expert_b, sampling_params, messages_a, "Baseline Warmup Expert-B")
    
    baseline_results = benchmark_multi_requests(
        llm_baseline, expert_a, expert_b, sampling_params, "Baseline Multi-Requests"
    )
    
    # 清理 baseline
    del llm_baseline
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ========== 测试 2: Optimized ==========
    print("\n" + "=" * 80)
    print("  测试 2: Optimized (有 SGMV 优化)")
    print("=" * 80)
    
    llm_optimized = create_llm(enable_sgmv=True)
    
    # Warmup with multiple experts
    warmup(llm_optimized, expert_a, sampling_params, messages_a, "Optimized Warmup Expert-A")
    warmup(llm_optimized, expert_b, sampling_params, messages_a, "Optimized Warmup Expert-B")
    
    optimized_results = benchmark_multi_requests(
        llm_optimized, expert_a, expert_b, sampling_params, "Optimized Multi-Requests"
    )
    
    # ========== 性能对比 ==========
    print("\n" + "=" * 80)
    print("  性能对比总结")
    print("=" * 80)
    print()
    
    baseline_avg = baseline_results["avg_ms"]
    optimized_avg = optimized_results["avg_ms"]
    
    speedup = baseline_avg / optimized_avg if optimized_avg > 0 else 0
    improvement = ((baseline_avg - optimized_avg) / baseline_avg) * 100 if baseline_avg > 0 else 0
    
    print(f"\n  {'Metric':<20} {'Baseline':>15} {'Optimized':>15} {'Improvement':>15}")
    print("  " + "-" * 75)
    print(f"  {'Avg Latency':<20} {baseline_avg:>15.2f}ms {optimized_avg:>15.2f}ms {improvement:>14.2f}%")
    print(f"  {'Min Latency':<20} {baseline_results['min_ms']:>15.2f}ms {optimized_results['min_ms']:>15.2f}ms {((baseline_results['min_ms']-optimized_results['min_ms'])/baseline_results['min_ms'])*100:>14.2f}%")
    print(f"  {'Max Latency':<20} {baseline_results['max_ms']:>15.2f}ms {optimized_results['max_ms']:>15.2f}ms {((baseline_results['max_ms']-optimized_results['max_ms'])/baseline_results['max_ms'])*100:>14.2f}%")
    print(f"  {'P50 Latency':<20} {baseline_results['p50_ms']:>15.2f}ms {optimized_results['p50_ms']:>15.2f}ms {((baseline_results['p50_ms']-optimized_results['p50_ms'])/baseline_results['p50_ms'])*100:>14.2f}%")
    print(f"  {'P99 Latency':<20} {baseline_results['p99_ms']:>15.2f}ms {optimized_results['p99_ms']:>15.2f}ms {((baseline_results['p99_ms']-optimized_results['p99_ms'])/baseline_results['p99_ms'])*100:>14.2f}%")
    print(f"  {'Avg TTFT':<20} {baseline_results['avg_ttft_ms']:>15.2f}ms {optimized_results['avg_ttft_ms']:>15.2f}ms {((baseline_results['avg_ttft_ms']-optimized_results['avg_ttft_ms'])/baseline_results['avg_ttft_ms'])*100:>14.2f}%")
    print(f"  {'Avg TPOT':<20} {baseline_results['avg_tpot_ms']:>15.2f}ms {optimized_results['avg_tpot_ms']:>15.2f}ms {((baseline_results['avg_tpot_ms']-optimized_results['avg_tpot_ms'])/baseline_results['avg_tpot_ms'])*100:>14.2f}%")
    print(f"  {'Avg Output':<20} {baseline_results['avg_output_length']:>15.1f} tokens {optimized_results['avg_output_length']:>15.1f} tokens {((optimized_results['avg_output_length']-baseline_results['avg_output_length'])/baseline_results['avg_output_length'])*100:>14.2f}%")
    print(f"  {'ROOF':<20} {baseline_results['roof_tokens_per_s']:>15.1f} tokens/s {optimized_results['roof_tokens_per_s']:>15.1f} tokens/s {((optimized_results['roof_tokens_per_s']-baseline_results['roof_tokens_per_s'])/baseline_results['roof_tokens_per_s'])*100:>14.2f}%")
    print()
    print(f"  加速比: {speedup:.3f}x")
    print()
    
    if speedup > 1.05:
        print(f"  ✅ SGMV 优化显著提升性能 ({speedup:.3f}x, 提升 {improvement:.2f}%)")
    elif speedup > 1.02:
        print(f"  ⚠️  SGMV 优化略有提升 ({speedup:.3f}x, 提升 {improvement:.2f}%)")
    elif speedup > 0.98:
        print(f"  - SGMV 优化提升不明显 ({speedup:.3f}x, 提升 {improvement:.2f}%)")
    else:
        print(f"  ❌ SGMV 优化反而变慢 ({speedup:.3f}x, 降低 {abs(improvement):.2f}%)")
    
    print()
    
    # 保存结果
    results = {
        "baseline": baseline_results,
        "optimized": optimized_results,
        "speedup": speedup,
        "improvement_pct": improvement,
    }
    
    output_file = Path("./bench_results/sgmv_performance.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"  结果已保存到: {output_file}")
    print("=" * 80)
    
    return results


def run_baseline_mode():
    """Baseline 模式：仅测试无优化版本"""
    print("=" * 80)
    print("  Baseline 测试 (无 SGMV 优化)")
    print("=" * 80)
    
    lora_rank = detect_lora_rank(EXPERT_A_PATH)
    print(f"检测到 LoRA rank: {lora_rank}")
    
    expert_a = LoRARequest("finance-expert-a", 1, EXPERT_A_PATH)
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
    )
    
    messages_a = [
        {"role": "system", "content": "你是金融分析专家A，擅长宏观经济分析。"},
        {"role": "user", "content": "请分析2024年A股市场的整体走势和主要驱动因素。"},
    ]
    
    llm = create_llm(enable_sgmv=False)
    warmup(llm, expert_a, sampling_params, messages_a, "Baseline Warmup")
    
    results = benchmark_inference(llm, expert_a, sampling_params, messages_a, "Baseline Expert-A")
    
    print("=" * 80)
    print(f"  Baseline Avg: {results['avg_ms']:.2f}ms")
    print("=" * 80)
    
    return results


def run_optimized_mode():
    """Optimized 模式：仅测试有优化版本"""
    print("=" * 80)
    print("  Optimized 测试 (有 SGMV 优化)")
    print("=" * 80)
    
    lora_rank = detect_lora_rank(EXPERT_A_PATH)
    print(f"检测到 LoRA rank: {lora_rank}")
    
    expert_a = LoRARequest("finance-expert-a", 1, EXPERT_A_PATH)
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
    )
    
    messages_a = [
        {"role": "system", "content": "你是金融分析专家A，擅长宏观经济分析。"},
        {"role": "user", "content": "请分析2024年A股市场的整体走势和主要驱动因素。"},
    ]
    
    llm = create_llm(enable_sgmv=True)
    warmup(llm, expert_a, sampling_params, messages_a, "Optimized Warmup")
    
    results = benchmark_inference(llm, expert_a, sampling_params, messages_a, "Optimized Expert-A")
    
    print("=" * 80)
    print(f"  Optimized Avg: {results['avg_ms']:.2f}ms")
    print("=" * 80)
    
    return results


def run_concurrent_mode(total_requests: int = 8, concurrent_workers: int = 2):
    """Concurrent 模式：真正并发压测（Baseline vs Optimized）"""
    print("=" * 80)
    print("  Concurrent 并发测试 (Baseline vs SGMV Optimized)")
    print("=" * 80)

    lora_rank = detect_lora_rank(EXPERT_A_PATH)
    print(f"检测到 LoRA rank: {lora_rank}")

    expert_a = LoRARequest("finance-expert-a", 1, EXPERT_A_PATH)
    expert_b = LoRARequest("finance-expert-b", 2, EXPERT_B_PATH)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
    )

    # ========== Baseline 并发 ==========
    print("=" * 80)
    print("  测试 1: Baseline 并发 (无 SGMV 优化)")
    print("=" * 80)

    llm_baseline = create_llm(enable_sgmv=False)

    # 简单 Warmup（单请求即可让模型完成基本加载）
    warmup_messages = [
        {"role": "system", "content": "你是金融分析专家，擅长宏观经济分析。"},
        {"role": "user", "content": "请简单分析一下当前股市走势。"},
    ]
    warmup(llm_baseline, expert_a, sampling_params, warmup_messages, "Baseline Warmup")

    baseline_results = benchmark_concurrent(
        llm_baseline,
        expert_a,
        expert_b,
        sampling_params,
        total_requests=total_requests,
        concurrent_workers=concurrent_workers,
        label="Baseline Concurrent",
    )

    # 清理 baseline，释放显存
    del llm_baseline
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========== Optimized 并发 ==========
    print("\n" + "=" * 80)
    print("  测试 2: Optimized 并发 (有 SGMV 优化)")
    print("=" * 80)

    llm_optimized = create_llm(enable_sgmv=True)
    warmup(llm_optimized, expert_a, sampling_params, warmup_messages, "Optimized Warmup")

    optimized_results = benchmark_concurrent(
        llm_optimized,
        expert_a,
        expert_b,
        sampling_params,
        total_requests=total_requests,
        concurrent_workers=concurrent_workers,
        label="Optimized Concurrent",
    )

    # ========== 性能对比 ==========
    print("\n" + "=" * 80)
    print("  并发性能对比总结")
    print("=" * 80)
    print()

    baseline_avg = baseline_results["avg_ms"]
    optimized_avg = optimized_results["avg_ms"]
    baseline_tp = baseline_results["throughput_tokens_per_s"]
    optimized_tp = optimized_results["throughput_tokens_per_s"]
    baseline_qps = baseline_results["qps"]
    optimized_qps = optimized_results["qps"]

    latency_speedup = baseline_avg / optimized_avg if optimized_avg > 0 else 0.0
    tp_speedup = baseline_tp / optimized_tp if optimized_tp > 0 else 0.0
    qps_speedup = baseline_qps / optimized_qps if optimized_qps > 0 else 0.0

    def pct_improve(old, new):
        if old <= 0:
            return 0.0
        return (new - old) / old * 100

    print(f"  {'Metric':<22} {'Baseline':>15} {'Optimized':>15} {'Improvement':>15}")
    print("  " + "-" * 80)
    print(
        f"  {'Avg Latency':<22} "
        f"{baseline_avg:>15.2f}ms {optimized_avg:>15.2f}ms "
        f"{((baseline_avg - optimized_avg) / baseline_avg * 100 if baseline_avg > 0 else 0):>14.2f}%"
    )
    print(
        f"  {'Throughput':<22} "
        f"{baseline_tp:>15.1f} {optimized_tp:>15.1f} "
        f"{pct_improve(baseline_tp, optimized_tp):>14.2f}%"
    )
    print(
        f"  {'QPS':<22} "
        f"{baseline_qps:>15.2f} {optimized_qps:>15.2f} "
        f"{pct_improve(baseline_qps, optimized_qps):>14.2f}%"
    )
    print()
    print(f"  Latency Speedup:   {latency_speedup:.3f}x")
    print(f"  Throughput Speedup:{tp_speedup:.3f}x")
    print(f"  QPS Speedup:       {qps_speedup:.3f}x")
    print()

    # 保存结果
    results = {
        "baseline": baseline_results,
        "optimized": optimized_results,
        "latency_speedup": latency_speedup,
        "throughput_speedup": tp_speedup,
        "qps_speedup": qps_speedup,
        "total_requests": total_requests,
        "concurrent_workers": concurrent_workers,
    }

    output_file = Path("./bench_results/sgmv_concurrent_performance.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  并发测试结果已保存到: {output_file}")
    print("=" * 80)

    return results


def run_longrun_mode(
    duration_hours: float = 24.0,
    total_requests: int = 8,
    concurrent_workers: int = 2,
):
    """Longrun 模式：持续 20~30 小时的长时间压力测试

    设计思想：
    - 在给定时长内循环多种策略（单请求、多请求、不同并发度）；
    - 每种策略都跑 Baseline 和 Optimized 两个版本，做成「A/B 对比」；
    - 持续累积统计信息，定期覆盖写入一个 JSON 文件，方便中途查看或中断恢复。
    """
    print("=" * 80)
    print("  Longrun 长时间并发/延迟混合压力测试 (Baseline vs SGMV Optimized)")
    print("=" * 80)

    lora_rank = detect_lora_rank(EXPERT_A_PATH)
    print(f"检测到 LoRA rank: {lora_rank}")

    expert_a = LoRARequest("finance-expert-a", 1, EXPERT_A_PATH)
    expert_b = LoRARequest("finance-expert-b", 2, EXPERT_B_PATH)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
    )

    # 单请求用的典型对话
    single_messages = [
        {"role": "system", "content": "你是金融分析专家A，擅长宏观经济分析。"},
        {"role": "user", "content": "请分析2024年A股市场的整体走势和主要驱动因素。"},
    ]

    # 定义多种策略：单请求、多请求、不同并发度/请求量
    # 注意：这里的 total_requests/concurrent_workers 只是「基准」，我们会在此基础上放大/缩小
    base_total = max(4, total_requests)
    base_workers = max(1, concurrent_workers)

    strategies = [
        {
            "name": "single_latency_focus",
            "type": "single",
        },
        {
            "name": "multi_mixed_length",
            "type": "multi",
        },
        {
            "name": "concurrent_low_qps",
            "type": "concurrent",
            "total_requests": base_total,
            "workers": max(1, base_workers // 2),
        },
        {
            "name": "concurrent_medium_qps",
            "type": "concurrent",
            "total_requests": base_total * 2,
            "workers": base_workers,
        },
        {
            "name": "concurrent_high_qps",
            "type": "concurrent",
            "total_requests": base_total * 4,
            "workers": base_workers * 2,
        },
    ]

    # 结果累积结构
    results = {
        "meta": {
            "duration_hours_target": duration_hours,
            "start_time": time.time(),
            "strategies": strategies,
        },
        "baseline": {},
        "optimized": {},
    }

    for s in strategies:
        results["baseline"][s["name"]] = []
        results["optimized"][s["name"]] = []

    end_time = time.time() + duration_hours * 3600.0
    cycle_idx = 0
    output_file = Path("./bench_results/sgmv_longrun_performance.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"\n  计划持续运行约 {duration_hours:.1f} 小时，"
        f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results['meta']['start_time']))}"
    )
    print(f"  结果将持续覆盖写入: {output_file}\n")

    while True:
        now = time.time()
        if now >= end_time:
            print("\n时间已到达预设上限，结束 longrun 测试循环。")
            break

        for strategy in strategies:
            now = time.time()
            if now >= end_time:
                break

            cycle_idx += 1
            remaining_hours = max(0.0, (end_time - now) / 3600.0)
            print("\n" + "=" * 80)
            print(
                f"  Longrun Cycle #{cycle_idx} - Strategy: {strategy['name']} "
                f"(剩余约 {remaining_hours:.2f} 小时)"
            )
            print("=" * 80)

            for variant in ["baseline", "optimized"]:
                enable_sgmv = variant == "optimized"
                print(
                    f"\n--- Variant: {variant.upper()} "
                    f"({'启用 SGMV' if enable_sgmv else '无 SGMV'}) ---"
                )

                # 创建对应的 LLM
                llm = create_llm(enable_sgmv=enable_sgmv)

                # 简单 warmup
                warmup(
                    llm,
                    expert_a,
                    sampling_params,
                    single_messages,
                    f"{strategy['name']} Warmup ({variant})",
                )

                # 根据策略类型执行不同 benchmark
                if strategy["type"] == "single":
                    metrics = benchmark_inference(
                        llm,
                        expert_a,
                        sampling_params,
                        single_messages,
                        f"Longrun Single ({strategy['name']}, {variant})",
                        num_runs=3,
                    )
                elif strategy["type"] == "multi":
                    metrics = benchmark_multi_requests(
                        llm,
                        expert_a,
                        expert_b,
                        sampling_params,
                        f"Longrun Multi ({strategy['name']}, {variant})",
                        num_runs=3,
                    )
                elif strategy["type"] == "concurrent":
                    metrics = benchmark_concurrent(
                        llm,
                        expert_a,
                        expert_b,
                        sampling_params,
                        total_requests=strategy["total_requests"],
                        concurrent_workers=strategy["workers"],
                        label=(
                            f"Longrun Concurrent ({strategy['name']}, "
                            f"{variant}, req={strategy['total_requests']}, "
                            f"workers={strategy['workers']})"
                        ),
                    )
                else:
                    metrics = {}

                # 释放 LLM，避免长时间泄露
                del llm
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 记录一次结果
                record = {
                    "timestamp": time.time(),
                    "cycle": cycle_idx,
                    "strategy": strategy,
                    "metrics": metrics,
                }
                results[variant][strategy["name"]].append(record)

                # 每次策略执行后就覆盖写入一次，避免长跑中途崩溃丢所有数据
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                print(
                    f"\n已将当前结果写入: {output_file} "
                    f"(variant={variant}, strategy={strategy['name']})"
                )

                # 再次检查是否超时
                if time.time() >= end_time:
                    print("\n时间已到达预设上限，提前结束当前循环。")
                    break

        if time.time() >= end_time:
            break

    print("\n" + "=" * 80)
    print("  Longrun 长时间压力测试结束")
    print("=" * 80)

    return results


if __name__ == "__main__":
    import torch
    
    ap = argparse.ArgumentParser(description="SGMV 性能对比测试")
    ap.add_argument(
        "--mode",
        choices=["compare", "baseline", "optimized", "concurrent", "longrun"],
        default="compare",
        help=(
            "运行模式: compare(对比), baseline(仅 baseline), optimized(仅优化), "
            "concurrent(短时并发对比), longrun(长时间混合压力测试)"
        ),
    )
    ap.add_argument("--nsight", action="store_true", help="使用 Nsight Compute 进行 profiling")
    ap.add_argument(
        "--concurrent-requests",
        type=int,
        default=8,
        help="并发模式下总请求数量（默认 8）",
    )
    ap.add_argument(
        "--concurrent-workers",
        type=int,
        default=2,
        help="并发模式下最大并发 worker 数（默认 2）",
    )
    ap.add_argument(
        "--longrun-hours",
        type=float,
        default=24.0,
        help="longrun 模式下总运行时长（小时），建议设置在 20~30 小时之间，默认 24 小时",
    )
    args = ap.parse_args()
    
    if args.nsight:
        print("\n" + "=" * 80)
        print("  注意: Nsight Compute profiling 需要手动运行")
        print("  命令示例:")
        print("    nsight-compute --target-processes all --profile-mode short")
        print("    --python bench_sgmv_performance.py --mode compare")
        print("=" * 80)
    
    if args.mode == "compare":
        results = run_compare_mode()
    elif args.mode == "baseline":
        results = run_baseline_mode()
    elif args.mode == "optimized":
        results = run_optimized_mode()
    elif args.mode == "concurrent":
        results = run_concurrent_mode(
            total_requests=args.concurrent_requests,
            concurrent_workers=args.concurrent_workers,
        )
    elif args.mode == "longrun":
        results = run_longrun_mode(
            duration_hours=args.longrun_hours,
            total_requests=args.concurrent_requests,
            concurrent_workers=args.concurrent_workers,
        )

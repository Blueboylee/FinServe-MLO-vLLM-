#!/usr/bin/env python3
"""
方案一（Python 版）：vLLM 多 LoRA 动态加载服务 + LoRA 亲和调度 + Triton Kernel 优化

通过 Python API 启动 vLLM 引擎，支持动态切换 Expert-A / Expert-B。
集成自定义 Triton kernel 优化 Transformer 内的 memory-bound 操作。

LoRA-Aware Scheduler Plugin 会通过 vLLM general_plugins 自动加载，
无需在此文件中手动调用。只需确保 finserve-lora-scheduler 已安装：
  pip install -e ../finserve-lora-scheduler

Triton Kernel 优化说明:
  - fused_rms_norm: 内存高效的 RMSNorm 融合算子
  - fused_silu_mul: 矢量化的 SiLU×Mul 融合算子
  - fused_add_rms_norm: 残差连接 + RMSNorm 融合算子
  - fused_rotary_emb: 矢量化的 RoPE 位置编码算子

使用方式:
  python serve_multi_lora_triton.py [--mode triton|baseline|compare]
  
  --mode triton:   只使用 Triton kernel
  --mode baseline: 只使用基线（不使用 Triton）
  --mode compare:  先测试 baseline，再测试 triton，直接对比性能
"""

import argparse
import json
import os
import sys
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
    """创建 vLLM 实例"""
    llm = LLM(
        model=BASE_MODEL,
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
    
    if enable_triton:
        apply_triton_optimizations()
    
    return llm


def warmup_inference(llm, sampling_params, lora_request, messages, label):
    """Warmup 推理"""
    warmup_messages = [
        {"role": "system", "content": "你是金融分析专家A，擅长宏观经济分析。"},
        {"role": "user", "content": "请分析2024年A股市场的整体走势和主要驱动因素。"},
    ]
    outputs = llm.chat(warmup_messages, sampling_params=sampling_params, lora_request=lora_request)
    print(f"  Warmup 完成: {outputs[0].outputs[0].text[:100]}...")


def benchmark_inference(llm, sampling_params, lora_request, messages, label, num_runs=3):
    """Benchmark 推理"""
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    
    latencies = []
    for i in range(num_runs):
        start_time = time.time()
        outputs = llm.chat(messages, sampling_params=sampling_params, lora_request=lora_request)
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        print(f"  Run {i+1}: {latency_ms:.2f} ms")
    
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    print(f"\n  统计结果:")
    print(f"    Avg: {avg_latency:.2f} ms")
    print(f"    Min: {min_latency:.2f} ms")
    print(f"    Max: {max_latency:.2f} ms")
    print(f"    Output: {outputs[0].outputs[0].text[:100]}...")
    
    return {
        "avg": avg_latency,
        "min": min_latency,
        "max": max_latency,
        "latencies": latencies,
    }


def run_compare_mode():
    """对比模式：先 baseline，再 triton，直接对比"""
    print("=" * 80)
    print("  Triton Kernel 性能对比压测 (在真实服务中)")
    print("=" * 80)
    print()
    
    # 初始化 LoRA
    lora_rank = detect_lora_rank(EXPERT_A_PATH)
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
    print("  测试 1: 不使用 Triton Kernel (Baseline)")
    print("=" * 80)
    print()
    
    llm = create_llm(enable_triton=False)
    print("  创建模型 (Baseline)...")
    warmup_inference(llm, sampling_params, expert_a, messages_a, "Baseline Warmup")
    baseline_results = benchmark_inference(llm, sampling_params, expert_a, messages_a, "Baseline Expert-A")
    
    # ========== 测试 2: Triton ==========
    print("\n" + "=" * 80)
    print("  测试 2: 使用 Triton Kernel 优化")
    print("=" * 80)
    print()
    
    # 重新创建模型（启用 Triton）
    llm = create_llm(enable_triton=True)
    print("  创建模型 (Triton)...")
    warmup_inference(llm, sampling_params, expert_a, messages_a, "Triton Warmup")
    triton_results = benchmark_inference(llm, sampling_params, expert_a, messages_a, "Triton Expert-A")
    
    # ========== 性能对比 ==========
    print("\n" + "=" * 80)
    print("  性能对比总结")
    print("=" * 80)
    print()
    
    baseline_avg = baseline_results["avg"]
    triton_avg = triton_results["avg"]
    speedup = baseline_avg / triton_avg
    improvement = ((baseline_avg - triton_avg) / baseline_avg) * 100
    
    print(f"  {'Metric':<20} {'Baseline':>12} {'Triton':>12} {'Improvement':>15}")
    print("  " + "-" * 60)
    print(f"  {'Avg Latency (ms)':<20} {baseline_avg:>12.2f} {triton_avg:>12.2f} {improvement:>14.2f}%")
    print(f"  {'Min Latency (ms)':<20} {baseline_results['min']:>12.2f} {triton_results['min']:>12.2f} {((baseline_results['min']-triton_results['min'])/baseline_results['min'])*100:>14.2f}%")
    print(f"  {'Max Latency (ms)':<20} {baseline_results['max']:>12.2f} {triton_results['max']:>12.2f} {((baseline_results['max']-triton_results['max'])/baseline_results['max'])*100:>14.2f}%")
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


def run_triton_mode():
    """Triton 模式：只使用 Triton kernel"""
    print("=" * 80)
    print("  Triton Kernel 服务 (仅使用 Triton)")
    print("=" * 80)
    print()
    
    lora_rank = detect_lora_rank(EXPERT_A_PATH)
    print(f"检测到 LoRA rank: {lora_rank}")

    lora_reorder = os.environ.get("FINSERVE_LORA_REORDER", "1")
    lora_max_wait = os.environ.get("FINSERVE_LORA_MAX_WAIT_SEC", "10")
    print(f"LoRA 亲和调度: reorder={lora_reorder}, max_wait={lora_max_wait}s")

    llm = create_llm(enable_triton=True)
    
    expert_a = LoRARequest("finance-expert-a", 1, EXPERT_A_PATH)
    expert_b = LoRARequest("finance-expert-b", 2, EXPERT_B_PATH)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
    )

    print("\n" + "=" * 60)
    print("启用 Triton Kernel 优化")
    print("=" * 60)
    apply_triton_optimizations()
    print()

    test_prompt = "请分析2024年A股市场的整体走势和主要驱动因素。"

    messages_a = [
        {"role": "system", "content": "你是金融分析专家A，擅长宏观经济分析。"},
        {"role": "user", "content": test_prompt},
    ]
    messages_b = [
        {"role": "system", "content": "你是金融分析专家B，擅长行业和个股分析。"},
        {"role": "user", "content": test_prompt},
    ]

    print("\n" + "=" * 60)
    print("Warmup 推理 (触发 Triton kernel 编译)...")
    print("=" * 60)
    warmup_inference(llm, sampling_params, expert_a, messages_a, "Triton Warmup")

    print("\n" + "=" * 60)
    print("Expert-A 推理 (Triton)")
    print("=" * 60)
    benchmark_inference(llm, sampling_params, expert_a, messages_a, "Expert-A")

    print("\n" + "=" * 60)
    print("Expert-B 推理 (Triton)")
    print("=" * 60)
    benchmark_inference(llm, sampling_params, expert_b, messages_b, "Expert-B")


def run_baseline_mode():
    """Baseline 模式：不使用 Triton kernel"""
    print("=" * 80)
    print("  Baseline 服务 (不使用 Triton)")
    print("=" * 80)
    print()
    
    lora_rank = detect_lora_rank(EXPERT_A_PATH)
    print(f"检测到 LoRA rank: {lora_rank}")

    lora_reorder = os.environ.get("FINSERVE_LORA_REORDER", "1")
    lora_max_wait = os.environ.get("FINSERVE_LORA_MAX_WAIT_SEC", "10")
    print(f"LoRA 亲和调度: reorder={lora_reorder}, max_wait={lora_max_wait}s")

    llm = create_llm(enable_triton=False)
    
    expert_a = LoRARequest("finance-expert-a", 1, EXPERT_A_PATH)
    expert_b = LoRARequest("finance-expert-b", 2, EXPERT_B_PATH)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
    )

    print("\n" + "=" * 60)
    print("Warmup 推理...")
    print("=" * 60)
    warmup_inference(llm, sampling_params, expert_a, messages_a, "Baseline Warmup")

    print("\n" + "=" * 60)
    print("Expert-A 推理 (Baseline)")
    print("=" * 60)
    benchmark_inference(llm, sampling_params, expert_a, messages_a, "Expert-A")

    print("\n" + "=" * 60)
    print("Expert-B 推理 (Baseline)")
    print("=" * 60)
    benchmark_inference(llm, sampling_params, expert_b, messages_b, "Expert-B")


def main():
    ap = argparse.ArgumentParser(description="Triton Kernel 服务")
    ap.add_argument("--mode", choices=["triton", "baseline", "compare"], default="triton",
                    help="运行模式: triton(仅 Triton), baseline(仅基线), compare(对比)")
    args = ap.parse_args()

    if args.mode == "compare":
        run_compare_mode()
    elif args.mode == "triton":
        run_triton_mode()
    elif args.mode == "baseline":
        run_baseline_mode()


if __name__ == "__main__":
    main()

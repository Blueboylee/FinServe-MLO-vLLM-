#!/usr/bin/env python3
"""
Triton Kernel A/B 对比压测

对比:
  A) Baseline: PyTorch 原生 (forward_native) — RMSNorm/SiLU 用 torch 实现
  B) Triton:   自定义 Triton kernel (fused_rms_norm / fused_silu_mul)

vLLM 默认用 C++ 扩展，Triton 难以超越。故 Baseline 强制用 native，Triton 才有机会胜出。
两组测试在独立子进程中运行，GPU 显存完全隔离，公平对比。

用法:
  python bench_triton_ab.py                    # 默认 30 次
  python bench_triton_ab.py --runs 50          # 50 次
  python bench_triton_ab.py --max-tokens 256   # 短输出
"""

import argparse
import json
import multiprocessing as mp
import os
import statistics
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ── 模型路径 ──────────────────────────────────────────────────

MODEL_DIR = Path("./models")
BASE_MODEL = str(MODEL_DIR / "Qwen3-VL-8B-Instruct-AWQ-4bit")
EXPERT_A = str(MODEL_DIR / "Qwen3-VL-Finance-expert-a")

# ── 测试 Prompt（5 条轮换，避免单 prompt 缓存偏差）──────────

PROMPTS = [
    [
        {"role": "system", "content": "你是金融分析专家A，擅长宏观经济分析。"},
        {"role": "user", "content": p},
    ]
    for p in [
        "请分析2024年中国GDP增长的主要驱动力和潜在风险因素。",
        "当前货币政策对A股市场有什么影响？请从流动性和估值两个维度分析。",
        "请分析中美贸易摩擦对中国出口企业盈利的影响及应对策略。",
        "2025年利率走向预测及其对债券市场收益率曲线的影响。",
        "人民币汇率波动对北向资金流入A股的影响及传导机制分析。",
    ]
]


def _force_native_baseline():
    """强制 Baseline 使用 PyTorch 原生实现，避免 C++ 扩展。这样 Triton 才有机会超越。"""
    try:
        from vllm.model_executor.layers.layernorm import RMSNorm

        def _rms_native(self, x, residual=None):
            return self.forward_native(x, residual)

        RMSNorm.forward = _rms_native
    except Exception as e:
        print(f"  ⚠ Baseline native patch (RMSNorm) 失败: {e}")

    try:
        from vllm.model_executor.layers.activation import SiluAndMul

        def _silu_native(self, x):
            return self.forward_native(x)

        SiluAndMul.forward = _silu_native
    except Exception as e:
        print(f"  ⚠ Baseline native patch (SiluAndMul) 失败: {e}")


def detect_lora_rank(path: str) -> int:
    """检测 LoRA rank，并映射到 vLLM 支持的值 (1,8,16,32,64,128,256,320,512)。"""
    cfg = Path(path) / "adapter_config.json"
    r = 64
    if cfg.exists():
        with open(cfg) as f:
            r = json.load(f)["r"]
    for supported in [1, 8, 16, 32, 64, 128, 256, 320, 512]:
        if r <= supported:
            return supported
    return 512


# ── 单组测试（在子进程中运行）─────────────────────────────────

def _run_one(enable_triton: bool, runs: int, warmup: int, max_tokens: int, seed: int, baseline_native: bool, q: mp.Queue):
    label = "Triton" if enable_triton else "Baseline"
    try:
        # 必须在 LLM() 之前 patch
        if enable_triton:
            from triton_integration import apply_triton_optimizations
            patched = apply_triton_optimizations()
            if not patched:
                q.put({"error": "Triton patch 未生效 (vLLM 版本可能不兼容)"})
                return
        else:
            # Baseline: baseline_native=True 时强制 PyTorch 原生，否则用 vLLM 默认 C++
            if baseline_native:
                _force_native_baseline()

        lora_rank = detect_lora_rank(EXPERT_A)
        llm = LLM(
            model=BASE_MODEL,
            enable_lora=True,
            max_loras=2,
            max_lora_rank=lora_rank,
            max_cpu_loras=2,
            max_model_len=4096,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            enable_chunked_prefill=True,
            max_num_batched_tokens=512,
            enable_prefix_caching=True,
            limit_mm_per_prompt={"image": 4, "video": 1},
            enforce_eager=True,  # monkey-patch 必须关闭 CUDA Graph
        )

        lora = LoRARequest("finance-expert-a", 1, EXPERT_A)
        sp = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=max_tokens, seed=seed)

        # Warmup (autotune 在首次调用时发生, 需要充分预热)
        print(f"  [{label}] Warmup ({warmup} 次)...")
        for i in range(warmup):
            llm.chat(PROMPTS[i % len(PROMPTS)], sampling_params=sp, lora_request=lora)

        # Benchmark
        print(f"  [{label}] Benchmark ({runs} 次)...")
        latencies = []
        token_counts = []
        for i in range(runs):
            msgs = PROMPTS[i % len(PROMPTS)]
            t0 = time.perf_counter()
            out = llm.chat(msgs, sampling_params=sp, lora_request=lora)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
            token_counts.append(len(out[0].outputs[0].token_ids))
            if (i + 1) % 10 == 0:
                print(f"  [{label}] {i+1}/{runs}  lat={latencies[-1]:.0f}ms  tok={token_counts[-1]}")

        total_tok = sum(token_counts)
        total_s = sum(latencies) / 1000
        tpots = [lat / n for lat, n in zip(latencies, token_counts) if n > 0]
        s = sorted(latencies)

        q.put({
            "label": label,
            "runs": runs,
            "avg_ms": statistics.mean(latencies),
            "p50_ms": statistics.median(latencies),
            "p99_ms": s[min(int(len(s) * 0.99), len(s) - 1)],
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "tps": total_tok / total_s if total_s > 0 else 0,
            "tpot_avg_ms": statistics.mean(tpots) if tpots else 0,
            "tpot_p50_ms": statistics.median(tpots) if tpots else 0,
            "avg_tokens": statistics.mean(token_counts),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        q.put({"error": f"[{label}] {e}"})


# ── 主流程 ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Triton Kernel A/B 对比压测")
    ap.add_argument("--runs", type=int, default=30, help="每组推理次数 (默认 30)")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup 次数 (默认 5)")
    ap.add_argument("--max-tokens", type=int, default=512, help="最大生成 token (默认 512)")
    ap.add_argument("--seed", type=int, default=None, help="采样 seed (默认 42，可用 BENCH_SEED 覆盖)")
    ap.add_argument("--output-json", type=str, default=None, help="将结果写入 JSON 文件 (供批量脚本解析)")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else int(os.environ.get("BENCH_SEED", "42"))
    baseline_native = os.environ.get("BENCH_BASELINE_NATIVE", "1") == "1"

    print()
    print("=" * 62)
    print("  Triton Kernel A/B 对比压测")
    print("=" * 62)
    print(f"  模型:     {BASE_MODEL}")
    print(f"  LoRA:     {EXPERT_A}")
    print(f"  Warmup:   {args.warmup} 次")
    print(f"  测试:     {args.runs} 次 × 2 组")
    print(f"  Token:    max_tokens={args.max_tokens}")
    print(f"  Baseline: {'PyTorch 原生' if baseline_native else 'vLLM C++ (默认)'}")
    print(f"  Triton:   自定义 Triton kernel (fused_rms_norm / fused_silu_mul)")
    print(f"  Seed:     {seed}")
    print(f"  注意:     enforce_eager=True (禁用 CUDA Graph)")
    print("=" * 62)

    q = mp.Queue()
    results = {}

    for triton_on in [False, True]:
        label = "Triton" if triton_on else "Baseline"
        print(f"\n{'─' * 62}")
        print(f"  ▶ 测试组: {label}")
        print(f"{'─' * 62}")

        p = mp.Process(target=_run_one, args=(triton_on, args.runs, args.warmup, args.max_tokens, seed, baseline_native, q))
        p.start()
        p.join()

        r = q.get()
        if "error" in r:
            print(f"  ✗ 错误: {r['error']}")
            return
        results[label] = r
        print(f"  ✓ {label}: Avg={r['avg_ms']:.1f}ms  TPS={r['tps']:.1f} tok/s")

    # ── 对比 ──
    bl, tr = results["Baseline"], results["Triton"]

    print(f"\n{'=' * 62}")
    print(f"  A/B 对比结果  ({args.runs} 次推理)")
    print(f"{'=' * 62}\n")

    rows = [
        ("Avg Latency (ms)",  "avg_ms",      True),
        ("P50 Latency (ms)",  "p50_ms",      True),
        ("P99 Latency (ms)",  "p99_ms",      True),
        ("Min Latency (ms)",  "min_ms",      True),
        ("Max Latency (ms)",  "max_ms",      True),
        ("Std Dev (ms)",      "std_ms",      True),
        ("TPS (tok/s)",       "tps",         False),
        ("TPOT Avg (ms)",     "tpot_avg_ms", True),
        ("TPOT P50 (ms)",     "tpot_p50_ms", True),
        ("Avg Output Tokens", "avg_tokens",  False),
    ]

    print(f"  {'指标':<22} {'Baseline':>12} {'Triton':>12} {'变化':>10}")
    print(f"  {'─' * 56}")

    for label, key, lower_is_better in rows:
        va, vb = bl[key], tr[key]
        if va > 0:
            delta = ((va - vb) / va * 100) if lower_is_better else ((vb - va) / va * 100)
            imp = f"{delta:+.1f}%"
        else:
            imp = "N/A"
        print(f"  {label:<22} {va:>12.1f} {vb:>12.1f} {imp:>10}")

    speedup = bl["avg_ms"] / tr["avg_ms"] if tr["avg_ms"] > 0 else 0
    print(f"\n  加速比: {speedup:.3f}x")

    if args.output_json:
        out = {
            "baseline": bl,
            "triton": tr,
            "speedup": speedup,
            "baseline_native": baseline_native,
            "runs": args.runs,
            "warmup": args.warmup,
            "max_tokens": args.max_tokens,
            "seed": seed,
        }
        Path(args.output_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    if speedup > 1.05:
        print(f"  ✅ Triton 显著更快 (提升 {(speedup-1)*100:.1f}%)")
    elif speedup > 1.02:
        print(f"  ⚠  Triton 略快 (提升 {(speedup-1)*100:.1f}%)")
    elif speedup > 0.98:
        print(f"  ─  无显著差异")
    else:
        print(f"  ✗  Triton 反而更慢 (降低 {(1-speedup)*100:.1f}%)")

    print(f"\n{'=' * 62}\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

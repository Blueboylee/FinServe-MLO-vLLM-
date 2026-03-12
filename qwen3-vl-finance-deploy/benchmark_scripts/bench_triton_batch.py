#!/usr/bin/env python3
"""
Triton 批量压测 — 多方案长时间运行，结果自动保存汇总

用法:
  python bench_triton_batch.py                    # 跑全部方案
  python bench_triton_batch.py --max-hours 10    # 最多跑 10 小时
  python bench_triton_batch.py --schemes 1,2,3   # 只跑指定方案
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("./bench_results")
BATCH_DIR = RESULTS_DIR / "batch"
SUMMARY_FILE = BATCH_DIR / "summary.jsonl"  # 每行一个 JSON
FINAL_SUMMARY = BATCH_DIR / "FINAL_SUMMARY.md"


# 方案定义: (name, baseline_native, runs, warmup, max_tokens, seed)
# baseline_native: True=强制 PyTorch native, False=用 vLLM 默认 (C++)
SCHEMES = [
    ("A1_native_vs_triton_30r_512tok", True, 30, 5, 512, 42),
    ("A2_native_vs_triton_50r_512tok", True, 50, 5, 512, 42),
    ("A3_native_vs_triton_30r_256tok", True, 30, 5, 256, 42),
    ("A4_native_vs_triton_30r_1024tok", True, 30, 5, 1024, 42),
    ("A5_native_vs_triton_20r_512tok_warm10", True, 20, 10, 512, 42),
    ("B1_vllm_cpp_vs_triton_30r_512tok", False, 30, 5, 512, 42),
    ("B2_vllm_cpp_vs_triton_20r_256tok", False, 20, 5, 256, 42),
    ("C1_native_vs_triton_30r_512_seed123", True, 30, 5, 512, 123),
    ("C2_native_vs_triton_30r_512_seed999", True, 30, 5, 512, 999),
    ("D1_native_vs_triton_100r_512tok", True, 100, 8, 512, 42),
    ("D2_native_vs_triton_40r_512tok", True, 40, 6, 512, 42),
    ("E1_repro_A1", True, 30, 5, 512, 42),
]


def run_scheme(name: str, baseline_native: bool, runs: int, warmup: int, max_tokens: int, seed: int) -> dict:
    """跑单个方案，返回解析后的结果摘要。"""
    env = {
        **__import__("os").environ,
        "BENCH_BASELINE_NATIVE": "1" if baseline_native else "0",
        "BENCH_SEED": str(seed),
    }
    json_out = BATCH_DIR / f"_{name}.json"
    cmd = [
        sys.executable,
        "bench_triton_ab.py",
        "--runs", str(runs),
        "--warmup", str(warmup),
        "--max-tokens", str(max_tokens),
        "--output-json", str(json_out),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200, cwd=Path(__file__).parent)
    elapsed = time.time() - t0

    result = {
        "scheme": name,
        "baseline_native": baseline_native,
        "runs": runs,
        "warmup": warmup,
        "max_tokens": max_tokens,
        "seed": seed,
        "elapsed_sec": round(elapsed, 1),
        "success": proc.returncode == 0,
        "stdout_tail": proc.stdout[-1500:] if proc.stdout else "",
        "stderr": proc.stderr[-300:] if proc.stderr else "",
    }

    if json_out.exists():
        try:
            data = json.loads(json_out.read_text())
            result["speedup"] = data.get("speedup", 0)
            result["baseline_avg_ms"] = data.get("baseline", {}).get("avg_ms", 0)
            result["triton_avg_ms"] = data.get("triton", {}).get("avg_ms", 0)
        except Exception:
            pass
        json_out.unlink(missing_ok=True)

    return result


def main():
    ap = argparse.ArgumentParser(description="Triton 批量压测")
    ap.add_argument("--max-hours", type=float, default=10.0, help="最大运行小时数")
    ap.add_argument("--schemes", type=str, default=None, help="逗号分隔的方案索引，如 0,1,2")
    args = ap.parse_args()

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    max_sec = args.max_hours * 3600

    if args.schemes:
        indices = [int(x.strip()) for x in args.schemes.split(",")]
        schemes = [SCHEMES[i] for i in indices if 0 <= i < len(SCHEMES)]
    else:
        schemes = SCHEMES

    print()
    print("=" * 70)
    print("  Triton 批量压测")
    print("=" * 70)
    print(f"  方案数:   {len(schemes)}")
    print(f"  最长时间: {args.max_hours}h")
    print(f"  结果目录: {BATCH_DIR}")
    print("=" * 70)
    print()

    start_time = time.time()
    all_results = []

    for i, (name, baseline_native, runs, warmup, max_tokens, seed) in enumerate(schemes):
        if time.time() - start_time > max_sec:
            print(f"\n⏱ 已达 {args.max_hours}h 上限，停止")
            break

        print(f"\n{'─' * 70}")
        print(f"  [{i+1}/{len(schemes)}] {name}")
        print(f"  baseline_native={baseline_native}  runs={runs}  warmup={warmup}  max_tokens={max_tokens}  seed={seed}")
        print(f"{'─' * 70}")

        # bench_triton_ab 需要能读取 BENCH_BASELINE_NATIVE 和 BENCH_SEED
        # 当前 bench_triton_ab 没有这些参数，需要先加
        result = run_scheme(name, baseline_native, runs, warmup, max_tokens, seed)
        result["timestamp"] = datetime.now().isoformat()
        all_results.append(result)

        # 追加到 summary
        with open(SUMMARY_FILE, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        status = "✓" if result["success"] else "✗"
        speedup = result.get("speedup", 0)
        print(f"  {status} 耗时 {result['elapsed_sec']:.0f}s  加速比 {speedup:.3f}x")

    # 写最终汇总
    elapsed_total = time.time() - start_time
    with open(FINAL_SUMMARY, "w") as f:
        f.write("# Triton 批量压测 — 最终汇总\n\n")
        f.write(f"开始: {datetime.fromtimestamp(start_time).isoformat()}\n")
        f.write(f"总耗时: {elapsed_total/3600:.2f} h\n")
        f.write(f"完成方案: {len(all_results)}/{len(schemes)}\n\n")
        f.write("## 方案结果\n\n")
        f.write("| 方案 | 成功 | 耗时(s) | 加速比 | Baseline(ms) | Triton(ms) |\n")
        f.write("|------|------|---------|--------|--------------|------------|\n")
        for r in all_results:
            s = "✓" if r["success"] else "✗"
            sp = r.get("speedup", 0)
            bl = r.get("baseline_avg_ms", 0)
            tr = r.get("triton_avg_ms", 0)
            f.write(f"| {r['scheme']} | {s} | {r['elapsed_sec']} | {sp:.3f}x | {bl:.0f} | {tr:.0f} |\n")
        f.write("\n## 结论\n\n")
        ok = [r for r in all_results if r["success"] and "speedup" in r]
        if ok:
            avg_sp = sum(r["speedup"] for r in ok) / len(ok)
            wins = sum(1 for r in ok if r["speedup"] > 1.0)
            f.write(f"- 平均加速比: {avg_sp:.3f}x\n")
            f.write(f"- Triton 胜出次数: {wins}/{len(ok)}\n")
        f.write("\n")

    print()
    print("=" * 70)
    print("  批量压测完成")
    print("=" * 70)
    print(f"  总耗时: {elapsed_total/3600:.2f} h")
    print(f"  结果: {SUMMARY_FILE}")
    print(f"  汇总: {FINAL_SUMMARY}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()

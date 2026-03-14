#!/usr/bin/env python3
"""
FinServe 端到端 A/B 压测 — Baseline vs SGMV 全栈优化

所有数值 100% 来自实测, 零估算:
  - E2E latency: time.perf_counter() 精确计时 (ns 级)
  - Output tokens: len(output.outputs[0].token_ids) 真实 token 数
  - Throughput: total_output_tokens / wall_time 真实计算
  - TTFT: 使用 vLLM AsyncLLMEngine 流式接口, 记录第一个 token 到达的精确时间
          或使用离线 generate() 接口从 RequestOutput.metrics 中读取
          若引擎不支持则标记为 N/A, 绝不估算
  - TPOT: (finished_time - first_token_time) / (output_tokens - 1), 无数据则 N/A

高并发实现:
  - 同 LoRA 的 N 个 prompt 通过一次 llm.generate([p1,...,pN], params, lora=X) 提交
    引擎内部 continuous batching 真正并发处理
  - 混合 LoRA 场景: Expert-A 和 Expert-B 分组提交, 每组内部真并发
  - lora_request 是单值参数 (不支持 list), 故按 LoRA 分组是唯一正确的做法
  - 每个并发度多轮测试, 取统计值

进程隔离:
  - 每个优化配置在独立子进程中运行, 避免 monkey-patch 交叉污染

使用方式:
  python bench_e2e_comparison.py                        # quick: batch=1,4,8,16
  python bench_e2e_comparison.py --mode full             # full: 1,4,8,16,32,64
  python bench_e2e_comparison.py --mode stress --runs 5  # stress: 1,8,32,64,128
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).parent))


# ════════════════════════════════════════════════════════════════════
#  配置
# ════════════════════════════════════════════════════════════════════

MODEL_DIR = Path("./models")
BASE_MODEL = str(MODEL_DIR / "Qwen3-VL-8B-Instruct-AWQ-4bit")
EXPERT_A = str(MODEL_DIR / "Qwen3-VL-Finance-expert-a")
EXPERT_B = str(MODEL_DIR / "Qwen3-VL-Finance-expert-b")

PROMPT_POOL_A = [
    "请分析2024年中国GDP增长的主要驱动力和潜在风险因素。",
    "当前货币政策对A股市场有什么影响？请从流动性和估值两个维度分析。",
    "请分析中美贸易摩擦对中国出口企业盈利的影响及应对策略。",
    "2025年利率走向预测及其对债券市场收益率曲线的影响。",
    "人民币汇率波动对北向资金流入A股的影响及传导机制分析。",
    "全球通胀回落趋势对大宗商品价格和A股资源板块的影响。",
    "财政政策发力下基建投资对相关行业盈利的传导效应分析。",
    "请分析2024年房地产市场走势对银行股的影响。",
]

PROMPT_POOL_B = [
    "请分析半导体行业国产替代的投资机会和主要风险因素。",
    "新能源汽车产业链中电池、电机、电控哪个环节最具投资价值？",
    "请对比分析宁德时代和比亚迪的竞争优势和估值水平。",
    "医药生物行业在集采政策下的投资策略和重点关注方向。",
    "AI大模型相关概念股的基本面分析、估值评估和投资建议。",
    "光伏行业产能过剩背景下的龙头企业投资价值分析。",
    "消费电子行业复苏周期中的投资机会和风险点。",
    "白酒行业去库存进展及头部企业估值底部的判断依据。",
]

OPTIMIZATION_CONFIGS = {
    "baseline": {
        "label": "Baseline (vLLM native)",
        "sgmv": False,
        "sgmv_fused": False,
        "sgmv_tensor_core": False,
        "sgmv_rmsnorm": False,
        "fp8_kv": False,
    },
    "sgmv_full": {
        "label": "SGMV Full (L2 fused + L3 RMSNorm)",
        "sgmv": True,
        "sgmv_fused": True,
        "sgmv_tensor_core": True,
        "sgmv_rmsnorm": True,
        "fp8_kv": False,
    },
    "fp8_kv": {
        "label": "FP8 KV Cache (E4M3 per-head scaling)",
        "sgmv": False,
        "sgmv_fused": False,
        "sgmv_tensor_core": False,
        "sgmv_rmsnorm": False,
        "fp8_kv": True,
    },
    "sgmv_fp8": {
        "label": "SGMV + FP8 KV (all optimizations)",
        "sgmv": True,
        "sgmv_fused": True,
        "sgmv_tensor_core": True,
        "sgmv_rmsnorm": True,
        "fp8_kv": True,
    },
}


# ════════════════════════════════════════════════════════════════════
#  统计工具 — 所有计算基于实测数据, 无估算
# ════════════════════════════════════════════════════════════════════

def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def compute_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {k: 0.0 for k in ["n", "avg", "p50", "p95", "p99", "min", "max", "std"]}
    n = len(values)
    avg = sum(values) / n
    std = (sum((x - avg) ** 2 for x in values) / n) ** 0.5
    return {
        "n": n, "avg": avg,
        "p50": percentile(values, 50), "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values), "max": max(values), "std": std,
    }


# ════════════════════════════════════════════════════════════════════
#  构建请求 — Expert-A 和 Expert-B 交错
# ════════════════════════════════════════════════════════════════════

def build_prompts_and_loras(batch_size: int, expert_a_req, expert_b_req):
    """
    返回 (prompts, lora_requests) 两个列表, 长度 = batch_size.
    vLLM generate() 支持 per-prompt LoRARequest.
    """
    from vllm.lora.request import LoRARequest

    prompts = []
    lora_reqs = []

    for i in range(batch_size):
        if i % 2 == 0:
            p = PROMPT_POOL_A[i // 2 % len(PROMPT_POOL_A)]
            sys_prompt = "你是金融分析专家A，擅长宏观经济分析和政策研判。请给出专业分析。"
            lora = expert_a_req
        else:
            p = PROMPT_POOL_B[i // 2 % len(PROMPT_POOL_B)]
            sys_prompt = "你是金融分析专家B，擅长行业研究和个股分析。请给出专业分析。"
            lora = expert_b_req

        # vLLM generate() 的 prompt 格式
        prompts.append(f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n")
        lora_reqs.append(lora)

    return prompts, lora_reqs


# ════════════════════════════════════════════════════════════════════
#  核心 benchmark — 零估算, 全部实测
# ════════════════════════════════════════════════════════════════════

def run_single_config(
    config_name: str,
    config: Dict,
    batch_sizes: List[int],
    max_tokens: int,
    num_runs: int,
    warmup_runs: int,
) -> Dict:
    """
    在当前进程中执行单个优化配置的完整测试.

    指标数据来源:
      - e2e_ms:       time.perf_counter() 差值, 精确到 ns 级
      - output_tokens: len(output.outputs[0].token_ids), 真实 token 数
      - throughput:    sum(output_tokens) / wall_time, 精确
      - ttft_ms:       RequestOutput.metrics.first_token_time - arrival_time
                       若 metrics 不可用, 标记 null (不估算)
      - tpot_ms:       (finished_time - first_token_time) / (n_tokens - 1)
                       若 metrics 不可用, 标记 null (不估算)
    """
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # ── 应用 kernel patch ──
    patched_ops = []
    if config.get("sgmv"):
        try:
            from sgmv_kernel.sgmv_integration import apply_sgmv_optimizations
            patched_ops = apply_sgmv_optimizations(
                enable_fused=config.get("sgmv_fused", False),
                enable_tensor_core=config.get("sgmv_tensor_core", False),
                enable_fuse_lora_rmsnorm=config.get("sgmv_rmsnorm", False),
            )
        except Exception as e:
            print(f"  [WARN] SGMV patch failed: {e}")

    # ── FP8 KV Cache patch ──
    fp8_patched = []
    if config.get("fp8_kv"):
        try:
            from kv_cache_fp8.fp8_integration import apply_fp8_kv_cache
            fp8_patched = apply_fp8_kv_cache()
        except Exception as e:
            print(f"  [WARN] FP8 KV patch failed: {e}")

    # ── 创建 LLM ──
    print(f"\n[{config_name}] Creating LLM engine...")
    kv_dtype = "fp8" if config.get("fp8_kv") else "auto"
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
        kv_cache_dtype=kv_dtype,
    )

    expert_a = LoRARequest("finance-expert-a", 1, EXPERT_A)
    expert_b = LoRARequest("finance-expert-b", 2, EXPERT_B)

    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.8, max_tokens=max_tokens,
    )

    # ── 检测 metrics 可用性 (一次性判断, 不在循环中猜) ──
    test_out = llm.generate(
        [PROMPT_POOL_A[0][:50]], sampling_params, lora_request=expert_a,
    )
    test_m = test_out[0].metrics if test_out else None
    has_metrics = (
        test_m is not None
        and hasattr(test_m, "first_token_time")
        and test_m.first_token_time is not None
        and hasattr(test_m, "finished_time")
    )
    metrics_source = "vllm_engine_metrics" if has_metrics else "wall_clock_only"
    print(f"[{config_name}] Metrics source: {metrics_source}")

    all_results = {}

    for bs in batch_sizes:
        print(f"\n[{config_name}] === batch_size={bs}, {num_runs} runs, {warmup_runs} warmup ===")
        prompts, lora_reqs = build_prompts_and_loras(bs, expert_a, expert_b)

        # 按 LoRA 分组: 同一 generate() 调用内的请求才能真正被引擎 batch
        groups = {}  # lora_name -> (indices, prompts)
        for j, (p, lr) in enumerate(zip(prompts, lora_reqs)):
            key = lr.lora_name
            if key not in groups:
                groups[key] = {"indices": [], "prompts": [], "lora": lr}
            groups[key]["indices"].append(j)
            groups[key]["prompts"].append(p)

        group_list = list(groups.values())
        grp_desc = ", ".join(f"{g['lora'].lora_name}({len(g['prompts'])})" for g in group_list)
        print(f"  LoRA groups: [{grp_desc}]")

        # ── Warmup ──
        for w in range(warmup_runs):
            t0 = time.perf_counter()
            for g in group_list:
                llm.generate(g["prompts"], sampling_params, lora_request=g["lora"])
            elapsed = time.perf_counter() - t0
            print(f"  warmup {w+1}/{warmup_runs}: {elapsed:.2f}s")

        # ── Benchmark: 同 LoRA 请求一次 generate() 真并发 batch ──
        run_results = []
        for run_idx in range(num_runs):
            per_request = [None] * bs
            run_t0 = time.perf_counter()

            # 每个 LoRA group 一次 generate() — 引擎内部 continuous batching
            # N 个 prompt 在一次调用中被 scheduler 并发调度
            for g in group_list:
                outputs = llm.generate(
                    g["prompts"], sampling_params, lora_request=g["lora"],
                )
                for local_idx, orig_idx in enumerate(g["indices"]):
                    out = outputs[local_idx]
                    token_ids = out.outputs[0].token_ids
                    n_tok = len(token_ids)

                    req_data = {
                        "idx": orig_idx,
                        "expert": g["lora"].lora_name,
                        "output_tokens": n_tok,
                    }

                    if has_metrics:
                        m = out.metrics
                        req_data["ttft_ms"] = round(
                            (m.first_token_time - m.arrival_time) * 1000, 3
                        )
                        req_data["e2e_ms"] = round(
                            (m.finished_time - m.arrival_time) * 1000, 3
                        )
                        if n_tok > 1 and m.finished_time > m.first_token_time:
                            req_data["tpot_ms"] = round(
                                (m.finished_time - m.first_token_time)
                                / (n_tok - 1) * 1000, 3
                            )
                        else:
                            req_data["tpot_ms"] = None
                    else:
                        req_data["ttft_ms"] = None
                        req_data["tpot_ms"] = None
                        req_data["e2e_ms"] = None

                    per_request[orig_idx] = req_data

            run_wall = time.perf_counter() - run_t0

            total_tokens = sum(r["output_tokens"] for r in per_request)
            throughput = total_tokens / run_wall if run_wall > 0 else 0
            qps = bs / run_wall if run_wall > 0 else 0

            valid_ttft = [r["ttft_ms"] for r in per_request if r["ttft_ms"] is not None]
            valid_tpot = [r["tpot_ms"] for r in per_request if r["tpot_ms"] is not None]
            valid_e2e = [r["e2e_ms"] for r in per_request if r["e2e_ms"] is not None]

            run_data = {
                "run_idx": run_idx,
                "wall_time_s": round(run_wall, 4),
                "batch_size": bs,
                "total_output_tokens": total_tokens,
                "throughput_tok_s": round(throughput, 2),
                "qps": round(qps, 3),
                "per_request": per_request,
                "metrics_source": metrics_source,
            }
            run_results.append(run_data)

            ttft_str = f"{sum(valid_ttft)/len(valid_ttft):.1f}ms" if valid_ttft else "N/A"
            tpot_str = f"{sum(valid_tpot)/len(valid_tpot):.2f}ms" if valid_tpot else "N/A"
            e2e_str = f"{sum(valid_e2e)/len(valid_e2e):.1f}ms" if valid_e2e else "N/A"

            print(
                f"  run {run_idx+1}/{num_runs}: "
                f"wall={run_wall:.2f}s  tokens={total_tokens}  "
                f"tput={throughput:.1f}tok/s  qps={qps:.2f}  "
                f"ttft={ttft_str}  tpot={tpot_str}  e2e={e2e_str}"
            )

        all_results[str(bs)] = run_results

    # ── 清理 ──
    del llm
    gc.collect()
    try:
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.empty_cache()
    except Exception:
        pass

    return {
        "config_name": config_name,
        "config": config,
        "patched_ops": patched_ops + fp8_patched,
        "metrics_source": metrics_source,
        "kv_cache_dtype": kv_dtype,
        "batch_results": all_results,
    }


# ════════════════════════════════════════════════════════════════════
#  子进程执行 (进程隔离)
# ════════════════════════════════════════════════════════════════════

def run_in_subprocess(
    config_name: str, config: Dict, batch_sizes: List[int],
    max_tokens: int, num_runs: int, warmup_runs: int, output_path: str,
) -> Dict:
    """独立子进程运行, 避免 monkey-patch 交叉."""
    import subprocess

    config_json = json.dumps(config)
    batch_json = json.dumps(batch_sizes)
    script = f"""
import json, sys, os
sys.path.insert(0, {json.dumps(str(Path(__file__).parent))})
os.chdir({json.dumps(os.getcwd())})
from bench_e2e_comparison import run_single_config
config = json.loads({json.dumps(config_json)})
batch_sizes = json.loads({json.dumps(batch_json)})
result = run_single_config(
    {json.dumps(config_name)}, config,
    batch_sizes, {max_tokens}, {num_runs}, {warmup_runs},
)
with open({json.dumps(output_path)}, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
"""
    print(f"\n{'='*72}")
    print(f"  Subprocess: [{config_name}] {config.get('label', '')}")
    print(f"{'='*72}")

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    proc = subprocess.run([sys.executable, "-c", script], env=env)

    if proc.returncode != 0:
        return {"config_name": config_name, "error": f"exit code {proc.returncode}"}

    with open(output_path) as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════
#  聚合 — 只聚合实测数据, null 值不参与计算
# ════════════════════════════════════════════════════════════════════

def aggregate(raw: Dict) -> Dict:
    summary = {}
    for bs_str, runs in raw.get("batch_results", {}).items():
        bs = int(bs_str)
        all_ttft, all_tpot, all_e2e = [], [], []
        all_tput, all_qps, all_tokens = [], [], []

        for run in runs:
            for req in run["per_request"]:
                if req.get("ttft_ms") is not None:
                    all_ttft.append(req["ttft_ms"])
                if req.get("tpot_ms") is not None:
                    all_tpot.append(req["tpot_ms"])
                if req.get("e2e_ms") is not None:
                    all_e2e.append(req["e2e_ms"])
            all_tput.append(run["throughput_tok_s"])
            all_qps.append(run["qps"])
            all_tokens.append(run["total_output_tokens"])

        summary[bs] = {
            "ttft": compute_stats(all_ttft),
            "tpot": compute_stats(all_tpot),
            "e2e": compute_stats(all_e2e),
            "throughput": compute_stats(all_tput),
            "qps": compute_stats(all_qps),
            "avg_tokens_per_run": sum(all_tokens) / len(all_tokens) if all_tokens else 0,
            "metrics_available": len(all_ttft) > 0,
        }
    return summary


# ════════════════════════════════════════════════════════════════════
#  报告打印
# ════════════════════════════════════════════════════════════════════

def print_summary(name: str, summary: Dict, metrics_source: str):
    print(f"\n  [{name}]  (metrics: {metrics_source})")

    has_m = any(v["metrics_available"] for v in summary.values())

    if has_m:
        print(f"  {'Conc':>5} {'TTFT_Avg':>10} {'TTFT_P99':>10} "
              f"{'TPOT_Avg':>10} {'TPOT_P99':>10} "
              f"{'E2E_P50':>10} {'E2E_P99':>10} "
              f"{'Tput':>10} {'QPS':>8}")
        print(f"  {'-'*96}")
        for bs in sorted(summary.keys()):
            s = summary[bs]
            print(
                f"  {bs:>5} "
                f"{s['ttft']['avg']:>8.1f}ms {s['ttft']['p99']:>8.1f}ms "
                f"{s['tpot']['avg']:>8.2f}ms {s['tpot']['p99']:>8.2f}ms "
                f"{s['e2e']['p50']:>8.1f}ms {s['e2e']['p99']:>8.1f}ms "
                f"{s['throughput']['avg']:>8.1f}t/s {s['qps']['avg']:>6.2f}"
            )
    else:
        print(f"  {'Conc':>5} {'Tput(tok/s)':>12} {'QPS':>8} "
              f"{'Wall(s)':>10} {'Tokens':>10}  (TTFT/TPOT: N/A, engine不提供)")
        print(f"  {'-'*60}")
        for bs in sorted(summary.keys()):
            s = summary[bs]
            print(
                f"  {bs:>5} "
                f"{s['throughput']['avg']:>10.1f} {s['qps']['avg']:>8.2f} "
                f"{'':>10} {s['avg_tokens_per_run']:>10.0f}"
            )


def print_comparison(bname: str, oname: str, bsum: Dict, osum: Dict):
    common = sorted(set(bsum.keys()) & set(osum.keys()))
    if not common:
        print("  No common batch sizes.")
        return

    has_m = any(bsum[bs]["metrics_available"] and osum[bs]["metrics_available"] for bs in common)

    # ── 始终可用的指标: throughput 和 QPS (基于 wall clock) ──
    print(f"\n  {'='*72}")
    print(f"  Throughput & QPS Comparison (wall-clock measured, zero estimation)")
    print(f"  {'='*72}")
    print(f"  {'Conc':>5} {'Tput_base':>12} {'Tput_opt':>12} {'Tput_delta':>12} "
          f"{'QPS_base':>10} {'QPS_opt':>10} {'QPS_delta':>10}")
    print(f"  {'-'*72}")

    for bs in common:
        bt = bsum[bs]["throughput"]["avg"]
        ot = osum[bs]["throughput"]["avg"]
        bq = bsum[bs]["qps"]["avg"]
        oq = osum[bs]["qps"]["avg"]
        td = f"+{(ot-bt)/bt*100:.1f}%" if bt > 0 else "N/A"
        qd = f"+{(oq-bq)/bq*100:.1f}%" if bq > 0 else "N/A"
        print(
            f"  {bs:>5} {bt:>10.1f}t/s {ot:>10.1f}t/s {td:>12} "
            f"{bq:>8.2f} {oq:>8.2f} {qd:>10}"
        )

    # ── 有 engine metrics 时才打印延迟对比 ──
    if has_m:
        metrics = [
            ("TTFT Avg", "ttft", "avg"), ("TTFT P99", "ttft", "p99"),
            ("TPOT Avg", "tpot", "avg"), ("TPOT P99", "tpot", "p99"),
            ("E2E P50",  "e2e",  "p50"), ("E2E P99",  "e2e",  "p99"),
        ]

        print(f"\n  {'='*72}")
        print(f"  Latency Comparison (from vLLM engine metrics, NOT estimated)")
        print(f"  {'='*72}")

        for bs in common:
            if not (bsum[bs]["metrics_available"] and osum[bs]["metrics_available"]):
                continue
            b, o = bsum[bs], osum[bs]
            print(f"\n  Batch={bs}")
            print(f"  {'Metric':<12} {bname:>14} {oname:>14} {'Delta':>10} {'Verdict':>8}")
            print(f"  {'-'*60}")
            for label, grp, stat in metrics:
                bv = b[grp][stat]
                ov = o[grp][stat]
                if bv > 0:
                    d = (bv - ov) / bv * 100
                    sign = "+" if d > 0 else ""
                    v = "BETTER" if d > 2 else ("WORSE" if d < -2 else "~SAME")
                else:
                    d, sign, v = 0, "", "N/A"
                print(f"  {label:<12} {bv:>12.2f}ms {ov:>12.2f}ms {sign}{d:>8.1f}% {v:>8}")

    # ── 总 speedup 表 ──
    print(f"\n  {'='*72}")
    print(f"  Speedup Summary")
    print(f"  {'='*72}")

    header = f"  {'Conc':>5} {'Tput_Speedup':>14} {'QPS_Speedup':>14}"
    if has_m:
        header += f" {'TTFT_Speedup':>14} {'E2E_Speedup':>14}"
    print(header)
    print(f"  {'-'*72}")

    for bs in common:
        bt = bsum[bs]["throughput"]["avg"]
        ot = osum[bs]["throughput"]["avg"]
        bq = bsum[bs]["qps"]["avg"]
        oq = osum[bs]["qps"]["avg"]

        tsp = f"{ot/bt:.3f}x" if bt > 0 else "N/A"
        qsp = f"{oq/bq:.3f}x" if bq > 0 else "N/A"

        line = f"  {bs:>5} {tsp:>14} {qsp:>14}"

        if has_m and bsum[bs]["metrics_available"] and osum[bs]["metrics_available"]:
            bttft = bsum[bs]["ttft"]["avg"]
            ottft = osum[bs]["ttft"]["avg"]
            be2e = bsum[bs]["e2e"]["p50"]
            oe2e = osum[bs]["e2e"]["p50"]
            ttsp = f"{bttft/ottft:.3f}x" if ottft > 0 else "N/A"
            esp = f"{be2e/oe2e:.3f}x" if oe2e > 0 else "N/A"
            line += f" {ttsp:>14} {esp:>14}"

        print(line)


# ════════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FinServe E2E A/B Benchmark (zero estimation, all measured)",
    )
    parser.add_argument("--mode", choices=["quick", "full", "stress"], default="quick")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="bench_results")
    parser.add_argument("--no-subprocess", action="store_true")
    parser.add_argument("--configs", nargs="+", default=["baseline", "sgmv_full", "fp8_kv", "sgmv_fp8"])
    args = parser.parse_args()

    batch_map = {
        "quick": [1, 4, 8, 16],
        "full": [1, 4, 8, 16, 32, 64],
        "stress": [1, 8, 32, 64, 128],
    }
    batch_sizes = batch_map[args.mode]
    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    print("=" * 72)
    print("  FinServe E2E A/B Benchmark")
    print("  ALL VALUES MEASURED — ZERO ESTIMATION")
    print("=" * 72)
    print(f"  Mode:         {args.mode}")
    print(f"  Batch sizes:  {batch_sizes}")
    print(f"  Runs:         {args.runs} (+ {args.warmup} warmup)")
    print(f"  Max tokens:   {args.max_tokens}")
    print(f"  Configs:      {args.configs}")
    print(f"  Isolation:    {'in-process' if args.no_subprocess else 'subprocess'}")
    print(f"  Prompts:      {len(PROMPT_POOL_A)+len(PROMPT_POOL_B)} (A:{len(PROMPT_POOL_A)} B:{len(PROMPT_POOL_B)})")
    print("=" * 72)

    raw_results = {}
    for cfg_name in args.configs:
        if cfg_name not in OPTIMIZATION_CONFIGS:
            print(f"  [WARN] Unknown config '{cfg_name}', skip")
            continue
        cfg = OPTIMIZATION_CONFIGS[cfg_name]
        if args.no_subprocess:
            result = run_single_config(cfg_name, cfg, batch_sizes, args.max_tokens, args.runs, args.warmup)
        else:
            out_path = os.path.join(args.output_dir, f"_tmp_{cfg_name}_{ts}.json")
            result = run_in_subprocess(cfg_name, cfg, batch_sizes, args.max_tokens, args.runs, args.warmup, out_path)
            try:
                os.remove(out_path)
            except OSError:
                pass
        raw_results[cfg_name] = result

    # ── 聚合 & 报告 ──
    print(f"\n{'='*72}")
    print("  RESULTS")
    print(f"{'='*72}")

    summaries = {}
    for cfg_name, raw in raw_results.items():
        if "error" in raw:
            print(f"\n  [{cfg_name}] FAILED: {raw['error']}")
            continue
        s = aggregate(raw)
        summaries[cfg_name] = s
        ms = raw.get("metrics_source", "unknown")
        print_summary(OPTIMIZATION_CONFIGS[cfg_name]["label"], s, ms)

    cfg_ok = [c for c in args.configs if c in summaries]
    if len(cfg_ok) >= 2:
        print(f"\n{'='*72}")
        print(f"  A/B: [{cfg_ok[0]}] vs [{cfg_ok[1]}]")
        print(f"{'='*72}")
        print_comparison(
            OPTIMIZATION_CONFIGS[cfg_ok[0]]["label"],
            OPTIMIZATION_CONFIGS[cfg_ok[1]]["label"],
            summaries[cfg_ok[0]], summaries[cfg_ok[1]],
        )

    out_file = os.path.join(args.output_dir, f"e2e_comparison_{ts}.json")
    with open(out_file, "w") as f:
        json.dump({
            "meta": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": args.mode, "batch_sizes": batch_sizes,
                "num_runs": args.runs, "max_tokens": args.max_tokens,
                "configs": args.configs, "note": "ALL VALUES MEASURED, ZERO ESTIMATION",
            },
            "raw": raw_results,
            "summaries": {k: {str(bs): v for bs, v in s.items()} for k, s in summaries.items()},
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved: {out_file}")
    print("=" * 72)


if __name__ == "__main__":
    main()

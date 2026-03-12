#!/usr/bin/env python3
"""
FinServe vLLM 生产服务压测 (HTTP Streaming Endpoint)

针对 serve_multi_lora.sh 启动的 OpenAI-compatible vLLM 服务进行全面压测:
  ✓ 流式 SSE 精确测量 TTFT (Time To First Token)
  ✓ 混合 LoRA (Expert-A + Expert-B) 并发压力测试
  ✓ 多并发梯度: 1 → 2 → 4 → 8 → 16
  ✓ 完整指标: TTFT / TPOT / E2E / TPS / QPS / P50 / P99
  ✓ Per-model 拆分统计（验证 LoRA 亲和调度效果）
  ✓ 结果 JSON 导出 + A/B 对比模式

使用方式:
  # 1. 先启动服务
  DEPLOY_MODE=single ./serve_multi_lora.sh

  # 2. 基线压测
  python bench_serving.py --tag baseline

  # 3. 切换到 Triton 服务后再测一轮
  python bench_serving.py --tag triton

  # 4. 对比
  python bench_serving.py --compare baseline triton
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import aiohttp
except ImportError:
    print("ERROR: 需要 aiohttp\n  pip install aiohttp")
    sys.exit(1)


# ── 配置 ────────────────────────────────────────────────────────────────

DEFAULT_URL = "http://127.0.0.1:8000"
RESULTS_DIR = Path("./bench_results")

PROMPTS_EXPERT_A = [
    {
        "model": "finance-expert-a",
        "messages": [
            {"role": "system", "content": "你是金融分析专家A，擅长宏观经济分析。"},
            {"role": "user", "content": prompt},
        ],
    }
    for prompt in [
        "请分析2024年中国GDP增长的主要驱动力和潜在风险因素。",
        "当前货币政策对A股市场有什么影响？请从流动性和估值两个维度分析。",
        "请分析中美贸易摩擦对中国出口企业盈利的影响及应对策略。",
        "2025年利率走向预测及其对债券市场收益率曲线的影响。",
        "人民币汇率波动对北向资金流入A股的影响及传导机制分析。",
    ]
]

PROMPTS_EXPERT_B = [
    {
        "model": "finance-expert-b",
        "messages": [
            {"role": "system", "content": "你是金融分析专家B，擅长行业和个股分析。"},
            {"role": "user", "content": prompt},
        ],
    }
    for prompt in [
        "请分析半导体行业国产替代的投资机会和主要风险因素。",
        "新能源汽车产业链中电池、电机、电控哪个环节最具投资价值？",
        "请对比分析宁德时代和比亚迪的竞争优势和估值水平。",
        "医药生物行业在集采政策下的投资策略和重点关注方向。",
        "AI大模型相关概念股的基本面分析、估值评估和投资建议。",
    ]
]

PROMPTS_MIXED = PROMPTS_EXPERT_A + PROMPTS_EXPERT_B


# ── 流式请求 + 指标采集 ─────────────────────────────────────────────────

async def streaming_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: list,
    max_tokens: int,
    sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    """发送一条流式请求，解析 SSE 精确测量 TTFT / TPOT / E2E。"""
    url = f"{base_url}/v1/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.8,
        "stream": True,
    }

    async with sem:
        t0 = time.perf_counter()
        first_tok_t: Optional[float] = None
        last_tok_t: Optional[float] = None
        tok_count = 0

        try:
            async with session.post(url, json=body) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    return _fail(model, t0, f"HTTP {resp.status}: {txt[:200]}")

                buf = ""
                async for chunk in resp.content.iter_any():
                    buf += chunk.decode("utf-8", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break
                        try:
                            evt = json.loads(payload)
                            choices = evt.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            if delta.get("content"):
                                now = time.perf_counter()
                                if first_tok_t is None:
                                    first_tok_t = now
                                last_tok_t = now
                                tok_count += 1
                        except json.JSONDecodeError:
                            pass

            t1 = time.perf_counter()
            e2e = (t1 - t0) * 1000
            ttft = (first_tok_t - t0) * 1000 if first_tok_t else e2e

            if tok_count > 1 and first_tok_t and last_tok_t and last_tok_t > first_tok_t:
                tpot = (last_tok_t - first_tok_t) * 1000 / (tok_count - 1)
            else:
                tpot = 0.0

            return {
                "model": model, "ttft_ms": ttft, "e2e_ms": e2e,
                "output_tokens": tok_count, "tpot_ms": tpot, "success": True,
            }

        except Exception as e:
            return _fail(model, t0, str(e))


def _fail(model: str, t0: float, error: str) -> Dict[str, Any]:
    return {
        "model": model, "ttft_ms": 0, "e2e_ms": (time.perf_counter() - t0) * 1000,
        "output_tokens": 0, "tpot_ms": 0, "success": False, "error": error,
    }


# ── 统计工具 ─────────────────────────────────────────────────────────────

def pct(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def _agg(results: List[Dict], wall_s: float) -> Dict[str, Any]:
    """从一组请求结果中聚合指标。"""
    ok = [r for r in results if r["success"]]
    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] > 0]
    tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"] > 0]
    e2es = [r["e2e_ms"] for r in ok]
    total_toks = sum(r["output_tokens"] for r in ok)
    return {
        "successful": len(ok),
        "failed": len(results) - len(ok),
        "ttft_avg": round(statistics.mean(ttfts), 1) if ttfts else 0,
        "ttft_p50": round(pct(ttfts, 50), 1),
        "ttft_p99": round(pct(ttfts, 99), 1),
        "tpot_avg": round(statistics.mean(tpots), 1) if tpots else 0,
        "tpot_p50": round(pct(tpots, 50), 1),
        "tpot_p99": round(pct(tpots, 99), 1),
        "e2e_avg": round(statistics.mean(e2es), 1) if e2es else 0,
        "e2e_p50": round(pct(e2es, 50), 1),
        "e2e_p99": round(pct(e2es, 99), 1),
        "tps": round(total_toks / wall_s, 1) if wall_s > 0 else 0,
        "qps": round(len(ok) / wall_s, 2) if wall_s > 0 else 0,
        "total_tokens": total_toks,
    }


# ── 压测阶段 ─────────────────────────────────────────────────────────────

async def run_phase(
    session: aiohttp.ClientSession,
    base_url: str,
    name: str,
    prompts: list,
    concurrency: int,
    n_reqs: int,
    max_tokens: int,
) -> Dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        streaming_request(
            session, base_url,
            prompts[i % len(prompts)]["model"],
            prompts[i % len(prompts)]["messages"],
            max_tokens, sem,
        )
        for i in range(n_reqs)
    ]

    wall_t0 = time.perf_counter()
    results: List[Dict] = await asyncio.gather(*tasks)
    wall_s = time.perf_counter() - wall_t0

    agg = _agg(results, wall_s)
    phase: Dict[str, Any] = {
        "name": name,
        "concurrency": concurrency,
        "total_requests": n_reqs,
        "wall_time_s": round(wall_s, 2),
        **agg,
    }

    # Per-model 拆分（混合场景下显示各 LoRA 的独立指标）
    models_seen = set(r["model"] for r in results if r["success"])
    if len(models_seen) > 1:
        per_model: Dict[str, Any] = {}
        for m in sorted(models_seen):
            m_results = [r for r in results if r["model"] == m]
            m_ok = [r for r in m_results if r["success"]]
            m_ttfts = [r["ttft_ms"] for r in m_ok if r["ttft_ms"] > 0]
            m_tpots = [r["tpot_ms"] for r in m_ok if r["tpot_ms"] > 0]
            m_e2es = [r["e2e_ms"] for r in m_ok]
            per_model[m] = {
                "count": len(m_ok),
                "ttft_p50": round(pct(m_ttfts, 50), 1),
                "tpot_p50": round(pct(m_tpots, 50), 1),
                "e2e_p50": round(pct(m_e2es, 50), 1),
            }
        phase["per_model"] = per_model

    errors = [r["error"] for r in results if not r["success"] and "error" in r]
    if errors:
        phase["errors"] = errors[:5]

    return phase


# ── 打印 ─────────────────────────────────────────────────────────────────

def print_phase(p: Dict):
    w = 66
    print(f"\n  {'─' * w}")
    print(
        f"  {p['name']}  "
        f"(并发={p['concurrency']}, 请求={p['total_requests']}, "
        f"成功={p['successful']}, 失败={p['failed']})"
    )
    print(f"  {'─' * w}")

    if p["failed"] > 0 and "errors" in p:
        for e in p["errors"]:
            print(f"    ✗ {e}")

    print(f"  {'指标':<16} {'Avg':>10} {'P50':>10} {'P99':>10}")
    print(f"  {'─' * 46}")
    print(f"  {'TTFT (ms)':<16} {p['ttft_avg']:>10.1f} {p['ttft_p50']:>10.1f} {p['ttft_p99']:>10.1f}")
    print(f"  {'TPOT (ms)':<16} {p['tpot_avg']:>10.1f} {p['tpot_p50']:>10.1f} {p['tpot_p99']:>10.1f}")
    print(f"  {'E2E  (ms)':<16} {p['e2e_avg']:>10.1f} {p['e2e_p50']:>10.1f} {p['e2e_p99']:>10.1f}")
    print(f"  {'TPS (tok/s)':<16} {p['tps']:>10.1f}")
    print(f"  {'QPS (req/s)':<16} {p['qps']:>10.2f}")
    print(f"  {'Wall time (s)':<16} {p['wall_time_s']:>10.1f}")

    if "per_model" in p:
        print(f"\n  Per-model 拆分:")
        print(f"  {'Model':<24} {'N':>5} {'TTFT_P50':>10} {'TPOT_P50':>10} {'E2E_P50':>10}")
        print(f"  {'─' * 60}")
        for m, s in p["per_model"].items():
            print(
                f"  {m:<24} {s['count']:>5} "
                f"{s['ttft_p50']:>8.1f}ms {s['tpot_p50']:>8.1f}ms {s['e2e_p50']:>8.1f}ms"
            )


# ── 主流程 ───────────────────────────────────────────────────────────────

async def run_benchmark(args: argparse.Namespace):
    print()
    print("=" * 72)
    print("  FinServe vLLM 生产服务压测 (HTTP Streaming)")
    print("=" * 72)
    print()

    # ── Health check ──
    print("▶ 检查服务...")
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as s:
            async with s.get(f"{args.url}/v1/models") as r:
                data = await r.json()
                models = [m["id"] for m in data.get("data", [])]
                print(f"  ✓ 服务就绪 — 可用模型: {models}")
                has_a = any("expert-a" in m for m in models)
                has_b = any("expert-b" in m for m in models)
                if not (has_a and has_b):
                    print(
                        "  ⚠ 缺少 LoRA 模型 (需要 finance-expert-a + finance-expert-b)"
                    )
                    print(f"    当前模型: {models}")
    except Exception as e:
        print(f"  ✗ 连接失败: {e}")
        print("  请先启动: DEPLOY_MODE=single ./serve_multi_lora.sh")
        return
    print()

    phases: List[Dict] = []
    timeout = aiohttp.ClientTimeout(total=600)
    conn = aiohttp.TCPConnector(limit=args.max_concurrency + 8)

    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:

        # ── Warmup ──
        print("▶ Warmup (5 请求, 串行)...")
        warmup = await run_phase(
            session, args.url, "warmup", PROMPTS_MIXED[:2], 1, 5, args.max_tokens,
        )
        print(f"  ✓ 完成 ({warmup['wall_time_s']:.1f}s)")
        print()

        # ── Phase 1: 单 Expert 串行基线 ──
        print("▶ Phase 1 — 单 Expert 串行基线")

        pa = await run_phase(
            session, args.url, "Expert-A (serial)",
            PROMPTS_EXPERT_A, 1, args.num_requests, args.max_tokens,
        )
        print_phase(pa)
        phases.append(pa)

        pb = await run_phase(
            session, args.url, "Expert-B (serial)",
            PROMPTS_EXPERT_B, 1, args.num_requests, args.max_tokens,
        )
        print_phase(pb)
        phases.append(pb)

        # ── Phase 2: 混合 LoRA 并发梯度 ──
        print(f"\n▶ Phase 2 — 混合 LoRA 并发梯度")
        for c in [1, 2, 4, 8, 16]:
            if c > args.max_concurrency:
                break
            n = max(args.num_requests, c * 4)
            p = await run_phase(
                session, args.url, f"Mixed A+B (c={c})",
                PROMPTS_MIXED, c, n, args.max_tokens,
            )
            print_phase(p)
            phases.append(p)

    # ── 总结 ──
    print(f"\n{'=' * 72}")
    print(f"  总结  [tag={args.tag}]")
    print(f"{'=' * 72}\n")

    hdr = (
        f"  {'阶段':<24} {'并发':>4} "
        f"{'TTFT_P50':>10} {'TPOT_P50':>10} {'E2E_P50':>10} {'TPS':>8} {'QPS':>7}"
    )
    print(hdr)
    print(f"  {'─' * 73}")
    for p in phases:
        print(
            f"  {p['name']:<24} {p['concurrency']:>4} "
            f"{p['ttft_p50']:>8.1f}ms {p['tpot_p50']:>8.1f}ms "
            f"{p['e2e_p50']:>8.1f}ms {p['tps']:>7.1f} {p['qps']:>6.2f}"
        )

    # ── 保存 JSON ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.tag}.json"
    with open(out, "w") as f:
        json.dump(
            {
                "tag": args.tag,
                "url": args.url,
                "max_tokens": args.max_tokens,
                "num_requests": args.num_requests,
                "max_concurrency": args.max_concurrency,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "phases": phases,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n  ✓ 结果已保存: {out}")
    print(f"    对比命令: python bench_serving.py --compare baseline triton\n")


# ── 对比模式 ─────────────────────────────────────────────────────────────

def compare_results(tags: List[str]):
    tag_a, tag_b = tags
    fa, fb = RESULTS_DIR / f"{tag_a}.json", RESULTS_DIR / f"{tag_b}.json"
    for tag, path in [(tag_a, fa), (tag_b, fb)]:
        if not path.exists():
            print(f"  ✗ 找不到结果文件: {path}")
            return

    with open(fa) as f:
        da = json.load(f)
    with open(fb) as f:
        db = json.load(f)

    pa_map = {p["name"]: p for p in da["phases"]}
    pb_map = {p["name"]: p for p in db["phases"]}
    common = [name for name in pa_map if name in pb_map]
    if not common:
        print("  没有可对比的阶段（阶段名称不匹配）")
        return

    print()
    print("=" * 80)
    print(f"  压测对比: [{tag_a}] vs [{tag_b}]")
    print(f"  {tag_a}: {da['timestamp']}  |  {tag_b}: {db['timestamp']}")
    print("=" * 80)
    print()

    metrics = [
        ("ttft_p50", "TTFT-P50", True),
        ("tpot_p50", "TPOT-P50", True),
        ("e2e_p50", "E2E-P50", True),
        ("tps", "TPS", False),
        ("qps", "QPS", False),
    ]

    print(f"  {'阶段':<24} {'指标':<12} {tag_a:>10} {tag_b:>10} {'变化':>10}")
    print(f"  {'─' * 66}")

    for name in common:
        a, b = pa_map[name], pb_map[name]
        for key, label, lower_better in metrics:
            va, vb = a.get(key, 0), b.get(key, 0)
            if va > 0:
                delta = ((va - vb) / va * 100) if lower_better else ((vb - va) / va * 100)
                marker = "▲" if delta > 2 else ("▼" if delta < -2 else "─")
                imp = f"{marker}{delta:+.1f}%"
            else:
                imp = "  N/A"
            unit = "ms" if lower_better else ""
            print(
                f"  {name:<24} {label:<12} "
                f"{va:>9.1f}{unit} {vb:>9.1f}{unit} {imp:>10}"
            )
        print(f"  {'─' * 66}")

    # Per-model 对比（若有）
    for name in common:
        a, b = pa_map[name], pb_map[name]
        if "per_model" not in a or "per_model" not in b:
            continue
        common_models = set(a["per_model"]) & set(b["per_model"])
        if not common_models:
            continue
        print(f"\n  [{name}] Per-model 对比:")
        print(
            f"  {'Model':<24} {'TTFT(' + tag_a + ')':>12} {'TTFT(' + tag_b + ')':>12} "
            f"{'E2E(' + tag_a + ')':>12} {'E2E(' + tag_b + ')':>12}"
        )
        print(f"  {'─' * 72}")
        for m in sorted(common_models):
            sa, sb = a["per_model"][m], b["per_model"][m]
            print(
                f"  {m:<24} {sa['ttft_p50']:>10.1f}ms {sb['ttft_p50']:>10.1f}ms "
                f"{sa['e2e_p50']:>10.1f}ms {sb['e2e_p50']:>10.1f}ms"
            )

    print()


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="FinServe vLLM 生产服务压测 (HTTP Streaming)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python bench_serving.py --tag baseline          # 基线压测
  python bench_serving.py --tag triton            # Triton 压测
  python bench_serving.py --compare baseline triton  # 对比
  python bench_serving.py --num-requests 50 --max-concurrency 32  # 大规模
""",
    )
    ap.add_argument("--url", default=DEFAULT_URL, help="vLLM 服务地址 (默认 %(default)s)")
    ap.add_argument("--tag", default="default", help="压测标签，用于保存/对比")
    ap.add_argument("--num-requests", type=int, default=20, help="每阶段请求数 (默认 20)")
    ap.add_argument("--max-tokens", type=int, default=512, help="每请求最大生成 token (默认 512)")
    ap.add_argument("--max-concurrency", type=int, default=16, help="最大并发数 (默认 16)")
    ap.add_argument("--compare", nargs=2, metavar="TAG", help="对比两次压测: --compare A B")

    args = ap.parse_args()
    if args.compare:
        compare_results(args.compare)
    else:
        asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()

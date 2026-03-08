#!/usr/bin/env python
"""
简单 vLLM OpenAI 兼容接口压测脚本

指标：
- TTFT (Avg / P95)：Time To First Token，首个增量到达耗时
- TPOT (Avg)：Time Per Output Token，从首 token 到最后一个 token 的耗时
- Latency (P95 / P99)：端到端延迟分位数
- Throughput：基于 usage.completion_tokens 的整体吞吐，单位 token/s

实现说明：
- 使用 OpenAI chat/completions 流式接口 (stream=true)
- 依赖 httpx：请先 `pip install httpx`
- 尽量利用 OpenAI 协议的 `stream_options: {"include_usage": true}` 获取 usage 信息；
  如果后端暂不支持该字段，则回退为基于文本长度的粗略 token 估算。
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import httpx


@dataclass
class RequestMetrics:
    start_ts: float
    first_token_ts: float
    end_ts: float
    latency: float
    ttft: float
    tpot: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


def percentile(values: List[float], q: float) -> float:
    """简单百分位数实现，q 取 0~1 之间."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    return values[idx]


async def run_single_request(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    request_id: int,
) -> Optional[RequestMetrics]:
    """
    单个流式请求：
    - 记录首个 data chunk 到达时间 → TTFT
    - 记录流结束时间 → TPOT / 总延迟
    - 尝试从最后一个 chunk 中解析 usage
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        # OpenAI 新协议，要求后端实现后才能返回 usage
        "stream_options": {"include_usage": True},
    }

    start = time.perf_counter()
    first_token_ts: Optional[float] = None
    end: Optional[float] = None
    prompt_tokens = completion_tokens = total_tokens = None
    collected_text_parts: List[str] = []

    try:
        async with client.stream("POST", url, json=payload, timeout=None) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break

                now = time.perf_counter()
                if first_token_ts is None:
                    first_token_ts = now

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # 收集文本以便 fallback 估算 token 数
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                content_piece = delta.get("content")
                if isinstance(content_piece, str):
                    collected_text_parts.append(content_piece)

                # 有些实现会在最后一个 chunk 里带 usage
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    total_tokens = usage.get("total_tokens")

            end = time.perf_counter()

    except Exception as e:  # noqa: BLE001
        print(f"[request {request_id}] error: {e}", file=sys.stderr)
        return None

    if first_token_ts is None:
        # 没有任何 token，就把首 token 时间视为结束时间
        first_token_ts = end or start

    if end is None:
        end = time.perf_counter()

    ttft = first_token_ts - start
    latency = end - start
    tpot = max(0.0, latency - ttft)

    # 如果后端没有返回 usage，做一个粗略 token 估算（仅用于吞吐的近似统计）
    if completion_tokens is None and collected_text_parts:
        text = "".join(collected_text_parts)
        # 对中文/英文混合文本，一个「字符」~= 一个 token 的粗略估算
        completion_tokens = max(1, len(text))
        total_tokens = completion_tokens

    return RequestMetrics(
        start_ts=start,
        first_token_ts=first_token_ts,
        end_ts=end,
        latency=latency,
        ttft=ttft,
        tpot=tpot,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


async def run_load_test(
    base_url: str,
    model: str,
    prompt: str,
    total_requests: int,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    prompts: Optional[List[str]] = None,
) -> List[RequestMetrics]:
    """
    prompts: 若提供且 len==total_requests，则每个请求用对应 prompt；否则全部用 prompt。
    """
    url = f"{base_url}/v1/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    metrics: List[RequestMetrics] = []

    def get_prompt(req_id: int) -> str:
        if prompts is not None and len(prompts) == total_requests:
            return prompts[req_id]
        return prompt

    async with httpx.AsyncClient() as client:
        async def wrapped_request(req_id: int):
            async with sem:
                p = get_prompt(req_id)
                m = await run_single_request(
                    client, url, model, p, max_tokens, temperature, req_id
                )
                if m is not None:
                    metrics.append(m)

        tasks = [asyncio.create_task(wrapped_request(i)) for i in range(total_requests)]
        await asyncio.gather(*tasks)

    return metrics


def get_metrics_summary(metrics: List[RequestMetrics]) -> Optional[dict]:
    """从 metrics 计算汇总指标，返回字典供对比脚本使用；无有效数据时返回 None."""
    if not metrics:
        return None
    n = len(metrics)
    ttfts = [m.ttft for m in metrics]
    tpots = [m.tpot for m in metrics]
    latencies = [m.latency for m in metrics]
    ttft_avg = statistics.fmean(ttfts)
    ttft_p95 = percentile(ttfts, 0.95)
    tpot_avg = statistics.fmean(tpots)
    lat_p95 = percentile(latencies, 0.95)
    lat_p99 = percentile(latencies, 0.99)
    total_completion_tokens = 0
    for m in metrics:
        if m.completion_tokens is not None:
            total_completion_tokens += m.completion_tokens
        elif m.total_tokens is not None:
            total_completion_tokens += m.total_tokens
    earliest = min(m.start_ts for m in metrics)
    latest = max(m.end_ts for m in metrics)
    duration = max(1e-6, latest - earliest)
    throughput_toks_per_s = total_completion_tokens / duration
    return {
        "n": n,
        "duration_s": duration,
        "ttft_avg_ms": ttft_avg * 1000,
        "ttft_p95_ms": ttft_p95 * 1000,
        "tpot_avg_ms": tpot_avg * 1000,
        "lat_p95_ms": lat_p95 * 1000,
        "lat_p99_ms": lat_p99 * 1000,
        "throughput_tokens_per_s": throughput_toks_per_s,
    }


def summarize_metrics(metrics: List[RequestMetrics], phase_name: str = "") -> None:
    if not metrics:
        print("没有成功请求，无法统计。")
        return

    if phase_name:
        print(f"\n{'='*50}")
        print(f"  {phase_name}")
        print(f"{'='*50}")

    s = get_metrics_summary(metrics)
    if not s:
        return
    n, duration = s["n"], s["duration_s"]
    ttft_avg, ttft_p95 = s["ttft_avg_ms"], s["ttft_p95_ms"]
    tpot_avg = s["tpot_avg_ms"]
    lat_p95, lat_p99 = s["lat_p95_ms"], s["lat_p99_ms"]
    throughput_toks_per_s = s["throughput_tokens_per_s"]
    earliest = min(m.start_ts for m in metrics)
    latest = max(m.end_ts for m in metrics)

    print("========== Summary ==========")
    print(f"Total requests:        {n}")
    print(f"Test duration:         {duration:.3f} s")
    print("")
    print(f"TTFT Avg:              {ttft_avg:.2f} ms")
    print(f"TTFT P95:              {ttft_p95:.2f} ms")
    print("")
    print(f"TPOT Avg:              {tpot_avg:.2f} ms")
    print("")
    print(f"Latency P95:           {lat_p95:.2f} ms")
    print(f"Latency P99:           {lat_p99:.2f} ms")
    print("")
    print(f"Throughput:            {throughput_toks_per_s:.2f} tokens/s")

    # 简单的「趋势」：按时间片（10 个 bucket）统计 P95 / P99 Latency
    print("\n------ Latency Trend (P95 / P99 by time bucket) ------")
    start = earliest
    end = latest
    total_span = max(1e-6, end - start)
    num_buckets = 10
    bucket_stats = []

    for i in range(num_buckets):
        b_start = start + total_span * i / num_buckets
        b_end = start + total_span * (i + 1) / num_buckets
        bucket = [m for m in metrics if b_start <= m.start_ts < b_end]
        if not bucket:
            bucket_stats.append((i, 0, 0))
            continue
        b_lats = [m.latency for m in bucket]
        b_p95 = percentile(b_lats, 0.95) * 1000
        b_p99 = percentile(b_lats, 0.99) * 1000
        bucket_stats.append((i, b_p95, b_p99))

    for i, b_p95, b_p99 in bucket_stats:
        start_pct = i * 10
        end_pct = (i + 1) * 10
        if b_p95 == 0 and b_p99 == 0:
            print(f"{start_pct:3d}-{end_pct:3d}%:  (no samples)")
        else:
            print(
                f"{start_pct:3d}-{end_pct:3d}%:  "
                f"P95={b_p95:8.2f} ms, P99={b_p99:8.2f} ms"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM OpenAI chat/completions 压测脚本")
    parser.add_argument("--host", default="127.0.0.1", help="服务地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument(
        "--model",
        default="finance-expert-a",
        help="模型名称（多 LoRA 模式下常用 finance-expert-a / finance-expert-b）",
    )
    parser.add_argument(
        "--prompt",
        default="请简要分析一下当前A股市场的整体走势，用于压测。",
        help="用于压测的问句内容",
    )
    parser.add_argument(
        "--total-requests",
        type=int,
        default=1000,
        help="总请求数（每轮）；默认 1000",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=160,
        help="并发协程数量；默认 160",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="每次生成的最大 token 数 (max_tokens)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="两阶段对比：先跑「优化后」(相同 prompt，利于 Prefix Cache)，再跑「优化前」(每请求不同 prompt，冷缓存)",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=50,
        help="仅在 --compare 时生效：优化后阶段预热请求数，用于填满 Prefix Cache",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="对比模式下每阶段跑几轮，多轮结果合并后汇总，便于看平均优化效果；默认 5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    print("========== Load Test Config ==========")
    print(f"Base URL:       {base_url}")
    print(f"Model:          {args.model}")
    print(f"Prompt:         {args.prompt}")
    print(f"Total requests: {args.total_requests}")
    print(f"Concurrency:    {args.concurrency}")
    print(f"Max tokens:     {args.max_tokens}")
    print(f"Temperature:    {args.temperature}")
    if args.compare:
        print(f"Compare mode:    ON (优化后 → 优化前)")
        print(f"Rounds:         {args.rounds} (每阶段跑 {args.rounds} 轮，合并汇总)")
        print(f"Warmup:         {args.warmup_requests} requests")
    print("======================================\n")

    async def _run_all() -> None:
        if args.compare:
            rounds = max(1, args.rounds)
            # 阶段一：优化后 - 相同 prompt，利于 Prefix Cache，多轮合并
            print("\n[Phase 1] 优化后：相同 prompt，Prefix Cache 可复用")
            warmup = args.warmup_requests
            if warmup > 0:
                print(f"  预热 {warmup} 请求...")
                await run_load_test(
                    base_url=base_url,
                    model=args.model,
                    prompt=args.prompt,
                    total_requests=warmup,
                    concurrency=min(args.concurrency, warmup),
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
            m_opt_all: List[RequestMetrics] = []
            for r in range(rounds):
                print(f"  第 {r+1}/{rounds} 轮，每轮 {args.total_requests} 请求...")
                m_opt = await run_load_test(
                    base_url=base_url,
                    model=args.model,
                    prompt=args.prompt,
                    total_requests=args.total_requests,
                    concurrency=args.concurrency,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                m_opt_all.extend(m_opt)
            print(f"  优化后共 {len(m_opt_all)} 条样本（{rounds} 轮合并）")
            summarize_metrics(m_opt_all, "【优化后】相同 prompt，Prefix Cache 命中")

            # 阶段二：优化前 - 每请求不同 prompt，冷缓存，多轮合并
            print("\n[Phase 2] 优化前：每请求不同 prompt，无 Prefix Cache 复用")
            m_unopt_all: List[RequestMetrics] = []
            for r in range(rounds):
                print(f"  第 {r+1}/{rounds} 轮，每轮 {args.total_requests} 请求...")
                # 每轮用不同的 suffix，保证无缓存复用
                diverse_prompts = [
                    f"{args.prompt}\n[压测请求 #r{r}-{i}]"
                    for i in range(args.total_requests)
                ]
                m_unopt = await run_load_test(
                    base_url=base_url,
                    model=args.model,
                    prompt=args.prompt,
                    total_requests=args.total_requests,
                    concurrency=args.concurrency,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    prompts=diverse_prompts,
                )
                m_unopt_all.extend(m_unopt)
            print(f"  优化前共 {len(m_unopt_all)} 条样本（{rounds} 轮合并）")
            summarize_metrics(m_unopt_all, "【优化前】不同 prompt，冷缓存")

            # 简要对比
            if m_opt_all and m_unopt_all:
                _print_comparison(m_opt_all, m_unopt_all)
        else:
            metrics = await run_load_test(
                base_url=base_url,
                model=args.model,
                prompt=args.prompt,
                total_requests=args.total_requests,
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            summarize_metrics(metrics)

    asyncio.run(_run_all())


def _print_comparison(opt: List[RequestMetrics], unopt: List[RequestMetrics]) -> None:
    """打印优化前后简要对比表"""
    def _stats(m: List[RequestMetrics]):
        if not m:
            return 0.0, 0.0, 0.0, 0.0
        ttft = statistics.fmean([x.ttft for x in m]) * 1000
        ttft_p95 = percentile([x.ttft for x in m], 0.95) * 1000
        lat_p95 = percentile([x.latency for x in m], 0.95) * 1000
        tok = sum(x.completion_tokens or x.total_tokens or 0 for x in m)
        dur = max(1e-6, max(x.end_ts for x in m) - min(x.start_ts for x in m))
        thr = tok / dur
        return ttft, ttft_p95, lat_p95, thr

    o_ttft, o_ttft_p95, o_lat_p95, o_thr = _stats(opt)
    u_ttft, u_ttft_p95, u_lat_p95, u_thr = _stats(unopt)

    print("\n" + "=" * 50)
    print("  【对比汇总】")
    print("=" * 50)
    print(f"{'指标':<20} {'优化后':>12} {'优化前':>12} {'变化':>10}")
    print("-" * 56)
    for label, ov, uv in [
        ("TTFT Avg (ms)", o_ttft, u_ttft),
        ("TTFT P95 (ms)", o_ttft_p95, u_ttft_p95),
        ("Latency P95 (ms)", o_lat_p95, u_lat_p95),
        ("Throughput (tok/s)", o_thr, u_thr),
    ]:
        chg = ((ov - uv) / uv * 100) if uv else 0
        print(f"{label:<20} {ov:>12.2f} {uv:>12.2f} {chg:>+9.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()


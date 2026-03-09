#!/usr/bin/env python
"""
压测对比：未优化服务 vs 优化后服务

【服务端实际开关的优化】
- Chunked Prefill：长 Prefill 拆块与 Decode 交替，避免长请求队头阻塞短请求
- Prefix Caching：相同前缀复用 KV，减少重复计算
- LoRA Reorder：等待队列按 adapter 聚批，减少 LoRA 切换

【为何「全相同长 prompt」压测差异小】
- 全相同长 prompt：没有「长短混合」，Chunked Prefill 的「长不堵短」发挥不出来
- 只打 expert-a：没有多 adapter 混合，LoRA Reorder 无效果
- 总耗时里 decode 占大头时，Prefill 优化在整体数字里被稀释

【建议】加 --mixed：50% 短 prompt + 50% 长 prompt，才能明显看出 Chunked Prefill 带来的改善（短请求不再被长请求堵在后面）。
"""

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

# 复用 bench_load_test 的压测与汇总
from bench_load_test import get_metrics_summary, run_load_test


def wait_for_server(base_url: str, timeout_sec: float = 120, interval: float = 2.0) -> bool:
    """轮询 /v1/models 直到返回 200 或超时。"""
    url = f"{base_url}/v1/models"
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=5.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def start_server(cwd: Path, optimized: bool) -> subprocess.Popen:
    """启动 vLLM 服务。optimized=True 用默认配置，False 关闭 Chunked Prefill / Prefix Caching / LoRA Reorder。"""
    env = os.environ.copy()
    if not optimized:
        env["ENABLE_CHUNKED_PREFILL"] = "false"
        env["ENABLE_PREFIX_CACHING"] = "false"
        env["FINSERVE_LORA_REORDER"] = "0"
    proc = subprocess.Popen(
        ["bash", "serve_multi_lora.sh"],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    """终止服务进程。"""
    try:
        proc.terminate()
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass


async def run_one_phase(
    base_url: str,
    model: str,
    prompt: str,
    total_requests: int,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    mixed: bool = False,
) -> Optional[dict]:
    """跑一轮压测，返回 get_metrics_summary 的字典；失败返回 None。mixed=True 时 50% 短 50% 长 prompt。"""
    prompts: Optional[list] = None
    if mixed:
        prompts = [SHORT_PROMPT if i % 2 == 0 else DEFAULT_HEAVY_PROMPT for i in range(total_requests)]
    metrics = await run_load_test(
        base_url=base_url,
        model=model,
        prompt=prompt,
        total_requests=total_requests,
        concurrency=concurrency,
        max_tokens=max_tokens,
        temperature=temperature,
        prompts=prompts,
    )
    return get_metrics_summary(metrics)


def print_comparison(before: Optional[dict], after: Optional[dict]) -> None:
    """打印未优化 vs 优化后对比表。"""
    if not before or not after:
        print("缺少一侧数据，无法对比。")
        return
    print("\n" + "=" * 60)
    print("  【压测对比】未优化 vs 优化后（同一负载）")
    print("=" * 60)
    print(f"{'指标':<22} {'未优化':>14} {'优化后':>14} {'变化':>12}")
    print("-" * 64)
    for key, label in [
        ("ttft_avg_ms", "TTFT Avg (ms)"),
        ("ttft_p95_ms", "TTFT P95 (ms)"),
        ("lat_p95_ms", "Latency P95 (ms)"),
        ("lat_p99_ms", "Latency P99 (ms)"),
        ("throughput_tokens_per_s", "Throughput (tok/s)"),
    ]:
        bv = before.get(key)
        av = after.get(key)
        if bv is None or av is None:
            continue
        chg = ((av - bv) / bv * 100) if bv else 0
        # 延迟类指标：优化后变小为改善，变化为负更好；吞吐：优化后变大更好
        print(f"{label:<22} {bv:>14.2f} {av:>14.2f} {chg:>+11.1f}%")
    print("=" * 60)
    print("说明：TTFT/Latency 负变化表示优化后变快；Throughput 正变化表示优化后吞吐提升。")
    print("=" * 60)


# 混合负载用短 prompt（一句）
SHORT_PROMPT = "用一句话概括当前A股市场的主要风险。"

# 高压默认 prompt：长文本拉高 Prefill，才能体现 Chunked Prefill / Prefix Caching 差异
DEFAULT_HEAVY_PROMPT = """你是一位资深金融分析师。请基于以下研报摘要与数据，给出简明结论与风险提示。

【宏观与流动性】
2024年以来国内货币政策维持稳健偏松，M2与社融增速整体平稳，央行多次通过降准、结构性工具释放流动性。海外方面，美联储加息周期见顶后进入观察期，美债收益率与美元指数高位震荡，对新兴市场资金流向构成阶段性影响。国内无风险利率中枢下移，权益资产性价比有所抬升，但企业盈利与估值修复节奏仍依赖经济内生动力。

【行业与风格】
上半年顺周期与高股息资产表现占优，电力、煤炭、银行、石油石化等板块相对抗跌；成长风格在利率敏感与业绩预期分化下波动加大，TMT、新能源等经历估值回调。下半年需关注财政发力节奏、地产与消费政策效果，以及海外选举与贸易政策的不确定性。中观上建议关注：一是受益于出海与国产替代的高端制造与零部件；二是盈利稳定、分红率高的红利资产；三是供给出清、景气有望见底的周期细分。

【风险因素】
地缘冲突、全球通胀二次上行、国内需求恢复不及预期、部分行业产能过剩与价格战、汇率与资本流动波动等，均可能对A股节奏与结构造成扰动。

请用 3–5 段话总结：当前阶段对A股整体走势的判断、主要驱动与拖累因素、以及你建议的配置思路与需要重点跟踪的指标。"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="未优化 vs 优化后 服务压测对比")
    p.add_argument("--host", default="127.0.0.1", help="服务地址")
    p.add_argument("--port", type=int, default=8000, help="服务端口")
    p.add_argument("--model", default="finance-expert-a", help="模型名")
    p.add_argument("--prompt", default=DEFAULT_HEAVY_PROMPT, help="压测用 prompt（默认长文本以施加 Prefill 压力）")
    p.add_argument("--total-requests", type=int, default=2000, help="每轮总请求数")
    p.add_argument("--concurrency", type=int, default=64, help="并发数（提高以制造队列压力）")
    p.add_argument("--max-tokens", type=int, default=512, help="每请求最大生成 token 数（提高以拉长 decode）")
    p.add_argument("--temperature", type=float, default=0.7, help="temperature")
    p.add_argument("--wait-sec", type=float, default=120, help="等待服务就绪最长时间（秒）")
    p.add_argument("--mixed", action="store_true", help="混合负载：50%% 短 prompt + 50%% 长 prompt，便于看出 Chunked Prefill 对「长不堵短」的改善")
    p.add_argument("--no-start-server", action="store_true", help="不自动启停服务，仅对当前已运行服务跑两轮（需手动切换未优化/优化后并重跑）")
    return p.parse_args()


async def main_async(args: argparse.Namespace, deploy_dir: Path) -> None:
    base_url = f"http://{args.host}:{args.port}"
    workload = {
        "model": args.model,
        "prompt": args.prompt,
        "total_requests": args.total_requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "mixed": getattr(args, "mixed", False),
    }

    if args.no_start_server:
        print("当前为「仅压测」模式：请先手动启动未优化服务，本脚本将跑一轮并打印结果；")
        print("然后请手动改为优化后服务并再次运行本脚本做对比。")
        summary = await run_one_phase(base_url, **workload)
        if summary:
            print("\n当前一轮汇总:")
            for k, v in summary.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.2f}")
                else:
                    print(f"  {k}: {v}")
        return

    # 1) 启动未优化服务
    print("启动未优化服务（Chunked Prefill / Prefix Caching / LoRA Reorder 关闭）...")
    proc_unopt = start_server(deploy_dir, optimized=False)
    try:
        if not wait_for_server(base_url, timeout_sec=args.wait_sec):
            print("未优化服务在限定时间内未就绪，请检查模型与端口。", file=sys.stderr)
            return
        print("未优化服务已就绪，开始压测...")
        before = await run_one_phase(base_url, **workload)
        if not before:
            print("未优化压测无有效结果。", file=sys.stderr)
            return
        print(f"  未优化压测完成: {before['n']} 请求, TTFT P95={before['ttft_p95_ms']:.1f} ms, Throughput={before['throughput_tokens_per_s']:.1f} tok/s")
    finally:
        stop_server(proc_unopt)
        print("未优化服务已停止，等待端口释放...")
        time.sleep(5)

    # 2) 启动优化后服务
    print("启动优化后服务（默认开启 Chunked Prefill / Prefix Caching / LoRA Reorder）...")
    proc_opt = start_server(deploy_dir, optimized=True)
    try:
        if not wait_for_server(base_url, timeout_sec=args.wait_sec):
            print("优化后服务在限定时间内未就绪。", file=sys.stderr)
            return
        print("优化后服务已就绪，开始压测...")
        after = await run_one_phase(base_url, **workload)
        if not after:
            print("优化后压测无有效结果。", file=sys.stderr)
            return
        print(f"  优化后压测完成: {after['n']} 请求, TTFT P95={after['ttft_p95_ms']:.1f} ms, Throughput={after['throughput_tokens_per_s']:.1f} tok/s")
    finally:
        stop_server(proc_opt)

    print_comparison(before, after)


def main() -> None:
    args = parse_args()
    deploy_dir = Path(__file__).resolve().parent
    if not (deploy_dir / "serve_multi_lora.sh").exists():
        print("请在 qwen3-vl-finance-deploy 目录下运行本脚本。", file=sys.stderr)
        sys.exit(1)
    print("压测配置: total_requests=%d, concurrency=%d, max_tokens=%d, model=%s" % (args.total_requests, args.concurrency, args.max_tokens, args.model))
    if getattr(args, "mixed", False):
        print("负载: 混合（50%% 短 + 50%% 长 prompt）→ 便于体现 Chunked Prefill「长不堵短」")
    else:
        print("prompt 长度: %d 字符（约 %d token）" % (len(args.prompt), len(args.prompt) // 2))
    asyncio.run(main_async(args, deploy_dir))


if __name__ == "__main__":
    main()

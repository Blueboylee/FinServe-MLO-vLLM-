import argparse
import json
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    JSONResponse,
    StreamingResponse,
)


@dataclass
class RequestRecord:
    ts_start: float
    ts_end: float
    status_code: int
    success: bool
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Optional[str]


class MetricsStore:
    """简单内存指标，用于最近一段时间窗口的聚合统计。"""

    def __init__(self, window_sec: int = 300) -> None:
        self.window_sec = window_sec
        self.records: Deque[RequestRecord] = deque()
        self.active_requests: int = 0
        self.total_requests: int = 0
        self.total_success: int = 0
        self.total_error: int = 0

    def add_record(self, rec: RequestRecord) -> None:
        self.records.append(rec)
        self.total_requests += 1
        if rec.success:
            self.total_success += 1
        else:
            self.total_error += 1
        self._trim()

    def _trim(self) -> None:
        now = time.time()
        while self.records and now - self.records[0].ts_start > self.window_sec:
            self.records.popleft()

    def snapshot(self) -> Dict:
        self._trim()
        recs = list(self.records)
        if not recs:
            return {
                "window_sec": self.window_sec,
                "active_requests": self.active_requests,
                "requests": {
                    "total": self.total_requests,
                    "success": self.total_success,
                    "error": self.total_error,
                },
                "latency": {},
                "throughput": {},
                "reliability": {},
            }

        latencies = [(r.ts_end - r.ts_start) for r in recs]
        lat_ms = [x * 1000 for x in latencies]
        lat_ms_sorted = sorted(lat_ms)

        def pct(p: float) -> float:
            if not lat_ms_sorted:
                return 0.0
            k = max(0, min(len(lat_ms_sorted) - 1, int(len(lat_ms_sorted) * p)))
            return lat_ms_sorted[k]

        total_tokens = sum(r.total_tokens for r in recs if r.total_tokens >= 0)
        total_time = sum(latencies)
        tokens_per_s = total_tokens / total_time if total_time > 0 else 0.0

        by_status = Counter(r.status_code for r in recs)
        finish_counter = Counter(r.finish_reason or "unknown" for r in recs)

        success_recent = sum(1 for r in recs if r.success)
        error_recent = len(recs) - success_recent
        success_rate = success_recent / len(recs) * 100.0
        error_rate = error_recent / len(recs) * 100.0

        return {
            "window_sec": self.window_sec,
            "active_requests": self.active_requests,
            "requests": {
                "recent_count": len(recs),
                "total": self.total_requests,
                "success": self.total_success,
                "error": self.total_error,
            },
            "latency": {
                "avg_ms": sum(lat_ms) / len(lat_ms),
                "p50_ms": pct(0.5),
                "p95_ms": pct(0.95),
                "p99_ms": pct(0.99),
            },
            "throughput": {
                "total_tokens": total_tokens,
                "tokens_per_s": tokens_per_s,
            },
            "reliability": {
                "success_rate_percent": success_rate,
                "error_rate_percent": error_rate,
                "by_status": dict(by_status),
                "finish_reason": dict(finish_counter),
            },
        }


def _extract_text_from_messages(messages) -> str:
    """从 OpenAI Chat messages 中提取所有用户文本，支持多模态 content[]."""
    chunks: List[str] = []
    if not isinstance(messages, list):
        return ""
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text")
                    if isinstance(t, str):
                        chunks.append(t)
    return "\n".join(chunks)


def route_model(messages: List[Dict]) -> str:
    """简单关键字路由：在基座 / Expert-A / Expert-B 之间选择。"""
    text = _extract_text_from_messages(messages)
    t = text.lower()

    macro_keywords = [
        "宏观",
        "cpi",
        "ppi",
        "gdp",
        "通胀",
        "通货膨胀",
        "加息",
        "降息",
        "美联储",
        "fed",
        "央行",
        "经济周期",
        "货币政策",
        "财政政策",
        "汇率",
        "美元指数",
        "宏观经济",
        "经济数据",
        "宏观环境",
    ]
    stock_keywords = [
        "个股",
        "股票",
        "股价",
        "估值",
        "市盈率",
        "pe",
        "pb",
        "roe",
        "行业",
        "板块",
        "龙头",
        "赛道",
        "公司",
        "财报",
        "年报",
        "季报",
        "利润表",
        "资产负债表",
        "现金流",
        "券商",
        "银行",
        "地产",
        "新能源",
        "半导体",
        "科技股",
        "中字头",
        "a股",
        "港股",
        "美股",
        "代码",
        "ticker",
    ]

    macro_score = sum(1 for kw in macro_keywords if kw.lower() in t or kw in text)
    stock_score = sum(1 for kw in stock_keywords if kw.lower() in t or kw in text)

    if stock_score > macro_score and stock_score >= 1:
        return "finance-expert-b"
    if macro_score >= 1:
        return "finance-expert-a"

    # fallback：偏向宏观专家，如果明显谈行业/个股则给 B
    if any(kw in text for kw in ["行业", "板块", "个股", "股票", "公司"]):
        return "finance-expert-b"
    return "finance-expert-a"


def create_app(static_dir: Path, upstream: str) -> FastAPI:
    app = FastAPI()

    static_file = static_dir / "static-chat.html"
    dashboard_file = static_dir / "dashboard.html"
    metrics = MetricsStore(window_sec=300)

    @app.get("/")
    async def root():
        if static_file.exists():
            return RedirectResponse(url="/static-chat.html")
        return PlainTextResponse("static-chat.html not found", status_code=404)

    @app.get("/static-chat.html")
    async def chat_page():
        if not static_file.exists():
            return PlainTextResponse("static-chat.html not found", status_code=404)
        return FileResponse(static_file)

    @app.get("/dashboard.html")
    async def dashboard_page():
        if dashboard_file.exists():
            return FileResponse(dashboard_file)
        # 如果没有单独的 dashboard 文件，就简单返回 JSON 视图提示
        return PlainTextResponse(
            "No dashboard.html, but metrics are available at /dashboard-metrics",
            status_code=200,
        )

    @app.get("/dashboard-metrics")
    async def dashboard_metrics():
        snap = metrics.snapshot()

        # Best-effort: scrape vLLM /metrics for KV cache usage gauge.
        kv_usage_percent: Optional[float] = None
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{upstream}/metrics")
            if r.status_code == 200:
                values: List[float] = []
                for line in r.text.splitlines():
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("vllm:kv_cache_usage_perc") or line.startswith(
                        "vllm:gpu_cache_usage_perc"
                    ):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                values.append(float(parts[-1]))
                            except ValueError:
                                pass
                if values:
                    # metric is [0,1] where 1 = 100%
                    kv_usage_percent = max(values) * 100.0
        except Exception:
            kv_usage_percent = None

        snap["kv_cache"] = {"usage_percent": kv_usage_percent}
        return JSONResponse(snap)

    @app.get("/health")
    async def health():
        return {"ok": True, "upstream": upstream}

    # 反向代理 vLLM OpenAI 接口（同源调用，避免浏览器 CORS 问题）
    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def proxy_v1(path: str, request: Request):
        url = f"{upstream}/v1/{path}"
        headers = dict(request.headers)
        headers.pop("host", None)

        body = await request.body()
        # 如果前端传入 model=auto，则根据用户问题自动路由到 Expert-A / Expert-B / 基座
        if path.startswith("chat/completions") or path == "chat/completions":
            try:
                payload = json.loads(body or b"{}")
            except Exception:
                payload = None
            if isinstance(payload, dict):
                model_name = payload.get("model")
                if model_name in (None, "auto"):
                    routed = route_model(payload.get("messages") or [])
                    payload["model"] = routed
                    body = json.dumps(payload).encode("utf-8")

        is_stream = False
        if path.startswith("chat/completions") or path == "chat/completions":
            try:
                is_stream = json.loads(body).get("stream") is True
            except Exception:
                pass

        if is_stream:
            # 流式：直接转发上游流（透传 status/headers）
            excluded = {
                "connection",
                "keep-alive",
                "proxy-authenticate",
                "proxy-authorization",
                "te",
                "trailers",
                "transfer-encoding",
                "upgrade",
            }

            metrics.active_requests += 1
            client = httpx.AsyncClient(timeout=600)
            upstream_resp: Optional[httpx.Response] = None
            try:
                req = client.build_request(
                    method=request.method,
                    url=url,
                    params=dict(request.query_params),
                    headers=headers,
                    content=body,
                )
                upstream_resp = await client.send(req, stream=True)
                out_headers = {
                    k: v
                    for k, v in upstream_resp.headers.items()
                    if k.lower() not in excluded
                }

                async def stream_chunks():
                    try:
                        async for chunk in upstream_resp.aiter_bytes():
                            yield chunk
                    finally:
                        await upstream_resp.aclose()
                        await client.aclose()
                        metrics.active_requests -= 1

                return StreamingResponse(
                    stream_chunks(),
                    status_code=upstream_resp.status_code,
                    headers=out_headers,
                )
            except Exception:
                if upstream_resp is not None:
                    await upstream_resp.aclose()
                await client.aclose()
                metrics.active_requests -= 1
                raise

        ts_start = time.time()
        metrics.active_requests += 1
        try:
            async with httpx.AsyncClient(timeout=600) as client:
                resp = await client.request(
                    method=request.method,
                    url=url,
                    params=dict(request.query_params),
                    headers=headers,
                    content=body,
                )
            ts_end = time.time()
        finally:
            metrics.active_requests -= 1

        total_tokens = -1
        prompt_tokens = -1
        completion_tokens = -1
        finish_reason: Optional[str] = None
        success = 200 <= resp.status_code < 300

        if path.startswith("chat/completions") or path == "chat/completions":
            try:
                payload = resp.json()
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                usage = payload.get("usage") or {}
                total_tokens = int(usage.get("total_tokens") or -1)
                prompt_tokens = int(usage.get("prompt_tokens") or -1)
                completion_tokens = int(usage.get("completion_tokens") or -1)
                choices = payload.get("choices") or []
                if choices:
                    finish_reason = choices[0].get("finish_reason")

        rec = RequestRecord(
            ts_start=ts_start,
            ts_end=ts_end,
            status_code=resp.status_code,
            success=success,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )
        metrics.add_record(rec)

        excluded = {
            "connection", "keep-alive", "proxy-authenticate",
            "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade",
        }
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded}
        return Response(content=resp.content, status_code=resp.status_code, headers=out_headers)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--port",
        type=int,
        default=8188,
        help="用平台已映射的内网端口，比如 8188",
    )
    parser.add_argument("--static-dir", default=str(Path(__file__).parent))
    parser.add_argument(
        "--upstream",
        default="http://127.0.0.1:8000",
        help="vLLM OpenAI 服务地址",
    )
    args = parser.parse_args()

    import uvicorn

    app = create_app(Path(args.static_dir), args.upstream)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()



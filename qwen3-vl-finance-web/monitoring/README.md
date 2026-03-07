# FinServe Prometheus + Grafana 监控

用 Prometheus 采集 Web Proxy 的指标，用 Grafana 做专业展示（与内置 dashboard 数据一致，且支持长期存储与告警）。

## 1. 依赖

- Web Proxy 需安装 `prometheus_client`，才会暴露 `/metrics` 端点：
  ```bash
  pip install prometheus_client
  # 或
  pip install -r ../requirements.txt
  ```
- 启动 Web Proxy 后，访问 `http://<proxy_host>:<port>/metrics` 应看到 Prometheus 文本格式指标。

## 2. 指标说明（Prometheus）

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `finserve_request_duration_seconds` | Histogram | 请求 E2E 耗时（秒） |
| `finserve_ttft_seconds` | Histogram | 首 token 时间（秒） |
| `finserve_tpot_seconds` | Histogram | 每输出 token 耗时（秒） |
| `finserve_requests_total` | Counter | 请求总数（label: status=success/error） |
| `finserve_tokens_total` | Counter | 总 token 数 |
| `finserve_requests_in_flight` | Gauge | 当前处理中请求数 |
| `finserve_kv_cache_usage_percent` | Gauge | vLLM KV cache 使用率 0–100（/metrics 被拉取时从 vLLM 获取） |

## 3. 使用 Docker 启动 Prometheus + Grafana

在**能访问 Web Proxy 的机器**上执行（Proxy 默认端口 8188）：

```bash
cd /path/to/FinServe-MLO-vLLM-/qwen3-vl-finance-web/monitoring
```

**修改 `prometheus.yml`**：将 `targets` 改为你的 Proxy 地址。

- Proxy 与 Prometheus 同机：`127.0.0.1:8188`（本 compose 中 Prometheus 使用 host 网络，可直接抓本机 8188）
- Proxy 在其它机器：`<该机 IP>:8188`

启动：

```bash
docker compose -f docker-compose.monitoring.yml up -d
```

- **Grafana**: http://localhost:3000  默认账号 `admin` / `admin`
- **Prometheus**: http://localhost:9090

Grafana 已自动配置 Prometheus 数据源，并导入 **FinServe Inference (Prometheus)** 仪表盘，可直接查看 E2E 延迟、TTFT、TPOT、RPS、吞吐、成功率、KV Cache 等。

## 4. 不用 Docker 时

- **Prometheus**：从 [prometheus.io](https://prometheus.io/download/) 下载，用本目录下 `prometheus.yml` 作为配置，`--config.file=prometheus.yml`，确保能访问 Proxy 的 `http://<proxy>:8188/metrics`。
- **Grafana**：从 [grafana.com](https://grafana.com/grafana/download) 下载安装，添加 Prometheus 数据源（URL 为 Prometheus 地址），然后导入 `grafana/provisioning/dashboards/json/finserve.json` 即可。

## 5. 与内置 HTML Dashboard 的关系

- 内置 `dashboard.html` 使用 Web Proxy 的 `/dashboard-metrics`（内存 5 分钟窗口聚合）。
- Prometheus 拉取的是同一批请求的**原始指标**（Histogram/Counter/Gauge），由 Prometheus 做持久化与聚合，Grafana 用 PromQL 查询并展示。
- 两者数据源一致（都来自 Proxy），Prometheus + Grafana 更适合作长期监控、多实例汇总和告警。

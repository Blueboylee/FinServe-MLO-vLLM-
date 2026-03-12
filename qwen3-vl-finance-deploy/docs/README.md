# Qwen3-VL Finance 多专家 LoRA 部署

基于 Qwen3-VL-8B-Instruct-AWQ-4bit 基座模型，挂载两个金融领域 LoRA 微调专家，使用 vLLM 部署。

## 硬件要求

- GPU: 至少 1 张 16GB+ 显存的 NVIDIA GPU（推荐 A100/4090/3090）
- AWQ 4-bit 基座模型约需 6-8GB 显存，加载 LoRA 额外需要少量显存
- 内存: 32GB+

## 快速开始（conda 方式推荐）

### Step 1: 创建并准备 conda 环境

```bash
cd /root/下载/qwen3-vl-finance-deploy

# 一键创建/更新 conda 环境（环境名：qwen3-vllm）
bash setup_env.sh

# 激活环境
conda activate qwen3-vllm
```

> 如果你不想用 conda，也可以直接在系统 Python 里跑 `bash setup_env.sh`，脚本会自动回退到 `pip` 安装模式。

### Step 2: 下载模型（需在已激活的 conda 环境中执行）

```bash
bash download_models.sh
```

### Step 3: 启动服务

**方案一（推荐先试）：多 LoRA 动态加载**

一个 vLLM 进程同时服务基座模型 + 两个 LoRA 专家，通过请求中的 `model` 字段切换：

```bash
bash serve_multi_lora.sh
```

服务启动后，通过 API 调用不同专家：
```bash
# 调用 Expert-A
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "finance-expert-a",
    "messages": [{"role": "user", "content": "分析一下最近的A股走势"}],
    "temperature": 0.7,
    "max_tokens": 512
  }'

# 调用 Expert-B
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "finance-expert-b",
    "messages": [{"role": "user", "content": "分析一下最近的A股走势"}],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

**方案二（保底）：合并 LoRA 后部署**

如果方案一遇到 `AssertionError in lora_shrink_op` 错误（已知 Qwen3-VL LoRA bug），使用此方案：

```bash
# 1. 合并 LoRA 到基座模型
python merge_lora_and_serve.py --expert both

# 2. 启动 Expert-A（端口 8000）
bash serve_merged.sh a

# 3. 另一个终端启动 Expert-B（端口 8001）
bash serve_merged.sh b
```

### Step 4: 测试

```bash
# 单个专家测试
python test_client.py --model finance-expert-a --prompt "请分析贵州茅台最近的财报"

# 对比两个专家
python test_client.py --compare --prompt "请分析2024年A股市场走势"

# 多模态测试（传入图片）
python test_client.py --model finance-expert-a --prompt "请分析这张K线图" --image chart.png

# 合并模型模式，Expert-B 在 8001 端口
python test_client.py --port 8001 --model "Qwen3-VL-Finance-merged-expert-b" --prompt "分析一下..."
```

### 也可以用 Python API 离线推理

```bash
python serve_multi_lora.py
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `setup_env.sh` | 安装 vLLM、modelscope 等依赖 |
| `download_models.sh` | 下载基座模型和两个 LoRA 专家 |
| `serve_multi_lora.sh` | 方案一：vLLM 多 LoRA 服务（shell 版） |
| `serve_multi_lora.py` | 方案一：Python 离线推理示例 |
| `merge_lora_and_serve.py` | 方案二：合并 LoRA 到基座模型 |
| `serve_merged.sh` | 方案二：启动合并后的模型 |
| `test_client.py` | OpenAI 兼容 API 测试客户端 |

## 已知问题

Qwen3-VL 系列模型在 vLLM 中使用 LoRA 时存在已知 bug（[vllm#28186](https://github.com/vllm-project/vllm/issues/28186)），可能在加载 LoRA 适配器时报 `AssertionError`。如果遇到此问题，请使用方案二（合并模型）。建议先用最新版 vLLM 尝试方案一。

## 可选：做一个 ChatGPT 风格网页

很多云平台（你截图的“端口转发”面板）**只会把固定的内网端口映射到公网端口**，例如内网 `8188` → 公网 `11273`。这种情况下直接用 `python -m http.server 5173` 在公网是打不开的。

这里提供一个同源反向代理方案：

- 页面：`~/下载/qwen3-vl-finance-web/static-chat.html`
- 代理服务：`~/下载/qwen3-vl-finance-web/web_proxy_server.py`
- 浏览器访问：通过你平台「端口转发」里业务端口对应的访问地址（例如内网 8188 对应 `http://jq1.9gpu.com:11273`）

### 启动步骤

1) 确保 vLLM 已启动（内网 `127.0.0.1:8000`）：

```bash
conda activate qwen3-vllm
cd ~/下载/qwen3-vl-finance-deploy
bash serve_multi_lora.sh
```

2) 新开一个终端，启动网页 + 反向代理。**内网端口必须和「端口转发」里业务端口的内网端口一致**（例如 8188 或 7860）：

```bash
conda activate qwen3-vllm
pip install -U fastapi uvicorn httpx

cd ~/下载/qwen3-vl-finance-web
# 若端口转发里是「内网 8188 → 公网 11273」，用：
python web_proxy_server.py --port 8188
# 若端口转发里是「内网 7860 → 公网 11274」，用：
# python web_proxy_server.py --port 7860
```

3) 浏览器打开「端口转发」里该业务端口的访问地址，例如：

- 内网 8188 时：`http://jq1.9gpu.com:11273/`
- 内网 7860 时：`http://jq1.9gpu.com:11274/`


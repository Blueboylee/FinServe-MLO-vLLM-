# Qwen2.5 32B 专家模型推理

共享 Qwen2.5 32B 4bit GPTQ 基座，加载两个 QLoRA 专家（Expert A / Expert B）进行推理。适配 V100 GPU，所有模型从 **ModelScope** 下载。

## 环境要求

- Python 3.10
- CUDA（V100 或更高）
- 显存：建议 24GB+（32B 4bit GPTQ 约需 18–20GB）

## 安装

```bash
pip install -r requirements.txt
```

## 1. 下载模型

所有模型从 ModelScope 下载（不使用 HuggingFace）：

```bash
python download_models.py
```

默认保存到：
- 基座：`./models/base`（Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4）
- 专家 A：`./models/experts/expert-a`（GaryLeenene/qwen25-32b-expert-a-qlora）
- 专家 B：`./models/experts/expert-b`（GaryLeenene/qwen25-32b-expert-b-qlora）

可选参数：
- `--base-dir DIR`：基座保存目录
- `--experts-dir DIR`：专家保存根目录
- `--base-only`：仅下载基座
- `--experts-only`：仅下载专家

## 2. 离线推理

```bash
# 使用专家 A
python run_inference.py --expert expert-a --prompt "你的问题"

# 使用专家 B
python run_inference.py --expert expert-b --prompt "你的问题"

# 自定义参数
python run_inference.py --expert expert-a --prompt "你好" --max-tokens 256 --temperature 0.8
```

## 3. 启动 vLLM 多专家 HTTP 服务

**方式一：Shell 脚本**

```bash
chmod +x serve_experts.sh
./serve_experts.sh
```

**方式二：Python 脚本**

```bash
python serve_experts.py
```

指定目录和端口：

```bash
BASE_DIR=./models/base EXPERTS_DIR=./models/experts PORT=8000 ./serve_experts.sh
```

**测试服务**

```bash
python test_serve.py --url http://localhost:8000
```

请求时通过 `model` 参数切换专家：
- `expert-a`
- `expert-b`

示例（OpenAI 兼容 API）：

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "expert-a",
    "prompt": "你好，请介绍一下你自己。",
    "max_tokens": 256
  }'
```

## 模型来源

| 模型 | ModelScope ID |
|------|---------------|
| 基座（GPTQ 4bit） | Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 |
| 专家 A | GaryLeenene/qwen25-32b-expert-a-qlora |
| 专家 B | GaryLeenene/qwen25-32b-expert-b-qlora |

## 故障排查

1. **显存不足**：在 `run_inference.py` 或 `serve_experts.sh` 中调低 `gpu_memory_utilization`（如 0.8）。
2. **基座模型在 ModelScope 上不存在**：若 `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4` 不可用，可在 `download_models.py` 中修改 `BASE_MODEL_ID` 为 ModelScope 上可用的 GPTQ 版本。
3. **LoRA 加载失败**：确认专家模型为 QLoRA 格式，且与基座架构一致。

# Qwen2.5-32B 双专家模型 vLLM 部署

基于 vLLM 的 Qwen2.5-32B 双专家模型（Expert A 和 Expert B）部署方案，支持从 ModelScope 下载模型并在 V100 GPU 上运行。

## 环境要求

- Python 3.10
- NVIDIA V100 GPU (16GB)
- CUDA 11.8+
- vLLM 0.6.x+

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型

从 ModelScope 下载两个专家模型：

```bash
python download_models.py
```

这会自动下载：
- `GaryLeenene/qwen25-32b-expert-a-qlora`
- `GaryLeenene/qwen25-32b-expert-b-qlora`

### 3. 启动 API 服务器

```bash
python api_server.py --port 8000 --gpu-memory-utilization 0.85
```

### 4. 测试调用

```bash
python test_api.py
```

## 使用方式

### API 调用示例

```python
import requests

# 使用专家 A
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "你好，请介绍一下你自己。",
        "expert": "expert_a",
        "max_tokens": 512,
        "temperature": 0.7
    }
)
print(response.json()["text"])

# 使用专家 B
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "什么是机器学习？",
        "expert": "expert_b",
        "max_tokens": 512,
        "temperature": 0.7
    }
)
print(response.json()["text"])
```

### cURL 调用

```bash
# 健康检查
curl http://localhost:8000/health

# 使用专家 A 生成
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "你好",
    "expert": "expert_a",
    "max_tokens": 256
  }'

# 使用专家 B 生成
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "你好",
    "expert": "expert_b",
    "max_tokens": 256
  }'
```

## 配置参数

### api_server.py 参数

- `--host`: 监听地址 (默认：0.0.0.0)
- `--port`: 端口号 (默认：8000)
- `--max-model-len`: 最大序列长度 (默认：4096)
- `--gpu-memory-utilization`: GPU 内存利用率 (默认：0.85，V100 建议值)

### 生成参数

- `prompt`: 输入提示文本
- `expert`: 使用的专家模型 ("expert_a" 或 "expert_b")
- `max_tokens`: 最大生成 token 数 (1-2048)
- `temperature`: 温度参数 (0.0-2.0)
- `top_p`: Top-p 采样参数 (0.0-1.0)
- `top_k`: Top-k 采样参数 (-1 表示禁用)
- `presence_penalty`: 存在惩罚 (-2.0 到 2.0)
- `frequency_penalty`: 频率惩罚 (-2.0 到 2.0)

## API 文档

启动服务器后访问：http://localhost:8000/docs

## 文件说明

- `download_models.py`: 从 ModelScope 下载模型
- `api_server.py`: API 服务器主程序
- `deploy_experts.py`: 本地部署测试脚本
- `test_api.py`: API 测试脚本
- `requirements.txt`: Python 依赖包

## 故障排除

### 显存不足

降低 GPU 内存利用率：

```bash
python api_server.py --gpu-memory-utilization 0.75
```

### 模型下载失败

检查网络连接，或手动下载：

```bash
modelscope download --model GaryLeenene/qwen25-32b-expert-a-qlora
modelscope download --model GaryLeenene/qwen25-32b-expert-b-qlora
```

## 许可证

本项目仅供学习和研究使用。

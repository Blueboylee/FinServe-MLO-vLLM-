# 使用示例

## 快速开始

### 方式一：一键安装和启动

```bash
# 1. 安装所有依赖并下载模型
bash install.sh

# 2. 启动服务
bash start.sh --port 8000

# 3. 测试
python test_api.py
```

### 方式二：分步执行

#### 步骤 1: 安装依赖

```bash
pip install -r requirements.txt
```

#### 步骤 2: 下载模型

```bash
python download_models.py
```

这会自动从 ModelScope 下载两个专家模型到 `models/` 目录。

#### 步骤 3: 选择一种部署方式

##### 选项 A: API 服务器（推荐）

```bash
# 启动 API 服务器
python api_server.py --port 8000 --gpu-memory-utilization 0.85
```

然后在另一个终端测试：

```bash
python test_api.py
```

或者使用 curl：

```bash
# 健康检查
curl http://localhost:8000/health

# 使用专家 A
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好", "expert": "expert_a", "max_tokens": 256}'

# 使用专家 B
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好", "expert": "expert_b", "max_tokens": 256}'
```

##### 选项 B: 简单推理（单专家）

```bash
# 使用专家 A
python simple_inference.py --expert expert_a

# 使用专家 B
python simple_inference.py --expert expert_b
```

这会进入交互式对话模式。

##### 选项 C: 本地测试部署

```bash
python deploy_experts.py
```

## Python 代码调用示例

### 使用 requests 库

```python
import requests

BASE_URL = "http://localhost:8000"

# 专家 A 生成
response_a = requests.post(
    f"{BASE_URL}/generate",
    json={
        "prompt": "请介绍一下金融风险分析的主要方法。",
        "expert": "expert_a",
        "max_tokens": 512,
        "temperature": 0.7
    }
)
print("专家 A 回答:", response_a.json()["text"])

# 专家 B 生成
response_b = requests.post(
    f"{BASE_URL}/generate",
    json={
        "prompt": "请介绍一下金融风险分析的主要方法。",
        "expert": "expert_b",
        "max_tokens": 512,
        "temperature": 0.7
    }
)
print("专家 B 回答:", response_b.json()["text"])
```

### 批量生成

```python
import requests
from concurrent.futures import ThreadPoolExecutor

def generate_with_expert(prompt, expert):
    response = requests.post(
        "http://localhost:8000/generate",
        json={
            "prompt": prompt,
            "expert": expert,
            "max_tokens": 256
        }
    )
    return response.json()

prompts = [
    ("问题 1", "expert_a"),
    ("问题 2", "expert_b"),
    ("问题 3", "expert_a"),
]

with ThreadPoolExecutor() as executor:
    results = list(executor.map(lambda x: generate_with_expert(*x), prompts))

for i, result in enumerate(results, 1):
    print(f"结果 {i}: {result['text']}")
```

### 流式输出（需要额外实现）

```python
import requests
import json

def stream_generate(prompt, expert="expert_a"):
    """模拟流式输出"""
    response = requests.post(
        "http://localhost:8000/generate",
        json={
            "prompt": prompt,
            "expert": expert,
            "max_tokens": 512
        },
        stream=False  # vLLM 目前不支持流式，需要修改服务器
    )
    
    result = response.json()
    print(result["text"], end="", flush=True)
    return result["text"]

stream_generate("请写一首关于春天的诗。", expert="expert_a")
```

## 高级配置

### 调整 GPU 内存使用

对于 V100 16GB，如果遇到 OOM：

```bash
python api_server.py --gpu-memory-utilization 0.75 --max-model-len 2048
```

### 多卡部署（如果有多个 V100）

```bash
python api_server.py --tensor-parallel-size 2
```

### 自定义采样参数

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "写一个故事",
        "expert": "expert_a",
        "max_tokens": 1024,
        "temperature": 0.9,      # 更高创造性
        "top_p": 0.95,           # 更多样化
        "top_k": 50,             # 限制候选词数量
        "presence_penalty": 0.5,  # 鼓励新话题
        "frequency_penalty": 0.5  # 减少重复
    }
)
```

## 性能优化建议

1. **批处理请求**: 合并多个请求为一个批次
2. **调整 max_model_len**: 根据实际需求设置最大长度
3. **使用合适的 temperature**: 生产环境建议 0.5-0.7
4. **监控 GPU 使用**: 使用 `nvidia-smi` 监控显存

## 常见问题

### Q: 模型下载很慢怎么办？
A: 使用 ModelScope 镜像或手动下载后放到 `models/` 目录

### Q: 显存不足怎么办？
A: 降低 `--gpu-memory-utilization` 或 `--max-model-len`

### Q: 如何切换专家模型？
A: 在 API 请求中指定 `"expert": "expert_a"` 或 `"expert": "expert_b"`

### Q: 支持并发请求吗？
A: 支持，vLLM 内置了连续批处理

## 监控和日志

### 查看 GPU 使用

```bash
watch -n 1 nvidia-smi
```

### 查看服务器日志

服务器会自动输出请求日志到终端。

### 性能监控

```python
import time
import requests

start = time.time()
response = requests.post(
    "http://localhost:8000/generate",
    json={"prompt": "测试", "expert": "expert_a"}
)
elapsed = time.time() - start

print(f"响应时间：{elapsed:.2f}秒")
print(f"Token 数：{response.json()['usage']}")
```

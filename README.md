# Qwen2.5 32B QLoRA 部署说明

## 项目概述

本项目实现了基于Qwen2.5 32B模型，结合QLoRA和Unsloth技术的专家A和专家B模型部署方案。所有模型文件均从ModelScope下载，适配NVIDIA RTX 4080S 32G显卡环境。

## 目录结构

```
FinServe-MLO-vLLM/
├── requirements.txt              # Python依赖包列表
├── config.py                     # 部署配置文件
├── download_models.py            # 模型下载脚本
├── setup_environment.py          # 环境配置脚本
├── api_server.py                 # API服务器
├── start_expert_service.py       # 专家服务启动脚本
├── client.py                     # 命令行客户端
└── README.md                     # 本文档
```

## 环境准备

### 系统要求

- 操作系统: Ubuntu 20.04/22.04
- GPU: NVIDIA RTX 4080S (32GB显存)
- CUDA: 11.8 或更高版本
- conda: Anaconda或Miniconda

### 1. 创建conda环境

```bash
# 运行环境配置脚本
python setup_environment.py
```

或手动创建:

```bash
# 创建conda环境
conda create -n qwen_expert python=3.10 -y

# 激活环境
conda activate qwen_expert

# 安装依赖
pip install -r requirements.txt
```

### 2. 验证CUDA环境

```bash
# 检查GPU
nvidia-smi

# 检查CUDA编译器
nvcc --version
```

## 模型下载

### 从ModelScope下载模型

```bash
# 激活conda环境
conda activate qwen_expert

# 运行模型下载脚本
python download_models.py
```

该脚本会下载以下模型:
- Qwen2.5 32B基础模型 (4bit AWQ版本)
- 专家A模型 (qwen25-32b-expert-a-qlora)
- 专家B模型 (qwen25-32b-expert-b-qlora)

模型将下载到 `models/` 目录下。

## 服务部署

### 启动专家服务

#### 方式1: 使用API服务器 (推荐)

```bash
# 启动专家A服务
conda activate qwen_expert
python api_server.py --expert A --port 8001

# 启动专家B服务 (新开终端)
conda activate qwen_expert
python api_server.py --expert B --port 8002
```

#### 方式2: 使用专家服务启动脚本

```bash
# 启动专家A服务
python start_expert_service.py --expert A --port 8001

# 启动专家B服务
python start_expert_service.py --expert B --port 8002
```

### 验证服务状态

```bash
# 检查专家A服务
curl http://localhost:8001/health?expert=A

# 检查专家B服务
curl http://localhost:8002/health?expert=B
```

## API使用

### 1. 聊天补全接口

```bash
# 专家A
curl http://localhost:8001/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "你好"}],
    "expert": "A",
    "max_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9
  }'

# 专家B
curl http://localhost:8002/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "你好"}],
    "expert": "B",
    "max_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

### 2. 简单生成接口

```bash
# 专家A
curl http://localhost:8001/generate \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "请介绍一下你自己",
    "expert": "A",
    "max_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

### 3. Python客户端调用

```python
import requests

# 专家A
response = requests.post(
    "http://localhost:8001/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": "你好"}],
        "expert": "A",
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9
    }
)
print(response.json())

# 专家B
response = requests.post(
    "http://localhost:8002/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": "你好"}],
        "expert": "B",
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9
    }
)
print(response.json())
```

## 命令行交互

### 使用客户端脚本

```bash
# 交互式聊天
python client.py --expert A -i
python client.py --expert B -i

# 单次对话
python client.py --expert A --message "你好"
python client.py --expert B --message "你好"
```

## 配置说明

### GPU优化配置 (RTX 4080S 32G)

配置文件 [config.py](file:///Users/lixingchen/Documents/GitHub/FinServe-MLO-vLLM-/config.py) 中已针对RTX 4080S 32G显卡进行了优化:

- `gpu_memory_utilization`: 0.95 (95% GPU内存利用率)
- `tensor_parallel_size`: 1 (单卡运行)
- `max_model_len`: 32768 (最大模型长度)
- `max_num_seqs`: 256 (最大序列数)
- `dtype`: auto (自动精度)

如需调整，请修改 [config.py](file:///Users/lixingchen/Documents/GitHub/FinServe-MLO-vLLM-/config.py) 中的 `GPUConfig` 类。

## 故障排查

### 1. 模型加载失败

**问题**: 模型路径不存在

**解决方案**:
```bash
# 确认模型已下载
ls models/

# 重新下载模型
python download_models.py
```

### 2. GPU内存不足

**问题**: 显存不足导致加载失败

**解决方案**:
- 减少 `max_num_seqs` 参数
- 减少 `max_model_len` 参数
- 降低 `gpu_memory_utilization` 参数

### 3. 服务启动失败

**问题**: 端口被占用

**解决方案**:
```bash
# 查看端口占用
lsof -i:8001

# 使用其他端口
python api_server.py --expert A --port 8003
```

### 4. CUDA相关错误

**问题**: CUDA版本不匹配

**解决方案**:
```bash
# 检查CUDA版本
nvcc --version

# 检查PyTorch CUDA支持
python -c "import torch; print(torch.cuda.is_available())"
```

## 性能优化建议

1. **使用4bit量化**: 已在配置中启用，显著降低显存占用
2. **启用Flash Attention**: 在vLLM中自动启用
3. **调整批处理大小**: 根据显存情况调整 `max_num_seqs`
4. **使用Eager模式**: 如遇到图模式问题，设置 `enforce_eager=True`

## 技术栈

- **模型框架**: Qwen2.5 32B
- **量化技术**: QLoRA + 4bit量化
- **加速库**: Unsloth
- **推理引擎**: vLLM
- **API框架**: FastAPI
- **Python版本**: 3.10

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题，请检查:
1. 所有依赖是否正确安装
2. 模型是否完整下载
3. GPU驱动和CUDA版本是否兼容
4. 日志文件 `expert_service.log` 和 `api_server.log`

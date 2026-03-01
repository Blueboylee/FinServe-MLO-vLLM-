# Qwen2.5-32B 双专家模型部署说明

## 概述

本项目实现了 Qwen2.5 32B 模型通过 QLoRA+Unsloth 技术微调的两个专家模型（专家 A 和专家 B），共享同一个 4bit GPTQ 基础模型，优化显存使用。

## 硬件要求

- GPU: NVIDIA 4080S 或同等显卡（32GB 显存）
- CUDA 计算能力：7.0+
- 系统内存：64GB+
- 存储空间：100GB+

## 软件要求

- Ubuntu 22.04
- Python 3.10
- CUDA 12.1
- cuDNN 8.9.2
- conda

## 快速开始

### 1. 安装依赖

```bash
cd /Users/lixingchen/Documents/GitHub/FinServe-MLO-vLLM-/qwen25_dual_expert/scripts
bash install_dependencies.sh
```

### 2. 下载模型

```bash
python download_models.py
```

### 3. 系统验证

```bash
python verify_system.py
```

### 4. 启动服务

#### API 模式
```bash
bash startup.sh --auto
```

或手动启动：
```bash
conda activate qwen25-dual-expert
python deploy.py --mode api --port 8000
```

#### 交互模式
```bash
conda activate qwen25-dual-expert
python deploy.py --mode interactive
```

## API 使用

### 生成文本

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "什么是机器学习？",
    "expert": "A",
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

### 健康检查

```bash
curl http://localhost:8000/health
```

## 项目结构

```
qwen25_dual_expert/
├── scripts/
│   ├── download_models.py      # 模型下载脚本
│   ├── install_dependencies.sh # 依赖安装脚本
│   ├── model_loader.py         # 模型加载核心代码
│   ├── deploy.py               # 部署启动脚本
│   ├── startup.sh              # 启动流程脚本
│   └── verify_system.py        # 系统验证脚本
├── configs/
│   └── conda_environment.yml   # conda 环境配置
├── models/                     # 模型文件目录（下载后生成）
│   ├── base/                   # 基础模型
│   ├── expert-a/               # 专家 A 模型
│   └── expert-b/               # 专家 B 模型
└── logs/                       # 日志目录
```

## 核心功能

### 1. 模型共享机制

- 基础模型：Qwen2.5-32B-Instruct-GPTQ-Int4（4bit 量化）
- 专家 A：GaryLeenene/qwen25-32b-expert-a-qlora
- 专家 B：GaryLeenene/qwen25-32b-expert-b-qlora
- 通过 vLLM 的 LoRA 支持，两个专家共享基础模型权重

### 2. 显存优化

- 基础模型只加载一次
- LoRA 适配器动态切换
- 显存使用率控制在 85% 以内
- 支持 32GB 显存部署

### 3. 路由策略

- 手动选择专家：通过 `expert` 参数指定 A 或 B
- 自动路由：支持轮询和基于置信度的路由

## 日志文件

- 下载日志：`logs/download.log`
- 安装日志：`logs/setup.log`
- 启动日志：`logs/startup.log`
- 推理日志：`logs/inference.log`

## 故障排除

### 模型下载失败

检查网络连接，或手动下载：
```bash
modelscope download --model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4
modelscope download --model GaryLeenene/qwen25-32b-expert-a-qlora
modelscope download --model GaryLeenene/qwen25-32b-expert-b-qlora
```

### 显存不足

调整 `model_loader.py` 中的 `gpu_memory_utilization` 参数：
```python
gpu_memory_utilization=0.75  # 降低为 75%
```

### vLLM 初始化失败

检查 GPU 是否支持：
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## 性能优化

### 1. 调整批处理大小

```python
max_model_len=2048  # 减少最大序列长度
```

### 2. 使用张量并行

```python
tensor_parallel_size=2  # 使用多 GPU
```

### 3. 调整采样参数

```python
temperature=0.5  # 降低温度，提高确定性
top_p=0.85       # 调整 top-p 采样
```

## 更新日志

- v1.0.0: 初始版本
  - 支持 ModelScope 下载
  - 实现双专家共享基础模型
  - 提供 API 和交互两种模式
  - 完整的错误处理和日志记录

## 许可证

本项目遵循 Apache 2.0 许可证。

## 联系方式

如有问题，请查看日志文件或联系技术支持。

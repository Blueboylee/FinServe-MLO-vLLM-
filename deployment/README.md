# Qwen2.5 32B 专家模型部署指南

## 项目简介

本项目实现了基于Qwen2.5 32B模型结合QLoRA和Unsloth技术的专家系统部署方案，支持专家A和专家B两个专家模型，它们共享同一个Qwen2.5 32B 4bit AWQ基础模型。

## 系统要求

- **操作系统**: Ubuntu 20.04/22.04 (或兼容的Linux系统)
- **GPU**: NVIDIA RTX 4080S (32GB VRAM) 或类似规格的GPU
- **CUDA**: 12.1+
- **conda**: Miniconda或Anaconda

## 目录结构

```
FinServe-MLO-vLLM-
├── deployment/
│   ├── scripts/
│   │   ├── setup_environment.sh      # 环境准备脚本
│   │   ├── download_models.sh        # 模型下载脚本
│   │   ├── deploy_experts.py         # 专家模型部署服务
│   │   └── start_service.sh          # 服务启动脚本
│   ├── models/                        # 模型保存目录
│   └── logs/                          # 日志文件目录
```

## 快速开始

### 1. 环境准备

运行环境准备脚本，创建并配置conda环境：

```bash
cd deployment/scripts
chmod +x setup_environment.sh
./setup_environment.sh
```

该脚本会：
- 创建名为 `qwen_experts` 的conda环境（Python 3.10）
- 安装PyTorch、transformers、accelerate、peft、bitsandbytes等依赖
- 安装Unsloth库
- 安装ModelScope
- 安装vLLM

### 2. 下载模型

运行模型下载脚本，从ModelScope下载所有必需的模型：

```bash
chmod +x download_models.sh
./download_models.sh
```

该脚本会下载：
- 专家A模型：`GaryLeenene/qwen25-32b-expert-a-qlora`
- 专家B模型：`GaryLeenene/qwen25-32b-expert-b-qlora`
- Qwen2.5 32B 4bit AWQ基础模型：`qwen/Qwen2.5-32B-Instruct-AWQ`

模型将保存在 `deployment/models/` 目录下。

### 3. 启动服务

运行服务启动脚本：

#### 启动专家A服务
```bash
chmod +x start_service.sh
./start_service.sh deploy-a
```

#### 启动专家B服务
```bash
./start_service.sh deploy-b
```

#### 同时启动所有专家服务
```bash
./start_service.sh deploy-all
```

服务启动后：
- 专家A服务运行在 `http://localhost:8000`
- 专家B服务运行在 `http://localhost:8001`

## API接口

### 健康检查
```bash
curl http://localhost:8000/health
```

### 文本生成
```bash
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{
        "expert_id": "A",
        "prompt": "请介绍一下Qwen2.5模型的特点",
        "max_tokens": 1024,
        "temperature": 0.7
    }'
```

参数说明：
- `expert_id`: 专家ID（"A" 或 "B"）
- `prompt`: 输入提示
- `max_tokens`: 最大生成长度（默认：1024）
- `temperature`: 温度参数（默认：0.7）

### 停止服务
```bash
./start_service.sh stop
```

### 查看服务状态
```bash
./start_service.sh status
```

### 测试服务
```bash
./start_service.sh test A
./start_service.sh test B 8001
```

## 配置说明

### GPU内存优化配置

部署脚本针对RTX 4080S 32GB显存进行了优化：

- `tensor_parallel_size=1`: 单卡运行
- `max_num_seqs=16`: 最大序列数
- `max_model_len=4096`: 最大模型长度
- `gpu_memory_utilization=0.95`: GPU内存利用率
- `dtype='half'`: 使用半精度浮点数

### 修改配置

如需调整配置，请编辑 `deploy_experts.py` 文件中的 `ExpertDeployer` 类初始化参数。

## 技术栈

- **基础模型**: Qwen2.5 32B (4bit AWQ量化)
- **适配技术**: QLoRA + Unsloth
- **推理引擎**: vLLM
- **深度学习框架**: PyTorch
- **API框架**: FastAPI + Uvicorn

## 日志文件

服务日志保存在 `deployment/logs/` 目录：
- `expert_service.log`: 通用服务日志
- `expert_A.log`: 专家A服务日志
- `expert_B.log`: 专家B服务日志

## 故障排查

### 常见问题

1. **显存不足**
   - 减少 `max_num_seqs` 参数
   - 减少 `max_model_len` 参数
   - 确保没有其他进程占用GPU

2. **模型下载失败**
   - 检查网络连接
   - 确认ModelScope已正确安装
   - 查看 `download_models.sh` 日志

3. **服务启动失败**
   - 查看日志文件定位问题
   - 确认conda环境已正确激活
   - 检查GPU驱动和CUDA版本

## 许可证

本项目遵循Qwen2.5模型的原始许可证。

## 联系方式

如有问题，请查看日志文件或检查系统配置。

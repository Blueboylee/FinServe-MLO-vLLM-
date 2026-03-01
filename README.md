# Qwen2.5 32B 双 LoRA 专家模型部署指南

## 环境要求

- **操作系统**: Ubuntu 22.04
- **Python**: 3.10
- **GPU**: NVIDIA 显卡 (推荐 24GB+ 显存，如 RTX 4080S 32GB)
- **CUDA**: 12.1+

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境 (推荐)
python3.10 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装 vLLM (如果需要特定版本)
pip install vllm>=0.6.0
```

### 2. 下载模型

所有模型将从 ModelScope 自动下载：

```bash
python download_models.py
```

这将下载：
- **基座模型**: qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 (4bit 量化)
- **专家 A**: GaryLeenene/qwen25-32b-expert-a-qlora
- **专家 B**: GaryLeenene/qwen25-32b-expert-b-qlora

模型将保存在 `./models/` 目录

### 3. 启动服务

```bash
# 基础启动 (使用默认配置)
python serve_lora.py

# 自定义 GPU 内存利用率
python serve_lora.py --gpu-memory-utilization 0.9

# 自定义端口
python serve_lora.py --port 8000

# 完整参数示例
python serve_lora.py \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 \
  --port 8000
```

## 使用方式

### 交互式模式

启动后，你可以：

```
[用户]: :lora
当前 LoRA: None
可用的 LoRA: ['expert_a', 'expert_b']

[用户]: :lora expert_a
已切换到 expert_a

[用户]: 你好，请帮我分析一下这个财务报表
[expert_a 正在思考...]
[助手]: (专家 A 的专业回复)

[用户]: :lora expert_b
已切换到 expert_b

[用户]: 这个代码有什么优化建议
[expert_b 正在思考...]
[助手]: (专家 B 的专业回复)

[用户]: :lora base
已切换到基座模型

[用户]: :quit
```

### 命令说明

- `:lora` - 查看当前 LoRA 状态
- `:lora expert_a` - 切换到专家 A
- `:lora expert_b` - 切换到专家 B
- `:lora base` - 使用基座模型
- `:quit` - 退出

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `models/model_config.txt` | 模型配置文件路径 |
| `--gpu-memory-utilization` | `0.85` | GPU 内存利用率 (0-1) |
| `--max-model-len` | `4096` | 最大序列长度 |
| `--tensor-parallel-size` | `1` | 张量并行大小 |
| `--host` | `0.0.0.0` | 服务主机地址 |
| `--port` | `8000` | 服务端口 |

## 针对 4080S 32GB 的优化建议

由于你使用的是 RTX 4080S 32GB 特殊版本，建议：

```bash
python serve_lora.py --gpu-memory-utilization 0.9 --max-model-len 4096
```

- **gpu_memory_utilization=0.9**: 充分利用 32GB 显存
- **max_model_len=4096**: 平衡上下文长度和显存占用

## 模型文件结构

```
FinServe-MLO-vLLM/
├── download_models.py      # 模型下载脚本
├── serve_lora.py          # vLLM 部署脚本
├── requirements.txt       # 依赖列表
├── README.md             # 使用说明
└── models/               # 模型存储目录
    ├── model_config.txt  # 模型配置
    ├── qwen25-32b-gptq/  # 基座模型
    ├── expert-a-qlora/   # 专家 A LoRA
    └── expert-b-qlora/   # 专家 B LoRA
```

## 常见问题

### Q: 显存不足怎么办？
A: 降低 `--gpu-memory-utilization` 或 `--max-model-len`

### Q: 如何添加更多 LoRA 模型？
A: 修改 `download_models.py` 添加新模型，然后在 `serve_lora.py` 中加载

### Q: 支持 API 调用吗？
A: 当前是交互式模式，可以基于此扩展 FastAPI 接口

## 技术架构

- **基座模型**: Qwen2.5-32B-Instruct-GPTQ-Int4 (4bit 量化)
- **LoRA 适配器**: 两个独立的 QLoRA 微调模型
- **推理引擎**: vLLM (支持 LoRA 热切换)
- **下载源**: ModelScope (国内加速)

## 注意事项

1. 首次运行会自动下载模型，需要稳定的网络连接
2. 模型文件较大，确保有足够的磁盘空间 (约 20GB+)
3. 所有模型从 ModelScope 下载，无需配置 HF token
4. Python 版本必须为 3.10

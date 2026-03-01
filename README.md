# Qwen 2.5 32B 双专家 QLoRA 推理

基于共享 **Qwen 2.5 32B 4bit GPTQ** 基座，加载两个 QLoRA+Unsloth 微调专家（Expert A / Expert B），支持离线推理与 vLLM 服务部署。适用于 **V100 GPU**。

## 模型说明

| 类型 | 来源 | 模型 ID |
|------|------|---------|
| 基座 | HuggingFace | `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4` |
| 专家 A | ModelScope | `GaryLeenene/qwen25-32b-expert-a-qlora` |
| 专家 B | ModelScope | `GaryLeenene/qwen25-32b-expert-b-qlora` |

## 环境要求

- Python 3.10+
- CUDA 11.8+（V100 驱动）
- 显存：建议 **V100 32GB**；16GB 需降低 `--max-model-len` 或 `--gpu-memory-utilization`

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型

**方式一：Python 脚本（推荐）**

```bash
# 下载基座（HuggingFace）+ 两个专家（ModelScope）
python download_models.py

# 仅下载专家（基座已存在）
python download_models.py --skip-base

# 仅下载基座
python download_models.py --skip-experts
```

**方式二：ModelScope CLI**

```bash
# 安装 modelscope: pip install modelscope
modelscope download --model GaryLeenene/qwen25-32b-expert-a-qlora --local_dir ./models/experts/expert-a
modelscope download --model GaryLeenene/qwen25-32b-expert-b-qlora --local_dir ./models/experts/expert-b
```

### 3. 离线推理

```bash
# 使用专家 A
python run_inference.py --expert expert-a --prompt "你的问题"

# 使用专家 B
python run_inference.py --expert expert-b --prompt "你的问题"

# 仅用基座（不加载 LoRA）
python run_inference.py --expert base --prompt "你的问题"
```

### 4. 启动 vLLM 服务

```bash
chmod +x serve_experts.sh
./serve_experts.sh
```

服务启动后，通过 `model` 参数切换专家：

```bash
# 专家 A
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "expert-a",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 256
  }'

# 专家 B
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "expert-b",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 256
  }'
```

## 命令行参数

### download_models.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base-dir` | `./models/base` | 基座模型保存目录 |
| `--expert-dir` | `./models/experts` | 专家 LoRA 保存目录 |
| `--base-source` | `hf` | 基座来源：`hf` / `modelscope` |
| `--skip-base` | - | 跳过基座下载 |
| `--skip-experts` | - | 跳过专家下载 |

### run_inference.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base-model` | `./models/base` | 基座路径或 HF 模型 ID |
| `--expert-dir` | `./models/experts` | 专家 LoRA 目录 |
| `--expert` | `expert-a` | `expert-a` / `expert-b` / `base` |
| `--prompt` | - | 输入提示 |
| `--max-tokens` | 256 | 最大生成 token 数 |
| `--gpu-memory-utilization` | 0.9 | GPU 显存利用率 |

### 环境变量（serve_experts.sh）

```bash
BASE_MODEL=./models/base    # 或 HuggingFace 模型 ID
EXPERT_DIR=./models/experts
```

## 目录结构

```
.
├── download_models.py   # 模型下载脚本
├── run_inference.py    # 离线推理脚本
├── serve_experts.sh    # vLLM 服务启动脚本
├── requirements.txt
├── README.md
└── models/
    ├── base/           # 基座模型（GPTQ）
    └── experts/
        ├── expert-a/   # 专家 A LoRA
        └── expert-b/   # 专家 B LoRA
```

## 常见问题

**Q: V100 16GB 显存不够？**  
A: 降低 `--max-model-len 2048` 和 `--gpu-memory-utilization 0.85`。

**Q: 基座不下载，直接用 HuggingFace？**  
A: 可以。设置 `BASE_MODEL=Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4` 或 `--base-model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4`，脚本会自动使用远程模型。

**Q: 专家模型下载失败？**  
A: 确保已安装 `modelscope`，国内网络建议使用 ModelScope。若需 HuggingFace，可自行修改 `download_models.py` 中的下载逻辑。

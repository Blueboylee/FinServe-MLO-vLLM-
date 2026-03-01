#!/bin/bash
# Qwen 2.5 32B 双专家 vLLM 服务
# 共享 4bit GPTQ 基座，支持 expert-a / expert-b 切换
# 适用于 V100 GPU

set -e

BASE_MODEL="${BASE_MODEL:-./models/base}"
EXPERT_DIR="${EXPERT_DIR:-./models/experts}"

# 若本地路径不存在，使用 HuggingFace 模型 ID
USE_QUANT=""
if [ ! -d "$BASE_MODEL" ]; then
  BASE_MODEL="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
  USE_QUANT="--quantization gptq"
  echo "使用远程基座: $BASE_MODEL"
elif [ -f "$BASE_MODEL/quantize_config.json" ]; then
  USE_QUANT="--quantization gptq"
  echo "使用本地 GPTQ 基座: $BASE_MODEL"
fi

EXPERT_A="${EXPERT_DIR}/expert-a"
EXPERT_B="${EXPERT_DIR}/expert-b"

# 检查专家目录
for d in "$EXPERT_A" "$EXPERT_B"; do
  if [ ! -d "$d" ]; then
    echo "错误: 专家目录不存在 $d"
    echo "请先运行: python download_models.py"
    exit 1
  fi
done

# V100 优化参数
# --gpu-memory-utilization 0.9  充分利用显存
# --max-model-len 4096          降低可省显存
# --enable-lora --max-loras 2   支持 2 个专家

echo "启动 vLLM 服务..."
echo "基座: $BASE_MODEL"
echo "专家: expert-a, expert-b"
echo ""
echo "请求示例:"
echo "  专家A: curl -X POST http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\": \"expert-a\", \"messages\": [{\"role\": \"user\", \"content\": \"你好\"}], \"max_tokens\": 256}'"
echo "  专家B: curl -X POST http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\": \"expert-b\", \"messages\": [{\"role\": \"user\", \"content\": \"你好\"}], \"max_tokens\": 256}'"
echo ""

vllm serve "$BASE_MODEL" \
  $USE_QUANT \
  --enable-lora \
  --lora-modules expert-a="$EXPERT_A" expert-b="$EXPERT_B" \
  --max-loras 2 \
  --max-lora-rank 64 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000

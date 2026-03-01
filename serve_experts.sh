#!/bin/bash
# vLLM 服务：共享 Qwen2.5 32B GPTQ 基座 + 两个 QLoRA 专家
# 适配 V100 GPU

set -e

BASE_DIR="${BASE_DIR:-./models/base}"
EXPERTS_DIR="${EXPERTS_DIR:-./models/experts}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [[ ! -d "$BASE_DIR" ]]; then
  echo "错误: 基座模型目录不存在: $BASE_DIR"
  echo "请先运行: python download_models.py"
  exit 1
fi

EXPERT_A="${EXPERTS_DIR}/expert-a"
EXPERT_B="${EXPERTS_DIR}/expert-b"
if [[ ! -d "$EXPERT_A" ]] || [[ ! -d "$EXPERT_B" ]]; then
  echo "错误: 专家模型目录不存在"
  echo "请先运行: python download_models.py"
  exit 1
fi

echo "启动 vLLM 服务..."
echo "  基座: $BASE_DIR"
echo "  专家 A: $EXPERT_A"
echo "  专家 B: $EXPERT_B"
echo "  地址: http://${HOST}:${PORT}"
echo ""
echo "请求时通过 model 参数切换专家: expert-a 或 expert-b"
echo ""

vllm serve "$BASE_DIR" \
  --quantization gptq \
  --lora-modules "expert-a=$EXPERT_A" "expert-b=$EXPERT_B" \
  --host "$HOST" \
  --port "$PORT" \
  --gpu-memory-utilization 0.9 \
  --max-lora-rank 64 \
  --max-num-seqs 256

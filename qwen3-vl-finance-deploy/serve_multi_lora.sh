#!/bin/bash
# ============================================
#  方案一：vLLM 多 LoRA 动态加载服务
#  基座模型 + 两个 LoRA 专家共享 GPU 显存
#  通过请求中 model 字段切换专家
# ============================================

MODEL_DIR="./models"
BASE_MODEL="$MODEL_DIR/Qwen3-VL-8B-Instruct-AWQ-4bit"
EXPERT_A="$MODEL_DIR/Qwen3-VL-Finance-expert-a"
EXPERT_B="$MODEL_DIR/Qwen3-VL-Finance-expert-b"

HOST="0.0.0.0"
PORT=8000
MAX_MODEL_LEN=4096
GPU_UTIL=0.90

echo "============================================"
echo "  启动 vLLM 多 LoRA 服务"
echo "============================================"
echo "  基座模型: $BASE_MODEL"
echo "  Expert-A: $EXPERT_A"
echo "  Expert-B: $EXPERT_B"
echo "  端口: $PORT"
echo "============================================"

# 检查 LoRA adapter_config.json 中的 rank，并映射到 vLLM 支持的 max-lora-rank
# vLLM 允许的取值：1, 8, 16, 32, 64, 128, 256, 320, 512
LORA_RANK=64
if [ -f "$EXPERT_A/adapter_config.json" ]; then
    DETECTED_RANK=$(python3 -c "import json; print(json.load(open('$EXPERT_A/adapter_config.json'))['r'])" 2>/dev/null)
    if [ -n "$DETECTED_RANK" ]; then
        echo "  检测到 LoRA rank: $DETECTED_RANK"
        if   [ "$DETECTED_RANK" -le 1 ];   then LORA_RANK=1
        elif [ "$DETECTED_RANK" -le 8 ];   then LORA_RANK=8
        elif [ "$DETECTED_RANK" -le 16 ];  then LORA_RANK=16
        elif [ "$DETECTED_RANK" -le 32 ];  then LORA_RANK=32
        elif [ "$DETECTED_RANK" -le 64 ];  then LORA_RANK=64
        elif [ "$DETECTED_RANK" -le 128 ]; then LORA_RANK=128
        elif [ "$DETECTED_RANK" -le 256 ]; then LORA_RANK=256
        elif [ "$DETECTED_RANK" -le 320 ]; then LORA_RANK=320
        else LORA_RANK=512
        fi
        echo "  实际使用的 max-lora-rank: $LORA_RANK"
    fi
fi

vllm serve "$BASE_MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --trust-remote-code \
    --enable-lora \
    --max-loras 2 \
    --max-lora-rank "$LORA_RANK" \
    --max-cpu-loras 2 \
    --lora-modules "finance-expert-a=$EXPERT_A" "finance-expert-b=$EXPERT_B" \
    --limit-mm-per-prompt '{"image": 4, "video": 1}'

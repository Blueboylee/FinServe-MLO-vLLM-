#!/bin/bash
# ============================================
#  方案二：合并后模型的 vLLM 服务启动
#  需要先运行 merge_lora_and_serve.py 完成合并
#  两个专家分别在不同端口运行
# ============================================

MODEL_DIR="./models"
HOST="0.0.0.0"
MAX_MODEL_LEN=4096
GPU_UTIL=0.90

EXPERT=${1:-a}  # 默认启动 expert-a，可传参 a 或 b

if [ "$EXPERT" = "a" ]; then
    MODEL_PATH="$MODEL_DIR/Qwen3-VL-Finance-merged-expert-a"
    PORT=8000
    echo "启动合并后的 Expert-A (端口: $PORT)..."
elif [ "$EXPERT" = "b" ]; then
    MODEL_PATH="$MODEL_DIR/Qwen3-VL-Finance-merged-expert-b"
    PORT=8001
    echo "启动合并后的 Expert-B (端口: $PORT)..."
else
    echo "用法: $0 [a|b]"
    exit 1
fi

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --trust-remote-code \
    --limit-mm-per-prompt "image=4,video=1"

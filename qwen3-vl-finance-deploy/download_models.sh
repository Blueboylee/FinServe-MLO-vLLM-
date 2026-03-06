#!/bin/bash
set -e

MODEL_DIR="./models"
mkdir -p "$MODEL_DIR"

echo "============================================"
echo "  下载模型文件"
echo "============================================"

echo ""
echo "[1/3] 下载基座模型: Qwen3-VL-8B-Instruct-AWQ-4bit ..."
modelscope download --model cpatonn-mirror/Qwen3-VL-8B-Instruct-AWQ-4bit \
    --local_dir "$MODEL_DIR/Qwen3-VL-8B-Instruct-AWQ-4bit"

echo ""
echo "[2/3] 下载 LoRA Expert-A: Qwen3-VL-Finance-expert-a ..."
modelscope download --model GaryLeenene/Qwen3-VL-Finance-expert-a \
    --local_dir "$MODEL_DIR/Qwen3-VL-Finance-expert-a"

echo ""
echo "[3/3] 下载 LoRA Expert-B: Qwen3-VL-Finance-expert-b ..."
modelscope download --model GaryLeenene/Qwen3-VL-Finance-expert-b \
    --local_dir "$MODEL_DIR/Qwen3-VL-Finance-expert-b"

echo ""
echo "[✓] 所有模型下载完成！"
echo "    基座模型: $MODEL_DIR/Qwen3-VL-8B-Instruct-AWQ-4bit"
echo "    Expert-A: $MODEL_DIR/Qwen3-VL-Finance-expert-a"
echo "    Expert-B: $MODEL_DIR/Qwen3-VL-Finance-expert-b"

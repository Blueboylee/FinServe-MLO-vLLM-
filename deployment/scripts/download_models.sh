#!/bin/bash
set -e

echo "=========================================="
echo "开始下载专家模型"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$BASE_DIR/models"

echo "模型保存目录: $MODELS_DIR"
mkdir -p "$MODELS_DIR"

echo "[1/3] 激活conda环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen_experts

echo "[2/3] 下载专家A模型 (GaryLeenene/qwen25-32b-expert-a-qlora)..."
if [ -d "$MODELS_DIR/qwen25-32b-expert-a-qlora" ]; then
    echo "专家A模型已存在，跳过下载"
else
    modelscope download --model GaryLeenene/qwen25-32b-expert-a-qlora --local_dir "$MODELS_DIR/qwen25-32b-expert-a-qlora"
fi

echo "[3/3] 下载专家B模型 (GaryLeenene/qwen25-32b-expert-b-qlora)..."
if [ -d "$MODELS_DIR/qwen25-32b-expert-b-qlora" ]; then
    echo "专家B模型已存在，跳过下载"
else
    modelscope download --model GaryLeenene/qwen25-32b-expert-b-qlora --local_dir "$MODELS_DIR/qwen25-32b-expert-b-qlora"
fi

echo "=========================================="
echo "下载Qwen2.5 32B基础模型 (4bit AWQ版本)..."
echo "=========================================="

BASE_MODEL_DIR="$MODELS_DIR/qwen25-32b-awq"
if [ -d "$BASE_MODEL_DIR" ]; then
    echo "基础模型已存在，跳过下载"
else
    echo "下载Qwen2.5 32B 4bit AWQ基础模型..."
    modelscope download --model qwen/Qwen2.5-32B-Instruct-AWQ --local_dir "$BASE_MODEL_DIR"
fi

echo "=========================================="
echo "模型下载完成！"
echo "=========================================="
echo "专家A模型: $MODELS_DIR/qwen25-32b-expert-a-qlora"
echo "专家B模型: $MODELS_DIR/qwen25-32b-expert-b-qlora"
echo "基础模型:  $BASE_MODEL_DIR"
echo "=========================================="

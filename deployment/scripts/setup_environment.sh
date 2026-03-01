#!/bin/bash
set -e

echo "=========================================="
echo "开始部署Qwen2.5 32B专家模型系统"
echo "=========================================="

CONDA_ENV="qwen_experts"
PYTHON_VERSION="3.10"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

echo "[1/5] 检查conda环境..."
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到conda，请先安装Miniconda或Anaconda"
    exit 1
fi

echo "[2/5] 创建conda环境: $CONDA_ENV (Python $PYTHON_VERSION)..."
if conda info --envs | grep -q "$CONDA_ENV"; then
    echo "conda环境 $CONDA_ENV 已存在，跳过创建"
else
    conda create -y -n $CONDA_ENV python=$PYTHON_VERSION
fi

echo "[3/5] 激活conda环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV

echo "[4/5] 安装基础依赖..."
pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate peft bitsandbytes
pip install unsloth
pip install modelscope

echo "[5/5] 安装vLLM..."
pip install vllm

echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo "conda环境: $CONDA_ENV"
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA版本: $(python -c 'import torch; print(torch.version.cuda)')"
echo "=========================================="

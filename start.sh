#!/bin/bash
# Ubuntu 22.04 服务器启动脚本
# 用于一键启动 Qwen2.5 32B LoRA 服务

set -e

echo "=========================================="
echo "Qwen2.5 32B LoRA 服务启动脚本"
echo "=========================================="

# 检查 Python 版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "检测到 Python 版本：$PYTHON_VERSION"

if [[ ! $PYTHON_VERSION =~ ^3.10 ]]; then
    echo "警告：建议使用 Python 3.10，当前版本为 $PYTHON_VERSION"
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 检查依赖
echo "检查依赖..."
if ! python -c "import vllm" 2>/dev/null; then
    echo "安装依赖..."
    pip install -r requirements.txt -q
fi

# 检查模型
if [ ! -f "models/model_config.txt" ]; then
    echo "模型未下载，开始下载..."
    python download_models.py
fi

# 启动服务
echo ""
echo "=========================================="
echo "启动 vLLM 服务..."
echo "=========================================="
echo ""

python serve_lora.py "$@"

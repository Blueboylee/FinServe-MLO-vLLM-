#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/setup.log"

mkdir -p "$PROJECT_ROOT/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "开始安装 Qwen2.5 双专家模型依赖"
log "=========================================="

log "检查 conda 环境..."
if ! command -v conda &> /dev/null; then
    log "错误：未找到 conda，请先安装 conda"
    exit 1
fi

ENV_NAME="qwen25-dual-expert"

if conda env list | grep -q "^$ENV_NAME "; then
    log "环境 $ENV_NAME 已存在，正在删除..."
    conda env remove -n $ENV_NAME -y
fi

log "创建 conda 环境..."
conda env create -f "$SCRIPT_DIR/../configs/conda_environment.yml" -n $ENV_NAME

log "激活环境并安装 Python 依赖..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

log "安装 ModelScope..."
pip install modelscope==1.15.0

log "安装 vLLM..."
pip install vllm==0.6.1

log "安装 Unsloth..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

log "安装其他依赖..."
pip install transformers==4.44.2
pip install accelerate==0.33.0
pip install peft==0.12.0
pip install bitsandbytes==0.43.3
pip install protobuf
pip install sentencepiece

log "验证安装..."
python -c "import torch; print(f'PyTorch 版本：{torch.__version__}')"
python -c "import vllm; print(f'vLLM 版本：{vllm.__version__}')"
python -c "import modelscope; print(f'ModelScope 版本：{modelscope.__version__}')"

log "=========================================="
log "依赖安装完成！"
log "=========================================="
log "使用以下命令激活环境："
log "  conda activate $ENV_NAME"
log "=========================================="

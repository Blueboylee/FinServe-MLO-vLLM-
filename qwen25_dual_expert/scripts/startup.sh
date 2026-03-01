#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/startup.log"

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error_exit() {
    log "错误：$1"
    exit 1
}

log "=========================================="
log "Qwen2.5 双专家模型启动流程"
log "=========================================="

log "检查 conda 环境..."
if ! command -v conda &> /dev/null; then
    error_exit "未找到 conda，请先安装 conda"
fi

ENV_NAME="qwen25-dual-expert"

if ! conda env list | grep -q "^$ENV_NAME "; then
    log "环境 $ENV_NAME 不存在，正在创建..."
    bash "$SCRIPT_DIR/install_dependencies.sh" || error_exit "环境创建失败"
fi

log "激活 conda 环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME || error_exit "环境激活失败"

log "检查模型文件..."
MODELS_DIR="$PROJECT_ROOT/models"
if [ ! -d "$MODELS_DIR" ]; then
    log "模型目录不存在，开始下载模型..."
    python "$SCRIPT_DIR/download_models.py" || error_exit "模型下载失败"
fi

BASE_MODEL="$MODELS_DIR/base"
EXPERT_A="$MODELS_DIR/expert-a"
EXPERT_B="$MODELS_DIR/expert-b"

if [ ! -d "$BASE_MODEL" ] || [ ! -d "$EXPERT_A" ] || [ ! -d "$EXPERT_B" ]; then
    log "模型文件不完整，重新下载..."
    python "$SCRIPT_DIR/download_models.py" || error_exit "模型下载失败"
fi

log "验证模型文件..."
python -c "
import sys
from pathlib import Path

models = {
    'base': '$BASE_MODEL',
    'expert-a': '$EXPERT_A',
    'expert-b': '$EXPERT_B'
}

for name, path in models.items():
    p = Path(path)
    if not p.exists():
        print(f'错误：{name} 目录不存在')
        sys.exit(1)
    
    files = list(p.glob('*'))
    if len(files) == 0:
        print(f'错误：{name} 目录为空')
        sys.exit(1)
    
    print(f'✓ {name}: {len(files)} 个文件')

print('所有模型文件验证通过')
" || error_exit "模型验证失败"

log "=========================================="
log "启动准备完成！"
log "=========================================="
log ""
log "可用命令："
log "  API 模式：python $SCRIPT_DIR/deploy.py --mode api --port 8000"
log "  交互模式：python $SCRIPT_DIR/deploy.py --mode interactive"
log ""
log "API 端点："
log "  生成接口：POST http://localhost:8000/generate"
log "  健康检查：GET http://localhost:8000/health"
log ""
log "日志文件：$LOG_FILE"
log "=========================================="

if [ "$1" == "--auto" ]; then
    log "自动启动 API 服务..."
    python "$SCRIPT_DIR/deploy.py" --mode api --port 8000
else
    log "请手动选择启动模式"
fi

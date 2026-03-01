#!/bin/bash
# 快速修复脚本 - 创建日志目录并重新运行下载

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

echo "创建日志目录..."
mkdir -p "$LOG_DIR"

echo "日志目录已创建：$LOG_DIR"
echo ""
echo "现在可以重新运行下载脚本："
echo "  python $SCRIPT_DIR/download_models.py"

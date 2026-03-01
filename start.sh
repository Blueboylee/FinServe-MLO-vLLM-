#!/bin/bash
# 快速启动脚本

echo "=========================================="
echo "启动 Qwen2.5-32B 双专家模型服务"
echo "=========================================="

# 检查模型是否已下载
if [ ! -f "models/model_paths.txt" ]; then
    echo "✗ 模型未下载，请先运行：bash install.sh 或 python download_models.py"
    exit 1
fi

# 默认参数
PORT=8000
GPU_UTILIZATION=0.85
MAX_MODEL_LEN=4096

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --gpu-utilization)
            GPU_UTILIZATION="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        *)
            echo "未知参数：$1"
            echo "用法：./start.sh [--port 8000] [--gpu-utilization 0.85] [--max-model-len 4096]"
            exit 1
            ;;
    esac
done

echo ""
echo "配置信息:"
echo "  端口：$PORT"
echo "  GPU 内存利用率：$GPU_UTILIZATION"
echo "  最大序列长度：$MAX_MODEL_LEN"
echo ""

# 启动服务器
echo "启动 API 服务器..."
python api_server.py \
    --port $PORT \
    --gpu-memory-utilization $GPU_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN

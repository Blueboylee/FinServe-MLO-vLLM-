#!/bin/bash
# ============================================
#  FinServe vLLM 生产服务压测
#  针对 serve_multi_lora.sh 启动的 HTTP 端点
# ============================================
#
#  用法:
#    ./bench_serving.sh                       # 默认 tag=baseline
#    ./bench_serving.sh triton                # 指定 tag
#    ./bench_serving.sh baseline 30 512 16    # TAG NUM_REQ MAX_TOK MAX_CONC
#    ./bench_serving.sh --compare baseline triton  # 对比模式
#
# ============================================

cd "$(dirname "$0")"

# 安装依赖
python3 -c "import aiohttp" 2>/dev/null || {
    echo "[依赖] 安装 aiohttp..."
    pip install aiohttp -q
}

# 对比模式直接透传
if [ "$1" = "--compare" ]; then
    python3 bench_serving.py --compare "$2" "$3"
    exit $?
fi

TAG="${1:-baseline}"
NUM_REQUESTS="${2:-20}"
MAX_TOKENS="${3:-512}"
MAX_CONCURRENCY="${4:-16}"
URL="${BENCH_URL:-http://127.0.0.1:8000}"

echo "============================================"
echo "  FinServe vLLM 生产服务压测"
echo "============================================"
echo "  服务地址:     $URL"
echo "  压测标签:     $TAG"
echo "  每阶段请求:   $NUM_REQUESTS"
echo "  最大 Token:   $MAX_TOKENS"
echo "  最大并发:     $MAX_CONCURRENCY"
echo "============================================"
echo ""

# 检查服务
if ! curl -sf "$URL/v1/models" > /dev/null 2>&1; then
    echo "ERROR: vLLM 服务未启动 ($URL)"
    echo ""
    echo "请先运行:"
    echo "  DEPLOY_MODE=single ./serve_multi_lora.sh"
    exit 1
fi

python3 bench_serving.py \
    --url "$URL" \
    --tag "$TAG" \
    --num-requests "$NUM_REQUESTS" \
    --max-tokens "$MAX_TOKENS" \
    --max-concurrency "$MAX_CONCURRENCY"

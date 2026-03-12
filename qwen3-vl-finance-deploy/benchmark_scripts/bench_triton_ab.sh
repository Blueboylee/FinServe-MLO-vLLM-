#!/bin/bash
# ============================================
#  Triton Kernel A/B 对比压测
#  vLLM 原生 vs vLLM + 自定义 Triton kernel
# ============================================
#
#  用法:
#    ./bench_triton_ab.sh           # 默认 30 次
#    ./bench_triton_ab.sh 50        # 50 次
#    ./bench_triton_ab.sh 30 256    # 30次, max_tokens=256
#
#  不需要启动 HTTP 服务，脚本自己加载模型跑离线推理。
# ============================================

cd "$(dirname "$0")"

RUNS="${1:-30}"
MAX_TOKENS="${2:-512}"
WARMUP="${3:-5}"

echo "============================================"
echo "  Triton Kernel A/B 对比压测"
echo "============================================"
echo "  推理次数:   $RUNS"
echo "  Max Tokens: $MAX_TOKENS"
echo "  Warmup:     $WARMUP"
echo "============================================"
echo ""

python3 bench_triton_ab.py \
    --runs "$RUNS" \
    --warmup "$WARMUP" \
    --max-tokens "$MAX_TOKENS"

#!/bin/bash
# ============================================
#  Triton 批量压测 — 多方案长时间运行
#  适合睡前启动，睡醒看结果
# ============================================
#
#  用法:
#    ./bench_triton_batch.sh           # 默认跑 10 小时
#    ./bench_triton_batch.sh 6         # 跑 6 小时
#    ./bench_triton_batch.sh 10 0,1,2  # 10h 且只跑方案 0,1,2
#
#  结果:
#    bench_results/batch/summary.jsonl   — 每方案一行 JSON
#    bench_results/batch/FINAL_SUMMARY.md — 最终汇总
#
# ============================================

cd "$(dirname "$0")"

MAX_HOURS="${1:-10}"
SCHEMES="${2:-}"

echo "============================================"
echo "  Triton 批量压测"
echo "============================================"
echo "  最长时间: ${MAX_HOURS}h"
echo "  方案筛选: ${SCHEMES:-全部}"
echo "  结果目录: bench_results/batch/"
echo "============================================"
echo ""
echo "  启动时间: $(date)"
echo ""

if [ -n "$SCHEMES" ]; then
    python3 bench_triton_batch.py --max-hours "$MAX_HOURS" --schemes "$SCHEMES"
else
    python3 bench_triton_batch.py --max-hours "$MAX_HOURS"
fi

echo ""
echo "  结束时间: $(date)"
echo ""
echo "  查看汇总: cat bench_results/batch/FINAL_SUMMARY.md"
echo ""

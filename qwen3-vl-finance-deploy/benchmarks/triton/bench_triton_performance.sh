#!/bin/bash
# ============================================
#  Triton Kernel 性能对比压测
#  在真实服务中直接对比使用和不使用 Triton 的性能差异
# ============================================

cd "$(dirname "$0")"

echo "============================================"
echo "  Triton Kernel 性能对比压测 (真实服务)"
echo "============================================"
echo ""

# 检查 vLLM 是否可用
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "ERROR: vLLM 未安装"
    exit 1
fi

# 检查 Triton 是否可用
if ! python3 -c "import triton" 2>/dev/null; then
    echo "ERROR: Triton 未安装"
    exit 1
fi

# 运行性能对比压测
echo "运行 Triton 性能对比压测..."
echo ""

python3 serve_multi_lora_triton.py --mode compare

echo ""
echo "============================================"
echo "  压测完成！"
echo "============================================"

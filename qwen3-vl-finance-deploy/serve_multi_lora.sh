#!/bin/bash
# ============================================
#  vLLM 多 LoRA 服务 (PD 优化版)
#  支持: 单实例 Chunked Prefill / PD 分离架构
#  解决专家长 Prefill 对基座请求的队头阻塞
# ============================================

MODEL_DIR="./models"
BASE_MODEL="$MODEL_DIR/Qwen3-VL-8B-Instruct-AWQ-4bit"
EXPERT_A="$MODEL_DIR/Qwen3-VL-Finance-expert-a"
EXPERT_B="$MODEL_DIR/Qwen3-VL-Finance-expert-b"

HOST="0.0.0.0"
PORT=8000
MAX_MODEL_LEN=4096
GPU_UTIL=0.90

# ---------- 优化配置 ----------
# 部署模式: single (默认, 单 GPU) | disagg (PD 分离, 需 2+ GPU)
DEPLOY_MODE="${DEPLOY_MODE:-single}"

# Chunked Prefill: 将 Expert 的 2500~3000 token Prefill 拆为小块,
# 与 Decode 步骤交替执行, 直接消除队头阻塞
ENABLE_CHUNKED_PREFILL=true
MAX_NUM_BATCHED_TOKENS=512

# Prefix Caching: 缓存 Expert 的 system prompt 和固定格式前缀的 KV Cache
# Expert-A 的银行交易记录格式、Expert-B 的新闻分析格式可大幅复用
ENABLE_PREFIX_CACHING=true

# PD 分离配置 (仅 DEPLOY_MODE=disagg 时生效)
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
PREFILL_PORT=8000
DECODE_PORT=8001

echo "============================================"
echo "  vLLM 多 LoRA 服务 [PD 优化版]"
echo "============================================"
echo "  基座模型:        $BASE_MODEL"
echo "  Expert-A:        $EXPERT_A"
echo "  Expert-B:        $EXPERT_B"
echo "  部署模式:        $DEPLOY_MODE"
echo "  Chunked Prefill: $ENABLE_CHUNKED_PREFILL (chunk=$MAX_NUM_BATCHED_TOKENS)"
echo "  Prefix Caching:  $ENABLE_PREFIX_CACHING"
echo "============================================"

# 检查 LoRA adapter_config.json 中的 rank，并映射到 vLLM 支持的 max-lora-rank
# vLLM 允许的取值：1, 8, 16, 32, 64, 128, 256, 320, 512
LORA_RANK=64
if [ -f "$EXPERT_A/adapter_config.json" ]; then
    DETECTED_RANK=$(python3 -c "import json; print(json.load(open('$EXPERT_A/adapter_config.json'))['r'])" 2>/dev/null)
    if [ -n "$DETECTED_RANK" ]; then
        echo "  检测到 LoRA rank: $DETECTED_RANK"
        if   [ "$DETECTED_RANK" -le 1 ];   then LORA_RANK=1
        elif [ "$DETECTED_RANK" -le 8 ];   then LORA_RANK=8
        elif [ "$DETECTED_RANK" -le 16 ];  then LORA_RANK=16
        elif [ "$DETECTED_RANK" -le 32 ];  then LORA_RANK=32
        elif [ "$DETECTED_RANK" -le 64 ];  then LORA_RANK=64
        elif [ "$DETECTED_RANK" -le 128 ]; then LORA_RANK=128
        elif [ "$DETECTED_RANK" -le 256 ]; then LORA_RANK=256
        elif [ "$DETECTED_RANK" -le 320 ]; then LORA_RANK=320
        else LORA_RANK=512
        fi
        echo "  实际使用的 max-lora-rank: $LORA_RANK"
    fi
fi

# -------- 构建优化参数 --------
OPT_FLAGS=""
if [ "$ENABLE_CHUNKED_PREFILL" = true ]; then
    OPT_FLAGS="$OPT_FLAGS --enable-chunked-prefill --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS"
fi
if [ "$ENABLE_PREFIX_CACHING" = true ]; then
    OPT_FLAGS="$OPT_FLAGS --enable-prefix-caching"
fi

COMMON_FLAGS="--trust-remote-code --limit-mm-per-prompt '{\"image\": 4, \"video\": 1}'"

# -------- 按模式启动 --------
if [ "$DEPLOY_MODE" = "disagg" ]; then
    echo ""
    echo "[PD 分离] Prefill 节点 (GPU $PREFILL_GPU, :$PREFILL_PORT) → Expert A/B 长输入"
    echo "[PD 分离] Decode 节点  (GPU $DECODE_GPU, :$DECODE_PORT)  → 基座模型低延迟"
    echo ""

    # Prefill 节点: 挂载 LoRA 专家, 处理 2500~3000 token 长 Prefill
    CUDA_VISIBLE_DEVICES=$PREFILL_GPU vllm serve "$BASE_MODEL" \
        --host "$HOST" \
        --port "$PREFILL_PORT" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --trust-remote-code \
        --enable-lora \
        --max-loras 2 \
        --max-lora-rank "$LORA_RANK" \
        --max-cpu-loras 2 \
        --lora-modules "finance-expert-a=$EXPERT_A" "finance-expert-b=$EXPERT_B" \
        --limit-mm-per-prompt '{"image": 4, "video": 1}' \
        $OPT_FLAGS &
    PREFILL_PID=$!

    # Decode 节点: 纯基座模型, 无 LoRA 开销, 专注低延迟短请求
    CUDA_VISIBLE_DEVICES=$DECODE_GPU vllm serve "$BASE_MODEL" \
        --host "$HOST" \
        --port "$DECODE_PORT" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --trust-remote-code \
        --limit-mm-per-prompt '{"image": 4, "video": 1}' \
        $OPT_FLAGS &
    DECODE_PID=$!

    echo "  Prefill PID=$PREFILL_PID  Decode PID=$DECODE_PID"
    echo ""
    echo "  启动代理 (多上游模式):"
    echo "    python web_proxy_server.py \\"
    echo "      --upstream-expert http://127.0.0.1:$PREFILL_PORT \\"
    echo "      --upstream-base http://127.0.0.1:$DECODE_PORT"

    cleanup() { kill $PREFILL_PID $DECODE_PID 2>/dev/null; }
    trap cleanup EXIT INT TERM
    wait -n $PREFILL_PID $DECODE_PID

else
    # 单实例模式: Chunked Prefill + Prefix Caching 优化
    vllm serve "$BASE_MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --trust-remote-code \
        --enable-lora \
        --max-loras 2 \
        --max-lora-rank "$LORA_RANK" \
        --max-cpu-loras 2 \
        --lora-modules "finance-expert-a=$EXPERT_A" "finance-expert-b=$EXPERT_B" \
        --limit-mm-per-prompt '{"image": 4, "video": 1}' \
        $OPT_FLAGS
fi

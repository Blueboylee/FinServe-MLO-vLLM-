#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/data_env.sh"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-$DATA_ROOT/models/Qwen3-VL-8B-Instruct}"
OUTPUT_NAME="${OUTPUT_NAME:-Qwen3-VL-Finance-expert-c}"
DATASET_NAME="${DATASET_NAME:-nohurry/Opus-4.6-Reasoning-3000x-filtered}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-200}"
MAX_STEPS="${MAX_STEPS:--1}"

echo "============================================"
echo "  Qwen3-VL Unsloth QLoRA 微调"
echo "============================================"
echo "DATA_ROOT      : $DATA_ROOT"
echo "BASE_MODEL_DIR : $BASE_MODEL_DIR"
echo "DATASET        : $DATASET_NAME"
echo "SLICE          : $START_INDEX..$END_INDEX"
echo "OUTPUT_NAME    : $OUTPUT_NAME"
echo "MAX_STEPS      : $MAX_STEPS"
echo ""

if ! python - <<'PY'
import importlib.util
mods = ["datasets", "trl", "unsloth", "modelscope"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
raise SystemExit(0 if not missing else 1)
PY
then
  echo "[x] 当前 Python 环境缺少训练依赖。"
  echo "    请先激活训练环境，例如："
  echo "    conda activate /data/qwen3-vl-finance-expert-c/conda/envs/qwen3-vl-unsloth"
  exit 1
fi

python "$SCRIPT_DIR/train_qwen3_vl_unsloth_qlora.py" \
  --model_name_or_path "$BASE_MODEL_DIR" \
  --dataset_name "$DATASET_NAME" \
  --start_index "$START_INDEX" \
  --end_index "$END_INDEX" \
  --data_root "$DATA_ROOT" \
  --output_name "$OUTPUT_NAME" \
  --max_steps "$MAX_STEPS"

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/data_env.sh"

REPO_ID="${REPO_ID:-GaryLeenene/Qwen3-VL-Finance-expert-c}"
OUTPUT_NAME="${OUTPUT_NAME:-Qwen3-VL-Finance-expert-c}"
VISIBILITY="${VISIBILITY:-public}"
CHINESE_NAME="${CHINESE_NAME:-Qwen3-VL 金融专家}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Upload QLoRA adapter trained with Unsloth}"

if [[ $# -ge 1 ]]; then
  LOCAL_DIR="$1"
else
  latest_run="$(ls -dt "$DATA_ROOT/outputs/${OUTPUT_NAME}-"* 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest_run:-}" ]]; then
    echo "[x] 没有找到训练输出目录，请先运行训练脚本，或手动传入 adapter 路径。"
    exit 1
  fi
  LOCAL_DIR="$latest_run/adapter"
fi

echo "============================================"
echo "  上传 LoRA 到 ModelScope"
echo "============================================"
echo "Repo      : $REPO_ID"
echo "Local Dir : $LOCAL_DIR"
echo ""

if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("modelscope") else 1)
PY
then
  echo "[x] 当前 Python 环境缺少 modelscope 依赖。"
  echo "    请先激活训练环境，例如："
  echo "    conda activate /data/qwen3-vl-finance-expert-c/conda/envs/qwen3-vl-unsloth"
  exit 1
fi

python "$SCRIPT_DIR/upload_lora_to_modelscope.py" \
  --local_dir "$LOCAL_DIR" \
  --repo_id "$REPO_ID" \
  --visibility "$VISIBILITY" \
  --chinese_name "$CHINESE_NAME" \
  --commit_message "$COMMIT_MESSAGE"

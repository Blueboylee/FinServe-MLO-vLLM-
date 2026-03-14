#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/data_env.sh"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-$DATA_ROOT/models/Qwen3-VL-8B-Instruct}"
DATASET_NAME="${DATASET_NAME:-nohurry/Opus-4.6-Reasoning-3000x-filtered}"
UPLOAD_AFTER_TRAIN="${UPLOAD_AFTER_TRAIN:-1}"
DELETE_LOCAL_AFTER_UPLOAD="${DELETE_LOCAL_AFTER_UPLOAD:-1}"
EXPORT_PURE_LORA="${EXPORT_PURE_LORA:-1}"
START_LETTER="${START_LETTER:-c}"
END_LETTER="${END_LETTER:-l}"

if [[ "$UPLOAD_AFTER_TRAIN" == "1" && -z "${MODELSCOPE_API_TOKEN:-}" ]]; then
  echo "[x] 需要先设置 MODELSCOPE_API_TOKEN，才能批量上传。"
  exit 1
fi

echo "============================================"
echo "  批量训练并上传 Expert-C ~ Expert-L"
echo "============================================"
echo "BASE_MODEL_DIR            : $BASE_MODEL_DIR"
echo "DATASET                   : $DATASET_NAME"
echo "UPLOAD_AFTER_TRAIN        : $UPLOAD_AFTER_TRAIN"
echo "DELETE_LOCAL_AFTER_UPLOAD : $DELETE_LOCAL_AFTER_UPLOAD"
echo "EXPORT_PURE_LORA          : $EXPORT_PURE_LORA"
echo ""

letters=(c d e f g h i j k l)
starts=(0 200 401 601 801 1001 1201 1401 1601 1801)
ends=(200 400 600 800 1000 1200 1400 1600 1800 2000)

for idx in "${!letters[@]}"; do
  letter="${letters[$idx]}"

  if [[ "$letter" < "$START_LETTER" || "$letter" > "$END_LETTER" ]]; then
    continue
  fi

  start_index="${starts[$idx]}"
  end_index="${ends[$idx]}"
  upper_letter="$(printf '%s' "$letter" | tr '[:lower:]' '[:upper:]')"
  output_name="Qwen3-VL-Finance-expert-$letter"
  repo_id="GaryLeenene/$output_name"

  echo "--------------------------------------------"
  echo "训练 Expert-$upper_letter: $start_index..$end_index"
  echo "Repo: $repo_id"
  echo "--------------------------------------------"

  OUTPUT_NAME="$output_name" \
  START_INDEX="$start_index" \
  END_INDEX="$end_index" \
  DATASET_NAME="$DATASET_NAME" \
  BASE_MODEL_DIR="$BASE_MODEL_DIR" \
  bash "$SCRIPT_DIR/run_train_qlora_unsloth.sh"

  latest_run="$(ls -dt "$DATA_ROOT/outputs/${output_name}-"* 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest_run:-}" ]]; then
    echo "[x] 未找到训练输出目录: $output_name"
    exit 1
  fi

  publish_dir="$latest_run/adapter"
  if [[ "$EXPORT_PURE_LORA" == "1" ]]; then
    publish_dir="$latest_run/publish_lora"
    python "$SCRIPT_DIR/prepare_pure_lora_adapter.py" \
      --source_dir "$latest_run/adapter" \
      --target_dir "$publish_dir"
  fi

  if [[ "$UPLOAD_AFTER_TRAIN" == "1" ]]; then
    REPO_ID="$repo_id" \
    OUTPUT_NAME="$output_name" \
    CHINESE_NAME="Qwen3-VL 金融专家 ${upper_letter}" \
    COMMIT_MESSAGE="Upload Expert-${upper_letter} pure LoRA trained on shard ${start_index}-${end_index}" \
    bash "$SCRIPT_DIR/run_upload_modelscope.sh" "$publish_dir"
  fi

  if [[ "$UPLOAD_AFTER_TRAIN" == "1" && "$DELETE_LOCAL_AFTER_UPLOAD" == "1" ]]; then
    rm -rf "$latest_run"
    echo "[i] 已清理本地训练目录: $latest_run"
  fi
done

echo ""
echo "[✓] Expert-C ~ Expert-L 批处理完成"

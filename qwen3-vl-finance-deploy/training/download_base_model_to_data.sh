#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/data_env.sh"

BASE_MODEL_ID="${BASE_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-$DATA_ROOT/models/Qwen3-VL-8B-Instruct}"

mkdir -p "$(dirname "$BASE_MODEL_DIR")"

echo "============================================"
echo "  下载 Qwen3-VL-8B-Instruct 到 /data"
echo "============================================"
echo "Model ID   : $BASE_MODEL_ID"
echo "Local Path : $BASE_MODEL_DIR"
echo "Cache Path : $MODELSCOPE_CACHE"
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

cd "$DATA_ROOT"

python - <<PY
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download

try:
    snapshot_download(
        model_id="${BASE_MODEL_ID}",
        cache_dir="${MODELSCOPE_CACHE}",
        local_dir="${BASE_MODEL_DIR}",
    )
except UnicodeEncodeError as exc:
    print(f"[i] ModelScope 下载触发编码问题，自动回退到 Hugging Face: {exc}")
    from huggingface_hub import snapshot_download as hf_snapshot_download

    Path("${BASE_MODEL_DIR}").mkdir(parents=True, exist_ok=True)
    hf_snapshot_download(
        repo_id="${BASE_MODEL_ID}",
        local_dir="${BASE_MODEL_DIR}",
        local_dir_use_symlinks=False,
        cache_dir="${HUGGINGFACE_HUB_CACHE}",
    )
PY

echo ""
echo "[✓] 基座模型下载完成: $BASE_MODEL_DIR"

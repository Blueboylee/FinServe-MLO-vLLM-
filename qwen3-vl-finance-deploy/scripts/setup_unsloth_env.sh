#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/data_env.sh"

ENV_NAME="${ENV_NAME:-qwen3-vl-unsloth}"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/environment.unsloth.yml}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$SCRIPT_DIR/requirements.unsloth.txt}"
CONDA_DATA_ROOT="${CONDA_DATA_ROOT:-$DATA_ROOT/conda}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$CONDA_DATA_ROOT/pkgs}"
CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$CONDA_DATA_ROOT/envs}"
ENV_PREFIX="${ENV_PREFIX:-$CONDA_ENVS_PATH/$ENV_NAME}"
ENV_PYTHON="$ENV_PREFIX/bin/python"

mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH"

echo "============================================"
echo "  Qwen3-VL Unsloth 训练环境安装"
echo "  （conda 独立环境，写入 /data）"
echo "============================================"
echo "ENV_NAME        : $ENV_NAME"
echo "ENV_FILE        : $ENV_FILE"
echo "REQ_FILE        : $REQUIREMENTS_FILE"
echo "ENV_PREFIX      : $ENV_PREFIX"
echo "CONDA_PKGS_DIRS : $CONDA_PKGS_DIRS"
echo ""

if ! command -v conda >/dev/null 2>&1; then
  for candidate in \
    "/miniconda3/etc/profile.d/conda.sh" \
    "/root/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"
  do
    if [[ -f "$candidate" ]]; then
      # shellcheck disable=SC1090
      source "$candidate"
      break
    fi
  done
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[x] 环境文件不存在: $ENV_FILE"
  exit 1
fi

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  echo "[x] 依赖文件不存在: $REQUIREMENTS_FILE"
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  if [[ -d "$ENV_PREFIX" ]]; then
    echo "检测到环境已存在，开始更新..."
    CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS" \
    CONDA_ENVS_PATH="$CONDA_ENVS_PATH" \
    conda env update -p "$ENV_PREFIX" -f "$ENV_FILE" --prune
  else
    echo "检测到环境不存在，开始创建..."
    CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS" \
    CONDA_ENVS_PATH="$CONDA_ENVS_PATH" \
    conda env create -p "$ENV_PREFIX" -f "$ENV_FILE"
  fi
else
  if [[ ! -x "$ENV_PYTHON" ]]; then
    echo "[x] 当前 shell 未初始化 conda，且现有环境不存在：$ENV_PREFIX"
    echo "    请先加载 conda，或确认 conda 安装路径。"
    exit 1
  fi
  echo "[i] 当前 shell 未初始化 conda，跳过 conda create/update，直接修复现有环境。"
fi

echo ""
echo "开始安装固定版本 Python 训练依赖..."
"$ENV_PYTHON" -m pip install --upgrade pip
"$ENV_PYTHON" -m pip uninstall -y torch torchvision torchaudio transformers trl peft datasets huggingface_hub unsloth unsloth_zoo xformers bitsandbytes modelscope || true
"$ENV_PYTHON" -m pip install --upgrade --no-cache-dir --force-reinstall -r "$REQUIREMENTS_FILE"

echo ""
echo "[✓] conda 环境准备完成"
echo "后续请执行："
echo "  export CONDA_PKGS_DIRS=\"$CONDA_PKGS_DIRS\""
echo "  conda activate \"$ENV_PREFIX\""
echo "  bash download_base_model_to_data.sh"
echo "  bash run_train_qlora_unsloth.sh"

#!/bin/bash
set -e

echo "============================================"
echo "  Qwen3-VL Finance Multi-LoRA 部署环境安装"
echo "  （基于 conda 虚拟环境）"
echo "============================================"

ENV_NAME="qwen3-vllm"

if command -v conda >/dev/null 2>&1; then
  echo "检测到 conda，将使用 conda 创建/更新环境: $ENV_NAME"

  # 如果环境不存在则创建
  if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境不存在，使用 environment.yml 创建..."
    conda env create -n "$ENV_NAME" -f environment.yml
  else
    echo "环境已存在，使用 environment.yml 更新依赖..."
    conda env update -n "$ENV_NAME" -f environment.yml --prune
  fi

  echo ""
  echo "[✓] 环境准备完成"
  echo "请在后续终端中先执行："
  echo "  conda activate $ENV_NAME"
  echo "然后再运行其它脚本，例如："
  echo "  bash download_models.sh"
  echo "  bash serve_multi_lora.sh"
else
  echo "未检测到 conda，退回到系统 Python + pip 安装方式。"
  echo "如需使用 conda，请先安装 Miniconda/Anaconda 并将 conda 加入 PATH。"

  pip install --upgrade pip
  pip install -U vllm
  pip install -U modelscope
  pip install transformers accelerate peft

  echo ""
  echo "[✓] 使用系统 pip 安装完成"
  echo "    vLLM 版本: $(python -c 'import vllm; print(vllm.__version__)')"
  echo "    ModelScope 版本: $(python -c 'import modelscope; print(modelscope.__version__)')"
fi

#!/bin/bash

# 统一把训练、下载、缓存、临时文件都放到 /data，避免占用系统盘。
export DATA_ROOT="${DATA_ROOT:-/data/qwen3-vl-finance-expert-c}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$DATA_ROOT/modelscope}"
export TORCH_HOME="${TORCH_HOME:-$DATA_ROOT/torch}"
export TMPDIR="${TMPDIR:-$DATA_ROOT/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$DATA_ROOT/triton}"
export UNSLOTH_CACHE_DIR="${UNSLOTH_CACHE_DIR:-$DATA_ROOT/unsloth}"

mkdir -p \
  "$DATA_ROOT" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$MODELSCOPE_CACHE" \
  "$TORCH_HOME" \
  "$TMPDIR" \
  "$TRITON_CACHE_DIR" \
  "$UNSLOTH_CACHE_DIR" \
  "$DATA_ROOT/models" \
  "$DATA_ROOT/datasets" \
  "$DATA_ROOT/outputs" \
  "$DATA_ROOT/logs"

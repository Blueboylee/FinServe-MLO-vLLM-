#!/bin/bash
# ============================================
#  未优化版：关闭 Chunked Prefill / Prefix Caching / LoRA Reorder
#  用于与 serve_multi_lora.sh（优化后）做压测对比
# ============================================
export ENABLE_CHUNKED_PREFILL=false
export ENABLE_PREFIX_CACHING=false
export FINSERVE_LORA_REORDER=0
exec bash "$(dirname "$0")/serve_multi_lora.sh"

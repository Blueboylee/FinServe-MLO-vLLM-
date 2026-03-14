#!/usr/bin/env python3
"""
FinServe 同进程启动器

解决原 serve_multi_lora.sh 的核心缺陷:
  旧方案: shell → subprocess.Popen(vllm serve) → 父进程 apply patch
         → patch 只存在于父进程, 对子进程内的 vLLM 引擎完全无效

  新方案: shell → python serve_launcher.py (同一进程):
         1. apply_sgmv_optimizations()  — patch vLLM LoRA kernel
         2. apply_triton_optimizations() — patch RMSNorm/SiLU
         3. vllm.entrypoints.openai.api_server — 在同进程内启动 API Server
         引擎初始化时 CUDA Graph capture 已包含自研 kernel

同进程 patch 的关键约束:
  - 所有 monkey-patch 必须在 Engine 创建前完成
  - vLLM 的 CUDA Graph capture 发生在首批推理时,
    此时 Python 层的 forward 已被替换, graph 自然包含 Triton kernel
  - 若使用 torch.library.custom_op 注册, 则 torch.compile 也能 trace 到
"""

import os
import sys
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _apply_kernel_patches():
    """在引擎创建前完成所有 kernel patch."""
    patched_all = []

    enable_sgmv = os.environ.get("ENABLE_SGMV_OPTIMIZATION", "1") == "1"
    enable_fused = os.environ.get("SGMV_ENABLE_FUSED", "1") == "1"

    if enable_sgmv:
        try:
            from sgmv_kernel.sgmv_integration import apply_sgmv_optimizations
            patched = apply_sgmv_optimizations(
                enable_fused=enable_fused,
                enable_tensor_core=True,
                enable_fuse_lora_rmsnorm=enable_fused,
            )
            patched_all.extend(patched)
        except Exception as e:
            print(f"[WARN] SGMV patch failed: {e}")

        try:
            from sgmv_kernel.sgmv_cuda_graph import apply_cuda_graph_sgmv
            info = apply_cuda_graph_sgmv(hidden_dim=4096, rank=64)
            patched_all.append(info)
        except Exception as e:
            print(f"[WARN] CUDA Graph SGMV registration failed: {e}")

    try:
        from triton_integration import apply_triton_optimizations
        patched = apply_triton_optimizations()
        patched_all.extend(patched)
    except Exception as e:
        print(f"[WARN] Triton patch failed: {e}")

    enable_fp8_kv = os.environ.get("ENABLE_FP8_KV_CACHE", "0") == "1"
    if enable_fp8_kv:
        try:
            from kv_cache_fp8.fp8_integration import apply_fp8_kv_cache
            patched = apply_fp8_kv_cache()
            patched_all.extend(patched)
        except Exception as e:
            print(f"[WARN] FP8 KV Cache patch failed: {e}")

    return patched_all


def main():
    # ── Step 1: Patch BEFORE any vLLM engine initialization ──
    print("=" * 60)
    print("[FinServe] Applying kernel patches in-process...")
    print("=" * 60)

    patched = _apply_kernel_patches()

    print(f"\n[FinServe] {len(patched)} ops patched (CUDA Graph will capture these)")
    for p in patched:
        print(f"  + {p}")
    print("=" * 60 + "\n")

    # ── Step 2: Launch vLLM OpenAI API Server in same process ──
    # vllm.entrypoints.openai.api_server uses sys.argv for arg parsing
    # serve_multi_lora.sh passes all --model/--host/... flags directly to us
    sys.argv = ["vllm-serve"] + sys.argv[1:]

    from vllm.entrypoints.openai.api_server import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    main()

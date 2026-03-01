#!/usr/bin/env python3
"""
启动 vLLM 服务：Qwen2.5-32B 基座（仅 GPTQ，兼容 V100）+ 可选专家 LoRA。

用法:
  1. 必须指定基座路径（服务器上已下载的 GPTQ 目录）:
     python scripts/run_serve.py --base-model /root/.cache/modelscope/hub/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4

  2. 若已在项目根目录有 lora_paths.json（运行 download_experts.py 时用 --output-config lora_paths.json），
     会自动加载专家 LoRA；否则只启动基座。

  3. 额外参数会原样传给 vllm，例如:
     python scripts/run_serve.py --base-model /path/to/gptq --tensor-parallel-size 2
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LORA_CONFIG = "lora_paths.json"

# 用于修复 ModelScope 下载的 GPTQ 目录里错误 config（如误标 Qwen3 等）
QWEN25_32B_GPTQ_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 27648,
    "max_position_embeddings": 32768,
    "max_window_layers": 70,
    "model_type": "qwen2",
    "num_attention_heads": 40,
    "num_hidden_layers": 64,
    "num_key_value_heads": 8,
    "quantization_config": {
        "bits": 4,
        "group_size": 128,
        "quant_method": "gptq",
        "sym": True,
    },
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "sliding_window": 131072,
    "tie_word_embeddings": False,
    "torch_dtype": "float16",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 152064,
}


def ensure_gptq_config(model_dir: Path) -> None:
    """若 config.json 不是 qwen2+gptq，则写回正确配置（避免 ModelScope 错包）。"""
    model_dir = Path(model_dir).resolve()
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        print(f"警告: 未找到 {config_path}，跳过 config 修复")
        return
    need_fix = True
    with open(config_path, encoding="utf-8") as f:
        cur = json.load(f)
    if cur.get("model_type") == "qwen2":
        qc = cur.get("quantization_config") or {}
        if qc.get("quant_method") == "gptq":
            need_fix = False
    if need_fix:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(QWEN25_32B_GPTQ_CONFIG, f, indent=2, ensure_ascii=False)
        print(f"已修复 GPTQ config: {config_path}")


def load_lora_paths(config_path: Path | None = None) -> dict[str, str] | None:
    """若存在 lora_paths.json 则返回 name -> path；否则返回 None。"""
    if config_path is None:
        config_path = ROOT / LORA_CONFIG
    if not config_path.is_file():
        return None
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="启动 vLLM（Qwen2.5-32B GPTQ + 可选专家 LoRA）",
        epilog="未在上面的参数会原样传给 vllm serve，如 --tensor-parallel-size 2",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=os.environ.get("VLLM_BASE_MODEL", ""),
        required=False,
        help="基座模型本地目录（GPTQ），例如 ~/.cache/modelscope/hub/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
    )
    parser.add_argument(
        "--lora-config",
        type=str,
        default=str(ROOT / LORA_CONFIG),
        help="lora_paths.json 路径；不存在则只起基座",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--max-loras", type=int, default=2, help="最大 LoRA 数")
    parser.add_argument("--max-lora-rank", type=int, default=64, help="LoRA 最大 rank")
    parser.add_argument(
        "extra",
        nargs="*",
        help="传给 vllm 的额外参数，如 --tensor-parallel-size 2",
    )
    args = parser.parse_args()

    base_model = (args.base_model or "").strip()
    if not base_model:
        print("错误: 请指定 --base-model（GPTQ 模型目录），或设置环境变量 VLLM_BASE_MODEL")
        print("示例: --base-model /root/.cache/modelscope/hub/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4")
        sys.exit(1)

    base_path = Path(base_model).expanduser().resolve()
    if not base_path.is_dir():
        print(f"错误: 基座模型目录不存在: {base_path}")
        sys.exit(1)

    ensure_gptq_config(base_path)
    base_model_str = str(base_path)

    lora_paths = load_lora_paths(Path(args.lora_config))
    # 构建 vllm serve 的参数列表
    vllm_argv = [
        "vllm",
        "serve",
        base_model_str,
        "--quantization", "gptq",
        "--host", args.host,
        "--port", str(args.port),
    ]
    if lora_paths:
        lora_pairs = []
        for name, path in lora_paths.items():
            path_resolved = str(Path(path).expanduser().resolve())
            if not Path(path_resolved).is_dir():
                print(f"警告: LoRA 路径不存在，已跳过: {name} -> {path_resolved}")
                continue
            lora_pairs.append(f"{name}={path_resolved}")
        if lora_pairs:
            vllm_argv += ["--enable-lora", "--max-loras", str(args.max_loras), "--max-lora-rank", str(args.max_lora_rank)]
            vllm_argv += ["--lora-modules"] + lora_pairs
            print("LoRA 已启用:", [p.split("=")[0] for p in lora_pairs])
        else:
            print("lora_paths.json 中路径均无效，仅启动基座")
    else:
        print("未找到 lora_paths.json，仅启动基座（无 LoRA）")
    vllm_argv += args.extra

    print("执行: vllm", " ".join(vllm_argv[2:]))  # 跳过 "vllm" "serve"
    os.chdir(ROOT)

    # 在进程内调用 vLLM CLI main，避免 python -m vllm / python -m vllm.entrypoints.cli 在某些环境下不可用
    backup_argv = sys.argv
    try:
        sys.argv = vllm_argv
        from vllm.entrypoints.cli import main as vllm_main
        sys.exit(vllm_main() or 0)
    except ImportError as e:
        sys.argv = backup_argv
        # 回退：用子进程执行 vllm 可执行文件（若存在）
        bindir = Path(sys.executable).parent
        vllm_bin = bindir / "vllm"
        if vllm_bin.is_file() or (bindir / "vllm.exe").is_file():
            cmd = [str(vllm_bin.resolve()) if vllm_bin.is_file() else str((bindir / "vllm.exe").resolve())] + vllm_argv[1:]
            sys.exit(subprocess.run(cmd).returncode)
        print(f"错误: 无法导入 vllm.entrypoints.cli.main（{e}），且未找到 vllm 可执行文件", file=sys.stderr)
        sys.exit(1)
    finally:
        sys.argv = backup_argv


if __name__ == "__main__":
    main()

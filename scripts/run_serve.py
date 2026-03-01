#!/usr/bin/env python3
"""
启动 vLLM 服务：Qwen2.5-32B 基座 + 专家 A / 专家 B 双 LoRA。
需先运行 scripts/download_experts.py 下载 LoRA 并生成 lora_paths.json。

因 vLLM CLI 对多 LoRA 传参有 bug，本脚本用 Python 直接调 vLLM 的 run_server，
在代码里传入 lora_modules 列表，绕过 --lora-modules 解析问题。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent

# 4bit 量化基座：GPTQ 兼容 V100 等算力 70；AWQ 需算力 75+
DEFAULT_BASE_MODEL_AWQ = "Qwen/Qwen2.5-32B-Instruct-AWQ"
DEFAULT_BASE_MODEL_GPTQ = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"

# LoRA 配置文件名
LORA_CONFIG = "lora_paths.json"


def load_lora_paths(config_path: Path | None = None) -> dict[str, str]:
    if config_path is None:
        config_path = ROOT / LORA_CONFIG
    if not config_path.is_file():
        print(f"未找到 {config_path}，请先运行：")
        print("  python scripts/download_experts.py")
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


# 与 HuggingFace Qwen2.5-32B-Instruct-GPTQ-Int4 一致的 config，用于修复 ModelScope 错包
_QWEN25_32B_GPTQ_CONFIG = {
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
        "batch_size": 1,
        "bits": 4,
        "block_name_to_quantize": None,
        "cache_block_outputs": True,
        "damp_percent": 0.01,
        "dataset": None,
        "desc_act": False,
        "exllama_config": {"version": 1},
        "group_size": 128,
        "max_input_length": None,
        "model_seqlen": None,
        "module_name_preceding_first_block": None,
        "modules_in_block_to_quantize": None,
        "pad_token_id": None,
        "quant_method": "gptq",
        "sym": True,
        "tokenizer": None,
        "true_sequential": True,
        "use_cuda_fp16": False,
        "use_exllama": True,
    },
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "sliding_window": 131072,
    "tie_word_embeddings": False,
    "torch_dtype": "float16",
    "transformers_version": "4.39.3",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 152064,
}


def _ensure_gptq_config(model_dir: Path) -> None:
    """若本地 GPTQ 目录的 config 不是 Qwen2+gptq，则写回正确 config，避免 vLLM 报错。"""
    model_dir = model_dir.resolve()
    config_path = model_dir / "config.json"
    need_fix = True
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            cur = json.load(f)
        if cur.get("model_type") == "qwen2" and (cur.get("quantization_config") or {}).get("quant_method") == "gptq":
            need_fix = False
    if need_fix:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(_QWEN25_32B_GPTQ_CONFIG, f, indent=2, ensure_ascii=False)
        print(f"已修复 {config_path} 为 Qwen2.5-32B GPTQ 配置")
        quant_path = model_dir / "quantize_config.json"
        q = _QWEN25_32B_GPTQ_CONFIG["quantization_config"].copy()
        for k in list(q.keys()):
            if q[k] is None or k in ("exllama_config", "dataset", "tokenizer"):
                q.pop(k, None)
        with open(quant_path, "w", encoding="utf-8") as f:
            json.dump(q, f, indent=2)
        print(f"已写入 {quant_path}")


def _run_with_vllm_api(
    base_model: str,
    paths: dict[str, str],
    *,
    quantization: str = "gptq",
    no_quantization: bool = False,
    host: str = "0.0.0.0",
    port: int = 8000,
    max_loras: int = 2,
    max_lora_rank: int = 64,
    vllm_extra: list[str],
) -> None:
    """用 vLLM Python API 启动服务，直接传入 lora_modules 列表，避免 CLI 解析问题。"""
    from vllm.entrypoints.utils import cli_env_setup
    cli_env_setup()

    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.entrypoints.openai.models.protocol import LoRAModulePath
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    # 构造等价于 "vllm serve <model> --quantization gptq --enable-lora --max-loras 2 ..." 的 argv，不传 --lora-modules
    argv = [
        "vllm_serve",
        base_model,
        "--enable-lora",
        "--max-loras", str(max_loras),
        "--max-lora-rank", str(max_lora_rank),
        "--host", host,
        "--port", str(port),
    ]
    if not no_quantization:
        argv += ["--quantization", quantization]
    argv += vllm_extra

    old_argv = sys.argv
    try:
        sys.argv = argv
        parser = make_arg_parser(FlexibleArgumentParser(description="vLLM OpenAI API server"))
        args = parser.parse_args()
    finally:
        sys.argv = old_argv

    # 在代码里注入多 LoRA，绕过 CLI 无法正确传 list 的问题
    args.lora_modules = [
        LoRAModulePath(name=name, path=path)
        for name, path in paths.items()
    ]
    validate_parsed_serve_args(args)

    import uvloop
    from vllm.entrypoints.openai.api_server import run_server

    print("启动 vLLM 服务（Python API，多 LoRA 已注入）:", base_model, "LoRAs:", list(paths.keys()))
    uvloop.run(run_server(args))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="启动 vLLM（Qwen2.5-32B + 专家 LoRA）")
    parser.add_argument(
        "--base-model",
        type=str,
        default=os.environ.get("VLLM_BASE_MODEL", ""),
        help="基座模型 ID 或本地路径；不传时按 --quantization 自动选 AWQ 或 GPTQ 默认",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=("awq", "gptq"),
        default=os.environ.get("VLLM_QUANTIZATION", "gptq"),
        help="量化方式：gptq 兼容 V100/算力 70；awq 需算力 75+（默认: gptq）",
    )
    parser.add_argument(
        "--lora-config",
        type=str,
        default=str(ROOT / LORA_CONFIG),
        help="lora_paths.json 路径",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="服务 host",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="服务 port",
    )
    parser.add_argument(
        "--max-loras", type=int, default=2, help="同时启用的 LoRA 数量",
    )
    parser.add_argument(
        "--max-lora-rank", type=int, default=64, help="LoRA 最大 rank（与训练一致）",
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="使用全精度基座（显存约 60GB+），与 --quantization 二选一",
    )
    parser.add_argument(
        "vllm_extra",
        nargs="*",
        help="传给 vllm 的额外参数，如 --tensor-parallel-size 2",
    )
    args = parser.parse_args()

    base_model = (args.base_model or "").strip()
    if args.no_quantization:
        if not base_model or base_model in (DEFAULT_BASE_MODEL_AWQ, DEFAULT_BASE_MODEL_GPTQ):
            base_model = "Qwen/Qwen2.5-32B-Instruct"
        print("提示：已启用 --no-quantization，使用全精度基座，显存需求较大")
    else:
        if not base_model:
            base_model = DEFAULT_BASE_MODEL_GPTQ if args.quantization == "gptq" else DEFAULT_BASE_MODEL_AWQ
        if base_model == "Qwen/Qwen2.5-32B-Instruct":
            print("错误：全精度基座请使用 --no-quantization")
            sys.exit(1)

    paths = load_lora_paths(Path(args.lora_config))

    # 使用本地 GPTQ 目录时，先确保 config 为 Qwen2 + gptq，避免 ModelScope 错包导致 "Qwen3-0.6B" / "Cannot find the config file for gptq"
    if args.quantization == "gptq" and not args.no_quantization:
        base_path = Path(base_model).resolve()
        if not base_path.is_dir():
            print("提示：若出现 'Cannot find the config file for gptq'，请先运行：")
            print(f"  python scripts/fix_gptq_config.py {base_model}")
        else:
            _ensure_gptq_config(base_path)

    os.chdir(ROOT)
    _run_with_vllm_api(
        base_model,
        paths,
        quantization=args.quantization,
        no_quantization=args.no_quantization,
        host=args.host,
        port=args.port,
        max_loras=args.max_loras,
        max_lora_rank=args.max_lora_rank,
        vllm_extra=args.vllm_extra,
    )


if __name__ == "__main__":
    main()

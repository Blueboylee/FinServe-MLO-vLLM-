#!/usr/bin/env python3
"""
修复 ModelScope 下载的 Qwen2.5-32B-Instruct-GPTQ-Int4 目录下错误或缺失的 config.json。
若 config 里被识别成 Qwen3 或缺少 quantization_config，会写回正确的 Qwen2 GPTQ 配置。
用法: python scripts/fix_gptq_config.py /path/to/Qwen2___5-32B-Instruct-GPTQ-Int4
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 与 HuggingFace 上 Qwen2.5-32B-Instruct-GPTQ-Int4 一致的 config（含 quantization_config）
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


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python scripts/fix_gptq_config.py <GPTQ 模型目录>")
        print("示例: python scripts/fix_gptq_config.py /root/.cache/modelscope/hub/Qwen/Qwen2___5-32B-Instruct-GPTQ-Int4")
        sys.exit(1)
    model_dir = Path(sys.argv[1]).resolve()
    config_path = model_dir / "config.json"
    if not model_dir.is_dir():
        print(f"错误: 目录不存在 {model_dir}")
        sys.exit(1)
    if not config_path.is_file():
        print(f"在 {model_dir} 下未找到 config.json，将创建默认 Qwen2.5-32B GPTQ 配置")
        need_fix = True
    else:
        with open(config_path, encoding="utf-8") as f:
            current = json.load(f)
        model_type = current.get("model_type", "")
        arch = (current.get("architectures") or [""])[0]
        quant = current.get("quantization_config") or {}
        quant_method = quant.get("quant_method", "")
        need_fix = (
            model_type != "qwen2"
            or arch != "Qwen2ForCausalLM"
            or quant_method != "gptq"
        )
        if need_fix:
            print(f"当前 config: model_type={model_type!r}, architectures={current.get('architectures')}, quant_method={quant_method!r}")
            print("将写回正确的 Qwen2.5-32B GPTQ 配置")
        else:
            print("config.json 已是 Qwen2 GPTQ 格式，无需修改")
            return
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(QWEN25_32B_GPTQ_CONFIG, f, indent=2, ensure_ascii=False)
    print(f"已写入 {config_path}")

    # 部分 vLLM 版本会找单独的 quantize_config.json，也写一份
    quant_path = model_dir / "quantize_config.json"
    quant_cfg = QWEN25_32B_GPTQ_CONFIG["quantization_config"].copy()
    for k in list(quant_cfg.keys()):
        if quant_cfg[k] is None or k in ("exllama_config", "dataset", "tokenizer"):
            quant_cfg.pop(k, None)
    with open(quant_path, "w", encoding="utf-8") as f:
        json.dump(quant_cfg, f, indent=2)
    print(f"已写入 {quant_path}")


if __name__ == "__main__":
    main()

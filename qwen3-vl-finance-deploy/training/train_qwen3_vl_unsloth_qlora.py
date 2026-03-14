#!/usr/bin/env python3
"""使用 Unsloth + QLoRA 对 Qwen3-VL-8B-Instruct 做文本 SFT 微调。

说明：
1. 基座模型仍然是 Qwen3-VL-8B-Instruct，但本脚本训练的是纯文本推理数据。
2. 为了避免无意义地更新视觉塔，LoRA 仅作用于语言侧模块。
3. 所有缓存、数据切片、日志、输出都应放到 /data 下，建议配合 data_env.sh 使用。
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import unsloth  # noqa: F401
from datasets import load_dataset
from transformers import set_seed
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel, is_bfloat16_supported


DEFAULT_SYSTEM_PROMPT = (
    "你是一名严谨的金融推理专家。请先准确理解题目，再给出结构化、可验证、"
    "尽量完整的分析过程与最终答案。若题目涉及数字、条件或约束，请显式检查它们。"
)
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "up_proj",
    "k_proj",
    "gate_proj",
    "down_proj",
    "o_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unsloth QLoRA fine-tuning for Qwen3-VL-8B-Instruct"
    )
    parser.add_argument(
        "--model_name_or_path",
        default="/data/qwen3-vl-finance-expert-c/models/Qwen3-VL-8B-Instruct",
        help="基座模型路径，建议先用 ModelScope 下载到 /data",
    )
    parser.add_argument(
        "--dataset_name",
        default="nohurry/Opus-4.6-Reasoning-3000x-filtered",
        help="Hugging Face 数据集名",
    )
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="数据起始下标，含当前下标",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=200,
        help="数据结束下标，含当前下标",
    )
    parser.add_argument(
        "--data_root",
        default=os.environ.get("DATA_ROOT", "/data/qwen3-vl-finance-expert-c"),
    )
    parser.add_argument(
        "--output_name",
        default="Qwen3-VL-Finance-expert-c",
        help="LoRA 输出名，也会用于 ModelScope 发布目录",
    )
    parser.add_argument("--max_seq_length", type=int, default=3072)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="最大训练步数；>0 时优先于 num_train_epochs",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=DEFAULT_TARGET_MODULES,
        help="LoRA 目标模块列表",
    )
    parser.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="写入每条样本的 system 指令",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, content: Any) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def build_model_card(
    repo_id: str,
    base_model: str,
    dataset_name: str,
    start_index: int,
    end_index: int,
    output_name: str,
) -> str:
    return f"""---
base_model: {base_model}
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:{base_model}
- lora
- qlora
- unsloth
- transformers
- finance
---

# {repo_id}

这是一个基于 `{base_model}` 的 LoRA 适配器，目标是增强金融推理类问答能力。

## 训练数据

- 数据集：`{dataset_name}`
- 使用字段：`problem` -> 用户问题，`solution` -> 助手答案
- 使用区间：`{start_index}..{end_index}`（含两端）

## 训练方式

- 框架：Unsloth + TRL SFTTrainer
- 方法：QLoRA
- 适配器名称：`{output_name}`
- 训练类型：文本 SFT（不额外训练视觉塔）

## 使用方式

先加载基座模型 `{base_model}`，再挂载当前 LoRA 适配器即可。
"""


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = ensure_dir(Path(args.data_root))
    datasets_root = ensure_dir(data_root / "datasets")
    outputs_root = ensure_dir(data_root / "outputs")

    run_name = f"{args.output_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_output_dir = ensure_dir(outputs_root / run_name)
    adapter_output_dir = ensure_dir(run_output_dir / "adapter")
    dataset_artifact_dir = ensure_dir(datasets_root / args.output_name)

    print("=" * 80)
    print("Qwen3-VL + Unsloth QLoRA 微调开始")
    print("=" * 80)
    print(f"基座模型: {args.model_name_or_path}")
    print(f"数据集  : {args.dataset_name} [{args.dataset_split}]")
    print(f"区间    : {args.start_index}..{args.end_index} (inclusive)")
    print(f"输出目录: {adapter_output_dir}")
    print("")

    raw_dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        cache_dir=os.environ.get("HF_DATASETS_CACHE"),
    )

    if args.start_index < 0 or args.end_index < args.start_index:
        raise ValueError("start_index / end_index 设置不合法。")

    max_end = len(raw_dataset) - 1
    if args.start_index > max_end:
        raise ValueError(
            f"start_index={args.start_index} 超出数据集范围，当前最大下标为 {max_end}。"
        )

    end_index = min(args.end_index, max_end)
    selected_indices = list(range(args.start_index, end_index + 1))
    sliced_dataset = raw_dataset.select(selected_indices)

    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    tokenizer.padding_side = "right"

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
        target_modules=args.target_modules,
    )
    FastVisionModel.for_training(model)

    def format_example(example: dict[str, Any]) -> dict[str, Any]:
        problem = normalize_text(example.get("problem"))
        solution = normalize_text(example.get("solution"))
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {
            "text": text,
            "problem": problem,
            "solution": solution,
        }

    processed_dataset = sliced_dataset.map(
        format_example,
        desc="Formatting dataset to chat template",
    )

    dataset_snapshot_path = dataset_artifact_dir / (
        f"slice_{args.start_index}_{end_index}.jsonl"
    )
    with dataset_snapshot_path.open("w", encoding="utf-8") as f:
        for row in processed_dataset:
            payload = {
                "problem": row["problem"],
                "solution": row["solution"],
                "text": row["text"],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    train_args = SFTConfig(
        output_dir=str(run_output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        # 只保留最终导出的 LoRA，避免中途保存 optimizer checkpoint 占用过大并写盘失败。
        save_strategy="no",
        save_total_limit=1,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=processed_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=args.preprocessing_num_workers,
        packing=False,
        args=train_args,
    )

    train_result = trainer.train()

    model.save_pretrained(str(adapter_output_dir))
    tokenizer.save_pretrained(str(adapter_output_dir))

    write_json(
        adapter_output_dir / "configuration.json",
        {"framework": "pytorch", "task": "text-generation", "model_type": "qwen3_vl"},
    )

    model_card = build_model_card(
        repo_id=f"GaryLeenene/{args.output_name}",
        base_model="Qwen/Qwen3-VL-8B-Instruct",
        dataset_name=args.dataset_name,
        start_index=args.start_index,
        end_index=end_index,
        output_name=args.output_name,
    )
    (adapter_output_dir / "README.md").write_text(model_card, encoding="utf-8")

    metrics = dict(train_result.metrics)
    summary = {
        "repo_id": f"GaryLeenene/{args.output_name}",
        "base_model": "Qwen/Qwen3-VL-8B-Instruct",
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "start_index": args.start_index,
        "end_index": end_index,
        "num_samples": len(processed_dataset),
        "run_output_dir": str(run_output_dir),
        "adapter_output_dir": str(adapter_output_dir),
        "dataset_snapshot_path": str(dataset_snapshot_path),
        "hyperparameters": {
            "max_seq_length": args.max_seq_length,
            "num_train_epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
        "metrics": metrics,
    }
    write_json(run_output_dir / "training_summary.json", summary)

    print("")
    print("[✓] 训练完成")
    print(f"[✓] LoRA 输出目录: {adapter_output_dir}")
    print(f"[✓] 数据切片快照 : {dataset_snapshot_path}")
    print(f"[✓] 训练摘要文件 : {run_output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 ModelScope 下载 Qwen2.5 32B 基座模型和两个 LoRA 专家模型
"""

import os
import sys
from pathlib import Path

try:
    from modelscope import snapshot_download
except ImportError:
    print("正在安装 modelscope...")
    os.system("pip install modelscope -q")
    from modelscope import snapshot_download


def download_model(model_id: str, save_dir: str):
    """下载模型到指定目录"""
    print(f"\n{'='*60}")
    print(f"正在下载模型：{model_id}")
    print(f"保存目录：{save_dir}")
    print(f"{'='*60}\n")
    
    try:
        model_dir = snapshot_download(
            model_id,
            cache_dir=save_dir,
            revision='master'
        )
        print(f"✓ 模型下载完成：{model_dir}")
        return model_dir
    except Exception as e:
        print(f"✗ 模型下载失败：{e}")
        sys.exit(1)


def main():
    # 设置模型根目录
    model_root = Path(__file__).parent / "models"
    model_root.mkdir(exist_ok=True)
    
    print(f"模型存储根目录：{model_root.absolute()}")
    
    # 定义模型
    models = {
        "base": {
            "model_id": "qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
            "save_dir": str(model_root / "qwen25-32b-gptq"),
            "description": "Qwen2.5 32B GPTQ 4bit 基座模型"
        },
        "expert_a": {
            "model_id": "GaryLeenene/qwen25-32b-expert-a-qlora",
            "save_dir": str(model_root / "expert-a-qlora"),
            "description": "专家 A LoRA 模型"
        },
        "expert_b": {
            "model_id": "GaryLeenene/qwen25-32b-expert-b-qlora",
            "save_dir": str(model_root / "expert-b-qlora"),
            "description": "专家 B LoRA 模型"
        }
    }
    
    # 下载所有模型
    downloaded_models = {}
    for key, model_info in models.items():
        print(f"\n开始下载 {model_info['description']}")
        model_dir = download_model(
            model_info['model_id'],
            model_info['save_dir']
        )
        downloaded_models[key] = {
            'path': model_dir,
            'model_id': model_info['model_id']
        }
    
    # 生成配置信息
    print(f"\n{'='*60}")
    print("所有模型下载完成！")
    print(f"{'='*60}")
    print("\n模型路径信息：")
    for key, info in downloaded_models.items():
        print(f"  {key}: {info['path']}")
    
    # 保存配置到文件
    config_file = model_root / "model_config.txt"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write("# 模型配置文件\n")
        f.write(f"base_model_path={downloaded_models['base']['path']}\n")
        f.write(f"expert_a_path={downloaded_models['expert_a']['path']}\n")
        f.write(f"expert_b_path={downloaded_models['expert_b']['path']}\n")
    
    print(f"\n配置已保存到：{config_file}")
    print("\n下一步：运行 python serve_lora.py 启动 vLLM 服务")


if __name__ == "__main__":
    main()

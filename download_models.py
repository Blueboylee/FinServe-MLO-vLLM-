#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 ModelScope 下载专家模型脚本
下载 Qwen2.5-32B 的两个专家模型 A 和 B
"""

import os
import subprocess
import sys


def check_modelscope():
    """检查 modelscope 是否已安装"""
    try:
        import modelscope
        print(f"✓ modelscope 已安装，版本：{modelscope.__version__}")
        return True
    except ImportError:
        print("✗ modelscope 未安装")
        return False


def install_modelscope():
    """安装 modelscope"""
    print("正在安装 modelscope...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope", "-q"])
    print("✓ modelscope 安装完成")


def download_model(model_id, output_dir):
    """从 ModelScope 下载模型"""
    print(f"\n{'='*60}")
    print(f"正在下载模型：{model_id}")
    print(f"保存目录：{output_dir}")
    print(f"{'='*60}")
    
    try:
        from modelscope import snapshot_download
        
        model_dir = snapshot_download(
            model_id,
            cache_dir=output_dir,
            revision='master'
        )
        
        print(f"✓ 模型下载完成：{model_dir}")
        return model_dir
        
    except Exception as e:
        print(f"✗ 模型下载失败：{e}")
        raise


def main():
    """主函数"""
    print("="*60)
    print("Qwen2.5-32B 专家模型下载脚本")
    print("="*60)
    
    # 检查并安装 modelscope
    if not check_modelscope():
        install_modelscope()
    
    # 配置模型信息
    models = {
        "expert_a": {
            "model_id": "GaryLeenene/qwen25-32b-expert-a-qlora",
            "dir_name": "qwen25-32b-expert-a-qlora"
        },
        "expert_b": {
            "model_id": "GaryLeenene/qwen25-32b-expert-b-qlora",
            "dir_name": "qwen25-32b-expert-b-qlora"
        }
    }
    
    # 设置下载目录
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"\n模型下载目录：{base_dir}")
    
    # 下载所有模型
    downloaded_models = {}
    for expert_name, expert_info in models.items():
        model_dir = os.path.join(base_dir, expert_info["dir_name"])
        model_path = download_model(expert_info["model_id"], base_dir)
        downloaded_models[expert_name] = model_path
    
    # 保存模型路径配置
    config_file = os.path.join(base_dir, "model_paths.txt")
    with open(config_file, 'w', encoding='utf-8') as f:
        for expert_name, model_path in downloaded_models.items():
            f.write(f"{expert_name}={model_path}\n")
    
    print(f"\n{'='*60}")
    print("所有模型下载完成！")
    print(f"{'='*60}")
    print(f"\n模型路径配置已保存到：{config_file}")
    print("\n下载的模型:")
    for expert_name, model_path in downloaded_models.items():
        print(f"  - {expert_name}: {model_path}")
    
    print("\n下一步：运行 python deploy_experts.py 启动服务")


if __name__ == "__main__":
    main()

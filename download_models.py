#!/usr/bin/env python3
"""
模型下载脚本
从ModelScope下载Qwen2.5 32B基础模型和专家模型
"""

import os
import sys
from modelscope import snapshot_download


def download_model(model_id: str, model_name: str, cache_dir: str = None) -> str:
    """
    从ModelScope下载模型
    
    Args:
        model_id: ModelScope模型ID
        model_name: 模型名称（用于日志）
        cache_dir: 缓存目录
        
    Returns:
        模型下载路径
    """
    print(f"开始下载 {model_name}...")
    print(f"模型ID: {model_id}")
    
    try:
        if cache_dir:
            model_dir = snapshot_download(
                model_id=model_id,
                cache_dir=cache_dir
            )
        else:
            model_dir = snapshot_download(model_id=model_id)
        
        print(f"{model_name} 下载完成: {model_dir}")
        return model_dir
        
    except Exception as e:
        print(f"下载 {model_name} 失败: {e}")
        raise


def main():
    """主函数：下载所有必需的模型"""
    
    print("=" * 60)
    print("Qwen2.5 32B QLoRA 模型下载脚本")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"创建模型目录: {models_dir}")
    
    cache_dir = os.path.join(models_dir, "cache")
    
    print(f"模型将下载到: {models_dir}")
    print("-" * 60)
    
    base_model_id = "qwen/Qwen2.5-32B"
    expert_a_id = "GaryLeenene/qwen25-32b-expert-a-qlora"
    expert_b_id = "GaryLeenene/qwen25-32b-expert-b-qlora"
    
    base_model_path = os.path.join(models_dir, "Qwen2.5-32B")
    expert_a_path = os.path.join(models_dir, "qwen25-32b-expert-a-qlora")
    expert_b_path = os.path.join(models_dir, "qwen25-32b-expert-b-qlora")
    
    if not os.path.exists(base_model_path):
        try:
            print("\n[1/3] 下载Qwen2.5 32B基础模型（4bit AWQ版本）...")
            download_model(base_model_id, "Qwen2.5 32B基础模型", cache_dir)
        except Exception as e:
            print(f"基础模型下载失败: {e}")
            print("请检查网络连接或ModelScope访问...")
            sys.exit(1)
    else:
        print("\n[1/3] Qwen2.5 32B基础模型已存在，跳过下载")
    
    if not os.path.exists(expert_a_path):
        try:
            print("\n[2/3] 下载专家A模型...")
            download_model(expert_a_id, "专家A模型", cache_dir)
        except Exception as e:
            print(f"专家A模型下载失败: {e}")
            sys.exit(1)
    else:
        print("\n[2/3] 专家A模型已存在，跳过下载")
    
    if not os.path.exists(expert_b_path):
        try:
            print("\n[3/3] 下载专家B模型...")
            download_model(expert_b_id, "专家B模型", cache_dir)
        except Exception as e:
            print(f"专家B模型下载失败: {e}")
            sys.exit(1)
    else:
        print("\n[3/3] 专家B模型已存在，跳过下载")
    
    print("\n" + "=" * 60)
    print("所有模型下载完成！")
    print("=" * 60)
    print(f"\n模型目录结构:")
    print(f"  {models_dir}/")
    print(f"  ├── Qwen2.5-32B/                    (基础模型)")
    print(f"  ├── qwen25-32b-expert-a-qlora/     (专家A模型)")
    print(f"  └── qwen25-32b-expert-b-qlora/     (专家B模型)")
    print(f"\n缓存目录: {cache_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

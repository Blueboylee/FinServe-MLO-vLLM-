#!/usr/bin/env python3
"""
检查模型信息和磁盘空间
"""

import sys
import shutil
from pathlib import Path

try:
    from modelscope.hub.api import HubApi
    
    api = HubApi()
    
    print("="*60)
    print("模型信息检查")
    print("="*60)
    
    models_to_check = [
        "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
        "GaryLeenene/qwen25-32b-expert-a-qlora",
        "GaryLeenene/qwen25-32b-expert-b-qlora"
    ]
    
    for model_id in models_to_check:
        print(f"\n检查模型：{model_id}")
        try:
            model_info = api.get_model_info(model_id)
            print(f"  ✓ 模型存在")
            print(f"  模型 ID: {model_info.model_id}")
            if hasattr(model_info, 'file_size'):
                size_gb = model_info.file_size / (1024**3)
                print(f"  文件大小：{size_gb:.2f} GB")
        except Exception as e:
            print(f"  ✗ 模型不存在或无法访问：{str(e)}")
    
    print("\n" + "="*60)
    print("磁盘空间检查")
    print("="*60)
    
    check_path = Path.home()
    total, used, free = shutil.disk_usage(str(check_path))
    
    print(f"\n检查路径：{check_path}")
    print(f"总空间：{total / (1024**3):.2f} GB")
    print(f"已使用：{used / (1024**3):.2f} GB")
    print(f"可用空间：{free / (1024**3):.2f} GB")
    
    print("\n" + "="*60)
    print("建议")
    print("="*60)
    
    required_space = 80 * (1024**3)  # 80GB
    
    if free < required_space:
        needed = (required_space - free) / (1024**3)
        print(f"✗ 空间不足！还需要约 {needed:.2f} GB")
        print("\n解决方案：")
        print("1. 清理磁盘空间：")
        print("   df -h")
        print("   du -sh /root/*")
        print("   rm -rf /root/.cache/pip")
        print("   rm -rf /root/.cache/modelscope")
        print("")
        print("2. 使用符号链接到其他磁盘分区：")
        print("   mkdir -p /mnt/large_disk/models")
        print("   ln -s /mnt/large_disk/models ~/FinServe-MLO-vLLM-/qwen25_dual_expert/scripts/models")
        print("")
        print("3. 使用环境变量指定缓存目录：")
        print("   export MODELSCOPE_CACHE=/path/to/larger/disk")
    else:
        print("✓ 空间充足")
    
    print("="*60)
    
except ImportError:
    print("错误：请先安装 modelscope")
    print("pip install modelscope")
    sys.exit(1)
except Exception as e:
    print(f"错误：{str(e)}")
    sys.exit(1)

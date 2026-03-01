#!/usr/bin/env python3
"""
环境配置脚本
创建并配置conda环境
"""

import os
import sys
import subprocess
import shutil


def check_command(cmd: str) -> bool:
    """检查命令是否存在"""
    return shutil.which(cmd) is not None


def run_command(cmd: list, cwd: str = None) -> bool:
    """运行命令"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {' '.join(cmd)}")
        print(f"错误: {e.stderr}")
        return False


def create_conda_env(env_name: str = "qwen_expert", python_version: str = "3.10") -> bool:
    """创建conda环境"""
    print(f"\n{'=' * 60}")
    print(f"创建conda环境: {env_name}")
    print(f"Python版本: {python_version}")
    print('=' * 60)
    
    if not check_command("conda"):
        print("错误: conda未安装，请先安装Anaconda或Miniconda")
        return False
    
    print(f"\n检查是否存在环境: {env_name}")
    result = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True,
        text=True
    )
    
    if env_name in result.stdout:
        print(f"环境 {env_name} 已存在")
        return True
    
    print(f"创建conda环境...")
    cmd = [
        "conda", "create",
        "-n", env_name,
        f"python={python_version}",
        "-y"
    ]
    
    return run_command(cmd)


def install_dependencies(env_name: str = "qwen_expert") -> bool:
    """安装依赖"""
    print(f"\n{'=' * 60}")
    print("安装依赖")
    print('=' * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(base_dir, "requirements.txt")
    
    if not os.path.exists(requirements_path):
        print(f"错误: requirements.txt不存在: {requirements_path}")
        return False
    
    print(f"使用requirements.txt: {requirements_path}")
    
    cmd = [
        "conda", "run",
        "-n", env_name,
        "pip", "install",
        "-r", requirements_path
    ]
    
    return run_command(cmd)


def verify_cuda() -> bool:
    """验证CUDA环境"""
    print(f"\n{'=' * 60}")
    print("验证CUDA环境")
    print('=' * 60)
    
    if not check_command("nvidia-smi"):
        print("警告: nvidia-smi未找到")
        return False
    
    result = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    
    if not check_command("nvcc"):
        print("警告: nvcc未找到，请安装CUDA工具包")
        return False
    
    result = subprocess.run(
        ["nvcc", "--version"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    
    return True


def main():
    print("=" * 60)
    print("Qwen2.5 32B QLoRA 环境配置脚本")
    print("=" * 60)
    
    env_name = "qwen_expert"
    python_version = "3.10"
    
    if not check_command("conda"):
        print("错误: conda未安装")
        print("请先安装Anaconda或Miniconda")
        sys.exit(1)
    
    if not create_conda_env(env_name, python_version):
        print("环境创建失败")
        sys.exit(1)
    
    if not verify_cuda():
        print("CUDA验证失败，请确保已安装CUDA和驱动")
        sys.exit(1)
    
    if not install_dependencies(env_name):
        print("依赖安装失败")
        sys.exit(1)
    
    print(f"\n{'=' * 60}")
    print("环境配置完成！")
    print("=" * 60)
    print(f"\n使用以下命令激活环境:")
    print(f"  conda activate {env_name}")
    print(f"\n然后运行:")
    print(f"  python download_models.py")
    print(f"  python api_server.py --expert A")
    print(f"  python api_server.py --expert B")
    print("=" * 60)


if __name__ == "__main__":
    main()

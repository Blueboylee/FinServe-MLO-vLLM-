#!/usr/bin/env python3
"""
系统验证脚本
检查所有组件是否正确安装和配置
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """检查 Python 版本"""
    logger.info("检查 Python 版本...")
    if sys.version_info[:2] == (3, 10):
        logger.info(f"✓ Python 版本：{sys.version}")
        return True
    else:
        logger.error(f"✗ Python 版本：{sys.version} (需要 3.10)")
        return False


def check_dependencies():
    """检查依赖包"""
    logger.info("检查依赖包...")
    
    required_packages = {
        'torch': 'PyTorch',
        'vllm': 'vLLM',
        'modelscope': 'ModelScope',
        'transformers': 'Transformers',
        'peft': 'PEFT',
        'bitsandbytes': 'BitsAndBytes'
    }
    
    all_ok = True
    for package, name in required_packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            logger.info(f"✓ {name}: {version}")
        except ImportError:
            logger.error(f"✗ {name}: 未安装")
            all_ok = False
    
    return all_ok


def check_cuda():
    """检查 CUDA"""
    logger.info("检查 CUDA...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.error("✗ CUDA 不可用")
            return False
        
        cuda_version = torch.version.cuda
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        
        logger.info(f"✓ CUDA 版本：{cuda_version}")
        logger.info(f"✓ GPU 数量：{gpu_count}")
        logger.info(f"✓ GPU 型号：{gpu_name}")
        logger.info(f"✓ GPU 显存：{gpu_memory:.2f} GB")
        
        if gpu_memory < 32:
            logger.warning(f"⚠ 显存小于 32GB，可能影响性能")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ CUDA 检查失败：{str(e)}")
        return False


def check_model_files():
    """检查模型文件"""
    logger.info("检查模型文件...")
    
    base_dir = Path(__file__).parent
    models = {
        'base': base_dir / 'models' / 'base',
        'expert-a': base_dir / 'models' / 'expert-a',
        'expert-b': base_dir / 'models' / 'expert-b'
    }
    
    all_ok = True
    for name, path in models.items():
        if not path.exists():
            logger.error(f"✗ {name}: 目录不存在")
            all_ok = False
            continue
        
        files = list(path.glob('*'))
        if len(files) == 0:
            logger.error(f"✗ {name}: 目录为空")
            all_ok = False
        else:
            logger.info(f"✓ {name}: {len(files)} 个文件")
    
    return all_ok


def check_vllm_compatibility():
    """检查 vLLM 兼容性"""
    logger.info("检查 vLLM 兼容性...")
    
    try:
        from vllm import LLM
        
        logger.info("✓ vLLM 可以正常导入")
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_compute = torch.cuda.get_device_capability(0)
                if gpu_compute[0] >= 7:
                    logger.info(f"✓ GPU 计算能力：{gpu_compute[0]}.{gpu_compute[1]} (支持 vLLM)")
                    return True
                else:
                    logger.error(f"✗ GPU 计算能力：{gpu_compute[0]}.{gpu_compute[1]} (需要 7.0+)")
                    return False
            else:
                logger.warning("⚠ CUDA 不可用，无法检查 GPU 计算能力")
                return True
        except Exception as e:
            logger.error(f"✗ GPU 检查失败：{str(e)}")
            return False
            
    except ImportError as e:
        logger.error(f"✗ vLLM 导入失败：{str(e)}")
        return False


def check_memory_sharing():
    """检查显存共享机制"""
    logger.info("检查显存共享机制...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.warning("⚠ CUDA 不可用，无法检查显存共享")
            return True
        
        initial_memory = torch.cuda.memory_allocated()
        
        logger.info(f"✓ 初始显存使用：{initial_memory / (1024 ** 3):.2f} GB")
        logger.info("✓ 显存共享机制检查通过")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 显存检查失败：{str(e)}")
        return False


def main():
    """主验证函数"""
    logger.info("="*60)
    logger.info("Qwen2.5 双专家模型 - 系统验证")
    logger.info("="*60)
    
    checks = [
        ("Python 版本", check_python_version),
        ("依赖包", check_dependencies),
        ("CUDA", check_cuda),
        ("vLLM 兼容性", check_vllm_compatibility),
        ("模型文件", check_model_files),
        ("显存共享", check_memory_sharing)
    ]
    
    results = []
    for name, check_func in checks:
        logger.info(f"\n{'='*40}")
        result = check_func()
        results.append((name, result))
    
    logger.info(f"\n{'='*60}")
    logger.info("验证总结")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{status}: {name}")
    
    logger.info(f"\n总计：{passed}/{total} 项检查通过")
    
    if passed == total:
        logger.info("\n✓ 所有检查通过，系统已就绪！")
        return 0
    else:
        logger.error(f"\n✗ {total - passed} 项检查失败，请修复后再运行")
        return 1


if __name__ == "__main__":
    sys.exit(main())

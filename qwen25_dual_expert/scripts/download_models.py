#!/usr/bin/env python3
"""
模型下载脚本 - 从 ModelScope 下载基础模型和专家模型
"""

import os
import sys
import logging
from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download

# 确保日志目录存在
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / 'download.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


BASE_MODEL_ID = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
EXPERT_A_MODEL_ID = "GaryLeenene/qwen25-32b-expert-a-qlora"
EXPERT_B_MODEL_ID = "GaryLeenene/qwen25-32b-expert-b-qlora"


def check_disk_space(path: Path, required_gb: int = 100) -> bool:
    """
    检查磁盘空间
    
    Args:
        path: 要检查的路径
        required_gb: 需要的最小空间（GB）
    
    Returns:
        bool: 空间是否足够
    """
    import shutil
    
    total, used, free = shutil.disk_usage(str(path))
    free_gb = free / (1024 ** 3)
    
    logger.info(f"磁盘总空间：{total / (1024 ** 3):.2f} GB")
    logger.info(f"磁盘已使用：{used / (1024 ** 3):.2f} GB")
    logger.info(f"磁盘可用空间：{free_gb:.2f} GB")
    logger.info(f"需要空间：约 {required_gb} GB")
    
    if free_gb < required_gb:
        logger.error(f"磁盘空间不足！需要至少 {required_gb} GB，当前只有 {free_gb:.2f} GB")
        return False
    
    logger.info("✓ 磁盘空间充足")
    return True


def setup_directories():
    """创建必要的目录结构"""
    base_dir = Path(__file__).parent
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    base_model_dir = models_dir / "base"
    expert_a_dir = models_dir / "expert-a"
    expert_b_dir = models_dir / "expert-b"
    
    for dir_path in [base_model_dir, expert_a_dir, expert_b_dir]:
        dir_path.mkdir(exist_ok=True)
    
    return {
        'base': str(base_model_dir),
        'expert_a': str(expert_a_dir),
        'expert_b': str(expert_b_dir)
    }


def download_model(model_id: str, save_dir: str, description: str):
    """
    从 ModelScope 下载模型
    
    Args:
        model_id: 模型 ID
        save_dir: 保存目录
        description: 模型描述（用于日志）
    
    Returns:
        bool: 下载是否成功
    """
    try:
        logger.info(f"开始下载 {description}: {model_id}")
        logger.info(f"保存路径：{save_dir}")
        
        snapshot_download(
            model_id=model_id,
            local_dir=save_dir,
            revision="master"
        )
        
        logger.info(f"{description} 下载完成")
        return True
        
    except Exception as e:
        logger.error(f"{description} 下载失败：{str(e)}")
        return False


def verify_download(model_dir: str, description: str) -> bool:
    """
    验证模型文件是否完整
    
    Args:
        model_dir: 模型目录
        description: 模型描述
    
    Returns:
        bool: 验证是否通过
    """
    required_files = {
        'base': ['config.json', 'generation_config.json', 'tokenizer.json'],
        'expert-a': ['adapter_config.json', 'adapter_model.safetensors'],
        'expert-b': ['adapter_config.json', 'adapter_model.safetensors']
    }
    
    model_type = description.lower().replace(' ', '-')
    required = required_files.get(model_type, ['config.json'])
    
    for file_name in required:
        file_path = Path(model_dir) / file_name
        if not file_path.exists():
            logger.error(f"缺少必要文件：{file_name}")
            return False
        logger.info(f"验证通过：{file_name}")
    
    return True


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("开始下载 Qwen2.5-32B 双专家模型")
    logger.info("=" * 60)
    
    model_dirs = setup_directories()
    
    # 检查磁盘空间
    models_dir = Path(model_dirs['base']).parent
    if not check_disk_space(models_dir, required_gb=80):
        logger.error("请先清理磁盘空间或使用更大容量的存储设备")
        return 1
    
    download_tasks = [
        (BASE_MODEL_ID, model_dirs['base'], "基础模型"),
        (EXPERT_A_MODEL_ID, model_dirs['expert_a'], "专家 A"),
        (EXPERT_B_MODEL_ID, model_dirs['expert_b'], "专家 B")
    ]
    
    success_count = 0
    for model_id, save_dir, description in download_tasks:
        logger.info(f"\n{'='*40}")
        if download_model(model_id, save_dir, description):
            if verify_download(save_dir, description):
                success_count += 1
                logger.info(f"✓ {description} 下载并验证成功")
            else:
                logger.error(f"✗ {description} 验证失败")
        else:
            logger.error(f"✗ {description} 下载失败")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"下载完成：{success_count}/{len(download_tasks)} 个模型成功")
    logger.info("=" * 60)
    
    if success_count == len(download_tasks):
        logger.info("所有模型下载成功！")
        return 0
    else:
        logger.error("部分模型下载失败，请检查日志")
        return 1


if __name__ == "__main__":
    sys.exit(main())

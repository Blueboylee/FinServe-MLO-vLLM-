#!/usr/bin/env python3
"""
模型下载脚本 - 从 ModelScope 下载基础模型和专家模型
"""

import os
import sys
import logging
from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/download.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


BASE_MODEL_ID = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
EXPERT_A_MODEL_ID = "GaryLeenene/qwen25-32b-expert-a-qlora"
EXPERT_B_MODEL_ID = "GaryLeenene/qwen25-32b-expert-b-qlora"


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

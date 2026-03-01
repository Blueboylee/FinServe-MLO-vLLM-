#!/usr/bin/env python3
"""
专家服务启动脚本
启动专家A或专家B的vLLM服务
"""

import os
import sys
import time
import argparse
import logging
from typing import Optional

from config import config
from unsloth import FastLanguageModel


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('expert_service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_expert_model(
    expert_path: str,
    base_model_path: str,
    gpu_config: dict
) -> tuple:
    """
    加载专家模型
    
    Args:
        expert_path: 专家模型路径
        base_model_path: 基础模型路径
        gpu_config: GPU配置参数
        
    Returns:
        模型和tokenizer
    """
    try:
        logger.info(f"加载基础模型: {base_model_path}")
        logger.info(f"加载专家模型: {expert_path}")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_path,
            max_seq_length=32768,
            dtype=None,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("加载QLoRA适配器...")
        model = FastLanguageModel.load_lora(model, expert_path)
        model.eval()
        
        logger.info("模型加载完成")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise


def create_service(
    expert_name: str,
    expert_path: str,
    port: int,
    gpu_config: dict
):
    """
    创建并启动专家服务
    
    Args:
        expert_name: 专家名称
        expert_path: 专家模型路径
        port: 服务端口
        gpu_config: GPU配置参数
    """
    logger.info(f"启动 {expert_name} 服务...")
    logger.info(f"服务端口: {port}")
    
    try:
        model, tokenizer = load_expert_model(
            expert_path=expert_path,
            base_model_path=config.model_config.base_model_path,
            gpu_config=gpu_config
        )
        
        logger.info(f"{expert_name} 服务启动成功")
        logger.info(f"访问 http://localhost:{port} 进行API调用")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"{expert_name} 服务启动失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="启动专家服务")
    parser.add_argument(
        "--expert",
        type=str,
        required=True,
        choices=["A", "B"],
        help="专家名称 (A or B)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="服务端口 (默认: A=8001, B=8002)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务主机地址"
    )
    
    args = parser.parse_args()
    
    expert_name = f"专家{args.expert}"
    expert_key = f"expert_{args.expert.lower()}_path"
    port = args.port or (8001 if args.expert == "A" else 8002)
    
    expert_path = getattr(config.model_config, expert_key)
    
    if not os.path.exists(expert_path):
        logger.error(f"专家模型路径不存在: {expert_path}")
        logger.error("请先运行 download_models.py 下载模型")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info(f"部署配置")
    logger.info("=" * 60)
    logger.info(f"专家: {expert_name}")
    logger.info(f"模型路径: {expert_path}")
    logger.info(f"服务端口: {port}")
    logger.info(f"主机: {args.host}")
    logger.info("=" * 60)
    
    try:
        model, tokenizer = create_service(
            expert_name=expert_name,
            expert_path=expert_path,
            port=port,
            gpu_config=config.gpu_config.vllm_kwargs
        )
        
        logger.info(f"{expert_name} 服务正在运行...")
        logger.info("按 Ctrl+C 停止服务")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info(f"\n{expert_name} 服务已停止")
        sys.exit(0)
    except Exception as e:
        logger.error(f"服务运行异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

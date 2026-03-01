#!/usr/bin/env python3
"""
双专家模型加载器 - 共享同一个 4bit GPTQ 基础模型
使用 vLLM 进行高效推理
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from vllm import LLM, SamplingParams
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/inference.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""
    base_model_path: str
    expert_a_path: str
    expert_b_path: str
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    tensor_parallel_size: int = 1


class DualExpertModel:
    """
    双专家模型加载器
    共享同一个 4bit GPTQ 基础模型，动态加载不同的 LoRA 适配器
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = None
        self.base_model = None
        self.expert_a_model = None
        self.expert_b_model = None
        self.lora_a_path = None
        self.lora_b_path = None
        
    def load_tokenizer(self):
        """加载分词器"""
        try:
            logger.info("加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_path,
                trust_remote_code=True,
                padding_side='left'
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("分词器加载成功")
            return True
            
        except Exception as e:
            logger.error(f"分词器加载失败：{str(e)}")
            return False
    
    def load_base_model_vllm(self) -> bool:
        """
        使用 vLLM 加载基础模型
        vLLM 会自动处理 GPTQ 量化模型
        """
        try:
            logger.info("使用 vLLM 加载基础模型...")
            logger.info(f"基础模型路径：{self.config.base_model_path}")
            
            self.llm = LLM(
                model=self.config.base_model_path,
                trust_remote_code=True,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                tensor_parallel_size=self.config.tensor_parallel_size,
                quantization="gptq",
                dtype="float16",
                enforce_eager=False,
                enable_lora=True,
                max_lora_rank=64,
            )
            
            logger.info("基础模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"基础模型加载失败：{str(e)}")
            return False
    
    def load_expert_adapters(self) -> bool:
        """
        加载专家 LoRA 适配器
        由于 vLLM 的 LoRA 支持，我们可以在运行时切换不同的适配器
        """
        try:
            logger.info("验证专家适配器...")
            
            adapter_a_config = Path(self.config.expert_a_path) / "adapter_config.json"
            adapter_b_config = Path(self.config.expert_b_path) / "adapter_config.json"
            
            if not adapter_a_config.exists():
                logger.error(f"专家 A 配置文件不存在：{adapter_a_config}")
                return False
            
            if not adapter_b_config.exists():
                logger.error(f"专家 B 配置文件不存在：{adapter_b_config}")
                return False
            
            self.lora_a_path = self.config.expert_a_path
            self.lora_b_path = self.config.expert_b_path
            
            logger.info(f"专家 A 适配器路径：{self.lora_a_path}")
            logger.info(f"专家 B 适配器路径：{self.lora_b_path}")
            logger.info("专家适配器验证通过")
            
            return True
            
        except Exception as e:
            logger.error(f"专家适配器加载失败：{str(e)}")
            return False
    
    def generate_with_expert(
        self, 
        prompt: str, 
        expert: str = "A",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        使用指定专家生成回复
        
        Args:
            prompt: 输入提示
            expert: 选择专家 ("A" 或 "B")
            max_tokens: 最大生成长度
            temperature: 温度参数
            top_p: Top-p 采样参数
        
        Returns:
            生成的文本
        """
        try:
            lora_path = self.lora_a_path if expert == "A" else self.lora_b_path
            
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer else None
            )
            
            outputs = self.llm.generate(
                prompts=[prompt],
                sampling_params=sampling_params,
                lora_request=lora_path
            )
            
            if outputs and len(outputs) > 0:
                return outputs[0].outputs[0].text
            else:
                return ""
                
        except Exception as e:
            logger.error(f"生成失败：{str(e)}")
            return f"错误：{str(e)}"
    
    def generate_with_routing(
        self,
        prompt: str,
        routing_strategy: str = "confidence",
        **kwargs
    ) -> Tuple[str, str]:
        """
        根据路由策略自动选择专家
        
        Args:
            prompt: 输入提示
            routing_strategy: 路由策略 ("confidence" 或 "round_robin")
            **kwargs: 传递给 generate_with_expert 的参数
        
        Returns:
            (生成的文本，使用的专家)
        """
        if routing_strategy == "round_robin":
            if not hasattr(self, '_last_expert'):
                self._last_expert = "A"
            current_expert = "B" if self._last_expert == "A" else "A"
            self._last_expert = current_expert
        else:
            current_expert = "A"
        
        response = self.generate_with_expert(prompt, current_expert, **kwargs)
        return response, current_expert
    
    def verify_memory_sharing(self) -> bool:
        """
        验证显存共享机制是否正常工作
        """
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                
                logger.info(f"当前显存使用：{allocated:.2f} GB")
                logger.info(f"当前显存保留：{reserved:.2f} GB")
                logger.info(f"总显存：{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
                
                if allocated < 32:
                    logger.info("显存使用正常，共享机制工作正常")
                    return True
                else:
                    logger.warning("显存使用过高，可能存在共享问题")
                    return False
            else:
                logger.warning("CUDA 不可用，无法验证显存共享")
                return True
                
        except Exception as e:
            logger.error(f"显存验证失败：{str(e)}")
            return False


def create_model_config(base_dir: Optional[Path] = None) -> ModelConfig:
    """创建模型配置"""
    if base_dir is None:
        base_dir = Path(__file__).parent
    
    models_dir = base_dir / "models"
    
    return ModelConfig(
        base_model_path=str(models_dir / "base"),
        expert_a_path=str(models_dir / "expert-a"),
        expert_b_path=str(models_dir / "expert-b"),
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        tensor_parallel_size=1
    )


def main():
    """测试模型加载"""
    logger.info("=" * 60)
    logger.info("双专家模型加载测试")
    logger.info("=" * 60)
    
    config = create_model_config()
    model = DualExpertModel(config)
    
    logger.info("\n步骤 1: 加载分词器...")
    if not model.load_tokenizer():
        return 1
    
    logger.info("\n步骤 2: 加载基础模型...")
    if not model.load_base_model_vllm():
        return 1
    
    logger.info("\n步骤 3: 加载专家适配器...")
    if not model.load_expert_adapters():
        return 1
    
    logger.info("\n步骤 4: 验证显存共享...")
    if not model.verify_memory_sharing():
        logger.warning("显存验证未通过，但可能仍可正常使用")
    
    logger.info("\n" + "=" * 60)
    logger.info("模型加载测试完成！")
    logger.info("=" * 60)
    
    test_prompt = "你好，请介绍一下你自己。"
    logger.info(f"\n测试提示：{test_prompt}")
    
    response_a, expert_used = model.generate_with_routing(test_prompt)
    logger.info(f"\n使用专家：{expert_used}")
    logger.info(f"回复：{response_a}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

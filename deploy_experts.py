#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 vLLM 部署 Qwen2.5-32B 双专家模型
支持通过 API 切换不同的专家模型
"""

import os
import sys
import time
import requests
from typing import Optional
from vllm import LLM, SamplingParams
from vllm.server import serve


class ExpertModelDeployer:
    """专家模型部署器"""
    
    def __init__(self, base_model_path: str, expert_a_path: str, expert_b_path: str):
        """
        初始化部署器
        
        Args:
            base_model_path: 基础 Qwen2.5-32B 4bit GPTQ 模型路径
            expert_a_path: 专家 A 的 LoRA 适配器路径
            expert_b_path: 专家 B 的 LoRA 适配器路径
        """
        self.base_model_path = base_model_path
        self.expert_a_path = expert_a_path
        self.expert_b_path = expert_b_path
        self.llm: Optional[LLM] = None
        self.current_expert: Optional[str] = None
        
    def load_base_model(self, max_model_len: int = 4096, gpu_memory_utilization: float = 0.9):
        """
        加载基础模型
        
        Args:
            max_model_len: 最大序列长度
            gpu_memory_utilization: GPU 内存利用率 (0-1)
        """
        print("="*60)
        print("加载基础模型 Qwen2.5-32B...")
        print("="*60)
        
        self.llm = LLM(
            model=self.base_model_path,
            load_format="gptq",
            dtype="auto",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            trust_remote_code=True,
        )
        
        print("✓ 基础模型加载完成")
        
    def generate(self, prompt: str, expert: str = "expert_a", 
                 max_tokens: int = 512, temperature: float = 0.7,
                 top_p: float = 0.9) -> str:
        """
        使用指定专家模型生成文本
        
        Args:
            prompt: 输入提示
            expert: 使用的专家模型 ("expert_a" 或 "expert_b")
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: Top-p 采样参数
            
        Returns:
            生成的文本
        """
        if self.llm is None:
            raise RuntimeError("请先调用 load_base_model() 加载基础模型")
        
        # 确定 LoRA 路径
        if expert == "expert_a":
            lora_path = self.expert_a_path
        elif expert == "expert_b":
            lora_path = self.expert_b_path
        else:
            raise ValueError(f"未知的专家模型：{expert}，请使用 'expert_a' 或 'expert_b'")
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=[151643, 151644, 151645],  # Qwen 的特殊 token
        )
        
        # 生成
        print(f"\n使用专家模型：{expert}")
        print(f"输入：{prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        
        outputs = self.llm.generate(
            [prompt],
            sampling_params,
            lora_request=None  # TODO: 如果需要 LoRA，需要配置 lora_request
        )
        
        generated_text = outputs[0].outputs[0].text
        print(f"输出：{generated_text}")
        
        return generated_text


def load_model_paths():
    """从配置文件加载模型路径"""
    config_file = os.path.join(os.path.dirname(__file__), "models", "model_paths.txt")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"模型配置文件不存在：{config_file}\n"
            "请先运行：python download_models.py"
        )
    
    model_paths = {}
    with open(config_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                model_paths[key.strip()] = value.strip()
    
    return model_paths


def main():
    """主函数"""
    print("="*60)
    print("Qwen2.5-32B 双专家模型 vLLM 部署")
    print("="*60)
    
    # 加载模型路径
    try:
        model_paths = load_model_paths()
        expert_a_path = model_paths.get("expert_a")
        expert_b_path = model_paths.get("expert_b")
        
        if not expert_a_path or not expert_b_path:
            raise ValueError("模型路径配置不完整")
            
    except Exception as e:
        print(f"✗ 加载模型路径失败：{e}")
        sys.exit(1)
    
    # 注意：由于两个专家模型都是基于同一个 Qwen2.5-32B 基础模型训练的，
    # 我们需要指定基础模型路径。这里我们使用 expert_a 作为基础模型，
    # 因为 LoRA 适配器会在此基础上加载。
    base_model_path = expert_a_path  # 或者指定单独的 base model 路径
    
    print(f"\n基础模型：{base_model_path}")
    print(f"专家 A: {expert_a_path}")
    print(f"专家 B: {expert_b_path}")
    
    # 创建部署器
    deployer = ExpertModelDeployer(
        base_model_path=base_model_path,
        expert_a_path=expert_a_path,
        expert_b_path=expert_b_path
    )
    
    # 加载基础模型
    try:
        deployer.load_base_model(
            max_model_len=4096,
            gpu_memory_utilization=0.85  # V100 16GB 建议设置 0.85
        )
    except Exception as e:
        print(f"✗ 模型加载失败：{e}")
        print("\n提示：如果是显存不足，尝试降低 gpu_memory_utilization 参数")
        sys.exit(1)
    
    # 测试生成
    print("\n" + "="*60)
    print("测试专家模型生成")
    print("="*60)
    
    test_prompts = [
        "你好，请介绍一下你自己。",
        "什么是机器学习？",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[测试 {i}/{len(test_prompts)}]")
        
        # 使用专家 A
        print("\n--- 使用专家 A ---")
        try:
            deployer.generate(prompt, expert="expert_a", max_tokens=200)
        except Exception as e:
            print(f"生成失败：{e}")
        
        # 使用专家 B
        print("\n--- 使用专家 B ---")
        try:
            deployer.generate(prompt, expert="expert_b", max_tokens=200)
        except Exception as e:
            print(f"生成失败：{e}")
    
    print("\n" + "="*60)
    print("部署测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()

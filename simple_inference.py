#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的部署脚本（不依赖 LoRA）
直接加载专家模型进行推理
"""

import os
import sys
import argparse
from typing import Optional
from vllm import LLM, SamplingParams


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


class SimpleExpertInference:
    """简单的专家模型推理（每个专家独立加载）"""
    
    def __init__(self, model_path: str, max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.85):
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm: Optional[LLM] = None
        
    def load(self):
        """加载模型"""
        print(f"\n加载模型：{self.model_path}")
        
        self.llm = LLM(
            model=self.model_path,
            load_format="gptq",  # 如果是 GPTQ 格式
            # load_format="auto",  # 或者自动检测
            dtype="auto",
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=1,
            trust_remote_code=True,
        )
        
        print("✓ 模型加载完成")
    
    def generate(self, prompt: str, max_tokens: int = 512,
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        """生成文本"""
        if self.llm is None:
            raise RuntimeError("请先加载模型")
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=[151643, 151644, 151645],
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简单的专家模型推理脚本")
    parser.add_argument("--expert", type=str, default="expert_a",
                       choices=["expert_a", "expert_b"],
                       help="选择专家模型")
    parser.add_argument("--max-model-len", type=int, default=4096,
                       help="最大序列长度")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                       help="GPU 内存利用率")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Qwen2.5-32B 专家模型推理")
    print("="*60)
    
    # 加载模型路径
    try:
        model_paths = load_model_paths()
        expert_path = model_paths.get(args.expert)
        
        if not expert_path:
            raise ValueError(f"未找到专家模型：{args.expert}")
            
    except Exception as e:
        print(f"✗ 加载模型路径失败：{e}")
        sys.exit(1)
    
    print(f"\n使用专家：{args.expert}")
    print(f"模型路径：{expert_path}")
    
    # 创建推理器
    inferencer = SimpleExpertInference(
        model_path=expert_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # 加载模型
    try:
        inferencer.load()
    except Exception as e:
        print(f"\n✗ 模型加载失败：{e}")
        print("\n提示:")
        print("  1. 如果是显存不足，尝试降低 --gpu-memory-utilization")
        print("  2. 如果是格式错误，检查模型是否为 GPTQ 格式")
        sys.exit(1)
    
    # 交互式对话
    print("\n" + "="*60)
    print("开始对话（输入 'quit' 退出）")
    print("="*60)
    
    while True:
        try:
            prompt = input("\n你：").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            if not prompt:
                continue
            
            print(f"\n{args.expert}:", end=" ", flush=True)
            response = inferencer.generate(prompt, max_tokens=512)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n中断退出")
            break
        except Exception as e:
            print(f"\n错误：{e}")


if __name__ == "__main__":
    main()

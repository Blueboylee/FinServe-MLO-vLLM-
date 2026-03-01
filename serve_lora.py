#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 vLLM 部署 Qwen2.5 32B + 双 LoRA 专家模型
支持运行时动态切换 LoRA 适配器
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class QwenLoRAServer:
    """Qwen2.5 LoRA 服务类"""
    
    def __init__(
        self,
        base_model_path: str,
        lora_models: Dict[str, str],
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1
    ):
        """
        初始化 LoRA 服务
        
        Args:
            base_model_path: 基座模型路径
            lora_models: LoRA 模型字典 {name: path}
            gpu_memory_utilization: GPU 内存利用率
            max_model_len: 最大序列长度
            tensor_parallel_size: 张量并行大小
        """
        print(f"加载基座模型：{base_model_path}")
        
        self.base_model_path = base_model_path
        self.lora_models = lora_models
        self.loaded_loras: Dict[str, LoRARequest] = {}
        
        # 初始化 vLLM 模型
        self.llm = LLM(
            model=base_model_path,
            enable_lora=True,
            max_loras=2,  # 最多同时加载 2 个 LoRA
            max_lora_rank=128,  # 根据 LoRA 秩调整
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            dtype="auto",
            trust_remote_code=True,
        )
        
        print("✓ 基座模型加载完成")
        
        # 预加载所有 LoRA 适配器
        self._preload_loras()
    
    def _preload_loras(self):
        """预加载所有 LoRA 适配器"""
        print("\n预加载 LoRA 适配器:")
        for name, path in self.lora_models.items():
            print(f"  - 加载 {name}: {path}")
            lora_request = LoRARequest(
                lora_name=name,
                lora_int_id=hash(name) % 10000,
                lora_path=path,
            )
            self.loaded_loras[name] = lora_request
            print(f"    ✓ {name} 加载完成")
        print()
    
    def generate(
        self,
        prompts: List[str],
        lora_name: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ):
        """
        生成文本
        
        Args:
            prompts: 输入提示列表
            lora_name: 使用的 LoRA 适配器名称，None 表示使用基座模型
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: Top-p 采样参数
        """
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        # 选择 LoRA 适配器
        lora_request = None
        if lora_name:
            if lora_name not in self.loaded_loras:
                raise ValueError(f"未知的 LoRA 适配器：{lora_name}")
            lora_request = self.loaded_loras[lora_name]
            print(f"使用 LoRA 适配器：{lora_name}")
        else:
            print("使用基座模型（无 LoRA）")
        
        # 生成
        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            lora_request=lora_request
        )
        
        return outputs
    
    def generate_single(
        self,
        prompt: str,
        lora_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """单个提示生成，返回纯文本"""
        outputs = self.generate([prompt], lora_name=lora_name, **kwargs)
        return outputs[0].outputs[0].text
    
    def list_loras(self) -> List[str]:
        """列出所有可用的 LoRA 适配器"""
        return list(self.loaded_loras.keys())


def load_model_config(config_path: str) -> Dict:
    """加载模型配置文件"""
    config = {}
    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config


def main():
    parser = argparse.ArgumentParser(description='Qwen2.5 LoRA vLLM 服务')
    parser.add_argument(
        '--config',
        type=str,
        default='models/model_config.txt',
        help='模型配置文件路径'
    )
    parser.add_argument(
        '--gpu-memory-utilization',
        type=float,
        default=0.85,
        help='GPU 内存利用率 (0-1)'
    )
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=4096,
        help='最大序列长度'
    )
    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        default=1,
        help='张量并行大小'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='服务主机地址'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='服务端口'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误：配置文件不存在：{config_path}")
        print("请先运行：python download_models.py")
        return
    
    config = load_model_config(str(config_path))
    
    # 构建 LoRA 模型字典
    lora_models = {
        'expert_a': config.get('expert_a_path', ''),
        'expert_b': config.get('expert_b_path', ''),
    }
    
    # 验证路径
    base_model_path = config.get('base_model_path', '')
    if not Path(base_model_path).exists():
        print(f"错误：基座模型路径不存在：{base_model_path}")
        return
    
    for name, path in lora_models.items():
        if not Path(path).exists():
            print(f"错误：LoRA 模型路径不存在 {name}: {path}")
            return
    
    # 创建服务
    print("="*60)
    print("Qwen2.5 32B + 双 LoRA 专家模型服务")
    print("="*60)
    print(f"基座模型：{base_model_path}")
    print(f"LoRA 模型：{list(lora_models.keys())}")
    print(f"GPU 内存利用率：{args.gpu_memory_utilization}")
    print(f"最大序列长度：{args.max_model_len}")
    print("="*60)
    
    server = QwenLoRAServer(
        base_model_path=base_model_path,
        lora_models=lora_models,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size
    )
    
    # 交互式测试
    print("\n服务已启动！输入文本进行测试")
    print("命令:")
    print("  :quit - 退出")
    print("  :lora expert_a - 切换到专家 A")
    print("  :lora expert_b - 切换到专家 B")
    print("  :lora base - 使用基座模型")
    print("="*60)
    
    current_lora = None
    
    while True:
        try:
            user_input = input("\n[用户]: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == ':quit':
                print("退出服务")
                break
            
            if user_input.lower().startswith(':lora'):
                parts = user_input.split()
                if len(parts) == 2:
                    lora_name = parts[1]
                    if lora_name == 'base':
                        current_lora = None
                        print("已切换到基座模型")
                    elif lora_name in server.list_loras():
                        current_lora = lora_name
                        print(f"已切换到 {lora_name}")
                    else:
                        print(f"未知的 LoRA: {lora_name}")
                        print(f"可用的 LoRA: {server.list_loras()}")
                else:
                    print(f"当前 LoRA: {current_lora if current_lora else 'base'}")
                    print(f"可用的 LoRA: {server.list_loras()}")
                continue
            
            # 生成回复
            print(f"\n[{current_lora or 'base'} 正在思考...]")
            response = server.generate_single(
                prompt=user_input,
                lora_name=current_lora,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9
            )
            print(f"[助手]: {response}")
            
        except KeyboardInterrupt:
            print("\n\n中断退出")
            break
        except Exception as e:
            print(f"错误：{e}")


if __name__ == "__main__":
    main()

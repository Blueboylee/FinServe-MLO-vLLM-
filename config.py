#!/usr/bin/env python3
"""
部署配置文件
配置模型路径、GPU参数和vLLM服务参数
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """模型配置"""
    base_model_name: str = "Qwen2.5-32B"
    expert_a_name: str = "qwen25-32b-expert-a-qlora"
    expert_b_name: str = "qwen25-32b-expert-b-qlora"
    
    @property
    def base_model_path(self) -> str:
        return os.path.join(self.models_dir, self.base_model_name)
    
    @property
    def expert_a_path(self) -> str:
        return os.path.join(self.models_dir, self.expert_a_name)
    
    @property
    def expert_b_path(self) -> str:
        return os.path.join(self.models_dir, self.expert_b_name)
    
    @property
    def models_dir(self) -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "models")


@dataclass
class GPUConfig:
    """GPU配置 - 适配RTX 4080S 32G"""
    num_gpus: int = 1
    gpu_memory_utilization: float = 0.95
    tensor_parallel_size: int = 1
    max_model_len: int = 32768
    max_num_seqs: int = 256
    dtype: str = "auto"
    quantization: Optional[str] = None
    enforce_eager: bool = False
    disable_custom_all_reduce: bool = True
    
    @property
    def vllm_kwargs(self) -> dict:
        return {
            "num_gpus": self.num_gpus,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "enforce_eager": self.enforce_eager,
            "disable_custom_all_reduce": self.disable_custom_all_reduce,
        }


@dataclass
class ServerConfig:
    """服务器配置"""
    host: str = "0.0.0.0"
    port_a: int = 8001
    port_b: int = 8002
    workers: int = 1
    timeout_keep_alive: int = 120
    
    @property
    def expert_a_url(self) -> str:
        return f"http://{self.host}:{self.port_a}"
    
    @property
    def expert_b_url(self) -> str:
        return f"http://{self.host}:{self.port_b}"


@dataclass
class GenerationConfig:
    """生成参数配置"""
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    use_beam_search: bool = False
    stop_token_ids: list = None
    
    def __post_init__(self):
        if self.stop_token_ids is None:
            self.stop_token_ids = [151645]
    
    def get_generation_params(self) -> dict:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "use_beam_search": self.use_beam_search,
            "stop_token_ids": self.stop_token_ids,
        }


class DeploymentConfig:
    """部署配置管理器"""
    
    def __init__(self):
        self.model_config = ModelConfig()
        self.gpu_config = GPUConfig()
        self.server_config = ServerConfig()
        self.generation_config = GenerationConfig()
    
    def print_config(self):
        """打印配置信息"""
        print("\n" + "=" * 60)
        print("部署配置信息")
        print("=" * 60)
        
        print("\n[模型配置]")
        print(f"  基础模型: {self.model_config.base_model_path}")
        print(f"  专家A: {self.model_config.expert_a_path}")
        print(f"  专家B: {self.model_config.expert_b_path}")
        
        print("\n[GPU配置]")
        print(f"  GPU数量: {self.gpu_config.num_gpus}")
        print(f"  GPU内存利用率: {self.gpu_config.gpu_memory_utilization}")
        print(f"  张量并行: {self.gpu_config.tensor_parallel_size}")
        print(f"  最大模型长度: {self.gpu_config.max_model_len}")
        print(f"  最大序列数: {self.gpu_config.max_num_seqs}")
        
        print("\n[服务器配置]")
        print(f"  专家A服务: {self.server_config.expert_a_url}")
        print(f"  专家B服务: {self.server_config.expert_b_url}")
        
        print("\n[生成参数]")
        gen_params = self.generation_config.get_generation_params()
        for key, value in gen_params.items():
            print(f"  {key}: {value}")
        
        print("=" * 60 + "\n")


config = DeploymentConfig()

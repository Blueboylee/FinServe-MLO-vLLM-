#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen2.5-32B 双专家模型 API 服务器
提供 RESTful API 接口，支持切换不同专家模型
"""

import os
import sys
import argparse
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from vllm import LLM, SamplingParams


class GenerationRequest(BaseModel):
    """生成请求模型"""
    prompt: str = Field(..., description="输入提示文本")
    expert: str = Field(default="expert_a", description="使用的专家模型：expert_a 或 expert_b")
    max_tokens: int = Field(default=512, ge=1, le=2048, description="最大生成 token 数")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p 采样参数")
    top_k: int = Field(default=-1, ge=-1, le=100, description="Top-k 采样参数")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="存在惩罚")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="频率惩罚")


class GenerationResponse(BaseModel):
    """生成响应模型"""
    text: str
    expert: str
    prompt: str
    usage: dict
    success: bool = True


class ExpertModelServer:
    """专家模型 API 服务器"""
    
    def __init__(self, base_model_path: str, expert_a_path: str, expert_b_path: str,
                 max_model_len: int = 4096, gpu_memory_utilization: float = 0.9):
        self.base_model_path = base_model_path
        self.expert_a_path = expert_a_path
        self.expert_b_path = expert_b_path
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm: Optional[LLM] = None
        self.app = FastAPI(
            title="Qwen2.5-32B Expert Models API",
            description="双专家模型推理服务",
            version="1.0.0"
        )
        self._setup_routes()
        self._setup_cors()
        
    def _setup_cors(self):
        """配置 CORS"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.get("/")
        async def root():
            """健康检查"""
            return {
                "status": "running",
                "model": "Qwen2.5-32B",
                "experts": ["expert_a", "expert_b"],
                "base_model": self.base_model_path
            }
        
        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            return {"status": "healthy"}
        
        @self.app.post("/generate", response_model=GenerationResponse)
        async def generate(request: GenerationRequest):
            """文本生成接口"""
            try:
                # 验证专家模型
                if request.expert not in ["expert_a", "expert_b"]:
                    raise HTTPException(
                        status_code=400,
                        detail=f"无效的专家模型：{request.expert}，请使用 expert_a 或 expert_b"
                    )
                
                # 确定 LoRA 路径
                lora_path = self.expert_a_path if request.expert == "expert_a" else self.expert_b_path
                
                # 设置采样参数
                sampling_params = SamplingParams(
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    top_k=request.top_k if request.top_k > 0 else None,
                    presence_penalty=request.presence_penalty,
                    frequency_penalty=request.frequency_penalty,
                    stop_token_ids=[151643, 151644, 151645],
                )
                
                # 生成
                outputs = self.llm.generate(
                    prompts=[request.prompt],
                    sampling_params=sampling_params,
                    lora_request=None  # TODO: 配置 LoRA request
                )
                
                generated_text = outputs[0].outputs[0].text
                
                return GenerationResponse(
                    text=generated_text,
                    expert=request.expert,
                    prompt=request.prompt,
                    usage={
                        "prompt_tokens": outputs[0].prompt_token_count,
                        "completion_tokens": outputs[0].output_token_count,
                        "total_tokens": outputs[0].prompt_token_count + outputs[0].output_token_count
                    }
                )
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"生成失败：{str(e)}")
        
        @self.app.get("/models")
        async def list_models():
            """列出可用模型"""
            return {
                "models": [
                    {
                        "id": "expert_a",
                        "name": "Qwen2.5-32B Expert A",
                        "path": self.expert_a_path
                    },
                    {
                        "id": "expert_b",
                        "name": "Qwen2.5-32B Expert B",
                        "path": self.expert_b_path
                    }
                ]
            }
    
    def load_model(self):
        """加载模型"""
        print("="*60)
        print("加载 Qwen2.5-32B 基础模型...")
        print("="*60)
        
        self.llm = LLM(
            model=self.base_model_path,
            load_format="gptq",
            dtype="auto",
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enable_lora=True,  # 启用 LoRA 支持
        )
        
        print("✓ 模型加载完成")
        print(f"  - 基础模型：{self.base_model_path}")
        print(f"  - 专家 A: {self.expert_a_path}")
        print(f"  - 专家 B: {self.expert_b_path}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """运行服务器"""
        print(f"\n启动 API 服务器：http://{host}:{port}")
        print(f"API 文档：http://{host}:{port}/docs")
        print("="*60)
        
        uvicorn.run(self.app, host=host, port=port)


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
    parser = argparse.ArgumentParser(description="Qwen2.5-32B 双专家模型 API 服务器")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser.add_argument("--max-model-len", type=int, default=4096, help="最大序列长度")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, 
                       help="GPU 内存利用率 (0-1)，V100 建议 0.85")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Qwen2.5-32B 双专家模型 API 服务器")
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
    
    # 使用 expert_a 作为基础模型路径
    # （因为两个专家都是基于同一个 Qwen2.5-32B 训练的）
    base_model_path = expert_a_path
    
    print(f"\n配置信息:")
    print(f"  - 基础模型：{base_model_path}")
    print(f"  - 专家 A: {expert_a_path}")
    print(f"  - 专家 B: {expert_b_path}")
    print(f"  - 最大序列长度：{args.max_model_len}")
    print(f"  - GPU 内存利用率：{args.gpu_memory_utilization}")
    
    # 创建并启动服务器
    server = ExpertModelServer(
        base_model_path=base_model_path,
        expert_a_path=expert_a_path,
        expert_b_path=expert_b_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    try:
        server.load_model()
        server.run(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"\n✗ 服务器错误：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

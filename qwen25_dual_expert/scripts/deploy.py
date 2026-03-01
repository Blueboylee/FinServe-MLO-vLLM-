#!/usr/bin/env python3
"""
vLLM 部署启动脚本
提供 API 服务和交互式命令行两种模式
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from model_loader import DualExpertModel, ModelConfig, create_model_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GenerationRequest(BaseModel):
    """生成请求模型"""
    prompt: str = Field(..., description="输入提示")
    expert: str = Field(default="A", description="选择专家：A 或 B")
    max_tokens: int = Field(default=512, ge=1, le=2048, description="最大生成长度")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p 参数")


class GenerationResponse(BaseModel):
    """生成响应模型"""
    response: str
    expert_used: str
    prompt: str
    success: bool


app = FastAPI(title="Qwen2.5 双专家模型 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_instance: Optional[DualExpertModel] = None


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global model_instance
    
    logger.info("正在加载模型...")
    config = create_model_config()
    model_instance = DualExpertModel(config)
    
    if not model_instance.load_tokenizer():
        raise RuntimeError("分词器加载失败")
    
    if not model_instance.load_base_model_vllm():
        raise RuntimeError("基础模型加载失败")
    
    if not model_instance.load_expert_adapters():
        raise RuntimeError("专家适配器加载失败")
    
    logger.info("模型加载完成，服务已就绪")


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """生成文本接口"""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        response = model_instance.generate_with_expert(
            prompt=request.prompt,
            expert=request.expert,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return GenerationResponse(
            response=response,
            expert_used=request.expert,
            prompt=request.prompt,
            success=True
        )
        
    except Exception as e:
        logger.error(f"生成失败：{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查接口"""
    if model_instance is None:
        return {"status": "unhealthy", "message": "模型未加载"}
    
    try:
        memory_ok = model_instance.verify_memory_sharing()
        return {
            "status": "healthy" if memory_ok else "degraded",
            "message": "服务运行正常" if memory_ok else "显存使用过高"
        }
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}


def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """启动 API 服务器"""
    logger.info(f"启动 API 服务器：http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


def run_interactive_mode():
    """交互式命令行模式"""
    logger.info("进入交互式模式，输入 'quit' 退出")
    
    config = create_model_config()
    model = DualExpertModel(config)
    
    if not model.load_tokenizer():
        logger.error("分词器加载失败")
        return 1
    
    if not model.load_base_model_vllm():
        logger.error("基础模型加载失败")
        return 1
    
    if not model.load_expert_adapters():
        logger.error("专家适配器加载失败")
        return 1
    
    print("\n" + "="*60)
    print("Qwen2.5 双专家模型 - 交互式模式")
    print("="*60)
    print("命令格式：[专家 A/B] 你的问题")
    print("示例：A 什么是机器学习？")
    print("输入 'quit' 退出")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("你：").strip()
            
            if user_input.lower() == 'quit':
                print("再见！")
                break
            
            if not user_input:
                continue
            
            expert = "A"
            prompt = user_input
            
            if user_input.upper().startswith("A "):
                expert = "A"
                prompt = user_input[2:]
            elif user_input.upper().startswith("B "):
                expert = "B"
                prompt = user_input[2:]
            
            logger.info(f"使用专家 {expert}，提示：{prompt}")
            response = model.generate_with_expert(prompt, expert)
            
            print(f"\n专家 {expert}: {response}\n")
            
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            logger.error(f"错误：{str(e)}")
            print(f"错误：{str(e)}\n")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5 双专家模型部署")
    parser.add_argument(
        "--mode",
        choices=["api", "interactive"],
        default="api",
        help="运行模式：api 或 interactive"
    )
    parser.add_argument("--host", default="0.0.0.0", help="API 服务器主机")
    parser.add_argument("--port", type=int, default=8000, help="API 服务器端口")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Qwen2.5 双专家模型部署")
    logger.info("="*60)
    
    if args.mode == "api":
        run_api_server(args.host, args.port)
    else:
        sys.exit(run_interactive_mode())


if __name__ == "__main__":
    main()

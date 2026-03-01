#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen2.5 LoRA API 服务
提供 RESTful API 接口调用 LoRA 模型
"""

import argparse
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from serve_lora import QwenLoRAServer, load_model_config


# 数据模型
class GenerateRequest(BaseModel):
    """生成请求"""
    prompt: str = Field(..., description="输入提示")
    lora_name: Optional[str] = Field(None, description="LoRA 适配器名称")
    max_tokens: int = Field(512, ge=1, le=2048, description="最大生成 token 数")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="温度参数")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p 参数")


class GenerateResponse(BaseModel):
    """生成响应"""
    text: str
    lora_used: Optional[str]
    prompt: str


class LoRAStatus(BaseModel):
    """LoRA 状态"""
    current_lora: Optional[str]
    available_loras: list[str]


# FastAPI 应用
app = FastAPI(
    title="Qwen2.5 LoRA API",
    description="Qwen2.5 32B + 双 LoRA 专家模型 API 服务",
    version="1.0.0"
)

# 全局服务实例
server: Optional[QwenLoRAServer] = None
current_lora: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """启动时初始化模型"""
    global server, current_lora
    
    config_path = Path("models/model_config.txt")
    if not config_path.exists():
        raise RuntimeError("模型配置文件不存在，请先运行：python download_models.py")
    
    config = load_model_config(str(config_path))
    
    lora_models = {
        'expert_a': config.get('expert_a_path', ''),
        'expert_b': config.get('expert_b_path', ''),
    }
    
    base_model_path = config.get('base_model_path', '')
    
    print(f"加载基座模型：{base_model_path}")
    print(f"加载 LoRA 模型：{list(lora_models.keys())}")
    
    server = QwenLoRAServer(
        base_model_path=base_model_path,
        lora_models=lora_models,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        tensor_parallel_size=1
    )
    
    print("✓ 服务初始化完成")


@app.get("/", tags=["健康检查"])
async def root():
    """根路径"""
    return {
        "service": "Qwen2.5 LoRA API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", tags=["健康检查"])
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


@app.get("/lora", response_model=LoRAStatus, tags=["LoRA 管理"])
async def get_lora_status():
    """获取 LoRA 状态"""
    if server is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    return LoRAStatus(
        current_lora=current_lora,
        available_loras=server.list_loras()
    )


@app.post("/lora/switch", tags=["LoRA 管理"])
async def switch_lora(lora_name: Optional[str] = None):
    """切换 LoRA 适配器"""
    global current_lora
    
    if server is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    if lora_name:
        if lora_name not in server.list_loras():
            raise HTTPException(
                status_code=400,
                detail=f"未知的 LoRA 适配器：{lora_name}"
            )
        current_lora = lora_name
        return {"message": f"已切换到 {lora_name}", "lora_name": lora_name}
    else:
        current_lora = None
        return {"message": "已切换到基座模型", "lora_name": None}


@app.post("/generate", response_model=GenerateResponse, tags=["生成"])
async def generate(request: GenerateRequest):
    """文本生成"""
    if server is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    try:
        lora_to_use = request.lora_name or current_lora
        
        response = server.generate_single(
            prompt=request.prompt,
            lora_name=lora_to_use,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return GenerateResponse(
            text=response,
            lora_used=lora_to_use,
            prompt=request.prompt
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=GenerateResponse, tags=["生成"])
async def chat(request: GenerateRequest):
    """聊天接口（别名）"""
    return await generate(request)


def main():
    parser = argparse.ArgumentParser(description='Qwen2.5 LoRA API 服务')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Qwen2.5 32B LoRA API 服务")
    print("="*60)
    print(f"服务地址：http://{args.host}:{args.port}")
    print(f"API 文档：http://{args.host}:{args.port}/docs")
    print("="*60)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1
    )


if __name__ == "__main__":
    main()

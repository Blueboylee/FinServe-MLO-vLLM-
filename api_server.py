#!/usr/bin/env python3
"""
专家服务API服务器
提供RESTful API接口进行模型推理
"""

import os
import sys
import logging
import argparse
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import config
from unsloth import FastLanguageModel


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Qwen2.5 32B Expert API",
    description="专家A和专家B的API服务",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models: Dict[str, Any] = {}
tokenizers: Dict[str, Any] = {}


class ChatMessage(BaseModel):
    role: str = Field(..., description="角色: user, assistant, system")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="对话消息列表")
    expert: str = Field(..., description="专家名称: A or B", pattern="^[AB]$")
    max_tokens: Optional[int] = Field(2048, description="最大生成token数")
    temperature: Optional[float] = Field(0.7, description="温度参数")
    top_p: Optional[float] = Field(0.9, description="top_p参数")
    top_k: Optional[int] = Field(50, description="top_k参数")
    stream: Optional[bool] = Field(False, description="是否流式输出")


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class HealthResponse(BaseModel):
    status: str
    expert: str
    model_loaded: bool


def load_expert_model(expert: str):
    """加载专家模型"""
    expert_key = f"expert_{expert.lower()}_path"
    expert_path = getattr(config.model_config, expert_key)
    
    if expert in models:
        return models[expert], tokenizers[expert]
    
    try:
        logger.info(f"加载专家{expert}模型...")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_config.base_model_path,
            max_seq_length=32768,
            dtype=None,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True
        )
        
        model = FastLanguageModel.load_lora(model, expert_path)
        model.eval()
        
        models[expert] = model
        tokenizers[expert] = tokenizer
        
        logger.info(f"专家{expert}模型加载完成")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"专家{expert}模型加载失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "Qwen2.5 32B Expert API Server",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check(expert: str = "A") -> HealthResponse:
    """健康检查"""
    model_loaded = expert in models
    return HealthResponse(
        status="healthy" if model_loaded else "not_loaded",
        expert=expert,
        model_loaded=model_loaded
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest) -> ChatResponse:
    """聊天补全接口"""
    try:
        if request.expert not in ["A", "B"]:
            raise HTTPException(status_code=400, detail="专家必须是 A 或 B")
        
        model, tokenizer = load_expert_model(request.expert)
        
        messages_text = ""
        for msg in request.messages:
            if msg.role == "user":
                messages_text += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                messages_text += f"Assistant: {msg.content}\n"
            elif msg.role == "system":
                messages_text += f"System: {msg.content}\n"
        
        messages_text += "Assistant: "
        
        FastLanguageModel.for_inference(model)
        
        inputs = tokenizer([messages_text], return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0])
        
        response_text = response.split("Assistant:")[-1].strip()
        
        input_tokens = len(inputs.input_ids[0])
        output_tokens = len(outputs[0]) - input_tokens
        
        return ChatResponse(
            id=f"chatcmpl-{id(request)}",
            object="chat.completion",
            created=int(__import__("time").time()),
            model=f"qwen25-32b-expert-{request.expert.lower()}",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"聊天补全失败: {e}")
        raise HTTPException(status_code=500, detail=f"聊天补全失败: {str(e)}")


@app.post("/generate")
async def generate(
    prompt: str,
    expert: str = "A",
    max_tokens: Optional[int] = 2048,
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = 0.9,
    top_k: Optional[int] = 50
) -> Dict[str, Any]:
    """简单生成接口"""
    try:
        if expert not in ["A", "B"]:
            raise HTTPException(status_code=400, detail="专家必须是 A 或 B")
        
        model, tokenizer = load_expert_model(expert)
        
        FastLanguageModel.for_inference(model)
        
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0])
        
        input_tokens = len(inputs.input_ids[0])
        output_tokens = len(outputs[0]) - input_tokens
        
        return {
            "prompt": prompt,
            "response": response,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@app.get("/models")
async def list_models() -> Dict[str, List[str]]:
    """列出可用模型"""
    return {
        "models": [
            "qwen25-32b-expert-a",
            "qwen25-32b-expert-b"
        ]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动API服务器")
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
    
    port = args.port or (8001 if args.expert == "A" else 8002)
    
    logger.info("=" * 60)
    logger.info("API服务器配置")
    logger.info("=" * 60)
    logger.info(f"专家: {args.expert}")
    logger.info(f"端口: {port}")
    logger.info(f"主机: {args.host}")
    logger.info("=" * 60)
    
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=port,
        workers=1,
        timeout_keep_alive=120
    )

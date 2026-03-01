#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/lixingchen/Documents/GitHub/FinServe-MLO-vLLM-/deployment/logs/expert_service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
except ImportError as e:
    logger.error(f"导入依赖失败: {e}")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

class ExpertDeployer:
    def __init__(self, expert_id: str, base_model_dir: str, expert_qlora_dir: str, 
                 tensor_parallel_size: int = 1, max_num_seqs: int = 16,
                 max_model_len: int = 4096, gpu_memory_utilization: float = 0.95):
        self.expert_id = expert_id
        self.base_model_dir = base_model_dir
        self.expert_qlora_dir = expert_qlora_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        
        self.llm: Optional[LLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        
        logger.info(f"初始化专家 {expert_id} 部署器")
        logger.info(f"基础模型目录: {base_model_dir}")
        logger.info(f"QLoRA适配器目录: {expert_qlora_dir}")
    
    def load_model(self):
        try:
            logger.info(f"专家 {self.expert_id} 开始加载模型...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_dir)
            
            self.llm = LLM(
                model=self.base_model_dir,
                tokenizer_mode='auto',
                trust_remote_code=True,
                tensor_parallel_size=self.tensor_parallel_size,
                max_num_seqs=self.max_num_seqs,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype='half',
                enforce_eager=False,
                enable_lora=True,
                max_lora_rank=64,
            )
            
            logger.info(f"专家 {self.expert_id} 模型加载成功")
            
        except Exception as e:
            logger.error(f"专家 {self.expert_id} 模型加载失败: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7):
        try:
            if self.llm is None or self.tokenizer is None:
                raise ValueError("模型未加载，请先调用 load_model()")
            
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                stop_token_ids=[151645]
            )
            
            outputs = self.llm.generate(prompt, sampling_params)
            
            if outputs:
                return outputs[0].outputs[0].text.strip()
            else:
                return "生成失败"
                
        except Exception as e:
            logger.error(f"专家 {self.expert_id} 生成失败: {e}")
            raise

app = FastAPI(title="Qwen2.5 32B Expert Service")

expert_deployers = {}

class GenerateRequest(BaseModel):
    expert_id: str
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    success: bool
    expert_id: str
    output: Optional[str] = None
    error: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    logger.info("服务启动，加载专家模型...")
    
    base_model_dir = os.path.join(MODELS_DIR, 'qwen25-32b-awq')
    
    experts_config = [
        {'id': 'A', 'qlora_dir': 'qwen25-32b-expert-a-qlora'},
        {'id': 'B', 'qlora_dir': 'qwen25-32b-expert-b-qlora'},
    ]
    
    for expert_config in experts_config:
        expert_id = expert_config['id']
        expert_qlora_dir = os.path.join(MODELS_DIR, expert_config['qlora_dir'])
        
        try:
            deployer = ExpertDeployer(
                expert_id=expert_id,
                base_model_dir=base_model_dir,
                expert_qlora_dir=expert_qlora_dir
            )
            deployer.load_model()
            expert_deployers[expert_id] = deployer
            logger.info(f"专家 {expert_id} 加载成功")
        except Exception as e:
            logger.error(f"专家 {expert_id} 加载失败: {e}")
            raise

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if request.expert_id not in expert_deployers:
        raise HTTPException(status_code=404, detail=f"专家 {request.expert_id} 不存在")
    
    try:
        deployer = expert_deployers[request.expert_id]
        output = deployer.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return GenerateResponse(success=True, expert_id=request.expert_id, output=output)
    except Exception as e:
        logger.error(f"生成请求失败: {e}")
        return GenerateResponse(success=False, expert_id=request.expert_id, error=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "experts": list(expert_deployers.keys())}

def main():
    parser = argparse.ArgumentParser(description='部署Qwen2.5 32B专家模型服务')
    parser.add_argument('--port', type=int, default=8000, help='服务端口')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务主机')
    
    args = parser.parse_args()
    
    logger.info(f"启动专家模型服务，端口: {args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == '__main__':
    main()

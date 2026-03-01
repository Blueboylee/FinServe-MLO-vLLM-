#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM-based Expert Model Server with Shared Base Model
Optimized for RTX 4080S 32GB with QLoRA adapters
"""

import os
import sys
import json
import logging
import argparse
from typing import Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ExpertServer")

class GenerationRequest(BaseModel):
    prompt: str
    expert: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

class GenerationResponse(BaseModel):
    response: str
    expert: str
    tokens_generated: int

class ExpertModelServer:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.llm = None
        self.app = FastAPI(title="Qwen Expert Server")
        self.setup_routes()
        
    def load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def initialize_vllm(self):
        """Initialize vLLM engine with shared base model"""
        logger.info("Initializing vLLM engine...")
        
        try:
            self.llm = LLM(
                model=self.config['base_model_path'],
                tensor_parallel_size=self.config['tensor_parallel_size'],
                max_model_len=self.config['max_model_len'],
                gpu_memory_utilization=self.config['gpu_memory_utilization'],
                dtype=self.config['dtype'],
                quantization=self.config['quantization'],
                enable_lora=True,
                max_loras=2,
                max_lora_rank=64,
                trust_remote_code=True
            )
            
            logger.info("vLLM engine initialized successfully")
            self._log_memory_usage()
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {str(e)}")
            raise
    
    def _log_memory_usage(self):
        """Log current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.post("/generate", response_model=GenerationResponse)
        async def generate(request: GenerationRequest):
            if self.llm is None:
                raise HTTPException(status_code=503, detail="Model not initialized")
            
            try:
                lora_path = (self.config['expert_a_path'] if request.expert == 'expert_a' 
                            else self.config['expert_b_path'])
                
                sampling_params = SamplingParams(
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    max_tokens=request.max_tokens
                )
                
                outputs = self.llm.generate(
                    [request.prompt],
                    sampling_params,
                    lora_request=LoRARequest(request.expert, 1, lora_path)
                )
                
                response = outputs[0].outputs[0].text
                tokens = len(outputs[0].outputs[0].token_ids)
                
                return GenerationResponse(
                    response=response,
                    expert=request.expert,
                    tokens_generated=tokens
                )
                
            except Exception as e:
                logger.error(f"Generation failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "experts_loaded": self.llm is not None}
        
        @self.app.get("/memory")
        async def memory_status():
            if torch.cuda.is_available():
                return {
                    "allocated_gb": torch.cuda.memory_allocated(0) / 1024**3,
                    "reserved_gb": torch.cuda.memory_reserved(0) / 1024**3,
                    "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
                }
            return {"error": "CUDA not available"}
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the server"""
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="info")

def main():
    parser = argparse.ArgumentParser(description="Expert Model Server")
    parser.add_argument("--config", type=str, default="./configs/deployment_config.json")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--download", action="store_true", help="Download models first")
    
    args = parser.parse_args()
    
    if args.download:
        from deploy_experts import ExpertModelConfig, SharedBaseModelManager
        config = ExpertModelConfig(args.config)
        manager = SharedBaseModelManager(config)
        manager.download_models_from_modelscope()
    
    server = ExpertModelServer(args.config)
    server.initialize_vllm()
    server.run(args.host, args.port)

if __name__ == "__main__":
    main()

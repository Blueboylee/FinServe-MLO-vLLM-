#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-32B QLoRA Expert Models Deployment Script
Implements shared base model mechanism for memory optimization
Compatible with Python 3.10 and vLLM
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Configure logging
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging configuration"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"deployment_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("QwenExpertDeployment")
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

logger = setup_logging()

class ExpertModelConfig:
    """Configuration for expert models"""
    def __init__(self, config_path: Optional[str] = None):
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        
        self.base_model_id = config.get('base_model_id', 'Qwen/Qwen2.5-32B-Instruct-AWQ')
        self.expert_a_id = config.get('expert_a_id', 'GaryLeenene/qwen25-32b-expert-a-qlora')
        self.expert_b_id = config.get('expert_b_id', 'GaryLeenene/qwen25-32b-expert-b-qlora')
        
        self.base_model_path = config.get('base_model_path', './models/base_model')
        self.expert_a_path = config.get('expert_a_path', './models/expert_a')
        self.expert_b_path = config.get('expert_b_path', './models/expert_b')
        
        self.tensor_parallel_size = config.get('tensor_parallel_size', 1)
        self.max_model_len = config.get('max_model_len', 4096)
        self.gpu_memory_utilization = config.get('gpu_memory_utilization', 0.85)
        
        logger.info("Expert model configuration loaded")

class SharedBaseModelManager:
    """Manages shared base model for multiple LoRA adapters"""
    
    def __init__(self, config: ExpertModelConfig):
        self.config = config
        self.base_model = None
        self.base_tokenizer = None
        self.lora_models: Dict[str, PeftModel] = {}
        
    def download_models_from_modelscope(self):
        """Download all models from ModelScope"""
        logger.info("Downloading models from ModelScope...")
        
        try:
            from modelscope import snapshot_download
            
            logger.info(f"Downloading base model: {self.config.base_model_id}")
            base_path = snapshot_download(
                model_id=self.config.base_model_id,
                local_dir=self.config.base_model_path,
                revision='master'
            )
            logger.info(f"Base model downloaded to: {base_path}")
            
            logger.info(f"Downloading Expert A: {self.config.expert_a_id}")
            expert_a_path = snapshot_download(
                model_id=self.config.expert_a_id,
                local_dir=self.config.expert_a_path,
                revision='master'
            )
            logger.info(f"Expert A downloaded to: {expert_a_path}")
            
            logger.info(f"Downloading Expert B: {self.config.expert_b_id}")
            expert_b_path = snapshot_download(
                model_id=self.config.expert_b_id,
                local_dir=self.config.expert_b_path,
                revision='master'
            )
            logger.info(f"Expert B downloaded to: {expert_b_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download models: {str(e)}")
            raise
    
    def load_base_model(self):
        """Load the shared 4bit GPTQ base model"""
        logger.info("Loading shared 4bit GPTQ base model...")
        
        try:
            from auto_gptq import AutoGPTQForCausalLM
            
            self.base_tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_path,
                trust_remote_code=True,
                padding_side='left'
            )
            
            if self.base_tokenizer.pad_token is None:
                self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            
            self.base_model = AutoGPTQForCausalLM.from_quantized(
                self.config.base_model_path,
                device="cuda:0",
                use_triton=True,
                inject_fused_attention=True,
                inject_fused_mlp=True,
                disable_exllama_v2=True,
                use_safetensors=True
            )
            
            logger.info("Base model loaded successfully")
            logger.info(f"Base model device map: {self.base_model.hf_device_map}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load base model: {str(e)}")
            raise
    
    def load_lora_adapter(self, expert_name: str, expert_path: str) -> PeftModel:
        """Load a LoRA adapter on top of the shared base model"""
        logger.info(f"Loading LoRA adapter for {expert_name}...")
        
        try:
            peft_config = PeftConfig.from_pretrained(expert_path)
            logger.info(f"LoRA config for {expert_name}: {peft_config}")
            
            lora_model = PeftModel.from_pretrained(
                self.base_model,
                expert_path,
                device_map="cuda:0",
                torch_dtype=torch.float16,
                is_trainable=False
            )
            
            lora_model.eval()
            self.lora_models[expert_name] = lora_model
            
            logger.info(f"LoRA adapter {expert_name} loaded successfully")
            return lora_model
            
        except Exception as e:
            logger.error(f"Failed to load LoRA adapter {expert_name}: {str(e)}")
            raise
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        return {}
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        
        for name in list(self.lora_models.keys()):
            del self.lora_models[name]
        
        if self.base_model is not None:
            del self.base_model
        
        if self.base_tokenizer is not None:
            del self.base_tokenizer
        
        torch.cuda.empty_cache()
        logger.info("Resources cleaned up")
        logger.info(f"Final memory state: {self.get_memory_usage()}")

class ExpertInferenceEngine:
    """Inference engine for expert models"""
    
    def __init__(self, model_manager: SharedBaseModelManager):
        self.model_manager = model_manager
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=1024,
            repetition_penalty=1.1,
            stop_token_ids=[151643, 151644, 151645]
        )
    
    def generate(self, expert_name: str, prompt: str, max_tokens: int = 512) -> str:
        """Generate response using specified expert"""
        if expert_name not in self.model_manager.lora_models:
            raise ValueError(f"Expert {expert_name} not loaded")
        
        try:
            lora_model = self.model_manager.lora_models[expert_name]
            tokenizer = self.model_manager.base_tokenizer
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = lora_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=self.sampling_params.temperature,
                    top_p=self.sampling_params.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
            
        except Exception as e:
            logger.error(f"Generation failed for {expert_name}: {str(e)}")
            raise
    
    def batch_generate(self, expert_name: str, prompts: List[str], max_tokens: int = 512) -> List[str]:
        """Batch generation using specified expert"""
        responses = []
        for prompt in prompts:
            try:
                response = self.generate(expert_name, prompt, max_tokens)
                responses.append(response)
            except Exception as e:
                logger.error(f"Failed to generate for prompt: {prompt[:50]}... Error: {str(e)}")
                responses.append(f"Error: {str(e)}")
        return responses

def validate_deployment(model_manager: SharedBaseModelManager, inference_engine: ExpertInferenceEngine) -> bool:
    """Validate the deployment with test queries"""
    logger.info("Validating deployment...")
    
    test_prompts = {
        'expert_a': "请介绍一下人工智能的基本概念。",
        'expert_b': "请解释一下量子计算的原理。"
    }
    
    try:
        for expert, prompt in test_prompts.items():
            if expert in model_manager.lora_models:
                response = inference_engine.generate(expert, prompt, max_tokens=100)
                logger.info(f"{expert} test response (first 100 chars): {response[:100]}...")
        
        logger.info("Deployment validation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Deployment validation failed: {str(e)}")
        return False

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Qwen2.5-32B QLoRA Expert Deployment")
    parser.add_argument("--config", type=str, default="./configs/deployment_config.json",
                       help="Path to configuration file")
    parser.add_argument("--download", action="store_true",
                       help="Download models from ModelScope")
    parser.add_argument("--experts", nargs="+", default=["expert_a", "expert_b"],
                       help="Experts to load")
    parser.add_argument("--validate", action="store_true",
                       help="Run validation tests after deployment")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Qwen2.5-32B QLoRA Expert Models Deployment")
    logger.info("=" * 60)
    
    try:
        config = ExpertModelConfig(args.config)
        model_manager = SharedBaseModelManager(config)
        
        if args.download:
            logger.info("Step 1: Downloading models...")
            model_manager.download_models_from_modelscope()
        
        logger.info("Step 2: Loading shared base model...")
        model_manager.load_base_model()
        
        logger.info("Step 3: Loading LoRA adapters...")
        for expert_name in args.experts:
            expert_path = config.expert_a_path if expert_name == "expert_a" else config.expert_b_path
            model_manager.load_lora_adapter(expert_name, expert_path)
        
        logger.info("Step 4: Memory optimization status...")
        memory_info = model_manager.get_memory_usage()
        logger.info(f"GPU Memory - Allocated: {memory_info.get('allocated_gb', 0):.2f}GB, "
                   f"Reserved: {memory_info.get('reserved_gb', 0):.2f}GB, "
                   f"Total: {memory_info.get('total_gb', 0):.2f}GB")
        
        inference_engine = ExpertInferenceEngine(model_manager)
        
        if args.validate:
            logger.info("Step 5: Running validation tests...")
            validate_deployment(model_manager, inference_engine)
        
        logger.info("=" * 60)
        logger.info("Deployment completed successfully!")
        logger.info("Ready for inference")
        logger.info("=" * 60)
        
        return model_manager, inference_engine
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

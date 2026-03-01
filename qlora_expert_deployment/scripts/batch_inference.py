#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Inference Script for Expert Models
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import csv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BatchInference")

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_input_file(input_path: str) -> List[Dict]:
    """Load input prompts from JSON or CSV file"""
    path = Path(input_path)
    
    if path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'prompts' in data:
                return data['prompts']
    
    elif path.suffix == '.csv':
        prompts = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append(row)
        return prompts
    
    elif path.suffix == '.txt':
        with open(path, 'r', encoding='utf-8') as f:
            return [{'prompt': line.strip()} for line in f if line.strip()]
    
    raise ValueError(f"Unsupported file format: {path.suffix}")

def save_results(results: List[Dict], output_path: str):
    """Save inference results to file"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == '.json':
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    elif path.suffix == '.csv':
        if results:
            with open(path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
    
    logger.info(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch Inference for Expert Models")
    parser.add_argument("--config", type=str, default="./configs/deployment_config.json")
    parser.add_argument("--input", type=str, required=True, help="Input file with prompts")
    parser.add_argument("--output", type=str, default="./outputs/batch_results.json")
    parser.add_argument("--expert", type=str, default="expert_a", 
                       choices=["expert_a", "expert_b"], help="Expert to use")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    
    args = parser.parse_args()
    
    logger.info("Loading configuration...")
    config = load_config(args.config)
    
    logger.info("Loading input prompts...")
    prompts = load_input_file(args.input)
    logger.info(f"Loaded {len(prompts)} prompts")
    
    logger.info("Initializing models...")
    from deploy_experts import ExpertModelConfig, SharedBaseModelManager, ExpertInferenceEngine
    
    expert_config = ExpertModelConfig(args.config)
    manager = SharedBaseModelManager(expert_config)
    manager.load_base_model()
    manager.load_lora_adapter(args.expert, config[f'{args.expert}_path'])
    
    engine = ExpertInferenceEngine(manager)
    
    logger.info(f"Starting batch inference with {args.expert}...")
    results = []
    
    for i, item in enumerate(prompts):
        prompt = item.get('prompt', '')
        metadata = {k: v for k, v in item.items() if k != 'prompt'}
        
        try:
            response = engine.generate(args.expert, prompt, max_tokens=args.max_tokens)
            
            result = {
                'id': i,
                'prompt': prompt,
                'response': response,
                'expert': args.expert,
                'timestamp': datetime.now().isoformat(),
                **metadata
            }
            results.append(result)
            
            logger.info(f"Processed {i+1}/{len(prompts)}")
            
        except Exception as e:
            logger.error(f"Failed to process prompt {i}: {str(e)}")
            results.append({
                'id': i,
                'prompt': prompt,
                'error': str(e),
                'expert': args.expert
            })
    
    save_results(results, args.output)
    
    logger.info("Batch inference completed")
    logger.info(f"Total: {len(results)}, Success: {sum(1 for r in results if 'response' in r)}")
    
    manager.cleanup()

if __name__ == "__main__":
    main()

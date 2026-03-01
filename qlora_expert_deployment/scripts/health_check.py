#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Health Check and Validation Script for Expert Models
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HealthCheck")

class HealthChecker:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
    
    def check_python_version(self) -> bool:
        """Check Python version"""
        logger.info("Checking Python version...")
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        expected = "3.10"
        
        if version == expected:
            logger.info(f"✓ Python version: {version}")
            self.checks_passed += 1
            return True
        else:
            logger.warning(f"⚠ Python version: {version} (expected {expected})")
            self.warnings.append(f"Python version mismatch: {version} vs {expected}")
            self.checks_passed += 1
            return True
    
    def check_cuda_availability(self) -> bool:
        """Check CUDA availability"""
        logger.info("Checking CUDA...")
        
        if not torch.cuda.is_available():
            logger.error("✗ CUDA not available")
            self.checks_failed += 1
            return False
        
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        logger.info(f"✓ CUDA version: {cuda_version}")
        logger.info(f"✓ GPU count: {device_count}")
        logger.info(f"✓ GPU model: {device_name}")
        logger.info(f"✓ GPU memory: {device_memory:.2f}GB")
        
        if device_memory < 30:
            self.warnings.append(f"GPU memory {device_memory:.2f}GB may be insufficient")
        
        self.checks_passed += 1
        return True
    
    def check_model_files(self) -> bool:
        """Check if model files exist"""
        logger.info("Checking model files...")
        
        required_paths = [
            self.config['base_model_path'],
            self.config['expert_a_path'],
            self.config['expert_b_path']
        ]
        
        all_exist = True
        for path in required_paths:
            if Path(path).exists():
                logger.info(f"✓ Found: {path}")
                files = list(Path(path).glob("*"))
                logger.info(f"  Files: {len(files)} items")
            else:
                logger.error(f"✗ Missing: {path}")
                all_exist = False
                self.checks_failed += 1
        
        if all_exist:
            self.checks_passed += 1
        
        return all_exist
    
    def check_dependencies(self) -> bool:
        """Check required Python packages"""
        logger.info("Checking dependencies...")
        
        required_packages = {
            'torch': 'PyTorch',
            'vllm': 'vLLM',
            'transformers': 'Transformers',
            'peft': 'PEFT',
            'modelscope': 'ModelScope',
            'auto_gptq': 'AutoGPTQ'
        }
        
        all_installed = True
        for package, name in required_packages.items():
            try:
                mod = __import__(package)
                version = getattr(mod, '__version__', 'unknown')
                logger.info(f"✓ {name}: {version}")
            except ImportError:
                logger.error(f"✗ {name}: Not installed")
                all_installed = False
                self.checks_failed += 1
        
        if all_installed:
            self.checks_passed += 1
        
        return all_installed
    
    def check_gpu_memory(self) -> bool:
        """Check GPU memory availability"""
        logger.info("Checking GPU memory...")
        
        if not torch.cuda.is_available():
            return False
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        free_memory = total_memory - allocated
        
        logger.info(f"✓ Total: {total_memory:.2f}GB")
        logger.info(f"✓ Allocated: {allocated:.2f}GB")
        logger.info(f"✓ Free: {free_memory:.2f}GB")
        
        if free_memory < 20:
            logger.warning("⚠ Less than 20GB free memory")
            self.warnings.append(f"Low free memory: {free_memory:.2f}GB")
        
        self.checks_passed += 1
        return True
    
    def run_all_checks(self) -> bool:
        """Run all health checks"""
        logger.info("=" * 60)
        logger.info("Starting Health Checks")
        logger.info("=" * 60)
        
        checks = [
            self.check_python_version,
            self.check_cuda_availability,
            self.check_dependencies,
            self.check_model_files,
            self.check_gpu_memory
        ]
        
        for check in checks:
            try:
                check()
            except Exception as e:
                logger.error(f"Check failed with error: {str(e)}")
                self.checks_failed += 1
        
        logger.info("=" * 60)
        logger.info(f"Checks Passed: {self.checks_passed}")
        logger.info(f"Checks Failed: {self.checks_failed}")
        
        if self.warnings:
            logger.warning("Warnings:")
            for w in self.warnings:
                logger.warning(f"  - {w}")
        
        if self.checks_failed == 0:
            logger.info("✓ All checks passed!")
            return True
        else:
            logger.error("✗ Some checks failed")
            return False

def main():
    parser = argparse.ArgumentParser(description="Health Check for Expert Models")
    parser.add_argument("--config", type=str, default="./configs/deployment_config.json")
    parser.add_argument("--output", type=str, help="Output results to JSON file")
    
    args = parser.parse_args()
    
    checker = HealthChecker(args.config)
    success = checker.run_all_checks()
    
    if args.output:
        results = {
            "checks_passed": checker.checks_passed,
            "checks_failed": checker.checks_failed,
            "warnings": checker.warnings,
            "success": success
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

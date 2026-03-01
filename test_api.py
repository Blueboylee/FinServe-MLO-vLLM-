#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API 测试脚本
测试双专家模型的 API 接口
"""

import requests
import json
import time
from typing import List


class APITester:
    """API 测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def check_health(self) -> bool:
        """检查服务器健康状态"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"✗ 健康检查失败：{e}")
            return False
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"✗ 获取模型信息失败：{e}")
            return {}
    
    def list_models(self) -> List[dict]:
        """列出可用模型"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            print(f"✗ 列出模型失败：{e}")
            return []
    
    def generate(self, prompt: str, expert: str = "expert_a", 
                 max_tokens: int = 256, **kwargs) -> dict:
        """生成文本"""
        payload = {
            "prompt": prompt,
            "expert": expert,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"✗ 请求超时")
            return {"error": "timeout"}
        except Exception as e:
            print(f"✗ 生成失败：{e}")
            return {"error": str(e)}
    
    def test_all_endpoints(self):
        """测试所有端点"""
        print("="*60)
        print("API 端点测试")
        print("="*60)
        
        # 1. 健康检查
        print("\n[1/4] 健康检查...")
        if self.check_health():
            print("✓ 服务器运行正常")
        else:
            print("✗ 服务器未响应，请先启动 api_server.py")
            return False
        
        # 2. 获取模型信息
        print("\n[2/4] 获取模型信息...")
        model_info = self.get_model_info()
        if model_info:
            print(f"✓ 模型：{model_info.get('model', 'Unknown')}")
            print(f"  可用专家：{model_info.get('experts', [])}")
        else:
            print("✗ 无法获取模型信息")
        
        # 3. 列出模型
        print("\n[3/4] 列出可用模型...")
        models = self.list_models()
        if models:
            for model in models:
                print(f"  - {model['id']}: {model['name']}")
        else:
            print("  无可用模型")
        
        # 4. 测试生成
        print("\n[4/4] 测试文本生成...")
        test_prompts = [
            ("你好，请介绍一下你自己。", "expert_a"),
            ("什么是人工智能？", "expert_b"),
        ]
        
        for prompt, expert in test_prompts:
            print(f"\n  测试 {expert}:")
            print(f"  输入：{prompt}")
            
            start_time = time.time()
            result = self.generate(prompt, expert=expert, max_tokens=200)
            elapsed = time.time() - start_time
            
            if "error" not in result:
                print(f"  输出：{result['text']}")
                print(f"  耗时：{elapsed:.2f}秒")
                print(f"  Token 统计：{result.get('usage', {})}")
            else:
                print(f"  ✗ 错误：{result['error']}")
        
        return True


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="API 测试脚本")
    parser.add_argument("--url", type=str, default="http://localhost:8000",
                       help="API 服务器地址")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Qwen2.5-32B 双专家模型 API 测试")
    print("="*60)
    print(f"服务器地址：{args.url}")
    
    tester = APITester(args.url)
    tester.test_all_endpoints()
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()

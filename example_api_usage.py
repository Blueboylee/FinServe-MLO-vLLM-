#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API 调用示例
展示如何使用 HTTP API 调用 Qwen2.5 LoRA 服务
"""

import requests

# 服务地址
BASE_URL = "http://localhost:8000"


def health_check():
    """健康检查"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"健康检查：{response.json()}")


def get_lora_status():
    """获取 LoRA 状态"""
    response = requests.get(f"{BASE_URL}/lora")
    print(f"LoRA 状态：{response.json()}")
    return response.json()


def switch_lora(lora_name: str = None):
    """切换 LoRA 适配器"""
    params = {"lora_name": lora_name} if lora_name else {}
    response = requests.post(f"{BASE_URL}/lora/switch", params=params)
    print(f"切换 LoRA: {response.json()}")


def generate_text(prompt: str, lora_name: str = None, **kwargs):
    """生成文本"""
    payload = {
        "prompt": prompt,
        "lora_name": lora_name,
        "max_tokens": kwargs.get("max_tokens", 512),
        "temperature": kwargs.get("temperature", 0.7),
        "top_p": kwargs.get("top_p", 0.9)
    }
    
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    result = response.json()
    
    print(f"\n提示：{result['prompt']}")
    print(f"使用 LoRA: {result['lora_used']}")
    print(f"回复：{result['text']}")
    
    return result


def main():
    print("="*60)
    print("Qwen2.5 LoRA API 调用示例")
    print("="*60)
    
    # 1. 健康检查
    print("\n1. 健康检查")
    try:
        health_check()
    except requests.exceptions.ConnectionError:
        print("错误：无法连接到服务，请先启动 API 服务")
        print("运行：python api_server.py")
        return
    
    # 2. 查看 LoRA 状态
    print("\n2. 查看 LoRA 状态")
    status = get_lora_status()
    
    # 3. 使用基座模型生成
    print("\n3. 使用基座模型生成")
    generate_text("你好，请介绍一下你自己")
    
    # 4. 切换到专家 A
    print("\n4. 切换到专家 A")
    switch_lora("expert_a")
    generate_text("请分析这个财务报表的关键指标")
    
    # 5. 切换到专家 B
    print("\n5. 切换到专家 B")
    switch_lora("expert_b")
    generate_text("这段代码有什么优化建议")
    
    # 6. 切换回基座模型
    print("\n6. 切换回基座模型")
    switch_lora(None)
    
    print("\n" + "="*60)
    print("示例完成！")
    print("="*60)


if __name__ == "__main__":
    main()

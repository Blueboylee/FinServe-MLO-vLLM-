#!/usr/bin/env python3
"""
命令行交互脚本
提供简单的命令行接口与专家模型进行对话
"""

import os
import sys
import requests
import argparse
import json


class ExpertClient:
    """专家模型客户端"""
    
    def __init__(self, expert: str = "A", base_url: str = "http://localhost:8001"):
        self.expert = expert
        self.base_url = base_url
        self.port = 8001 if expert == "A" else 8002
        self.url = f"{self.base_url}:{self.port}"
    
    def check_health(self) -> bool:
        """检查服务健康状态"""
        try:
            response = requests.get(f"{self.url}/health?expert={self.expert}", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def chat(self, message: str, max_tokens: int = 2048, 
             temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        与专家对话
        
        Args:
            message: 用户消息
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top_p参数
            
        Returns:
            模型回复
        """
        payload = {
            "messages": [
                {"role": "user", "content": message}
            ],
            "expert": self.expert,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        try:
            response = requests.post(
                f"{self.url}/v1/chat/completions",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"错误: {e}"
    
    def generate(self, prompt: str, max_tokens: int = 2048,
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        生成文本
        
        Args:
            prompt: 提示文本
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top_p参数
            
        Returns:
            生成的文本
        """
        payload = {
            "prompt": prompt,
            "expert": self.expert,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        try:
            response = requests.post(
                f"{self.url}/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result["response"]
        except requests.exceptions.RequestException as e:
            return f"错误: {e}"
    
    def interactive_chat(self):
        """交互式聊天模式"""
        print(f"\n{'=' * 60}")
        print(f"专家{self.expert} 交互式聊天")
        print(f"{'=' * 60}")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'clear' 清空对话")
        print(f"{'=' * 60}\n")
        
        conversation = []
        
        while True:
            user_input = input("你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit"]:
                print("退出聊天")
                break
            
            if user_input.lower() == "clear":
                conversation = []
                print("对话已清空")
                continue
            
            if len(conversation) > 10:
                conversation = conversation[-10:]
            
            messages = []
            for msg in conversation:
                messages.append({"role": "user", "content": msg["user"]})
                if "assistant" in msg:
                    messages.append({"role": "assistant", "content": msg["assistant"]})
            
            messages.append({"role": "user", "content": user_input})
            
            payload = {
                "messages": messages,
                "expert": self.expert,
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            try:
                response = requests.post(
                    f"{self.url}/v1/chat/completions",
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()
                assistant_reply = result["choices"][0]["message"]["content"]
                
                print(f"专家{self.expert}: {assistant_reply}")
                
                conversation.append({
                    "user": user_input,
                    "assistant": assistant_reply
                })
                
            except requests.exceptions.RequestException as e:
                print(f"错误: {e}")
                print("请检查服务是否运行")


def main():
    parser = argparse.ArgumentParser(description="专家模型命令行客户端")
    parser.add_argument(
        "--expert",
        type=str,
        default="A",
        choices=["A", "B"],
        help="专家名称 (A or B)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost",
        help="API服务地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="服务端口 (默认: A=8001, B=8002)"
    )
    parser.add_argument(
        "--message",
        type=str,
        help="单次消息，不指定则进入交互模式"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="交互式模式"
    )
    
    args = parser.parse_args()
    
    port = args.port or (8001 if args.expert == "A" else 8002)
    client = ExpertClient(
        expert=args.expert,
        base_url=args.url
    )
    
    if not client.check_health():
        print(f"错误: 专家{args.expert}服务未运行")
        print(f"请先启动服务: python api_server.py --expert {args.expert} --port {port}")
        sys.exit(1)
    
    print(f"已连接到专家{args.expert}服务")
    
    if args.message:
        response = client.chat(args.message)
        print(f"专家{args.expert}: {response}")
    elif args.interactive:
        client.interactive_chat()
    else:
        print("请输入消息或使用 -i 进入交互模式")
        print("使用 --help 查看帮助")


if __name__ == "__main__":
    main()

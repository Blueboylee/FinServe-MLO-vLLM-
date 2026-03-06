"""
测试客户端：调用 vLLM OpenAI 兼容 API

支持两种模式：
  - 方案一（多 LoRA）: model 填 "finance-expert-a" / "finance-expert-b"
  - 方案二（合并模型）: 直接连不同端口
"""

import argparse
import base64
import json
from pathlib import Path

import requests


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def chat_text(base_url: str, model: str, prompt: str, system: str = ""):
    """纯文本对话"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def chat_with_image(base_url: str, model: str, prompt: str, image_path: str):
    """图文多模态对话"""
    b64 = encode_image(image_path)
    ext = Path(image_path).suffix.lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext, "jpeg")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{mime};base64,{b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def list_models(base_url: str):
    """列出可用模型"""
    resp = requests.get(f"{base_url}/v1/models")
    resp.raise_for_status()
    models = resp.json()["data"]
    print("可用模型:")
    for m in models:
        print(f"  - {m['id']}")
    return models


def main():
    parser = argparse.ArgumentParser(description="vLLM 多专家测试客户端")
    parser.add_argument("--host", default="localhost", help="服务地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--model", default="finance-expert-a",
                        help="模型名 (多 LoRA 模式下用 finance-expert-a/b)")
    parser.add_argument("--prompt", default="请分析2024年A股市场的整体走势和主要驱动因素。",
                        help="提问内容")
    parser.add_argument("--image", default=None, help="图片路径 (可选，用于多模态测试)")
    parser.add_argument("--compare", action="store_true",
                        help="同时对比两个专家的回答 (仅多 LoRA 模式)")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print(f"连接到: {base_url}")
    print("-" * 60)

    try:
        list_models(base_url)
    except requests.ConnectionError:
        print(f"[错误] 无法连接到 {base_url}，请确认 vLLM 服务已启动。")
        return

    print("-" * 60)

    if args.compare:
        for expert in ["finance-expert-a", "finance-expert-b"]:
            print(f"\n{'=' * 60}")
            print(f"  {expert} 的回答")
            print(f"{'=' * 60}")
            if args.image:
                result = chat_with_image(base_url, expert, args.prompt, args.image)
            else:
                result = chat_text(base_url, expert, args.prompt)
            print(result)
    else:
        print(f"\n模型: {args.model}")
        print(f"提问: {args.prompt}")
        print("-" * 60)
        if args.image:
            result = chat_with_image(base_url, args.model, args.prompt, args.image)
        else:
            result = chat_text(base_url, args.model, args.prompt)
        print(result)


if __name__ == "__main__":
    main()

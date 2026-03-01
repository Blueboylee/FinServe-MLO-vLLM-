#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 vLLM 多专家服务是否正常。
用法: python test_serve.py [--url http://localhost:8000]
"""
import argparse
import sys

try:
    import requests
except ImportError:
    print("请先安装: pip install requests")
    sys.exit(1)


def test(url: str, model: str, prompt: str = "你好", max_tokens: int = 32):
    print(f"测试 model={model} ...")
    r = requests.post(
        f"{url}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens},
        timeout=60,
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["text"]
    print(f"  回复: {text[:100]}...")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000", help="服务地址")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    print(f"服务地址: {url}\n")
    for model in ["expert-a", "expert-b"]:
        try:
            test(url, model)
        except Exception as e:
            print(f"  失败: {e}")
            sys.exit(1)
    print("\n全部测试通过。")


if __name__ == "__main__":
    main()

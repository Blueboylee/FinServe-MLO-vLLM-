#!/bin/bash
# 快速安装脚本

set -e

echo "=========================================="
echo "Qwen2.5-32B 双专家模型 安装脚本"
echo "=========================================="

# 检查 Python 版本
echo -e "\n[1/4] 检查 Python 版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 版本：$python_version"

# 创建虚拟环境（可选）
read -p "是否创建虚拟环境？(y/n, 默认 n): " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ 虚拟环境已创建并激活"
fi

# 安装依赖
echo -e "\n[2/4] 安装 Python 依赖..."
pip install -r requirements.txt -q
echo "✓ 依赖安装完成"

# 下载模型
echo -e "\n[3/4] 下载模型..."
python download_models.py

# 验证安装
echo -e "\n[4/4] 验证安装..."
python -c "import vllm; import modelscope; print('✓ 所有依赖正常')"

echo -e "\n=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  1. 启动 API 服务器：python api_server.py"
echo "  2. 测试 API: python test_api.py"
echo "  3. 简单推理：python simple_inference.py --expert expert_a"
echo ""

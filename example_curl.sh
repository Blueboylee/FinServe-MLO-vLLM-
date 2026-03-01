#!/bin/bash
# API 调用示例 (curl)
# 用于快速测试 Qwen2.5 LoRA API 服务

BASE_URL="http://localhost:8000"

echo "=========================================="
echo "Qwen2.5 LoRA API 调用示例"
echo "=========================================="

# 1. 健康检查
echo -e "\n1. 健康检查"
curl -s $BASE_URL/health | jq .

# 2. 查看 LoRA 状态
echo -e "\n2. 查看 LoRA 状态"
curl -s $BASE_URL/lora | jq .

# 3. 使用基座模型生成
echo -e "\n3. 使用基座模型生成"
curl -s -X POST $BASE_URL/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "你好，请介绍一下你自己",
    "max_tokens": 256,
    "temperature": 0.7
  }' | jq .

# 4. 切换到专家 A
echo -e "\n4. 切换到专家 A"
curl -s -X POST $BASE_URL/lora/switch?lora_name=expert_a | jq .

# 5. 使用专家 A 生成
echo -e "\n5. 使用专家 A 生成"
curl -s -X POST $BASE_URL/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "请分析这个财务报表的关键指标",
    "max_tokens": 256,
    "temperature": 0.7
  }' | jq .

# 6. 切换到专家 B
echo -e "\n6. 切换到专家 B"
curl -s -X POST $BASE_URL/lora/switch?lora_name=expert_b | jq .

# 7. 使用专家 B 生成
echo -e "\n7. 使用专家 B 生成"
curl -s -X POST $BASE_URL/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "这段代码有什么优化建议",
    "max_tokens": 256,
    "temperature": 0.7
  }' | jq .

# 8. 切换回基座模型
echo -e "\n8. 切换回基座模型"
curl -s -X POST $BASE_URL/lora/switch | jq .

echo -e "\n=========================================="
echo "示例完成！"
echo "=========================================="

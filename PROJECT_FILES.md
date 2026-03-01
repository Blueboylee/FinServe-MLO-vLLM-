# 项目文件清单

## 核心脚本

### 1. download_models.py
**功能**: 从 ModelScope 下载专家模型
- 自动检查并安装 modelscope
- 下载两个专家模型到 `models/` 目录
- 生成模型路径配置文件

**使用**:
```bash
python download_models.py
```

### 2. api_server.py
**功能**: RESTful API 服务器
- 支持双专家模型切换
- 提供完整的 API 文档（Swagger UI）
- 支持多种采样参数

**使用**:
```bash
python api_server.py --port 8000 --gpu-memory-utilization 0.85
```

**API 端点**:
- `GET /` - 获取模型信息
- `GET /health` - 健康检查
- `GET /models` - 列出可用模型
- `POST /generate` - 文本生成

### 3. deploy_experts.py
**功能**: 本地部署测试脚本
- 加载基础模型和 LoRA 适配器
- 测试双专家模型生成
- 适合本地快速测试

**使用**:
```bash
python deploy_experts.py
```

### 4. simple_inference.py
**功能**: 简化的推理脚本
- 单专家模型加载
- 交互式对话模式
- 不依赖 LoRA 功能

**使用**:
```bash
python simple_inference.py --expert expert_a
```

### 5. test_api.py
**功能**: API 测试脚本
- 测试所有 API 端点
- 验证双专家模型功能
- 输出性能统计

**使用**:
```bash
python test_api.py
```

## 自动化脚本

### 6. install.sh
**功能**: 一键安装脚本
- 检查 Python 版本
- 可选创建虚拟环境
- 安装依赖
- 下载模型
- 验证安装

**使用**:
```bash
bash install.sh
```

### 7. start.sh
**功能**: 快速启动脚本
- 检查模型是否已下载
- 支持自定义参数
- 启动 API 服务器

**使用**:
```bash
bash start.sh --port 8000
```

## 配置文件

### 8. requirements.txt
**功能**: Python 依赖包列表
包含：
- torch
- transformers
- vllm
- modelscope
- fastapi
- uvicorn
- pydantic
- requests

### 9. models/model_paths.txt
**功能**: 模型路径配置（自动生成）
格式：
```
expert_a=/path/to/expert-a
expert_b=/path/to/expert-b
```

## 文档

### 10. README.md
**功能**: 项目说明文档
- 环境要求
- 快速开始指南
- API 使用示例
- 故障排除

### 11. USAGE.md
**功能**: 详细使用指南
- 多种部署方式
- Python 代码示例
- 高级配置
- 性能优化建议
- 常见问题

## 目录结构

```
FinServe-MLO-vLLM-/
├── download_models.py      # 模型下载脚本
├── api_server.py           # API 服务器
├── deploy_experts.py       # 本地部署测试
├── simple_inference.py     # 简化推理脚本
├── test_api.py            # API 测试
├── install.sh             # 安装脚本
├── start.sh               # 启动脚本
├── requirements.txt       # 依赖配置
├── README.md             # 项目说明
├── USAGE.md              # 使用指南
├── models/               # 模型目录（下载后生成）
│   ├── qwen25-32b-expert-a-qlora/
│   ├── qwen25-32b-expert-b-qlora/
│   └── model_paths.txt   # 路径配置
└── venv/                 # 虚拟环境（可选）
```

## 使用流程

### 新用户推荐流程

1. **安装**
   ```bash
   bash install.sh
   ```

2. **启动服务**
   ```bash
   bash start.sh --port 8000
   ```

3. **测试**
   ```bash
   python test_api.py
   ```

4. **调用 API**
   ```python
   import requests
   response = requests.post(
       "http://localhost:8000/generate",
       json={"prompt": "你好", "expert": "expert_a"}
   )
   ```

### 高级用户流程

1. **手动安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **下载模型**
   ```bash
   python download_models.py
   ```

3. **自定义启动**
   ```bash
   python api_server.py \
     --port 8000 \
     --gpu-memory-utilization 0.85 \
     --max-model-len 4096
   ```

4. **开发集成**
   - 参考 USAGE.md 中的代码示例
   - 根据需求修改 api_server.py

## 技术栈

- **推理引擎**: vLLM 0.6.x+
- **模型格式**: GPTQ 4bit 量化
- **API 框架**: FastAPI + Uvicorn
- **模型来源**: ModelScope
- **GPU 支持**: NVIDIA V100 (16GB)
- **Python 版本**: 3.10+

## 注意事项

1. **模型下载**: 确保网络畅通，或使用离线下载
2. **显存管理**: V100 16GB 建议 gpu-memory-utilization=0.85
3. **LoRA 支持**: 如需 LoRA 功能，需要配置 enable_lora=True
4. **并发处理**: vLLM 支持连续批处理，适合高并发场景
5. **模型更新**: 重新下载会覆盖旧版本

## 下一步

- [ ] 添加 LoRA 适配器动态加载
- [ ] 实现流式输出支持
- [ ] 添加性能监控
- [ ] 支持更多量化格式
- [ ] 多卡并行推理

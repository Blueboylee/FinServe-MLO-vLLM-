# 快速部署指南

## 一句话总结

本方案帮你在一台 Ubuntu 22.04 服务器 (RTX 4080S 32GB) 上部署 Qwen2.5 32B 基座模型 + 两个 LoRA 专家模型，所有模型从 ModelScope 下载，使用 vLLM 推理引擎。

## 文件清单

```
FinServe-MLO-vLLM/
├── download_models.py      # 模型下载脚本（从 ModelScope）
├── serve_lora.py          # 交互式 vLLM 服务
├── api_server.py          # RESTful API 服务
├── start.sh               # 一键启动脚本
├── requirements.txt       # Python 依赖
├── README.md              # 详细文档
├── example_api_usage.py   # Python API 调用示例
└── example_curl.sh        # Curl API 调用示例
```

## 3 分钟快速启动

### Step 1: 安装依赖

```bash
cd /path/to/FinServe-MLO-vLLM

# 创建虚拟环境
python3.10 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### Step 2: 下载模型

```bash
python download_models.py
```

自动下载：
- ✅ Qwen2.5-32B-Instruct-GPTQ-Int4 (基座，18GB)
- ✅ qwen25-32b-expert-a-qlora (专家 A)
- ✅ qwen25-32b-expert-b-qlora (专家 B)

### Step 3: 启动服务

**方式 A: 交互式模式**
```bash
python serve_lora.py
```

**方式 B: API 服务模式**
```bash
python api_server.py
```

**方式 C: 一键启动**
```bash
./start.sh
```

## 使用方式

### 交互式模式命令

```
:lora              # 查看状态
:lora expert_a     # 切换到专家 A
:lora expert_b     # 切换到专家 B
:lora base         # 使用基座模型
:quit              # 退出
```

### API 调用

```bash
# 查看 API 文档
浏览器打开：http://localhost:8000/docs

# Curl 测试
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好", "lora_name": "expert_a"}'

# 运行示例脚本
python example_api_usage.py
./example_curl.sh
```

## 针对 4080S 32GB 的优化

```bash
# 推荐配置
python serve_lora.py \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096
```

## 核心特性

✅ **共享基座**: 两个 LoRA 共享同一个 Qwen2.5 32B GPTQ 4bit 基座
✅ **热切换**: 运行时动态切换 LoRA 适配器，无需重启
✅ **ModelScope**: 所有模型从 ModelScope 下载，国内加速
✅ **vLLM**: 高性能推理引擎，支持并发请求
✅ **双模式**: 交互式 + API 两种服务模式
✅ **显存优化**: 针对 32GB 显存优化配置

## 常见问题

**Q: 下载速度慢？**
A: ModelScope 在国内有 CDN 加速，如果仍然慢，可以配置镜像源

**Q: 显存不足？**
A: 降低 `--gpu-memory-utilization` 参数，如 0.85 → 0.8

**Q: 如何验证模型下载成功？**
A: 查看 `models/model_config.txt` 文件，确认三个模型路径都存在

**Q: 支持更多 LoRA 模型吗？**
A: 支持，修改 `download_models.py` 添加模型，重启服务即可

## 技术细节

- **基座模型**: Qwen2.5-32B-Instruct-GPTQ-Int4 (4bit 量化，约 18GB)
- **LoRA 适配器**: QLoRA 微调，每个约几百 MB
- **推理引擎**: vLLM 0.6.0+ (支持 LoRA 热切换)
- **Python**: 3.10
- **CUDA**: 12.1+
- **显存占用**: 约 20-24GB (取决于配置)

## 下一步

1. 测试两个专家模型的效果
2. 根据实际需求调整参数
3. 如需生产环境，配置 nginx 反向代理
4. 监控显存和性能指标

---

**祝你使用愉快！有问题随时查看 README.md 详细文档。**

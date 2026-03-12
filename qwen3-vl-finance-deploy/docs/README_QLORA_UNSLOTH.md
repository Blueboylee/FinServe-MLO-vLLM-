# Qwen3-VL Unsloth QLoRA 微调说明

这套脚本用于把 `Qwen/Qwen3-VL-8B-Instruct` 作为基座，使用
`nohurry/Opus-4.6-Reasoning-3000x-filtered` 数据集里的 `problem` / `solution`
字段做文本 SFT 微调，并将产出的 LoRA 适配器上传到：

`GaryLeenene/Qwen3-VL-Finance-expert-c`

## 目录约束

所有缓存、模型、数据切片、日志、训练输出默认都放在：

`/data/qwen3-vl-finance-expert-c`

核心环境变量由 `data_env.sh` 统一配置，包括：

- `HF_HOME`
- `HF_DATASETS_CACHE`
- `TRANSFORMERS_CACHE`
- `MODELSCOPE_CACHE`
- `TORCH_HOME`
- `TMPDIR`

## 新增文件

- `data_env.sh`: 统一 `/data` 路径
- `environment.unsloth.yml`: conda 训练环境定义
- `setup_unsloth_env.sh`: 一键创建或更新 conda 训练环境
- `download_base_model_to_data.sh`: 下载基座模型到 `/data`
- `requirements.unsloth.txt`: 微调依赖
- `train_qwen3_vl_unsloth_qlora.py`: 训练主脚本
- `run_train_qlora_unsloth.sh`: 启动训练
- `upload_lora_to_modelscope.py`: 上传到 ModelScope
- `run_upload_modelscope.sh`: 启动上传

## 1. 创建 conda 训练环境

推荐使用独立的 conda 环境，并且把 conda 的环境目录和包缓存都放到 `/data`：

- 环境名：`qwen3-vl-unsloth`
- 环境文件：`environment.unsloth.yml`
- 环境实际路径：`/data/qwen3-vl-finance-expert-c/conda/envs/qwen3-vl-unsloth`
- conda 包缓存：`/data/qwen3-vl-finance-expert-c/conda/pkgs`

```bash
cd /root/下载/FinServe-MLO-vLLM-/qwen3-vl-finance-deploy
bash setup_unsloth_env.sh
export CONDA_PKGS_DIRS=/data/qwen3-vl-finance-expert-c/conda/pkgs
conda activate /data/qwen3-vl-finance-expert-c/conda/envs/qwen3-vl-unsloth
```

如果你机器的 CUDA 不是 `12.1`，请把 `environment.unsloth.yml` 里的 `pytorch-cuda=12.1` 改成匹配的版本后再创建环境。

## 2. 手动补装依赖（可选）

只有在 conda 创建后仍缺包时，才需要补这一步：

```bash
pip install -r requirements.unsloth.txt
```

## 3. 下载基座模型到 /data

```bash
cd /root/下载/FinServe-MLO-vLLM-/qwen3-vl-finance-deploy
bash download_base_model_to_data.sh
```

脚本内部执行的是：

```bash
modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir /data/qwen3-vl-finance-expert-c/models/Qwen3-VL-8B-Instruct
```

## 4. 训练 QLoRA

默认会使用数据集的 `0..200` 条样本，包含两端，也就是 201 条。

```bash
cd /root/下载/FinServe-MLO-vLLM-/qwen3-vl-finance-deploy
bash run_train_qlora_unsloth.sh
```

如需覆盖默认参数：

```bash
START_INDEX=0 \
END_INDEX=200 \
OUTPUT_NAME=Qwen3-VL-Finance-expert-c \
bash run_train_qlora_unsloth.sh
```

训练输出位置示例：

```text
/data/qwen3-vl-finance-expert-c/outputs/Qwen3-VL-Finance-expert-c-20260310-120000/adapter
```

其中会包含：

- `adapter_config.json`
- `adapter_model.safetensors`
- `README.md`
- `configuration.json`
- tokenizer 相关文件

同时会把切片后的数据快照保存到：

```text
/data/qwen3-vl-finance-expert-c/datasets/Qwen3-VL-Finance-expert-c/slice_0_200.jsonl
```

## 5. 上传到 ModelScope

先准备你的 ModelScope Token：

```bash
export MODELSCOPE_API_TOKEN="你的 token"
```

默认上传最近一次训练生成的 adapter：

```bash
cd /root/下载/FinServe-MLO-vLLM-/qwen3-vl-finance-deploy
bash run_upload_modelscope.sh
```

也可以手动指定本地目录：

```bash
bash run_upload_modelscope.sh /data/qwen3-vl-finance-expert-c/outputs/xxx/adapter
```

## 备注

- 这次数据集是纯文本，所以脚本走的是“文本 SFT 到 VL 基座”的方式。
- 脚本默认关闭视觉塔 LoRA，只更新语言侧模块，适合你当前这份 `problem/solution` 数据。
- 如果后续你要接图文金融数据，我可以再给你补一版真正的多模态 Unsloth 训练脚本。
- 如果根分区空间不足，不要再把 conda 环境装到默认位置；本方案已经改为强制写入 `/data`。

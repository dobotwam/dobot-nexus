# 1. 构建环境与准备权重

[新手路线](../../README.md#从哪里开始) · 下一步 [数据格式与转换](02_data.md)

本篇在 GPU 主机执行。完成后，应能导入 Nexus、看到 CUDA GPU，并定位 Pi0.5 初始权重。
主线使用 Python 3.11、Pi0.5 Torch 和双卡 FSDP 训练。

## 1.1 确认工作目录和系统环境

取得项目源码后，进入包含 `pyproject.toml`、`src/` 和 `examples/` 的目录。

```bash
cd /path/to/dobot-nexus
python3.11 --version
nvidia-smi
```

Python 至少为 3.10，后续命令统一使用 3.11。`nvidia-smi` 应列出你可使用的 GPU。
若命令缺失或驱动异常，先由主机管理员配置 NVIDIA 驱动和 GPU 容器访问权限。

## 1.2 创建 Pi0.5 虚拟环境

```bash
python3.11 scripts/env/create_model_env.py --model pi05 --extra websocket
source .venvs/pi05/bin/activate
python scripts/env/check_model_env.py --model pi05 --extra websocket
```

环境位于仓库的 `.venvs/pi05/`。脚本安装公共训练依赖、Pi0.5 依赖、WebSocket 服务依赖，
并以 editable 模式安装当前源码。检查成功时末尾会显示类似结果。

```text
OK model=pi05 extras=websocket python=3.11.x
```

每次打开新终端，执行以下两行。

```bash
cd /path/to/dobot-nexus
source .venvs/pi05/bin/activate
```

如果希望提前查看安装计划，在创建命令末尾添加 `--dry-run`。
已有 CUDA PyTorch 的训练镜像可在创建新环境时添加 `--system-site-packages`。
自定义环境路径和其他模型的依赖见[环境参数参考](../environments.md)。

## 1.3 检查 PyTorch 能否使用 GPU

```bash
python - <<'PY'
import sys
import torch
import dobot_nexus

print("Python", sys.executable)
print("Nexus", dobot_nexus.__file__)
print("Torch", torch.__version__)
print("CUDA runtime", torch.version.cuda)
assert torch.cuda.is_available(), "当前 PyTorch 无法使用 CUDA"
print("GPU 数量", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    with torch.cuda.device(index):
        print(index, torch.cuda.get_device_name(index), "BF16", torch.cuda.is_bf16_supported())
PY
```

训练命令中的 `--gpus 0,1` 对应准备使用的两张卡，默认算子精度为 BF16。
显存不足时先降低每卡 batch size，具体方法见训练篇。
若 `nvidia-smi` 正常而 PyTorch 的 CUDA 检查失败，请核对解释器、Torch 安装来源和驱动兼容性。

## 1.4 准备 Pi0.5 初始权重

从项目维护者或已有模型资产来源获取与本仓库兼容的 **Pi0.5 Torch 权重**。
首次微调通常使用包含 `model.safetensors` 的目录。已有 Nexus 训练产物中的
`deployment/pytorch/full_weights.pt` 也可以用于初始化新实验。

```text
/path/to/pi05_base/
└── model.safetensors
```

设置路径并用仓库的解析器检查文件位置。

```bash
export PI05_BASE_CKPT=/path/to/pi05_base

python - <<'PY'
import os
from dobot_nexus.models.pi05.training.checkpoint import resolve_checkpoint_file

print("将加载", resolve_checkpoint_file(os.environ["PI05_BASE_CKPT"]))
PY
```

这里检查文件定位，权重结构和维度会在构造模型时继续校验。加载器接受 Torch `.pt` 和
`model.safetensors` 等已有格式，权重提供者应确认文件属于 Pi0.5 兼容实现。
tokenizer 随仓库位于 `src/dobot_nexus/models/pi05/assets/pi05_tokenizer.model`。

## 1.5 分清三类环境

| 环境 | 运行位置 | 用途 |
|---|---|---|
| `.venvs/pi05` | GPU 主机 | 训练、离线推理、策略服务、通信客户端 |
| `.venvs/data` | 数据处理主机，可与 GPU 主机相同 | PKL 转换为 LeRobot，下一篇创建 |
| 机器人项目环境 | 机器人主机 | `dobot_environment`、机械臂 SDK、相机、夹爪和串口驱动 |

Pi0.5 环境的包版本由仓库 requirements 和当前软件源共同决定。保存环境清单可帮助复现实验。

```bash
mkdir -p runs/quickstart
python -m pip freeze > runs/quickstart/pi05-environment.txt
nvidia-smi > runs/quickstart/gpu-environment.txt
```

完成本篇后继续[整理数据](02_data.md)。模型权重、数据和 run 输出需要足够磁盘空间，
训练过程中会同时保存部署权重及优化器恢复状态。

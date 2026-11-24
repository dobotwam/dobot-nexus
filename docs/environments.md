# 模型环境与安装入口

第一次安装 Pi0.5，请先按[新手环境构建](getting_started/01_environment.md)逐步操作。
本文保留所有模型的安装选项、依赖归属和可选组件说明。

模型使用独立 pip 环境，共享 Nexus 源码，不默认共享 site-packages。
环境脚本只处理安装与导入检查，不加载模型权重、不启动服务。

## 安装职责

| 位置 | 职责 |
|---|---|
| `requirements.txt` | 环境脚本使用的公共训练、数据与 Artifact 依赖 |
| `src/dobot_nexus/models/<model>/requirements.txt` | 引用公共依赖，加上该模型的依赖边界 |
| `pyproject.toml` | Nexus 包元数据、最小运行依赖、模型 extras 和可选服务 extras |
| `scripts/env/create_model_env.py` | 创建或复用 venv，安装模型依赖和 editable Nexus |
| `scripts/env/check_model_env.py` | `pip check`、公共/模型导入和选定服务依赖检查 |
| `data_processing/requirements.txt` | 独立数据转换环境，不需要模型环境 |

| 模型 | 默认环境 | 算子/外部实现 | 主要依赖边界 |
|---|---|---|---|
| Pi0.5 BC / RECAP policy / Memory | `.venvs/pi05` | 本地 `models/pi05/torch_core/`，Torch 训练/部署 | `datasets`、`lerobot>=0.3,<0.5`、`sentencepiece`、`einops` |
| Pi0 BC / RECAP policy / Memory | `.venvs/pi0` | 独立 `models/pi0/training/vendor/rlinf_core/` | 同类数据依赖，独立维护算子 |

禁止把模型专属依赖移入公共 requirements。`pip install -e .` 只安装最小 Nexus 包，
不等于准备好了完整训练环境；从源码训练优先使用环境脚本。

## 创建与检查

包和环境脚本的最低 Python 版本为 3.10，模型环境推荐 Python 3.11。

```bash
python3.11 scripts/env/create_model_env.py --model pi05
source .venvs/pi05/bin/activate
python scripts/env/check_model_env.py --model pi05
```

其他模型替换 `--model` 即可。自定义环境路径或解释器：

```bash
python3 scripts/env/create_model_env.py \
  --model pi05 --venv /path/to/venvs/pi05 --python /path/to/python3.11 \
  --dry-run
```

`--dry-run` 只打印计划，不执行解释器、venv 或 pip；已有环境会按复用路径展示计划。
移除该参数后实际安装。已有环境会检查其 Python 版本，再安装依赖；不会删除重建环境。

若训练镜像已带匹配 CUDA 的 PyTorch，可在创建时使用 `--system-site-packages`；它只影响
新环境，不改变已有环境的继承关系。`--skip-tool-upgrade` 跳过 pip/setuptools/wheel 升级。
脚本使用标准 `venv` 和 `pip`，不要求 `uv`。pip 源通过 `PIP_INDEX_URL` 和
`PIP_EXTRA_INDEX_URL` 等标准变量配置。

## 可选服务依赖

| 选项 | 安装/检查的包 | 用途 |
|---|---|---|
| `--extra websocket` | `msgpack`、`websockets>=15,<16` | 真机 policy 与通用 WebSocket 客户端 |
| `--extra serve` | `fastapi`、`uvicorn` | Nexus HTTP 服务 |

创建、复用和检查时均可重复传入 `--extra`：

```bash
python3 scripts/env/create_model_env.py --model pi05 --extra websocket
.venvs/pi05/bin/python scripts/env/check_model_env.py --model pi05 --extra websocket

# 同时使用 HTTP 时再添加 --extra serve
python3 scripts/env/create_model_env.py --model pi05 --extra websocket --extra serve
```

若模型环境已经准备好，也可仅安装 extra：

```bash
.venvs/pi05/bin/python -m pip install -e '.[websocket]'
# HTTP: .venvs/pi05/bin/python -m pip install -e '.[serve]'
```

环境脚本先安装模型 requirements；未指定 extra 时以 `--no-deps -e` 安装 Nexus。
指定 extra 时由 `pyproject.toml` 解析附加服务依赖，不在脚本中复制包版本。

检查脚本会汇总 pip 冲突和导入失败，不在首个错误处隐藏其余问题。
导入检查通过只说明 Python 依赖可用，不能代替 GPU、权重、数据解码或硬件验证。

如果环境仅安装了依赖、未 editable 安装 Nexus，可从仓库根目录用
`PYTHONPATH=src:. python scripts/env/check_model_env.py --model pi05 --extra websocket`
检查源码运行方式。脚本默认不注入源码路径，因此能发现 Nexus 本身漏装的问题。

## 可选训练指标记录

W&B 和 TensorBoard 独立可选，不属于模型必需依赖：

```bash
.venvs/pi05/bin/python -m pip install -e '.[tracking,tensorboard]'
```

只使用本地 TensorBoard 时安装 `.[tensorboard]`；只使用 W&B 时安装 `.[tracking]`。
配置和离线/在线同步见 [训练日志](pi05_training.md#52-可选-wb-与-tensorboard-指标记录)。
`create_model_env.py --extra` 当前用于服务依赖；训练 tracking extra 使用上述 pip 命令安装。

## RECAP Value 与数据转换

RECAP policy 使用对应模型环境（`.venvs/pi05` 或 `.venvs/pi0`）；离线 Value/标注另需其依赖入口：

```bash
.venvs/pi05/bin/python -m pip install \
  -r src/dobot_nexus/algorithms/recap/value/requirements.txt
```

该文件引用公共 `modules/value/fusion/requirements.txt` 并补充 RECAP 数据 reader。
Value 固定 `transformers==4.53.2`，请使用对应的依赖入口安装。
只复用公共 Value 网络时直接安装公共 Fusion requirements，见
[Value 组件](../src/dobot_nexus/modules/value/README.md)。
独立转换脚本的依赖与用法见 [数据转换](../data_processing/README.md)。

## 外部边界与复现

- Pi0.5 / Pi0 使用各自内置的 Torch 算子。定制实现通过 `model.target` 接入。
- Pi0.5 GPU 训练/部署需匹配 PyTorch wheel 的 NVIDIA 驱动，多卡使用 NCCL。
  LeRobot v2.1 reader 直接依赖 `datasets`，视频解码使用 LeRobot 的 backend。
  机器人 SDK、相机/串口驱动与硬件配置在机器人侧准备，不由这些脚本安装。
- requirements 使用版本范围，并非 lockfile。实际 CUDA wheel 与小版本取决于软件源和基础镜像；
  需要复现实验时单独记录 Python、pip freeze、驱动及模型资产版本。

# Pi0.5 训练与部署配置指南

首次使用请从[新手路线](../README.md#从哪里开始)开始，按顺序完成
[数据转换](getting_started/02_data.md)、[训练](getting_started/03_training.md)、
[离线推理](getting_started/04_inference.md)和[真机部署](getting_started/05_deployment.md)。
本文作为完整参数、数据语义和实现细节参考。

本文面向使用本仓库训练自有 Pi0.5 任务的算法和工程用户，覆盖从环境、LeRobot 数据、任务
配置、单卡/多卡训练、断点续训，到 Artifact 验收和推理部署的完整流程。

仓库当前提供的是 Behavior Cloning（BC/SFT）训练闭环。Pi0.5 训练使用仓库内置的 PyTorch
operator core；每次保存 checkpoint 都会保存 PyTorch 部署权重和模型专属资源。
正式训练应根据数据规模和验证结果选择训练步数。

Memory 的分步操作见 [Memory 训练与推理操作指南](pi05_memory.md)。

## 0. 训练前置说明：LeRobot v2.1 格式规范

本项目的 Pi0.5 数据链路以本地 LeRobot v2.1 数据集为主要输入，同时兼容 v3.0。训练配置中的
`data.path` 必须指向单个数据集根目录，也就是直接包含 `meta/`、`data/` 的目录；
视频格式还包含 `videos/`，内嵌图像格式无需该目录。不能指向多个任务数据集的上一级目录。
本节先说明 v2.1 的磁盘格式；Nexus 在训练阶段
如何映射字段、生成 action chunk 和计算 normalization statistics，见第 4 节。

### 0.1 目录结构

标准的、使用视频保存图像的 v2.1 数据集按 episode 分文件：

```text
<dataset_root>/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet
│   │   └── episode_000001.parquet
│   └── chunk-001/
│       └── episode_001000.parquet
└── videos/
    ├── chunk-000/
    │   ├── observation.images.top_camera/
    │   │   ├── episode_000000.mp4
    │   │   └── episode_000001.mp4
    │   ├── observation.images.left_camera/
    │   │   ├── episode_000000.mp4
    │   │   └── episode_000001.mp4
    │   └── observation.images.right_camera/
    │       ├── episode_000000.mp4
    │       └── episode_000001.mp4
    └── chunk-001/
        └── ...
```

v2.1 默认 `chunks_size` 是 1000，所以 `episode_index` 为 0–999 的文件位于 `chunk-000`，
1000–1999 位于 `chunk-001`。通用计算关系为：

```text
episode_chunk = episode_index // chunks_size
```

每个 episode 对应一个 Parquet 文件，每路相机对应一个同编号 MP4。真实路径由
`meta/info.json` 的 `data_path` 和 `video_path` 模板定义；建议保持官方模板，不要自行改变
占位符或目录层级。

### 0.2 `meta/info.json`

`info.json` 是数据集的总清单和 feature schema。关键字段如下：

| 字段 | v2.1 约束 |
|---|---|
| `codebase_version` | 写为 `v2.1`；Nexus 据此选择本地 v2.1 reader |
| `robot_type` | 采集机器人的类型或项目内约定名称 |
| `fps` | 正整数；控制帧频率，也用于 action 时间偏移和视频对齐 |
| `chunks_size` | 每个 chunk 最多容纳的 episode 数，官方默认 1000 |
| `total_episodes` / `total_frames` | 分别等于 episode 数和所有 episode 的 `length` 之和 |
| `total_tasks` / `total_videos` / `total_chunks` | 与 task、相机视频和 chunk 的实际数量一致 |
| `splits` | 通常为 `{"train": "0:<total_episodes>"}` |
| `data_path` | `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet` |
| `video_path` | `videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4` |
| `features` | 声明所有 Parquet 列和视频流的 dtype、shape、维度名及可选媒体信息 |

Pi0.5 默认任务的 `features` 至少应表达下面的 schema；相机分辨率和 state/action 维度按实际
数据填写：

```json
{
  "observation.images.top_camera": {
    "dtype": "video",
    "shape": [480, 640, 3],
    "names": ["height", "width", "channels"]
  },
  "observation.images.left_camera": {
    "dtype": "video",
    "shape": [480, 640, 3],
    "names": ["height", "width", "channels"]
  },
  "observation.images.right_camera": {
    "dtype": "video",
    "shape": [480, 640, 3],
    "names": ["height", "width", "channels"]
  },
  "observation.state": {
    "dtype": "float32",
    "shape": [14],
    "names": null
  },
  "action": {
    "dtype": "float32",
    "shape": [14],
    "names": null
  },
  "timestamp": {"dtype": "float32", "shape": [1], "names": null},
  "frame_index": {"dtype": "int64", "shape": [1], "names": null},
  "episode_index": {"dtype": "int64", "shape": [1], "names": null},
  "index": {"dtype": "int64", "shape": [1], "names": null},
  "task_index": {"dtype": "int64", "shape": [1], "names": null}
}
```

这段内容是 `features` 对象的示例，不是完整的 `info.json`。视频 feature 还可以带 `info`，记录
codec、pixel format、视频 FPS、宽高、通道数和是否含音频。视频的 shape 使用 HWC 顺序；实际
输入模型前，Nexus 会负责解码和转换布局。

### 0.3 Parquet 的逐帧记录

一个 Parquet 行表示同一时刻的一帧机器人数据。使用 MP4 的 v2.1 数据集不把 RGB 像素重复写入
Parquet，而是通过该行的 `episode_index` 和 `timestamp` 到对应视频中解码。标准列及其语义为：

| 列 | 类型/形状 | 语义 |
|---|---|---|
| `observation.state` | `float32[state_dim]` | 当前帧机器人状态 |
| `action` | `float32[action_dim]` | 当前帧监督 action；是 absolute 还是 delta 必须由数据定义明确 |
| `timestamp` | `float32` 标量 | episode 内时间，通常为 `frame_index / fps`，单位为秒 |
| `frame_index` | `int64` 标量 | episode 内帧号，从 0 开始连续递增 |
| `episode_index` | `int64` 标量 | 所属 episode，与文件名和 metadata 对齐 |
| `index` | `int64` 标量 | 数据集范围内的全局帧号 |
| `task_index` | `int64` 标量 | 指向 `meta/tasks.jsonl` 中的任务文本 |

如果数据还有力矩、速度或其他传感器，可以增加 feature 和 Parquet 列；Pi0.5 只读取配置
`data.field_mapping` 所引用的字段。不要把 state/action 各维拆成一组未映射的独立列，默认训练
接口需要一个定长向量列。

### 0.4 episode、task 和统计 metadata

三个 JSONL 文件都是“一行一个 JSON object”，不是一个 JSON array。`episodes.jsonl` 按
`episode_index` 记录该 episode 的长度和出现过的任务，例如：

```json
{"episode_index": 0, "tasks": ["Pick up the red cube."], "length": 385}
{"episode_index": 1, "tasks": ["Pick up the red cube."], "length": 412}
```

`tasks.jsonl` 提供 `task_index` 到非空自然语言指令的映射：

```json
{"task_index": 0, "task": "Pick up the red cube."}
```

上游视频数据集中的 `episodes_stats.jsonl` 保存逐 episode feature statistics，每行包含
`episode_index` 和 `stats`，使用上游工具时应保留其生成的 metadata。
本仓库 `data_processing.convert` 输出内嵌 PNG 的 v2.1 布局，并生成 `meta/stats.json`、
`sources.jsonl` 和 `conversion.json`。Nexus 的内嵌图像 reader 使用该布局，完整文件清单见
[转换输出说明](getting_started/02_data.md#22-v21-的目录和字段)。
Pi0.5 训练按 action chunk 和 transform 后的实际数值重新计算归一化统计。

### 0.5 Nexus/Pi0.5 接入检查

除了文件存在，接入本仓库前还必须满足以下一致性条件：

- `episodes.jsonl` 按 `episode_index` 排序；所有 Parquet 行也按 episode 排序，且同一 episode 的
  行连续存放。
- 每个 episode 的 `frame_index` 从 0 开始连续递增，Parquet 行数等于 metadata 的 `length`；所有
  `length` 之和等于 `info.json.total_frames`。
- `timestamp` 在 episode 内单调递增，并与 `fps` 及 MP4 帧时间对齐。默认允许误差由
  `data.tolerance_s=1.0e-4` 控制。
- 每个 Parquet `task_index` 都能在 `tasks.jsonl` 找到非空文本；否则需要配置
  `data.default_prompt`。
- 若使用视频格式，三路 feature 均在 `info.json.features` 中声明为 `dtype: video`，每个 episode 都存在对应
  MP4，且视频可由当前 LeRobot backend 解码。
- `observation.state` 和 `action` 的末维分别等于 `model.state_dim` 和 `model.action_dim`；各维顺序、
  单位、关节方向、夹爪范围必须与部署端完全一致。
- 相机 feature 名可以不同，但必须通过 `data.field_mapping.images` 映射到 `base`、`left_wrist`、
  `right_wrist` 三个公共名称。

正式训练前检查 feature、维度、checkpoint 和 GPU 环境，并用少量真实数据走通读取、
解码和训练链路。实模 smoke 脚本保存在本地 `local_script/`，不随正式代码分发；
验收项见第 10 节。LeRobot 官方对 v2.1
episode-based 与 v3.0 file-based 布局的对照见
[Porting Large Datasets to LeRobot Dataset v3.0](https://huggingface.co/docs/lerobot/main/porting_datasets_v3#file-organization-v21-vs-v30)。

## 1. 最短可运行路径

假设已经有一个标准 LeRobot v2.1/v3 数据集和官方 Pi0.5 base checkpoint：

```bash
cd /path/to/dobot-nexus

# 1. 创建 Pi0.5 独立环境
python3.11 scripts/env/create_model_env.py --model pi05 --venv .venvs/pi05
.venvs/pi05/bin/python scripts/env/check_model_env.py --model pi05

# 2. 配置资产路径
export PI05_BASE_CKPT=/path/to/pi05_base
export VLA_DATASET_PATH=/path/to/lerobot_dataset

# 3. 双卡 FSDP 正式训练；run name 每次新实验都应唯一
examples/train.sh \
  --python .venvs/pi05/bin/python \
  --config examples/configs/train/pi05.yaml \
  --gpus 0,1 \
  --run-name pi05_my_task_b2_s10000 \
  --output-dir /path/to/training_runs \
  -- \
  data.batch_size=2 \
  train.max_steps=10000 \
  scheduler.total_steps=10000
```

模板里的 `data.batch_size=64` 是每个 rank 的 batch size，对显存要求很高。第一次在新硬件上
运行时建议从 `1` 或 `2` 开始，再根据实测显存逐步增大。

## 2. 代码和配置入口

```text
examples/configs/train/pi05.yaml                 通用 absolute/passthrough action 模板
examples/configs/train/pi05_recap.yaml           通用 RECAP 模板
examples/configs/train/pi05_memory.yaml          通用 Memory 模板
examples/recipes/                              USB、按钮等任务的差异配置
examples/train.sh                              统一训练入口
examples/infer.sh                              离线推理和 Nexus HTTP 服务入口
examples/configs/infer/pi05.yaml              Pi0.5 标准推理配置
src/dobot_nexus/models/pi05/                   Pi0.5 模型、数据、checkpoint 和部署实现
examples/pi05_real_robot/                      WebSocket 真机链路
```

用户通过任务 recipe 继承通用模板并覆盖任务字段。具体例子和 Memory 资源准备入口见
[`examples/recipes/README.md`](../examples/recipes/README.md)，不需要修改 Trainer 或算子。

## 3. 环境与资产准备

### 3.1 训练环境

推荐 Python 3.11。环境创建脚本会安装公共训练依赖、Pi0.5 专属依赖，并以 editable 模式安装
当前仓库：

```bash
python3.11 scripts/env/create_model_env.py \
  --model pi05 \
  --venv .venvs/pi05

.venvs/pi05/bin/python scripts/env/check_model_env.py --model pi05
```

如果训练镜像已经提供了匹配 CUDA 的 PyTorch，可以在创建新目录时使用
`--system-site-packages`。该选项不会改变已经存在的环境；需要彻底重建时应指定一个新的
`--venv` 路径。

仓库包含 Pi0.5 所需的 operator core 和 tokenizer。

按需要安装 HTTP / WebSocket 服务依赖（分别对应 `serve` / `websocket`）：

```bash
.venvs/pi05/bin/python -m pip install -e '.[serve,websocket]'
```

也可在创建环境时添加 `--extra websocket` / `--extra serve`，检查环境时传入相同选项。
环境安装与依赖说明见 [环境指南](environments.md)。

### 3.2 Base checkpoint

`model.checkpoint` 接受 checkpoint 文件，也接受包含以下任一文件的目录：

```text
model.safetensors
full_weights.pt
deployment/pytorch/full_weights.pt
```

首次微调通常使用官方目录：

```bash
export PI05_BASE_CKPT=/path/to/pi05_base
test -f "$PI05_BASE_CKPT/model.safetensors"
```

初始权重在模型构造时加载。Artifact 直接保存训练得到的 Torch 参数。

### 3.3 硬件和存储

- 训练必须使用 CUDA GPU；算子计算精度为 BF16。
- 默认 FSDP 配置至少需要两张 GPU。GPU 数量可以增加，但断点续训时 world size 必须保持一致。
- 单卡 local 模式会在一张卡上保留完整参数、梯度和优化器状态，只适合显存充足的机器或工程调试。
- 每个部署视图约包含 12.5–12.8 GiB PyTorch 权重；Memory 另带约 1.5 GiB 的冻结视觉权重。
  启用断点续训后还会保存分片模型、优化器和 RNG 状态。请按保存次数预留磁盘和 rank 0 主机内存。

### 3.4 依赖闭包和外部边界

Pi0.5 的不同入口应按下面的闭包部署，不能只依据顶层 `import` 判断环境是否完整：

| 链路 | Python 依赖 | 系统、源码和资产依赖 |
|---|---|---|
| 训练 | `torch`、`torchvision`、`accelerate`、`numpy`、`PyYAML`、`lerobot`、`datasets`、`pyarrow`、`Pillow`、`sentencepiece`、`einops`、`safetensors` | Linux；匹配 PyTorch wheel 的 NVIDIA driver/CUDA；训练数据、base `model.safetensors`、仓库内 tokenizer |
| 多卡 FSDP | 上述训练依赖和 `torch.distributed` | NCCL、卡间通信；断点恢复必须保持 world size |
| Torch 离线 policy | `torch`、`torchvision`、`numpy`、`PyYAML`、`Pillow`、`sentencepiece`、`einops`；读取 safetensors checkpoint 时另需 `safetensors` | Linux、匹配 PyTorch 的 driver/CUDA、完整 Artifact；不读取 LeRobot 数据 |
| Nexus HTTP policy | Torch policy 环境加 `fastapi`、`uvicorn` | 完整 Artifact；受控 HTTP 网络或网关 |
| Torch WebSocket policy | Torch policy 环境加 `msgpack`、`websockets` | 完整 Artifact；与 client 连通的网络 |
| 训练对比 benchmark | 对应训练/policy 环境；画图脚本额外使用 `matplotlib` | GPU profiler 额外调用 `nvidia-smi` |

这里的 `datasets` 是 `LocalLeRobotV21Dataset` 的直接 import，视频读取还依赖 LeRobot 选出的解码
backend（当前安装闭包会带入 `av`/`torchcodec` 等包）；Nexus 本身不调用系统 `ffmpeg` 命令。
`datasets` 已在 Pi0.5 requirements 和包 extra 中直接声明。`matplotlib` 仅供本地报告绘图，
不属于正式训练/部署依赖，需要时在本地诊断环境中安装。

训练 pip 脚本当前使用版本范围（例如 `torch>=2.10,<2.11`、`lerobot>=0.3,<0.5`），不是 lockfile；实际
CUDA wheel 来源取决于 `PIP_INDEX_URL`/`PIP_EXTRA_INDEX_URL` 或基础镜像。

训练期间需要访问原始 LeRobot 数据，Memory 保存时还要复制并校验冻结视觉权重。保存 Artifact
无需再读取 base checkpoint；Torch policy 部署只需完整 Artifact，不需要原始数据集或 base。
真机 `examples/pi05_real_robot/_runtime/dobot_dual_arm_14d.py`、
`dobot_environment`、机器人 SDK/驱动和设备权限属于机器人项目的外部边界，不属于 Nexus GPU
policy 的 pip 闭包。

## 4. LeRobot 数据契约

### 4.1 支持范围

`data.format` 当前应为 `lerobot`，`data.format_version` 支持 `auto`、`v2.1`、`v3.0`。推荐保留
`auto`，由 `meta/info.json` 自动识别版本和 FPS。

v2.1 的目录、metadata 和逐帧 schema 见第 0 节。本节只描述 Nexus/Pi0.5 在标准 LeRobot 数据之上
增加的运行时处理契约。

训练数据处理包括 action chunk、episode 尾部补齐、action transform、统计、图像 resize/pad、
prompt 编码和模型维度 padding。Dobot recorder PKL 可通过[数据转换工具](../data_processing/README.md)
生成 LeRobot 数据；其他原始日志需先转换为对应的数据集格式。

数据根目录至少要有 `meta/info.json`，并满足对应 LeRobot 版本的 parquet、episode/task metadata
和 video 布局。Pi0.5 默认字段契约如下：

| 公共含义 | 默认 LeRobot feature | 要求 |
|---|---|---|
| base camera | `observation.images.top_camera` | RGB video/image |
| left wrist | `observation.images.left_camera` | RGB video/image |
| right wrist | `observation.images.right_camera` | RGB video/image |
| robot state | `observation.state` | `float`，末维等于 `model.state_dim` |
| action | `action` | `float`，末维等于 `model.action_dim` |
| language | `task` 或 `task_index` metadata | 每条样本最终必须得到非空 prompt |

三路相机缺一不可，但 LeRobot feature 名可以通过 `data.field_mapping.images` 修改。进入公共推理
接口后，相机名固定为 `base`、`left_wrist`、`right_wrist`。

必须人工确认 state/action 每一维的顺序、单位、关节方向和夹爪范围与部署机器人一致。形状正确
不能证明控制语义正确。

### 4.2 Action chunk

`model.action_horizon` 默认是 50。数据加载器从当前帧向后读取 50 个 action；episode 末尾不足
50 步时，用该 episode 的最后一个 action 补齐，不会跨 episode 取数据。

如果数据集没有可解析的任务文本，配置固定任务 prompt：

```yaml
data:
  field_mapping:
    language: null
  default_prompt: "Pick up the red cube."
```

### 4.3 Absolute 和 delta 的准确含义

`data.transforms.action.type` 是 action codec，不应仅根据任务名称猜测：

| 配置 | 训练前处理 | 推理后处理 |
|---|---|---|
| `absolute` | 原始 action 原样进入 normalization | 模型输出直接返回 |
| `delta` | mask 为 true 的维度执行 `action - current_state` | 同一维度执行 `prediction + current_state` |

因此：

- 原始 action 已经是部署端需要的表示时，用 `absolute` 保持 passthrough。
- 需要把数据中的绝对关节目标转换成相对学习目标、部署时再还原为绝对目标时，用 `delta`。
- `delta` 要求 `state_dim == action_dim`，且 `delta_mask` 长度必须等于 `action_dim`。
- 双臂“6 个关节 + 1 个夹爪”常用 mask 是关节 `true`、夹爪 `false`：

```yaml
data:
  transforms:
    action:
      type: delta
      delta_mask:
        - true
        - true
        - true
        - true
        - true
        - true
        - false
        - true
        - true
        - true
        - true
        - true
        - true
        - false
```

如果数据本身已经保存 delta，再选择上述 `delta` 会重复减 state。修改 codec 后必须重新生成
normalization statistics。

### 4.4 自动 normalization statistics

正式训练建议省略 `data.norm_stats_path`。框架会按以下语义从 parquet 重新计算 state/action 的
mean/std 及归一化模式所需的分位数：

- 数据集内容和路径；
- state/action 字段；
- action horizon 和 episode 尾部补齐；
- absolute/delta 类型及 delta mask；
- state/action 维度和 FPS。

统计按 fingerprint 缓存到：

```text
<exp.output_dir>/<exp.name>/data/<fingerprint>/norm_stats.json
```

同一 run 和同一处理语义会复用缓存；任一关键语义变化会生成新 fingerprint。不要直接把
数据集自带的统计当成 delta/action-chunk 处理后的统计。

只有确认统计与当前字段和 transform 完全一致时，才使用高级覆盖：

```yaml
data:
  norm_stats_path: /path/to/transformed/norm_stats.json
```

## 5. 创建任务配置

通用模板保留公共默认值。在 `examples/recipes/pi05_my_task.yaml` 中只写任务差异：

```yaml
extends: ../configs/train/pi05.yaml
exp:
  name: pi05_my_task
data:
  path: ${env:VLA_DATASET_PATH,/path/to/my_task}
  default_prompt: "Perform the task."
```

```bash
examples/train.sh --config examples/recipes/pi05_my_task.yaml --gpus 0,1
```

`extends` 路径相对当前 YAML；只支持单文件继承，检测循环引用。映射递归合并，
列表、标量和 `null` 整体替换，命令行覆盖优先级最高。Artifact 保存展开后的完整配置。

通常需要检查或修改以下字段：

| 配置项 | 含义 | 建议 |
|---|---|---|
| `exp.name` | 默认 run 目录名 | 每个实验唯一，或用 `--run-name` 覆盖 |
| `exp.output_dir` | 所有 run 的父目录 | 指向容量足够的训练盘 |
| `exp.seed` | 初始化与数据采样的基础种子，训练采样使用基础种子加 global rank | 对比实验及续训保持一致 |
| `model.checkpoint` | 初始 Pi0.5 权重 | 指向官方 base 或兼容 checkpoint |
| `model.state_dim` / `action_dim` | 公共机器人维度 | 默认 14，不超过 `model.model_action_dim` |
| `model.model_action_dim` | 内部动作维度 | 默认 32；34 维机器人可设为 64 |
| `model.action_horizon` | action chunk 长度 | 默认 50，训练和部署必须一致 |
| `model.image_size` | operator 图像输入 | 当前必须保持 `[224, 224]` |
| `data.path` | LeRobot 根目录 | 推荐使用 `VLA_DATASET_PATH` 环境变量 |
| `data.batch_size` | 每个 rank 的 batch | 新硬件从 1–2 开始 |
| `data.num_workers` | 每个 rank 的加载进程 | 调试用 0，稳定后按 CPU/IO 调高 |
| `data.field_mapping` | 数据 feature 映射 | 必须与 `meta/info.json` 一致 |
| `data.default_prompt` | 无语言字段时的固定任务 | 必须是非空字符串 |
| `data.transforms.action` | action codec | 按上一节确认，不可盲用 delta |
| `optim.lr` | AdamW 学习率 | 模板为 `2.5e-5`，修改需重新验证 |
| `scheduler.total_steps` | 按成功 optimizer update 计数的 LR schedule 总步数 | 通常与计划的 `train.max_steps` 一致 |
| `train.max_steps` | 成功 optimizer update 次数 | 梯度累积和 AMP 溢出跳步不增加计数 |
| `train.grad_accum_steps` | 梯度累积 micro-batch 数 | 用于在小显存下保持有效 batch |
| `train.save_every` | Artifact 保存间隔 | Artifact 很大，不要设置得过密 |
| `train.resume_limit` | 保留可恢复状态的数量 | 默认 2；0 会禁用训练恢复状态 |
| `runtime.distributed_timeout_seconds` | Accelerate 进程组超时，单位秒 | 可选正整数；未配置时使用 Accelerate 默认值 |

Pi0.5 使用以下精度配置：

```yaml
model:
  precision: bf16
  parameter_precision: fp32

runtime:
  precision: bf16
```

持久参数和 Adam 状态使用 FP32，operator 执行使用 BF16。`model.precision` 与
`runtime.precision` 必须一致；`model.parameter_precision` 改成 BF16 会被显式拒绝。

较大维度的任务见 [KBD recipe](../examples/recipes/pi05_kbd_pick_0916_full34.yaml)：
公共 state/action 为 34 维，归一化后补零到内部 64 维，输出解码为 34 维。
从较小维度权重开始训练时，加载器保留已有动作投影的行列，仅为新增部分初始化
`N(0, 0.02)` 权重和零 bias；新增参数使用固定局部种子，不消耗训练随机数。
不支持缩小边界或忽略其他权重尺寸错误。BC、RECAP policy 和 Memory 的基础权重初始化
共用这一逻辑；Memory 的 `memory_initialization=trained` 仍要求权重完整且尺寸一致。
Artifact 记录内部维度，推理自动恢复该值并严格加载权重，不执行随机扩展。

三个维度都通过通用 `model` 配置指定，不根据机器人名称选择实现：

| 机器人数据示例 | `state_dim` 输入状态 | `action_dim` 输出动作 | `model_action_dim` 内部边界 |
|---|---:|---:|---:|
| 双臂，各 6 个关节与 1 个夹爪 | 14 | 14 | 32 |
| KBD | 34 | 34 | 64 |
| 12 维观测、7 维动作的机器人 | 12 | 7 | 32 |

训练 recipe 覆盖这三个字段，以及 `data.field_mapping`、动作表示、对应归一化统计。
内部边界必须是正整数且不小于两种公共维度；不要求 32 的倍数，但加载权重必须兼容。
state 与 action 可有不同维数；使用带有效 mask 的 delta 动作时，两者需维数相同、逐维对应。
同一套规则用于 BC、RECAP policy、Memory，不会从 34 维数据中隐式截出 14 维。
数据中各维的顺序、单位和夹爪表示由任务定义，维数相同不代表动作语义相同。

### 5.1 环境变量与命令行覆盖

YAML 支持 `${env:NAME,default}`。通用模板已经定义：

```bash
export PI05_BASE_CKPT=/path/to/pi05_base
export VLA_DATASET_PATH=/path/to/lerobot_dataset
export TRAINING_RUNS=/path/to/training_runs
```

也可以在 `--` 后使用点路径临时覆盖，不改 YAML：

```bash
examples/train.sh \
  --config examples/recipes/pi05_my_task.yaml \
  --gpus 0,1 \
  --run-name pi05_my_task_trial01 \
  --output-dir "$TRAINING_RUNS" \
  --dry-run \
  -- \
  data.batch_size=2 \
  data.num_workers=4 \
  optim.lr=1.0e-5
```

`--dry-run` 只打印最终启动命令，不训练。布尔、数字、`null` 会自动解析；包含空格的字符串要把
整个 `key=value` 参数加引号。列表建议直接写入 YAML；如果必须通过 CLI 传布尔列表，请使用
Python literal，例如 `[True,True,False]`，不要使用嵌在列表里的小写 `true/false`。

### 5.2 可选 W&B 与 TensorBoard 指标记录

BC、RECAP policy 和 Memory 共用训练日志配置。先在训练环境中安装：

```bash
python -m pip install -e '.[tracking,tensorboard]'
```

在任务 recipe 中加入：

```yaml
tracking:
  tensorboard:
    enabled: true
  wandb:
    enabled: true
    project: dobot-nexus
    mode: offline
    group: my_task
    tags: [pi05, bc]
    sync_tensorboard: true
```

训练模板默认启用 TensorBoard，W&B 默认关闭；非主 rank 不导入记录 SDK。
只由主 rank 按 `train.log_every`
记录 loss、学习率、梯度范数、吞吐和模型指标，首个 optimizer update 也会记录。
step 对应成功的 optimizer update。运行结束或异常退出时关闭记录。
可选 `name`、`entity`、`notes`、`dir`；名称默认使用 `exp.name`，目录默认是 run 下的 `wandb/`。
`offline` 保存到本地；在线记录先执行 `wandb login`，再设 `mode: online`。
每次启动创建新的 W&B run；训练续训仍由 `train.resume_from` 管理，日志 step 延续恢复的训练步数。

只需本地曲线时，设置 `tracking.tensorboard.enabled=true`，保持 W&B 关闭；仅安装
`.[tensorboard]` 即可。TensorBoard 默认写入 `<run_dir>/tensorboard/`，可通过
`tracking.tensorboard.dir` 修改目录：

```bash
python -m dobot_nexus.cli.main tensorboard --logdir runs --port 6006
```

同时启用两项时，W&B 默认同步 TensorBoard event 文件；上例显式开启同步。
训练只调用 `SummaryWriter.add_scalar`，避免同一指标再次通过 `wandb.log` 写入。
TensorBoard 的 step 和 W&B 的 `global_step` 均为 optimizer update；W&B 图表使用
`global_step` 作为横轴。结束时先关闭并刷新 event writer，再结束 W&B run。
每次记录会立即 flush，首步、末步也会记录。横轴使用成功的 optimizer update，
梯度累积不会重复计步；精确续训时隐藏 checkpoint 之后未保存的旧曲线。
`tracking.tensorboard.flush_secs`（默认 10）和 `max_queue`（默认 10）控制 writer 缓冲。
可查看 `loss`、模型附加 loss、`lr`、`grad_norm`、`steps_per_sec`、`eta_seconds`，
CUDA 训练另记录主 rank 的 `system/cuda_allocated_mib`、`system/cuda_reserved_mib`。
默认监听 `127.0.0.1:6006`；监控远程训练时通过已有 SSH 转发或受控入口访问。
自定义实验输出目录时，将 `--logdir` 指向该目录。关闭记录可覆盖
`tracking.tensorboard.enabled=false`。

设置 `sync_tensorboard=false` 可分别向两个后端记录；只启用 W&B 时使用 `wandb.log`。
启用同步必须同时启用 TensorBoard。`offline` 在本地保留 W&B run，可之后用
`wandb sync /path/to/offline-run-...` 上传。

## 6. 训练启动方式

### 6.1 多卡 FSDP（正式训练推荐）

模板默认启用 FSDP2 FULL_SHARD，至少两张 GPU：

```bash
export PI05_BASE_CKPT=/path/to/pi05_base
export VLA_DATASET_PATH=/path/to/lerobot_dataset

examples/train.sh \
  --python .venvs/pi05/bin/python \
  --config examples/recipes/pi05_my_task.yaml \
  --gpus 0,1,2,3 \
  --run-name pi05_my_task_b2x4_g4_s10000 \
  --output-dir "$TRAINING_RUNS" \
  -- \
  data.batch_size=2 \
  data.num_workers=4 \
  train.grad_accum_steps=4 \
  train.max_steps=10000 \
  scheduler.total_steps=10000
```

有效全局 batch：

```text
data.batch_size × GPU/rank 数 × train.grad_accum_steps
```

上例是 `2 × 4 × 4 = 32`。`train.max_steps` 按 optimizer update 计数，因此每次 update 会消费
`GPU 数 × grad_accum_steps` 个 per-rank micro-batch。
Local 与 Accelerate runtime 均跨 epoch 累积：不足一个窗口时继续读取下一 epoch，
直到每个 rank 收满 `grad_accum_steps` 批后更新，不在 epoch 尾部提前缩小有效 batch。

不要随意修改 `transformer_layer_cls_to_wrap`、`use_orig_params` 或 mixed precision 字段；这些值
决定当前已验证的 FSDP 包装和优化器行为。

FSDP2 是 runtime 和训练模板的默认版本，验证环境为 Torch 2.10.0 / Accelerate 1.14.0。
大权重保存时可设置 `runtime.distributed_timeout_seconds=7200`，为进程组通信留出更多等待时间。
它作用于整个进程组，不只影响保存；无法修复死锁、失联或存储错误。
例如 rank 0 正在写入大型 Artifact，其他 rank 已进入 barrier 或 broadcast 等待，超过进程组
期限会导致训练退出。延长期限适用于确认仍在正常推进的慢 IO/通信。
`7200` 是两小时通信等待上限，不是训练总时长、保存间隔或推理超时。
该字段在 Accelerate 初始化时生效，运行中改 YAML 不会改变已创建的进程组。
超时后应先排查各 rank 的首个错误、OOM、磁盘空间和网络，再决定是否增大。
需要 FSDP1 时，在原启动命令末尾增加 `runtime.fsdp.version=1`，启动器和模型 target
保持不变。FSDP2 的 `full_shard` 对应 `reshard_after_forward=true`，`shard_grad_op` 对应 `false`；
支持范围不包含 `no_shard`、`hybrid_shard`、显式前向预取和 `backward_post`。

v2 使用 `MixedPrecisionPolicy`，保持 BF16 compute、FP32 reduction，检查模型浮点 buffer
已符合配置 dtype；v1 的 `use_orig_params`、`sync_module_states`、`limit_all_gathers` 和
`backward_prefetch` 不传入 v2 插件，并输出语义说明。模型仍由各 rank 以同一基础 seed 构建
并加载相同权重。v1/v2 训练快照不能直接精确互续，现有恢复指纹会拒绝版本变化。
实模精度、恢复和性能需要按目标模型与训练配置验证；本地实验记录保存在 `local_script/`。

### 6.2 单卡 local

显存足够时，使用同一份 YAML，将 GPU 列表改为 `--gpus 0`。模板中的
`runtime.launcher: auto` 和 `runtime.fsdp.enabled: auto` 会选择 local 并关闭 FSDP。
多卡自动选择 Accelerate + FSDP2，规则见[统一训练配置](getting_started/03_training.md#单卡和多卡共用配置)。
单卡的 batch size 按可用显存调整，保存与恢复规则见第 7 节。

### 6.3 日志和运行目录

训练开始后主要检查：

```bash
tail -f "$TRAINING_RUNS/<run-name>/logs/train.log"
sed -n '1,240p' "$TRAINING_RUNS/<run-name>/config_resolved.yaml"
```

日志中的 `loss`、`bc_loss`、`flow_loss`、`grad_norm` 和 `lr` 应为有限值。`config_resolved.yaml`
是环境变量、CLI override 和自动数据资产解析后的最终配置，排查问题时应以它为准。

训练 CLI 先初始化分布式 runtime，再由主 rank 生成实验目录并广播给所有 rank；
`exp.auto_suffix: true` 的后缀也只生成一次。Python 调用推荐 `BCTrainer(config)`；
自建实验对象时，须将已初始化的 runtime 传入 `Experiment.from_config(config, runtime=runtime)`。
随后用 `BCTrainer(config, experiment, runtime=runtime)` 复用同一 runtime。

新实验必须使用新 run name。`exp.auto_suffix: false` 时，重复使用已有 run name 且不 resume，
可能在保存同一步 checkpoint 时得到 `Checkpoint already exists`。

## 7. 断点续训

生产配置应保持 `train.resume_limit` 大于 0。发生中断后，使用相同配置、相同 run name、相同
output dir 和相同 world size：

```bash
examples/train.sh \
  --python .venvs/pi05/bin/python \
  --config examples/recipes/pi05_my_task.yaml \
  --gpus 0,1,2,3 \
  --run-name pi05_my_task_b2x4_g4_s10000 \
  --output-dir "$TRAINING_RUNS" \
  -- \
  data.batch_size=2 \
  data.num_workers=4 \
  train.grad_accum_steps=4 \
  train.max_steps=10000 \
  scheduler.total_steps=10000 \
  train.resume_from=auto \
  train.trust_resume_checkpoint=true
```

DCP 的 `.metadata` 使用 Python pickle；仅对自己生成且存储未被篡改的断点启用
`train.trust_resume_checkpoint=true`。默认拒绝读取 DCP 元数据，外部模型请使用部署权重
初始化新实验。训练状态 v3 的 RNG 使用受限加载格式；旧断点请重新导出部署权重。

`auto` 解析 `<run>/checkpoints/latest_resume.txt`。恢复包含模型、optimizer、scheduler、数据位置、
各 rank 的 Python/NumPy/Torch/CUDA RNG、数据 Generator 和 epoch 起点状态，能够从下一批数据继续。

恢复时不可改变以下语义：模型结构和 checkpoint、数据路径/字段/transform、batch size、optimizer、
scheduler、`exp.seed`、精度、FSDP 配置和 `grad_accum_steps`。未填写 `scheduler.total_steps` 时，
实际使用的 `train.max_steps` 也进入兼容性检查；若计划分段训练，应提前显式设置 schedule 总长度。
当前 Pi 确定性 Dataset 可以调整 `data.num_workers`。DCP 还要求恢复时 world size 与保存时一致。

精确续训要求 `training_state_version: 3`。scheduler 每次成功更新只推进一次，与 world size
无关。只需加载模型权重开始新实验时，可将 `deployment/pytorch/full_weights.pt` 作为
`model.checkpoint`，并指定新的 run name。

模型构建前各 rank 使用同一基础 seed；构建和 FSDP prepare 完成后，模型内采样使用
`(exp.seed + global_rank) % 2**32`。sampler 使用独立 Generator，并由 Accelerate 同步后分片；
worker 初始化使用另一 Generator，避免重建 worker 改变样本顺序。保存/导出不推进训练 RNG。
设置 seed 保证受控随机序列，不保证不同硬件或不同 world size 下逐位相等。第三方 Dataset 的
worker 内随机状态和外部随机源不属于精确续训保证范围。

`train.resume_limit: 2` 只保留最近两个 checkpoint 的可恢复 DCP 状态；较老 checkpoint 的部署
权重目录不会自动删除。`null` 保留全部恢复状态，`0` 禁用恢复状态。需要另外规划部署 Artifact
的归档和磁盘清理策略。

## 8. Artifact 与训练验收

每次 `save_every` 和最后一步都会生成 schema v2 Artifact：

```text
<run>/checkpoints/
  latest.txt
  latest_deployment.txt
  latest_resume.txt                    # resume_limit > 0 时存在
  step_00004000/
    manifest.yaml
    config_resolved.yaml
    deployment/
      pytorch/full_weights.pt
      memory/vision.npz                # 仅 Memory 模型
      assets/norm_stats.json
      assets/data_tokenizer_path.model
    resume/                            # resume_limit > 0 且尚未被裁剪
      dcp/
      trainer_state.pt
      rng/rank_00000.pt                # 每个 rank 的训练 RNG 和数据恢复状态
```

部署统一使用 `latest_deployment.txt`；继续训练使用 `latest_resume.txt`。训练完成后至少检查：

```bash
RUN="$TRAINING_RUNS/pi05_my_task_b2x4_g4_s10000"
ARTIFACT_NAME="$(tr -d '\r\n' < "$RUN/checkpoints/latest_deployment.txt")"
ARTIFACT="$RUN/checkpoints/$ARTIFACT_NAME"

test -f "$ARTIFACT/manifest.yaml"
test -f "$ARTIFACT/deployment/pytorch/full_weights.pt"
test -f "$ARTIFACT/deployment/assets/norm_stats.json"
test -f "$ARTIFACT/deployment/assets/data_tokenizer_path.model"
du -sh "$ARTIFACT"
```

Artifact 已携带推理需要的字段映射、action transform、normalization 和 tokenizer；推理端不需要
再访问原始 LeRobot 数据集。不要只复制某一个 `.pt` 文件，部署时应保留完整 Artifact 目录和
manifest。

Memory 的 `files.memory_vision` 指向复制并校验后的 `deployment/memory/vision.npz`，
在线历史编码需要这个文件。

## 9. 推理和服务部署

### 9.1 输入格式

离线 `.npz` 的公共输入为：

```text
image.base          uint8 [B,H,W,3] 或 [H,W,3]
image.left_wrist    uint8 [B,H,W,3] 或 [H,W,3]
image.right_wrist   uint8 [B,H,W,3] 或 [H,W,3]
state               float32 [B,state_dim] 或 [state_dim]
prompt              每个 batch item 一个非空字符串
```

运行时输入使用公共相机名。图像会保持宽高比 resize/pad 到
224×224；部署前仍应确认训练和机器人图像的裁剪、视角、颜色顺序一致。

### 9.2 PyTorch 推理

训推共用模型位于 `src/dobot_nexus/models/pi05/torch_core/`，相关模块职责如下：

| 路径（相对于 `models/pi05/`） | 职责 |
| --- | --- |
| `modeling_pi05.py` | 公共训推入口、模型构造与 Artifact hooks |
| `torch_core/pi0.py`、`pi0_config.py` | 模型组装、flow-matching loss、prefix/suffix 和 Euler 采样 |
| `torch_core/gemma.py`、`siglip.py`、`lora.py` | 语言/动作专家、视觉编码器及可选 LoRA 层 |
| `torch_core/model.py`、`augmentation.py` | 内部 Observation、模型预处理和训练图像增强 |
| `torch_core/numerics.py` | 训推共用的数值策略 |
| `torch_core/source.py`、`NOTICE.md` | 普通包加载和上游来源记录 |
| `training/compile.py`、`checkpoint.py` | 训练小算子编译配置、权重加载/转换 |
| `openpi_torch/engine.py`、`legacy_ops.py`、`openpi_ops.py` | 推理编译、CUDA Graph 和两种数值策略的精度敏感算子边界 |
| `openpi_torch/conditioning.py`、`vision.py` | 原生时间条件投影缓存、冻结历史视觉的 Graph 回放 |

Pi0.5 加载本地 `torch_core/`，定制模型通过 `model.target` 接入。
模型导入时不执行 `torch.compile`，也不临时替换全局 `torch.compile`。训练的
`runtime.compile` 在模型构造时按实例选择 RoPE、gated residual、GELU-GLU 的编译版本；
推理实例从 eager 算子开始，再由推理引擎统一加速，两种配置可在同一进程内共存。
`pi0.py` 的文件名沿用上游架构名称，当前 Pi0.5 配置会设置 `pi05=True`。

Memory 和 RECAP 都经由 `Pi05Model` 的构造、训练和推理入口调用这套 core：

- Memory 的 `_configure_core()` 先调用基类配置 `openpi` 数值策略，再安装历史编码器和
  action expert 的 memory modulation。历史 Attention 复用所属模型选定的 RoPE callable。
  `MemoryDataAdapter` 生成的 `model_kwargs.memory_inputs` 经基础 adapter 搬运设备后，
  由 `compute_loss` / `sample_actions` 接收，历史特征每次请求刷新，prefix 不依赖历史。
- RECAP 通过 `_build_adapter()` 一次构造条件 adapter；训练只替换条件 observation，
  保留其余 batch 参数。推理的 base/positive/negative 条件先转换为 prompt/token，再调用
  相同 Torch 引擎。Value 网络仍独立使用 `modules/value/fusion`。
- 两个扩展从克隆配置构造，构建训练 DataLoader 时会把解析后的 `operator_numerics`、
  `max_token_len` 和训练 recipe 写回 Trainer 配置，保持 Artifact 配置与元数据一致。
  推理从 Artifact 恢复数值策略与 token 长度，并校验配置、元数据与显式覆盖是否一致。

RECAP 推理样例默认使用 Torch，和 BC 一样启用 `compile_mode=legacy` 与 CUDA Graph；
base/positive/negative 条件更新会刷新输入 token，保持训练的 legacy 数值策略。
`compile_mode=legacy` 只适用于 legacy 数值策略；Memory 使用 `openpi` 策略，推理样例默认
`runtime.compile=true runtime.compile_mode=openpi runtime.cuda_graph=true`。`openpi` 模式
复用 BC 的编译流程和原生算子边界，保留 FP32 输入投影的 dtype 提升、reciprocal(sqrt)
归一化及 Memory 算子的舍入顺序。Memory 每层历史 K/V 仅在每次预测开始时
计算一次，所有去噪步复用；这些计算位于 Graph 内，回放时读取本次历史，不跨请求缓存历史。
两种策略都在 Graph 捕获前缓存只依赖时间步和权重的 AdaRMSNorm 投影。
关闭 compile 可以单独使用原生 CUDA Graph。通用 `default`、`max-autotune-no-cudagraphs`
需要单独验证数值结果；CPU 运行需关闭 compile 和 cuda_graph。

Memory 服务使用 `memory/torch_vision.py`，直接加载 Artifact 内的
`deployment/memory/vision.npz`，复用本地 `torch_core/siglip.py`，按 SHA256 校验冻结视觉权重。
该编码器与策略视觉权重独立，保持 FP32 stem/归一化参数、BF16 Transformer、浮点归一化后
抗锯齿缩放及 4×4 空间池化。
服务将同一个 `runtime.cuda_graph` 开关传给冻结历史视觉编码器，编码和池化按实际 batch
大小捕获；图像预处理仍在图外。它与动作采样各自维护有上限的 Graph 缓存，不填充重复帧，
首次遇到新的 batch 大小会进行 warmup 和捕获。编码器权重在创建后保持冻结。

使用完整 Memory Artifact 启动 Torch 历史服务：

```bash
MEMORY_ARTIFACT=/path/to/memory/artifact PYTHONPATH=src:. \
  python -m dobot_nexus.models.pi05.deploy.memory_server \
  examples/configs/infer/pi05_memory.yaml --port 12021
```

服务依赖 Pi0.5 Torch 环境和 `.[websocket]` 中的 WebSocket/msgpack 传输包。
Memory 输入需要会话维护历史，完整操作见 [Memory 使用指南](pi05_memory.md)。

训练环境中可以直接回载 Artifact。标准推理配置及 WebSocket 服务默认使用 Torch；
下面的命令显式写出 backend，便于核对部署配置：

```bash
examples/infer.sh \
  --model pi05 \
  --python .venvs/pi05/bin/python \
  --artifact /path/to/run/checkpoints/latest_deployment.txt \
  --input /path/to/observation.npz \
  --output /path/to/actions.npz \
  --gpu 0 \
  --device cuda \
  -- \
  model.inference_backend=torch \
  inference.num_steps=10
```

输出 `actions` 的形状是 `[B, action_horizon, action_dim]`，并已经执行训练配置对应的 action
反变换。

通用 Pi0.5 推理模板中 `state_dim`、`action_dim`、`action_horizon` 为 `null`，从 Artifact
恢复；内部 `model_action_dim` 同样从 Artifact 恢复。显式覆盖必须与训练值一致。
不使用 Artifact、直接指定 `model.checkpoint` 时，必须明确填写公共维度、动作长度和数据资产；
内部边界省略时为 32。直接权重路径不含任务语义，推荐使用完整 Artifact。
policy 服务按恢复的维度接收状态并返回动作。`examples/pi05_real_robot/` 的硬件客户端要求
14 维，会在 warmup 和机器人连接前拒绝不匹配的服务；KBD 真机需要匹配其硬件协议的客户端。

标准 `examples/configs/infer/pi05.yaml` 使用 `legacy` 数值策略，默认开启编译与 CUDA Graph。
推理从 Artifact 恢复输入变换和动作反变换。

| 配置 | 标准推理 YAML 值 | 行为 |
| --- | --- | --- |
| `runtime.compile` | `true` | 编译动作专家的一次去噪计算，各采样步复用编译结果 |
| `runtime.compile_mode` | `legacy` | 保留原生矩阵乘、归约、激活及敏感 FP32 舍入边界，编译其余张量操作；前缀保留原生计算 |
| `runtime.compile_preserve_precision` | `true` | 保留 BF16/FP16 中间舍入；legacy 模式必须开启 |
| `runtime.cuda_graph` | `true` | 捕获完整采样过程，后续请求复制新输入并回放；要求 CUDA 和 eval 模式 |
| `runtime.cuda_graph_max_entries` | `2` | 按输入形状、dtype、采样步数缓存 Graph，超过容量淘汰最久未使用项 |

可以单独开启 CUDA Graph，保留 eager 算子运算。首次请求包含编译/调优或 Graph 捕获开销，
应在服务接收请求前 warmup；输入 shape、dtype 或采样步数变化会触发新的捕获。
编译参数已在 PyTorch 2.10/CUDA 12.8 上验证。编译失败会直接报错，不会静默退回 eager。
CPU 推理需覆盖 `runtime.compile=false runtime.cuda_graph=false`。

legacy 模式复用训练算子的计算顺序，不修改训练前向、反向或参数结构。RoPE 的 sin/cos
在各层之间共享；AdaRMSNorm 条件投影仅依赖权重和时间步，因此在 Graph 捕获前按原始
batch 大小逐步计算并缓存。它不会缓存观测、噪声或去噪轨迹。
legacy 模式仅支持无 Memory 的 Pi0.5 legacy 算子配置；`operator_numerics=openpi`
及 Memory 模型使用独立的 `compile_mode=openpi`，两种模式都会拒绝不匹配的数值策略。
通用编译配置 `default`、`max-autotune-no-cudagraphs`
仍可显式选择，但仅设置 `compile_preserve_precision=true` 不能保证它们与 eager 对齐。

启用加速后，应使用同一组观测和外部噪声比较 eager 与加速输出，并记录硬件、软件版本、
数值策略、batch/horizon 和采样步数。实模精度、延迟及真机成功率需分别验证；
测试脚本和报告放在 `local_script/`，历史测量值不作为当前环境的性能保证。

外部 `SamplingOptions.noise` 和模型内部生成的噪声均使用 FP32。
比较不同加速设置的输出时，应提供同一份噪声数组。
返回的 `model_ms` 包含 GPU 完成及结果回传时间，不只计 CPU 发射 Graph 的时间。

也可直接启动 Torch WebSocket policy 服务：

```bash
PYTHONPATH=src python -m dobot_nexus.models.pi05.deploy.server \
  --artifact /path/to/run/checkpoints/latest_deployment.txt \
  --backend torch --gpu 0 --device cuda \
  --torch-compile --compile-mode legacy --cuda-graph --num-steps 10 --port 8000
```

模型切换训练模式、移动设备或通过 `Pi05Model.load_state_dict` 恢复参数时会清理捕获和条件缓存。
若应用直接修改 `core_model` 权重（包括原地修改）或替换其 storage，
应先调用 `model.reset_inference_acceleration()`。

### 9.3 Nexus HTTP 服务

安装 `serve` extra 后启动：

```bash
examples/infer.sh \
  --model pi05 \
  --python .venvs/pi05/bin/python \
  --artifact /path/to/run/checkpoints/latest_deployment.txt \
  --serve \
  --host 127.0.0.1 \
  --port 12021 \
  --gpu 0 \
  --device cuda \
  -- \
  model.inference_backend=torch \
  inference.num_steps=10
```

接口为 `GET /healthz`、`GET /metadata`、`POST /predict`、`POST /act`、`POST /reset`。
`POST /sessions` 创建独立动作缓冲并返回随机 `session_id`；`/act` 和 `/reset` 请求必须携带
该字段，任务或环境变化前需要 reset。`DELETE /sessions/{session_id}` 释放会话。默认最多
16 个会话，空闲 15 分钟过期，请求体最多 8 MiB，`num_steps` 范围为 1–100。
默认绑定回环地址；非回环绑定必须设置 32 字节以上的 `NEXUS_POLICY_TOKEN`，
请求使用 `Authorization: Bearer <token>`（`/healthz` 除外）。对外服务需部署在提供 TLS 的网关后。
系统设置 HTTP 代理时，要把
`127.0.0.1`、`localhost` 和服务内网 IP 加入 `NO_PROXY/no_proxy`。

### 9.4 Torch 真机部署

`examples/pi05_real_robot/start_policy_server.sh` 使用 Torch 后端并读取 Artifact 的
`files.weights`。训练环境额外安装 `.[websocket]` 的 WebSocket/msgpack 协议依赖即可部署。

双臂真机使用 WebSocket/msgpack_numpy 链路。完整进程拓扑、
启动、dry-run 和正式执行参数见 [`examples/pi05_real_robot/README.md`](../examples/pi05_real_robot/README.md)。

## 10. 正式训练验收

- `config_resolved.yaml` 中数据路径、字段、action codec 和维度正确。
- 自动统计的 fingerprint 与当前数据和 action 语义一致。
- 训练 loss 和 gradient norm 有限，学习率符合 schedule。
- 每个 rank 的 GPU 利用率、显存和 dataloader 吞吐正常。
- `latest_resume.txt` 指向可恢复 checkpoint，并实际演练过一次 resume。
- `latest_deployment.txt` 的 PyTorch 权重、stats、tokenizer 齐全；Memory 还需冻结视觉权重。
- 使用真实相机/state 样本完成离线推理和服务请求。
- 在真机执行前单独验证 state/action 顺序、单位、限幅、急停和 dry-run。

## 11. 调参与故障排查

### 11.1 OOM

建议依次处理：

1. 降低 `data.batch_size`。
2. 增加 `train.grad_accum_steps` 保持有效全局 batch。
3. 保持 `train.gradient_checkpointing: true`。
4. 使用更多 GPU 的 FSDP，而不是在单卡关闭 FSDP。
5. 最后再评估 `runtime.fsdp.cpu_offload: true`；它会明显增加 CPU/PCIe 开销。

`data.num_workers` 主要影响 CPU、内存和 IO，不会直接解决模型显存不足。

### 11.2 常见错误

- `Config enables FSDP and requires at least two processes`：显式开启了 FSDP，但只启动一个进程；
  增加 GPU，或使用模板的 `runtime.launcher=auto runtime.fsdp.enabled=auto`。
- `missing configured fields`：`data.field_mapping` 与 `meta/info.json` 不一致。
- `No task prompt was found`：补齐 LeRobot task metadata，或设置 `data.default_prompt`。
- `delta_mask must contain ...`：mask 数量必须等于 `model.action_dim`；delta 还要求 state/action
  维度相同。
- `Checkpoint already exists`：新实验换 run name；继续已有实验则使用 `train.resume_from=auto`。
- `Resume checkpoint is incompatible`：恢复时修改了数据、optimizer、scheduler、精度、FSDP 或
  gradient accumulation 语义。回到保存 checkpoint 时的 resolved config。
- `Exact training resume requires the same world size`：使用与保存时相同数量的进程/GPU。
- Artifact 保存失败或磁盘暴涨：每次保存都包含大型部署权重，是否另存 wrapper 权重由模型 hook
  决定。调大 `save_every` 并检查训练盘空间；`resume_limit` 不会删除旧部署权重。
- 本地 HTTP/WebSocket 超时：检查端口占用，并配置 `NO_PROXY/no_proxy` 绕过全局代理。

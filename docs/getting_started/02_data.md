# 2. LeRobot v2.1 格式与数据转换

[新手路线](../../README.md#从哪里开始) · 上一步 [环境](01_environment.md) · 下一步 [训练](03_training.md)

本篇以 Dobot 双臂三相机数据为例。已有 LeRobot v2.1 数据可以阅读格式说明后执行第 2.4 节检查。
原始 Dobot recorder PKL 按第 2.3 节转换。
18 维叠衣服数据的字段和使用步骤见第 2.5 节。

### 根据手里的数据选择入口

| 手里的数据 | 当前入口 | 需要确认 |
|---|---|---|
| LeRobot v2.1，Parquet 内嵌 JPEG/PNG | 第 2.4 节直接检查 | metadata、向量维度、相机字段 |
| LeRobot v2.1，Parquet + MP4 | 第 2.4 节直接检查 | `video_path`、视频解码依赖和时间戳 |
| Dobot recorder，每个 PKL 保存整条轨迹 | 第 2.3 节转换 | 左右各 7 维，三路相机和逐帧 timestamp |
| 叠衣服采集，每条轨迹含逐帧 PKL 和 JPG 目录 | 第 2.5 节使用已转换的数据 | 从原始目录开始需要对应的格式转换器 |

训练 reader 的向量宽度来自数据，Pi0.5 配置中的 `state_dim`、`action_dim` 应分别与之匹配。
第 2.3 节转换器的输入格式固定为左右各 7 维。`data.field_mapping` 用于映射已经转换好的
LeRobot 列名，原始文件的解析由转换器负责。

## 2.1 一条训练数据里有什么

一次完整任务尝试叫一个 episode。每个 episode 由连续帧组成。

| 数据 | 示例 | 含义 |
|---|---|---|
| 图像 | 顶部、左腕、右腕 RGB 图像 | 机器人当前看到了什么 |
| 状态 | 14 维关节和夹爪状态 | 机器人当前处于什么姿态 |
| action | 14 维目标关节和夹爪命令 | 在这份观测后采取了什么动作 |
| task | `Pick up the red cube.` | 这次演示要完成什么任务 |

本示例的向量顺序为左臂 6 个关节、左夹爪、右臂 6 个关节、右夹爪。
关节使用 rad，夹爪范围沿用采集硬件协议。训练和部署使用相同顺序、单位和夹爪范围。
转换器逐帧保存记录的 state/action，选择哪些演示用于 BC 由数据整理者决定。

## 2.2 v2.1 的目录和字段

`data.path` 指向直接包含 `meta/` 和 `data/` 的数据集根目录。
本仓库转换器输出如下布局。

```text
my_task_v21/
├── meta/
│   ├── info.json                 # 版本、FPS、总帧数、字段 dtype/shape
│   ├── episodes.jsonl            # 每条轨迹的编号、长度、任务
│   ├── tasks.jsonl               # task_index 对应的任务文本
│   ├── stats.json                # 原始 state/action 统计
│   ├── sources.jsonl             # 来源文件、摘要、原始时间间隔
│   └── conversion.json           # 本次转换配置、ROI、图像编码
└── data/
    └── chunk-000/
        ├── episode_000000.parquet
        └── episode_000001.parquet
```

Parquet 是按列保存数据的二进制表格，一个文件对应一条 episode，每行对应一帧。
转换器每 1000 条 episode 使用一个 `chunk-NNN` 目录。

| Parquet 列 | 类型与形状 | 说明 |
|---|---|---|
| `observation.state` | `float32[14]` | 当前机器人状态 |
| `action` | `float32[14]` | 该观测后的目标命令 |
| `observation.images.top_camera` | image | 顶部图像 |
| `observation.images.left_camera` | image | 左腕图像 |
| `observation.images.right_camera` | image | 右腕图像 |
| `episode_index` | int64 | 轨迹编号 |
| `frame_index` | int64 | 轨迹内部帧号，从 0 开始 |
| `index` | int64 | 整个数据集的帧号 |
| `timestamp` | float32 | 轨迹内部时间，转换后为 `frame_index / fps` 秒 |
| `task_index` | int64 | 对应 `tasks.jsonl` 的指令 |
| `source_timestamp` / `source_frame_index` | float64 / int64 | 原始记录时间 / 帧号 |
| `is_success` / `is_human` | bool 或 null | 已记录的成败 / 人工操作信息 |

`episodes.jsonl` 与 `tasks.jsonl` 每行分别类似下面的内容。

```json
{"episode_index": 0, "tasks": ["Pick up the red cube."], "length": 300}
```

```json
{"task_index": 0, "task": "Pick up the red cube."}
```

### 图像有两种保存方式

- **内嵌图像**。本转换器在 Parquet 中保存 PNG bytes，相机在 `info.json.features` 中的
  `dtype` 为 `image`，图像为 `224×224×3`。Nexus 的共享 reader 直接读取这种布局。
- **视频图像**。已有 v2.1 数据也可以把相机声明为 `video`，像素保存在
  `videos/chunk-000/observation.images.top_camera/episode_000000.mp4` 等文件中。
  `info.json` 的 `video_path` 给出路径模板，reader 按 episode 和 timestamp 解码。

单个数据集的相机应统一使用一种存储方式。上游 v2.1 数据还可能带有
`episodes_stats.jsonl`。本转换器产出上图的文件，Nexus 按这些文件读取内嵌图像。
更完整的视频布局示例见[格式参考](../pi05_training.md#0-训练前置说明lerobot-v21-格式规范)。

### FPS 和动作长度

`fps` 应对应实际采集和执行频率。20 Hz 表示相邻步骤相隔约 0.05 秒。
转换器按现有记录生成索引时间，**保持原始步数**，并在 `sources.jsonl` 保存原始间隔统计。
更改配置中的 FPS 后，仍需确认记录的真实时间间隔与目标频率一致。

训练时，reader 从当前帧取未来 `action_horizon` 步动作，形成 action chunk。
Pi0.5 模板默认 50 步。episode 尾部通过重复最后一步补齐，窗口始终属于同一 episode。

## 2.3 把 Dobot recorder PKL 转换成 v2.1

### 先确认输入格式

转换器读取受信任的本地 PKL。每个文件包含一个非空 `list[dict]`，每个 dict 对应一帧。
Python pickle 加载会执行其序列化协议，因此来源应为自己的采集流程或已确认可信的数据。

| PKL 键 | 内容 |
|---|---|
| `timestamp` | 有限数值，同一轨迹内严格递增 |
| `left_dobot_arm_joints` / `right_dobot_arm_joints` | 每侧 7 维观测状态 |
| `action_left_dobot_arm_joints` / `action_right_dobot_arm_joints` | 每侧 7 维目标动作 |
| `top_camera_frame` / `left_camera_frame` / `right_camera_frame` | JPEG bytes 或编码字节数组 |
| `task` | 可选，可由转换 YAML 中的 `task` 统一提供 |
| `episode_success` / `is_human` | 可选成败 / 人工标签 |

目录输入查找该目录直属的 `*.pkl`。多个来源目录可写成多个 `sources` 条目。
其他机器人或其他原始格式需要先适配以上字段与动作语义。

### 创建转换环境与配置

在仓库根目录执行。后续用完整解释器路径区分数据环境和训练环境。

```bash
python3.11 -m venv .venvs/data
.venvs/data/bin/python -m pip install -r data_processing/requirements.txt
mkdir -p runs/quickstart
cp data_processing/example.yaml runs/quickstart/convert.yaml
```

编辑 `runs/quickstart/convert.yaml`。下面是使用原始全图、20 Hz 记录的完整基础示例。

```yaml
fps: 20
image_size: 224
top_roi_xywh: null
require_success: false
sources:
  - path: /path/to/my_recordings
    task: "Pick up the red cube."
```

将路径和任务文本改成自己的数据。`require_success: false` 允许 BC 数据缺少成败标签，
已知成败仍会保留，缺失结果为 null。RECAP 的严格标签规则和扩展字段见[转换参考](../../data_processing/README.md)。

明确填写 `top_roi_xywh`。全图写 `null`，裁剪写 `[x, y, width, height]`，
例如 `[145, 30, 400, 400]`。腕部使用全图，三路图像等比缩放补边至 224。
真机端需要使用相同裁剪，原始输入应对应这份配置使用的相机画面。

```bash
export VLA_DATASET_PATH=/path/to/datasets/my_task_v21
.venvs/data/bin/python -m data_processing.convert \
  --config runs/quickstart/convert.yaml \
  --output "$VLA_DATASET_PATH"
```

输出目录应是一个尚未存在的新路径。成功后会显示 `Wrote ...; recorded steps preserved`，
目录中出现 `meta/` 和 `data/`。转换过程校验数据并回读 Parquet 后发布完整目录。

## 2.4 用训练环境检查数据

切回 Pi0.5 环境。已有数据也从这里开始，填写自己的数据集根目录。

```bash
source .venvs/pi05/bin/activate
export VLA_DATASET_PATH=/path/to/datasets/my_task_v21

python - <<'PY'
import json
import os
from pathlib import Path
import numpy as np
from dobot_nexus.data.images import convert_image
from dobot_nexus.data.lerobot.source import build_source

root = Path(os.environ["VLA_DATASET_PATH"])
info = json.loads((root / "meta/info.json").read_text())
features = info["features"]
state_shape = tuple(features["observation.state"]["shape"])
action_shape = tuple(features["action"]["shape"])
assert len(state_shape) == len(action_shape) == 1
camera_keys = [key for key, feature in features.items()
               if feature["dtype"] in {"image", "video"}]
assert camera_keys, "metadata 中需要声明相机字段"
print("版本 / FPS", info["codebase_version"], info["fps"])
print("轨迹 / 帧数", info["total_episodes"], info["total_frames"])
print("state / action 维度", state_shape[0], action_shape[0])
source = build_source(data_path=str(root), action_key="action", language_key="task",
                      action_horizon=50, format_version_name="v2.1",
                      fps=None, tolerance_s=1e-4)
assert len(source) == info["total_frames"]
sample = source[0]
assert np.asarray(sample["observation.state"]).shape == state_shape
assert np.asarray(sample["action"]).shape == (50, *action_shape)
assert np.isfinite(np.asarray(sample["observation.state"])).all()
assert np.isfinite(np.asarray(sample["action"])).all()
assert isinstance(sample["prompt"], str) and sample["prompt"].strip()
for key in camera_keys:
    image = convert_image(sample[key])
    print(key, image.shape, image.dtype)
print("任务", sample["prompt"])
print("第一帧读取成功，action chunk", np.asarray(sample["action"]).shape)
PY
```

这一步验证 metadata、第一帧图像解码和动作块读取，按 metadata 自动识别向量维度和相机。
若 state/action 使用其他列名，同步修改检查代码中的键和下一篇的 `data.field_mapping`。
将打印出的维度填入 `model.state_dim/action_dim`，将需要使用的相机列名填入相机映射。
完整训练还会扫描统计并读取其他帧。

`meta/stats.json` 记录转换时的原始数值统计。Pi0.5 训练根据自己的动作变换和 chunk 重新生成
归一化统计，并将对应资产打包进 Artifact。新手配置保留自动统计，继续[创建训练任务](03_training.md)。

## 2.5 使用 18 维叠衣服数据

`collect_data` 的任务文本是 `Fold the T-shirt with both arms.`。这份已转换数据包含
100 条轨迹、135637 帧，索引频率为 20 Hz。先取得完整数据集，将下面的占位路径
替换为所在主机的数据集根目录。

```bash
export VLA_DATASET_PATH=/path/to/datasets/collect_data
test -f "$VLA_DATASET_PATH/meta/info.json"
```

设置变量后运行第 2.4 节的 Python 检查。预期打印 state/action 均为 18 维，
动作块为 `(50, 18)`，以及下面四路相机。

| 数据列 | 原始图像形状 H×W×C | Pi0.5 使用方式 |
|---|---|---|
| `observation.images.top_left` | `720×1280×3` | `base` |
| `observation.images.top_right` | `720×1280×3` | 数据保留该相机，策略选用另外三路 |
| `observation.images.wrist_left` | `480×640×3` | `left_wrist` |
| `observation.images.wrist_right` | `480×640×3` | `right_wrist` |

这份数据在 Parquet 内嵌原始 JPEG，Pi0.5 数据管线将选中的三路画面等比缩放补边到 224。
state/action 的 18 个元素保持采集数组的顺序，动作变换使用 `absolute`，数值归一化使用
`quantile`。各元素的物理含义和单位以采集协议为准。完整配置见[叠衣服训练](03_training.md#38-训练-18-维叠衣服任务)。

### 从叠衣服原始采集开始的要求

原始采集的 `collect_data` 根目录下，每个子目录是一条轨迹。

```text
collect_data/<episode>/
├── observation/0.pkl, 1.pkl, ...   # 每个 dict 包含 obs 和 action，各 float32[18]
├── top_left/0.jpg, 1.jpg, ...
├── top_right/0.jpg, 1.jpg, ...
├── wrist_left/0.jpg, 1.jpg, ...
├── wrist_right/0.jpg, 1.jpg, ...
└── trajectory.txt
```

这种原始布局需要专用转换器，负责对齐同帧 PKL/JPG、保留 18 维向量和四路相机，
并生成 episode、task、时间索引和 v2.1 metadata。仓库发布的 `data_processing.convert`
目前提供第 2.3 节的 recorder 输入适配。这项原始布局的转换入口仍待纳入发布代码，
本节训练步骤以完整的已转换 LeRobot 数据为前提。

# Pi0.5 Memory 训练与推理操作指南

流程：准备数据和冻结视觉权重 → 生成历史特征 → 训练 → 启动 policy → 连接机器人。
训练、历史编码和推理均使用 Torch。

本文使用 14 维双臂配置。除机器人端命令外，均在 Nexus 仓库根目录执行；先替换
`/path/to/...`、GPU 编号和机器人 IP。

## 1. 准备 Python 环境

首次使用时创建包含 WebSocket 服务依赖的 Pi0.5 环境：

```bash
python3.11 scripts/env/create_model_env.py --model pi05 --extra websocket
export NEXUS_PYTHON="$PWD/.venvs/pi05/bin/python"
export PYTHONPATH="$PWD/src:$PWD"
"$NEXUS_PYTHON" scripts/env/check_model_env.py --model pi05 --extra websocket
```

已有环境时跳过创建命令，将 `NEXUS_PYTHON` 改为该环境的 Python 路径，再执行检查。
新开终端时重新设置这两个环境变量。安装细节见[环境说明](environments.md)。

## 2. 准备训练资源

```bash
export PI05_BASE_CKPT=/path/to/pi05_base_torch
export VLA_DATASET_PATH=/path/to/lerobot_train_split
export MEMORY_VISION_CHECKPOINT=/path/to/frozen_vision.npz
export MEMORY_FEATURES=/path/to/memory_features/train_v1
export MEMORY_DATASET_ID=my_task_train_v1
```

| 变量 | 填写内容 |
|---|---|
| `PI05_BASE_CKPT` | Pi0.5 初始 Torch 权重目录，例如包含 `model.safetensors` 的目录 |
| `VLA_DATASET_PATH` | 已划分好的 LeRobot 训练集根目录，直接包含 `meta/`、`data/` |
| `MEMORY_VISION_CHECKPOINT` | 固定历史编码器的 `vision.npz`；可使用已有 Memory Artifact 的 `deployment/memory/vision.npz` |
| `MEMORY_FEATURES` | 历史特征输出目录，必须不存在或为空 |
| `MEMORY_DATASET_ID` | 此训练划分的固定版本标识；更改数据或重建资源时更新 |

`vision.npz` 必须提前准备好；本教程的特征准备命令不会从 policy 权重生成它。
保留同一份冻结编码器用于特征准备和训练，部署时会将它复制进 Artifact。

通用配置为 [pi05_memory.yaml](../examples/configs/train/pi05_memory.yaml)，继承
[pi05.yaml](../examples/configs/train/pi05.yaml)。默认读取：

| 内容 | LeRobot 字段 |
|---|---|
| 顶部图像 | `observation.images.top_camera` |
| 左腕 / 右腕图像 | `observation.images.left_camera` / `observation.images.right_camera` |
| 状态 / 动作 / 任务 | `observation.state` / `action` / `task` |

状态、动作均为 14 维，动作使用绝对值。每个 episode 的 `frame_index` 从 0 连续编号。
字段或动作表示不同的任务，应先编写继承通用配置的 recipe，后续准备与训练使用同一配置。
原始 PKL 的转换方法见[数据处理说明](../data_processing/README.md)；准备命令不会划分训练集和验证集。

## 3. 生成历史特征

```bash
CUDA_VISIBLE_DEVICES=0 "$NEXUS_PYTHON" -m dobot_nexus.models.pi05.memory.prepare \
  --config examples/configs/train/pi05_memory.yaml \
  --device cuda:0 --batch-size 16
```

完成后打印 `episodes` 和 `frames` 数量，输出目录包含 `manifest.json` 和每个 episode 的
`episode_*.npy`。这里的 `--batch-size` 只控制视觉特征提取，不改变训练 batch。

数据、prompt、相机映射或冻结编码器发生变化后，使用新的 `MEMORY_FEATURES` 和
`MEMORY_DATASET_ID` 重新准备。命令拒绝写入非空目录；中断后的不完整目录也不能直接续写。

## 4. 训练与续训

使用两张 GPU 启动通用 Memory 训练：

```bash
bash examples/train.sh \
  --config examples/configs/train/pi05_memory.yaml \
  --python "$NEXUS_PYTHON" --gpus 0,1 \
  --run-name pi05_memory_my_task --output-dir runs
```

首次执行可在命令末尾加 `--dry-run` 检查启动参数；去掉后才会训练。

| 默认配置 | 值 / 含义 |
|---|---|
| 训练方式 | BC，增加 Memory 历史输入；FSDP，至少两张 GPU |
| Batch | 每卡 16，梯度累积 1；上述两卡命令有效 batch 为 32 |
| 训练 / 保存 | 20,000 步，每 2,000 步保存；结束时也会保存 |
| 动作 / 归一化 | absolute / quantile；未指定统计文件时从训练集自动计算 |
| 历史 | 顶部相机；每次最多采样 32 帧、共 512 token；完整历史容量 8192 帧 |

显存不足时，在启动命令末尾添加 `-- data.batch_size=4 train.grad_accum_steps=4`，
上述两卡配置的有效 batch 仍为 32。修改总步数时同时设置
`train.max_steps`、`scheduler.total_steps`，并相应调整 `scheduler.warmup_steps`。

中断后，使用原数据、配置、GPU 数量和 run 名称续训：

```bash
bash examples/train.sh \
  --config examples/configs/train/pi05_memory.yaml \
  --python "$NEXUS_PYTHON" --gpus 0,1 \
  --run-name pi05_memory_my_task --output-dir runs \
  -- train.resume_from=auto train.trust_resume_checkpoint=true
```

若首次训练加过 batch 等覆盖参数，续训时一并保留。`auto` 读取
`checkpoints/latest_resume.txt`，需要对应 checkpoint 的恢复状态仍完整。

按钮任务已有 [pi05_button_memory.yaml](../examples/recipes/pi05_button_memory.yaml)：
四卡、有效 batch 64、1990 步、每 995 步保存、EMA 0.99。使用它时，特征准备和训练都改用
该 recipe，训练设置 `--gpus 0,1,2,3`，并提前设置
`MEMORY_NORM_STATS=/path/to/train_norm_stats.json`（训练划分的 `q01/q99` 统计）。
该配方与按钮 BC 的动作表示和训练预算不同，不能直接作为消融对照；详见[配方说明](../examples/recipes/README.md)。

## 5. 启动 Torch policy

上述通用训练产物位于 `runs/pi05_memory_my_task/checkpoints/`。每个部署 Artifact 至少保留：

```text
step_00002000/
├── manifest.yaml
├── config_resolved.yaml
└── deployment/
    ├── pytorch/full_weights.pt
    ├── memory/vision.npz
    └── assets/
        ├── norm_stats.json
        └── data_tokenizer_path.model
```

部署不需要训练数据、离线历史特征或 `resume/`，但不能只复制 `.pt`。
推理时由冻结编码器处理机器人上传的图像，并为每条连接维护历史。

在 GPU 主机启动服务；`GPU=0` 改为实际可用的卡：

```bash
POLICY_PYTHON="$NEXUS_PYTHON" GPU=0 POLICY_PORT=8001 \
bash examples/real_robot/start_policy_server.sh \
  runs/pi05_memory_my_task/checkpoints/latest_deployment.txt
```

也可直接传入指定的 `step_XXXXXXXX` 目录。启动器根据 Artifact 自动识别 Memory，
默认 BF16、10 步去噪，开启预热、`torch.compile` 和 CUDA Graph。
Memory 使用 `compile_mode=openpi`；这是本地 Torch 数值配置名称。
排查加速问题时，在同一启动命令末尾追加 `--no-torch-compile --no-cuda-graph`。

在另一个终端检查服务：

```bash
curl --noproxy '*' http://127.0.0.1:8001/healthz
```

Memory 的连续推理使用下节的历史桥接链路；普通单帧 `examples/infer.sh` 不负责维护历史会话。

## 6. 连接机器人

三个进程的部署位置：

```text
机器人主机：robot-server :12006 ← TCP → GPU 主机：client ← WebSocket → policy :8001
```

### 机器人主机

完整复制 `examples/real_robot/dobot_dual_arm_14d/`，保留 `_runtime/`，在复制目录的父目录执行。
机器人使用已有硬件环境，串口、机械臂 IP 和相机序列号按[真机部署说明](../examples/real_robot/dobot_dual_arm_14d/README.md)配置。
以下以 **30 Hz、顶部全图**的按钮任务为例；任务文本、频率和裁剪应与训练数据一致。

机器人环境需要安装 `msgpack`。在机器人服务终端和 GPU client 终端，分别通过密钥管理工具
或安全交互输入导出**相同的** `NEXUS_ROBOT_TOKEN`（至少 32 字节随机密钥）。新开终端时
重新注入该密钥，不要在两端各自生成不同的值，也不要写入配置或 Git。
生成与分发要求见 [Robot TCP 认证](../examples/real_robot/dobot_dual_arm_14d/README.md#robot-tcp-认证)。

```bash
: "${NEXUS_ROBOT_TOKEN:?请先注入与 GPU client 相同的机器人认证密钥}"
export ROBOT_PYTHON=/path/to/robot-env/bin/python
export ENV_SRC=/path/to/robot/environment/source
export ROBOT_BIND_HOST=ROBOT_IP
export TASK='press the yellow button, then press the red button, and then press the yellow button.'
export TOP_ROI_XYWH=none
export BRIDGE_MODE=memory
export STEP_DT=0.0333333333 CONTROL_FREQUENCY=30
export CAMERA_FPS=30 HISTORY_FPS=30
export ACTIONS_PER_RESPONSE=8 ROBOT_PORT=12006
ROBOT_EXECUTE=0 bash dobot_dual_arm_14d/start_dobot_dual_arm_14d_server.sh
```

此时采集图像和状态、接收预测，不执行模型动作。若训练使用顶部裁剪，设置
`TOP_ROI_XYWH='X Y W H'`；当前真机 Memory 桥接只传顶部相机历史。
`ROBOT_IP` 必须替换为机器人主机上供 GPU 主机访问的网卡地址；默认回环地址仅用于同机连接。

保持机器人服务在前台交互终端运行。初始状态为 `positioning_paused`，准备好姿态后，
在**运行该服务的终端**输入 `model` 并按 Enter；这是服务交互命令，不是 shell 命令。
如果打开过手动跟随，先输入 `adjust-left-on`、`adjust-right-on` 并分别按 Enter 关闭两臂跟随。
确认 `[control-state]` 日志中的 `state` 为 `model_waiting`，再启动下面的 client。
认证连接成功后状态为 `model_control`；dry-run 同样需要交接。

### GPU 主机：另开终端启动 client

先按第 1 节设置 `NEXUS_PYTHON`，在本终端注入与机器人端相同的 `NEXUS_ROBOT_TOKEN`，
回到 Nexus 仓库根目录执行：

```bash
: "${NEXUS_ROBOT_TOKEN:?请先注入与机器人端相同的认证密钥}"
CLIENT_PYTHON="$NEXUS_PYTHON" \
BRIDGE_MODE=memory HISTORY_FPS=30 ACTIONS_PER_RESPONSE=8 \
ROBOT_HOST=ROBOT_IP ROBOT_PORT=12006 POLICY_HOST=127.0.0.1 POLICY_PORT=8001 \
bash examples/real_robot/dobot_dual_arm_14d/start_robot_client.sh
```

两端必须设置 `BRIDGE_MODE=memory`，并使用相同的 `HISTORY_FPS` 和 `ACTIONS_PER_RESPONSE`。
历史采样频率与训练数据一致；client 默认沿用机器人发送的任务指令。
连接后应能看到历史上传、ACK 和动作预测日志。

### 执行与重新开始

结束 dry-run 后，停止 client 和 robot-server。保留机器人端环境变量，重新启动 robot-server：

```bash
ROBOT_EXECUTE=1 bash dobot_dual_arm_14d/start_dobot_dual_arm_14d_server.sh
```

默认启动时确认并回 Home。Home 完成后，在机器人服务终端重新输入 `model`，
确认 `model_waiting` 后再执行上面的 client 命令连接；动作执行仍需要人工确认。
按 `h` 回 Home 默认结束当前历史，新一轮从帧 0 开始。切换任务指令也应重新开始 episode；
连接断开后历史被清理，重连不会恢复上一轮记忆。
客户端断开或任务结束后，机器人恢复 `positioning_paused`；每次重连都需先重新输入 `model`
再启动 client。如果 client 连接过早被拒绝，完成交接后重新执行 client 命令。

## 常见问题

| 现象 | 处理 |
|---|---|
| 缺少 `vision.npz` | 训练前设置冻结编码器路径；部署时检查 Artifact 的 `deployment/memory/vision.npz` |
| 特征缓存身份或 SHA256 不匹配 | 核对训练划分、编码器和相机配置；用新目录、新版本标识重新生成 |
| `q01/q99` 缺失 | 通用配置移除外部 stats 覆盖以自动计算；按钮 recipe 使用对应训练集的 quantile stats |
| policy 正常但历史未上传 | 检查 robot-server、client 两端的 `BRIDGE_MODE=memory`、端口与历史采样频率 |
| 缺少 token 或认证失败 | 在机器人服务与 client 两个终端注入相同的 `NEXUS_ROBOT_TOKEN` |
| 远端连接机器人超时 | 检查 `ROBOT_BIND_HOST` 是机器人可达网卡地址，并核对 `ROBOT_HOST`、端口和网络 |
| 连接后立即断开，提示 handoff not ready | 在机器人服务终端输入 `model`，确认 `model_waiting` 后重启 client |
| 更换权重后行为异常 | 核对完整 Artifact，以及任务指令、裁剪、动作频率；重新连接以清空历史 |

## 部署资源限制

历史特征按收到的帧增长，reset 后释放。每条连接默认最多 4 个环境、256 MiB 历史特征，
每批最多 128 帧，Memory 服务最多 4 条连接。达到上限会返回错误，不静默截断全局历史。
可在推理配置的 `server.memory_max_environments` 和 `server.memory_max_history_bytes`
中设置环境数和历史字节数；模型的 `max_frames` / `window_frames` 仍由训练配置决定。
全局 8192 帧历史需要至少 1 GiB 特征预算，请按连接数计算主机内存。
WebSocket 单帧上限为 32 MiB，非回环绑定需要 `NEXUS_POLICY_TOKEN`。

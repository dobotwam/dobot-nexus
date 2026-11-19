# Pi0.5 任务 recipes

通用训练模板在 `../configs/train/`，本目录根部保存训练任务的差异，通过
`examples/train.sh --config ...` 启动，路径从仓库根目录使用。
`infer/` 保存部署配方，通过对应的策略服务入口使用。

| 配方 | 通用模板 | 任务与关键设置 |
|---|---|---|
| [pi05_usb_bc.yaml](pi05_usb_bc.yaml) | `pi05.yaml` | USB BC；absolute、quantile、七卡全局 batch 672、warmup 500、EMA 0.99 |
| [pi05_usb_recap.yaml](pi05_usb_recap.yaml) | `pi05_recap.yaml` | USB ACP；复用 BC 统计资产，读取优势标注，其余训练参数与 USB BC 对齐 |
| [pi05_button_bc.yaml](pi05_button_bc.yaml) | `pi05.yaml` | 按钮 BC 动作编码示例；关节 delta、夹爪 absolute、zscore、10k steps |
| [pi05_button_memory.yaml](pi05_button_memory.yaml) | `pi05_memory.yaml` | 按钮 Memory；absolute、quantile、四卡全局 batch 64、1990 steps、EMA 0.99 |
| [pi05_kbd_pick_0916_full34.yaml](pi05_kbd_pick_0916_full34.yaml) | `pi05.yaml` | KBD 抓取；34 维 state/action、64 维内部边界、三相机、absolute、FSDP2 |

两份按钮配方的动作表示、归一化和训练步数不同，不是直接比较 Memory 收益的消融对照。
需要比较时，应另外对齐数据划分、动作编码、统计资产、有效 batch 和训练预算。
这些配置用于启动新训练；续训应使用对应 Artifact 保存的配置和资源。

## 配置复用

每个 recipe 的 `extends` 指向一个通用模板，路径相对该 YAML。映射递归合并；
列表、标量和 `null` 整体替换。先合并文件，再解析环境变量，最后应用命令行覆盖。
数据、权重和输出的相对路径从运行目录解析。
`config_resolved.yaml` 不保留 `extends`，复制 Artifact 后不需要这些模板。

例如：

```yaml
extends: ../configs/train/pi05.yaml
exp:
  name: pi05_my_task
data:
  default_prompt: "Perform the task."
```

## ATOM 18 维部署

[infer/pi05_atom18.yaml](infer/pi05_atom18.yaml) 继承通用 Pi0.5 推理配置，
校验 18 维输入输出，horizon 从 Artifact 恢复；使用 BF16 参数/计算、当前 AMP 实现、
`legacy` 编译和 CUDA Graph。它不定义机型专用模型，也不改写训练时的字段与归一化。

```bash
examples/pi05_real_robot/start_policy_server.sh /path/to/atom18/artifact \
  --config examples/recipes/infer/pi05_atom18.yaml
```

相机、SDK、头部保持和完整动作块执行由 [ATOM 示例](../atom_robot/README.md) 管理。
本配方的动作长度不由 `inference.execute_steps` 裁剪。完整步骤及两机认证配置见该示例。

## XTrainer 14 维部署

[infer/pi05_xtrainer14.yaml](infer/pi05_xtrainer14.yaml) 使用现有双臂 14 维硬件入口。
布局为左臂 J1–J6、左夹爪、右臂 J1–J6、右夹爪，夹爪索引为 6、13；
字段定义复用 [dual6.json](../pi05_real_robot/layouts/dual6.json)。
该布局对应双 Nova 2 机械臂与双夹爪的 XTrainer 配置，硬件规格参考
[Dobot 官方介绍](https://www.dobot-robots.com/insights/news/dobot-x-trainer.html)。

模型输入输出固定为 14 维，horizon、归一化和训练数据映射从匹配的 Artifact 恢复。
相机对应 `top_camera → base`、`left_camera → left_wrist`、`right_camera → right_wrist`；
硬件端的相机序列号、顶部 ROI 和左右顺序必须与训练数据一致。
精度与 ATOM recipe 一致：BF16 参数/计算、`autocast=auto`、`legacy` 编译、
精度保护和 CUDA Graph。现有硬件桥使用自己的 ROI/缩放流程，不采用 ATOM 的填充预处理。

```bash
examples/pi05_real_robot/start_policy_server.sh /path/to/xtrainer14/artifact \
  --config examples/recipes/infer/pi05_xtrainer14.yaml
```

机器人侧使用 `start_dobot_dual_arm_14d_server.sh`，策略侧通信客户端使用
`start_robot_client.sh`。硬件项目需提供本仓库使用的 `DobotEnvironment` 接口。
机器人服务和通信客户端均设置 `ACTIONS_PER_RESPONSE=8`，且该值不能超过 Artifact 的 H；
硬件执行默认 `STEP_DT=0.05`，即 20 Hz。YAML 的 `inference.execute_steps=8`
不会自动修改这两个独立进程的执行参数。

两端认证、硬件地址、相机参数、启动顺序、`model` 控制权交接及 dry-run 流程见
[三进程部署指南](../../docs/getting_started/05_deployment.md)和
[真机启动器参数](../pi05_real_robot/README.md)。本配方面向 Pi0.5 BC，不替代 Memory 的部署配置。

## USB BC 与 RECAP

```bash
export PI05_BASE_CKPT=/path/to/pi05_base
export VLA_DATASET_PATH=/path/to/usb_dataset
examples/train.sh --config examples/recipes/pi05_usb_bc.yaml --gpus 0,1,2,3,4,5,6
```

BC 完成后采集 rollout、训练 value 并生成优势标注，步骤见
[RECAP 使用说明](../../docs/recap/README.md)。然后指定 rollout 数据及对应资产：

```bash
export VLA_DATASET_PATH=/path/to/usb_rollouts
export RECAP_BC_NORM_STATS=/path/to/bc/artifact/deployment/assets/norm_stats.json
export RECAP_ANNOTATIONS=/path/to/recap_annotations
examples/train.sh --config examples/recipes/pi05_usb_recap.yaml --gpus 0,1,2,3,4,5,6
```

这两份 USB recipe 校验有效 batch 为 672。改用两卡时可覆盖
`data.batch_size=4 train.grad_accum_steps=84 data.num_workers=4`。
通用 `configs/train/pi05_recap.yaml` 没有固定七卡 batch 或 USB prompt。

## KBD 34 维抓取

```bash
export PI05_BASE_CKPT=/path/to/pi05_base
export VLA_DATASET_PATH=/path/to/kbd_pick_0916_full34
examples/train.sh --config examples/recipes/pi05_kbd_pick_0916_full34.yaml --gpus 0,1
```

数据必须包含 34 维 `observation.state`、`action`，以及 `observation.images.head`、
`observation.images.left_wrist`、`observation.images.right_wrist` 三路图像。
默认自动计算 zscore 统计；已有匹配统计可通过 `data.norm_stats_path=/path/to/norm_stats.json` 指定。
每卡 batch 8、20k optimizer updates；两卡且不累积时全局 batch 为 16。
32 维基础权重的动作投影在训练初始化时扩展到 64 维，已有部分逐值保留。

TensorBoard 默认开启，W&B 默认关闭，安装及配置见 [训练日志](../../docs/pi05_training.md#52-可选-wb-与-tensorboard-指标记录)。
模型训练环境已包含 TensorBoard；监控命令为 `python -m dobot_nexus.cli.main tensorboard --logdir runs`。
需要 W&B 时可在启动命令末尾加 `-- tracking.wandb.enabled=true`，默认离线记录；
需要在线记录时先登录，再覆盖 `tracking.wandb.mode=online`。
recipe 的 `runtime.distributed_timeout_seconds=7200` 为分布式通信设置两小时超时。
需要同时记录 TensorBoard 并同步 W&B 时，增加
`tracking.tensorboard.enabled=true tracking.wandb.enabled=true`。
其他机器人沿用同一模板，在 recipe 中配置 `model.state_dim`、`model.action_dim`、
`model.model_action_dim` 和数据字段映射；内部边界必须覆盖输入、输出维度。

## Memory 资源准备与训练

通用配置使用 Torch，需准备 LeRobot 数据、Pi0.5 初始权重，以及独立冻结视觉资产
`vision.npz`。已有 Memory Artifact 的 `files.memory_vision` 指向该资产；也可使用同一
经过验证的固定 encoder 文件。准备入口直接读取 `vision.npz`；训练和在线历史编码应使用
同一份冻结权重。完整操作步骤见 [Memory 操作指南](../../docs/pi05_memory.md)。

```bash
export PYTHONPATH=src:.
export PI05_BASE_CKPT=/path/to/pi05_base
export VLA_DATASET_PATH=/path/to/lerobot_training_split
export MEMORY_VISION_CHECKPOINT=/path/to/frozen_vision.npz
export MEMORY_FEATURES=/path/to/new_torch_history_features
export MEMORY_DATASET_ID=my_training_split_v1

# 使用与在线服务相同的 Torch 视觉编码器，按 episode 写入完整稠密历史。
python -m dobot_nexus.models.pi05.memory.prepare \
  --config examples/configs/train/pi05_memory.yaml --device cuda:0 --batch-size 16

# 通用 Memory 训练：默认每卡 batch 16，不固定卡数。
examples/train.sh --config examples/configs/train/pi05_memory.yaml --gpus 0,1
```

准备阶段只读取相机帧和 prompt，不需要动作归一化统计或训练用的多卡环境。
必须使用新的空缓存目录。更改数据、
prompt、相机或冻结 encoder 后重新生成缓存，并更新相应的资源身份。
特征按实际 prompt 变化分段；训练采样只读取截至当前帧的历史。该命令不会划分数据集。

按钮 Memory recipe 使用四卡、absolute action、1990 steps，需要提供训练划分及由该划分
生成的统计资产和缓存。训练步数按 optimizer update 计数，应根据数据规模调整。

```bash
# 上面的数据、固定 encoder 和特征缓存必须对应这个训练划分。
export MEMORY_NORM_STATS=/path/to/button_training_norm_stats.json
python -m dobot_nexus.models.pi05.memory.prepare \
  --config examples/recipes/pi05_button_memory.yaml --device cuda:0
examples/train.sh --config examples/recipes/pi05_button_memory.yaml --gpus 0,1,2,3
```

上述两组准备命令是替代用法。切换到另一个数据划分时，重新设置
`MEMORY_FEATURES`、`MEMORY_DATASET_ID`，不要复用已有缓存目录。
`budget=512` 表示每次采样最多 32 个历史帧，每帧 16 个 token；不是仅缓存 32 帧。
CPU 资源准备可指定 `--device cpu`。完整模型训练仍需要适当的 GPU 显存。

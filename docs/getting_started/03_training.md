# 3. 第一次训练、正式训练与续训

[新手路线](../../README.md#从哪里开始) · 上一步 [数据](02_data.md) · 下一步 [离线推理](04_inference.md)

本篇使用 Pi0.5 BC、14 维双臂数据和两张 GPU。先完成环境和数据检查。
以下命令都在 GPU 主机的仓库根目录执行。
18 维 `collect_data` 叠衣服任务使用第 3.8 节的配置和命令。

## 3.1 设置自己的路径

```bash
source .venvs/pi05/bin/activate
export PI05_BASE_CKPT=/path/to/pi05_base
export VLA_DATASET_PATH=/path/to/datasets/my_task_v21
test -f "$VLA_DATASET_PATH/meta/info.json"
```

第一个变量指定初始权重，第二个指定单个数据集根目录。
所有示例输出写入仓库的 `runs/`。大规模训练可通过 `--output-dir /path/to/training_runs`
选择容量更大的磁盘，后面的 Artifact 路径也需要同步调整。

## 3.2 生成自己的任务配置

执行下面的命令创建 `runs/quickstart/pi05_my_task.yaml`。
该文件继承仓库模板，只覆盖新手实验需要调整的参数。

```bash
mkdir -p runs/quickstart
cat > runs/quickstart/pi05_my_task.yaml <<'YAML'
extends: ../../examples/configs/train/pi05.yaml
exp:
  name: pi05_my_task_train
data:
  batch_size: 1
  num_workers: 0
  normalization: zscore
  field_mapping:
    images:
      base: observation.images.top_camera
      left_wrist: observation.images.left_camera
      right_wrist: observation.images.right_camera
    state: observation.state
    action: action
    language: task
  transforms:
    action:
      type: absolute
train:
  max_steps: 10000
  save_every: 1000
  log_every: 10
  resume_limit: 2
scheduler:
  total_steps: 10000
YAML
```

`extends` 相对于这个 YAML 所在目录解析，因此请按上述位置保存文件。
权重和数据路径由父模板读取 `PI05_BASE_CKPT`、`VLA_DATASET_PATH`。

### 必须与数据一致的参数

| 参数 | 本示例 | 如何理解 |
|---|---|---|
| `model.state_dim` / `action_dim` | 继承 14 / 14 | 数据中 state/action 的向量长度 |
| `model.action_horizon` | 继承 50 | 一次预测的动作步数 |
| `data.field_mapping.images` | 三路相机映射 | 左侧是公共相机名，右侧是 Parquet 列名 |
| `data.transforms.action.type` | `absolute` | 保持数据中动作的原始数值语义 |
| `data.normalization` | `zscore` | 对 state/action 使用均值和标准差归一化 |
| `data.batch_size` | 1 | 每张 GPU、每次前向使用的样本数量 |
| `data.num_workers` | 0 | 先在主进程读取数据，便于看到完整错误 |

这条路线假设 recorder 保存的是绝对关节目标和夹爪命令。
如果已有数据保存增量、末端位姿或其他布局，先确认动作定义并选择对应 codec。
关节 delta、夹爪 absolute 的现成任务配置见[任务 recipes](../../examples/recipes/README.md)。

Pi0.5 也支持 `data.normalization: quantile`，其统计需要 q01/q99。
此参数控制 state/action 数值归一化，图像预处理由模型数据管线负责。
本篇采用 zscore 并自动生成统计资产，推理从 Artifact 恢复同一设置。

## 3.3 预览双卡启动命令

```bash
bash examples/train.sh \
  --python .venvs/pi05/bin/python \
  --config runs/quickstart/pi05_my_task.yaml \
  --gpus 0,1 \
  --run-name pi05_my_task_smoke \
  --dry-run \
  -- train.max_steps=10 train.save_every=10 train.log_every=1 train.resume_limit=1
```

输出应包含 `launcher=accelerate`、`processes=2` 和 `torch.distributed.run`。
`--dry-run` 完成配置解析并打印命令，实际加载权重和数据在下一步执行。

### TensorBoard 实时监控

模板默认开启本地 TensorBoard，日志位于 `<run_dir>/tensorboard/`。新终端使用同一训练环境：

```bash
.venvs/pi05/bin/python -m dobot_nexus.cli.main tensorboard --logdir runs --port 6006
```

浏览器打开 `http://127.0.0.1:6006`，在 Scalars 中比较不同 run 的 `loss`、`lr`、
`grad_norm`、`steps_per_sec` 和 `eta_seconds`。CUDA 任务还记录主 rank 的显存占用。
若 `TRAINING_RUNS` 指向自定义输出目录，将 `--logdir runs` 换成 `--logdir "$TRAINING_RUNS"`。
服务器训练时可通过已有 SSH 端口转发访问同一页面。
每 `train.log_every` 个 optimizer update 记录一次，首步和末步必记；恢复训练沿用原 run，
checkpoint 之后未保存的旧曲线会隐藏。现有环境缺少依赖时执行
`python -m pip install -e '.[tensorboard]'`。

## 3.4 运行 10 步训练检查

```bash
bash examples/train.sh \
  --python .venvs/pi05/bin/python \
  --config runs/quickstart/pi05_my_task.yaml \
  --gpus 0,1 \
  --run-name pi05_my_task_smoke \
  -- train.max_steps=10 train.save_every=10 train.log_every=1 train.resume_limit=1
```

首次启动会加载模型、扫描数据统计、准备 FSDP，再开始打印训练 step。
10 步检查用于验证读取、前向、反向和保存链路，任务效果需要后续训练与独立评测。

另开终端查看日志。

```bash
tail -f runs/pi05_my_task_smoke/logs/train.log
```

结束后检查保存指针。

```bash
cat runs/pi05_my_task_smoke/checkpoints/latest_deployment.txt
cat runs/pi05_my_task_smoke/checkpoints/latest_resume.txt
```

两者应指向 `step_00000010`。日志应有连续 step、有限的 loss 和保存成功的信息。
同名 smoke 已存在时，用新的 `--run-name` 重跑，并相应调整检查路径。

### 只有一张 GPU

显存足够容纳完整模型、梯度和 AdamW 状态时，可以用下面的单卡检查命令。

```bash
bash examples/train.sh \
  --python .venvs/pi05/bin/python \
  --config runs/quickstart/pi05_my_task.yaml \
  --gpus 0 \
  --run-name pi05_my_task_single_smoke \
  -- train.max_steps=10 train.save_every=10 train.log_every=1 train.resume_limit=1
```

单卡全量微调的显存压力较大。OOM 时优先核对 batch size 和硬件资源，多卡 FSDP 能分摊模型状态。
单卡和多卡使用同一份 YAML，启动器根据进程数选择执行方式。

### 单卡和多卡共用配置

Pi0、Pi0.5 训练模板以及继承它们的 Memory/RECAP policy recipes 使用：

```yaml
runtime:
  launcher: auto
  precision: bf16
  fsdp:
    enabled: auto
    version: 2
```

保留模板中的其他 FSDP 参数，上面仅列出运行方式相关字段。
`examples/train.sh --gpus 0` 启动一个 local 进程；`--gpus 0,1` 启动两个
Accelerate + FSDP2 进程。直接使用 Python / torchrun 入口时，训练端也按
`WORLD_SIZE` 解析；可见 GPU 数不会自动变成训练进程数。
`examples/train.sh` 的单卡分支会清除父环境遗留的 torchrun 进程身份和 rendezvous
变量（如 `WORLD_SIZE`、`RANK`、`LOCAL_RANK`、`MASTER_ADDR`），确保实际子进程与预检一致。

| 配置或行为 | 单卡 | 多卡 FSDP2 |
|---|---|---|
| 模型、数据映射、优化器、scheduler | 共用同一份配置 | 共用同一份配置 |
| `runtime.launcher: auto` | `local` | `accelerate` |
| `runtime.fsdp.enabled: auto` | `false` | `true` |
| 参数/梯度/优化器状态 | 单卡保存完整状态 | 跨 rank 分片 |
| `data.batch_size` | 每进程微批量 | 每进程微批量 |
| `train.grad_accum_steps` | 每次更新的累积批数 | 每个 rank 每次更新的累积批数 |
| `model.parameter_precision: fp32` | FP32 持久参数与 Adam 状态 | FP32 持久参数与 Adam 状态 |
| BF16 计算 | runtime autocast | FSDP mixed precision |

配置项统一不会自动改变批量或学习率。有效全局 batch 为：
`data.batch_size × WORLD_SIZE × train.grad_accum_steps`。
例如保持全局 batch 4，可以单卡 `batch_size=2, grad_accum_steps=2`，
双卡 `batch_size=1, grad_accum_steps=2`；也可以固定每卡 batch 1，单卡累积 4、双卡累积 2。
Pi0.5 部分 recipe 的 `data.global_batch_size` 是目标批量校验值，不是自动调整开关。
`data.num_workers` 也按进程配置，多卡时总 worker 数随进程数增加。

可继续显式设置 `runtime.launcher=local/accelerate` 和
`runtime.fsdp.enabled=true/false`。显式开启 FSDP 时仍要求至少两个进程，
显式 local 不能在多进程下运行。模板的精度、FSDP 包装、梯度检查点配置无需随卡数更改；
`model.precision` 与 `runtime.precision` 应保持一致，参数存储精度是独立含义。

实验的 `config_resolved.yaml` 保存解析后的执行方式。精确续训仍要求相同进程数、
批量、精度和 FSDP 设置；共用模板不表示单卡与多卡训练状态可以直接互续。

## 3.5 正式训练

使用独立实验名，按照任务配置训练 10000 步，每 1000 步保存一次。
步数是可调整的实验设置，应根据验证结果选择最终模型。

```bash
bash examples/train.sh \
  --python .venvs/pi05/bin/python \
  --config runs/quickstart/pi05_my_task.yaml \
  --gpus 0,1 \
  --run-name pi05_my_task_train
```

这份配置的有效 batch size 为 `1 × 2 × 1 = 2`，依次是每卡 batch、GPU 数、梯度累积次数。
调大 batch 或 `train.grad_accum_steps` 后，应新建实验并记录设置。
熟悉数据管线后可以提高 `data.num_workers` 改善读取吞吐。

## 3.6 中断后继续训练

确认 `latest_resume.txt` 存在，重新设置第 3.1 节的路径，然后使用原配置、原实验名和原 GPU 数量。

```bash
bash examples/train.sh \
  --python .venvs/pi05/bin/python \
  --config runs/quickstart/pi05_my_task.yaml \
  --gpus 0,1 \
  --run-name pi05_my_task_train \
  -- train.resume_from=auto train.trust_resume_checkpoint=true
```

这条命令从最近保存的状态恢复优化器、scheduler、RNG 和数据进度。
训练已达到 `max_steps` 时，恢复后无需再执行更新。
此前在 CLI 传入过其他训练覆盖项，也应原样保留。精确续训要求数据、batch、模型、
优化器、scheduler、随机种子、精度和 world size 等语义匹配。[恢复规则](../pi05_training.md#7-断点续训)

## 3.7 找到可用于推理的产物

```text
runs/pi05_my_task_train/
├── config_resolved.yaml              # 实际生效的配置
├── logs/train.log
├── data/<fingerprint>/norm_stats.json
└── checkpoints/
    ├── latest_deployment.txt          # 推理使用的指针
    ├── latest_resume.txt              # 续训使用的指针
    └── step_00001000/
        ├── manifest.yaml
        ├── config_resolved.yaml
        ├── deployment/
        │   ├── pytorch/full_weights.pt
        │   └── assets/               # 归一化统计、tokenizer
        └── resume/                   # 可恢复的训练状态
```

`resume_limit: 2` 保留最近两份恢复状态，历史部署权重另外占用磁盘。
指针会随保存推进，部署资产的实际文件名以 `manifest.yaml` 为准。

下一篇先加载 10 步 smoke 产物检查[离线推理](04_inference.md)。正式模型完成后换成对应 Artifact，
使用独立观测和任务评测判断效果。

## 3.8 训练 18 维叠衣服任务

先完成[叠衣服数据检查](02_data.md#25-使用-18-维叠衣服数据)和第 1 篇环境准备。
下面的配置适用于完整的 `collect_data` LeRobot v2.1 数据。路径按所在主机调整。

```bash
source .venvs/pi05/bin/activate
export PI05_BASE_CKPT=/path/to/pi05_base/model.safetensors
export VLA_DATASET_PATH=/path/to/datasets/collect_data
test -f "$PI05_BASE_CKPT"
test -f "$VLA_DATASET_PATH/meta/info.json"
mkdir -p runs/quickstart
cat > runs/quickstart/pi05_fold_tshirt.yaml <<'YAML'
extends: ../../examples/configs/train/pi05.yaml
exp:
  name: pi05_fold_tshirt
  seed: 2026
model:
  state_dim: 18
  action_dim: 18
  model_action_dim: 32
  action_horizon: 50
  operator_numerics: openpi
  max_token_len: 200
data:
  format_version: v2.1
  fps: 20
  batch_size: 32
  num_workers: 0
  normalization: quantile
  default_prompt: "Fold the T-shirt with both arms."
  field_mapping:
    images:
      base: observation.images.top_left
      left_wrist: observation.images.wrist_left
      right_wrist: observation.images.wrist_right
    state: observation.state
    action: action
    language: task
  transforms:
    action:
      type: absolute
runtime:
  compile: false
  distributed_timeout_seconds: 3600
train:
  max_steps: 20000
  grad_accum_steps: 4
  save_every: 2000
  log_every: 20
scheduler:
  total_steps: 20000
YAML
```

基础模板提供 BF16 计算、FP32 参数、FSDP2 和 AdamW，学习率为 `2.5e-5`。
推理参数精度在[推理配置](../precision.md)中单独指定，不从训练 Artifact 自动继承。
两卡正式训练的有效 batch 为 `32 × 2 × 4 = 256`。本示例先设置 `num_workers: 0`
以便定位数据读取异常，读取性能调优时可按内存和存储吞吐调整。

先用小 batch 完成 10 步检查。示例选择 GPU 1、2，请替换为所在主机的两张可用 GPU。
在命令中添加 `--dry-run` 可以先检查配置并预览启动命令。

```bash
bash examples/train.sh \
  --python .venvs/pi05/bin/python \
  --config runs/quickstart/pi05_fold_tshirt.yaml \
  --gpus 1,2 \
  --run-name pi05_fold_tshirt_smoke \
  -- data.batch_size=1 train.grad_accum_steps=1 \
  train.max_steps=10 train.save_every=10 train.log_every=1
```

按第 3.4 节核对日志和保存指针，目录改成 `runs/pi05_fold_tshirt_smoke`。
小 batch 检查完成后，用独立实验名按完整配置开始正式训练。

```bash
bash examples/train.sh \
  --python .venvs/pi05/bin/python \
  --config runs/quickstart/pi05_fold_tshirt.yaml \
  --gpus 1,2 \
  --run-name pi05_fold_tshirt_train
```

正式输出在 `runs/pi05_fold_tshirt_train/`。完成 20000 步后，部署 Artifact 为
`runs/pi05_fold_tshirt_train/checkpoints/step_00020000`。保存前中断的进度只能从已有
恢复状态重做，续训方法见第 3.6 节。训练 loss、Artifact 保存和实机叠衣服效果分别检查，
任务成功率通过独立评测获得。

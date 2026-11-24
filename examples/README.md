# Dobot Nexus 用户样例

第一次训练和部署请按[新手路线](../README.md#从哪里开始)操作。
其中的[训练步骤](../docs/getting_started/03_training.md)从每卡 batch size 1 开始，
并给出数据检查、10 步训练、正式训练与续训的完整命令。

`examples/` 保存用户可直接运行的配置与统一入口。`configs/train/` 保存通用训练模板，
`recipes/` 保存具体任务的差异配置。Pi0.5 的 BC、RECAP、Memory 分别使用
`pi05.yaml`、`pi05_recap.yaml`、`pi05_memory.yaml`（Pi0 对应 `pi0.yaml`、`pi0_recap.yaml`、`pi0_memory.yaml`），
统一经 `train.sh` 启动。

USB BC/RECAP、按钮 BC/Memory、KBD 抓取的参数和启动方法见 [任务 recipes](recipes/README.md)。
原始采集数据转换入口见 [`data_processing/`](../data_processing/README.md)。
独立 value 作业、数据结果标签和执行步骤见 [RECAP 使用说明](../docs/recap/README.md)。
Pi0.5 BC、RECAP 和 Memory 的推理样例默认启用 Torch 局部编译与 CUDA Graph：
BC/RECAP 使用 `compile_mode=legacy`，`configs/infer/pi05_memory.yaml` 使用 `openpi`。
Memory 的资源准备、训练和历史服务见 [Memory 操作指南](../docs/pi05_memory.md)。
CPU 推理配置见 [Pi0.5 部署说明](../docs/pi05_training.md#92-pytorch-推理)。
ATOM W 0759 的 18 维相机、DDS、夹爪与只读验证入口见
[ATOM 硬件示例](real_robot/atom_w_0759/README.md)，对应推理配方位于 `recipes/infer/pi05_atom18.yaml`。

ATOM D 的双臂叠衣入口为 `real_robot/atom_d/run_atom_d.sh`。权重选择、Home-start 和录制观测回放见 [ATOM D 部署指南](../docs/robots/atom_d_inference.md)。

```text
examples/
├── configs/
│   ├── train/               # 通用训练模板
│   ├── infer/               # 推理与服务配置
│   └── recap/               # 独立 value / 标注作业配置
├── recipes/                 # 具体任务配方，通过 extends 复用通用模板
├── train.sh                 # 统一训练入口
├── infer.sh                 # 统一推理入口
└── real_robot/              # 真机部署入口、各机型执行代码和共享协议
```

模块测试见 [`tests/README.md`](../tests/README.md)。性能、精度对齐和 smoke 实验保存在
Git 忽略的 `local_script/`，不属于用户入口。

## 训练

Pi0.5 标准 absolute-action 训练：

```bash
PI05_BASE_CKPT=/path/to/pi05_base \
VLA_DATASET_PATH=/path/to/lerobot_dataset \
examples/train.sh \
  --config examples/configs/train/pi05.yaml \
  --gpus 0,1
```

具体任务通过 `extends` 继承通用模板，只写需要调整的字段。仓库提供一个关节 delta、
夹爪 absolute 的双臂示例：

```bash
VLA_DATASET_PATH=/path/to/press_the_yellow_button \
examples/train.sh \
  --config examples/recipes/pi05_button_bc.yaml \
  --gpus 0,1 \
  -- \
  model.checkpoint=/path/to/pi05_base
```

`extends` 只接受一个 YAML 路径，相对当前配置文件定位；映射递归合并，列表、标量和
`null` 整体替换。合并后解析环境变量，最后应用命令行覆盖。Artifact 中的
`config_resolved.yaml` 保存完整结果，部署或恢复训练不依赖原始模板文件。

Memory 训练还需要固定 `vision.npz` 和逐帧历史特征。纯 Torch 的准备命令及必需资源见
[Memory recipe 说明](recipes/README.md#memory-资源准备与训练)。

字段映射、normalization、batch/FSDP、断点续训和 Artifact 规则见
[`Pi0.5 训练与部署配置指南`](../docs/pi05_training.md)。

Pi0 使用仓库内置 operator core 和与 Pi0.5 同形的 LeRobot pipeline；

```bash
VLA_DATASET_PATH=/path/to/lerobot_v3_dataset \
PI0_BASE_CKPT=/path/to/pi0_base \
examples/train.sh --config examples/configs/train/pi0.yaml --gpus 0
```

## 推理与服务

`infer.sh` 根据 `--model` 自动选择 `examples/configs/infer/<model>.yaml`。Pi0.5 从 Artifact
加载 Torch 权重和数据处理资产：

```bash
examples/infer.sh \
  --model pi05 \
  --python .venvs/pi05/bin/python \
  --artifact runs/pi05_bc/checkpoints/latest_deployment.txt \
  --input /path/to/observation.npz \
  --output runs/pi05_actions.npz \
  --gpu 0 \
  -- \
  model.inference_backend=torch
```

启动 Nexus HTTP 服务：

```bash
examples/infer.sh \
  --model pi05 \
  --python .venvs/pi05/bin/python \
  --artifact runs/pi05_bc/checkpoints/latest_deployment.txt \
  --serve \
  --host 127.0.0.1 \
  --port 12021 \
  -- \
  model.inference_backend=torch
```

Pi0.5 真机使用
WebSocket 链路，入口位于
[真机部署索引](real_robot/README.md)。

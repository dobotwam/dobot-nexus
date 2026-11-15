# Pi0 训练与推理

Pi0 的入口是 `dobot_nexus.models.pi0.modeling_pi0:Pi0Model`，使用独立的
`models/pi0/training/vendor/rlinf_core/` 算子。当前集成覆盖 BC、FSDP2、断点续训、
Artifact 导出、离线推理和通用 HTTP 服务。RECAP、Memory 和
`examples/pi05_real_robot/` 的专用部署入口属于 Pi0.5。

## 模型和权重

Pi0 将连续状态投影为独立 token，将时间嵌入与动作 token 拼接后送入 MLP，
动作专家使用普通 RMSNorm。Pi0.5 的时间条件和状态编码不同，不能交换初始权重。
公开 state/action 维度可以各自配置为 1–32；内部算子固定为 32 维，数据管线完成
补零和输出裁剪。增量动作要求 state/action 维度相同。

按[环境指南](environments.md)创建 `--model pi0` 环境。初始权重可使用
[ModelScope 的 `lerobot/pi0`](https://www.modelscope.cn/models/lerobot/pi0)，目录中
需包含 `model.safetensors`。保留下载的模型卡和来源信息；模型仓库的配置用于
说明来源，Nexus 训练参数来自本仓库 YAML。加载器会转换 LeRobot/OpenPI Torch
键名并严格检查参数，不会用随机参数补齐缺失权重。Tokenizer 随 Nexus 包提供。

## 数据和短训练

支持 LeRobot v2.1 内嵌图片，以及共享读取器支持的 v2.1/v3 视频数据。
三路公共相机名是 `base`、`left_wrist`、`right_wrist`，必须映射到数据集实际列名。
以下示例假定 18 维数据采用 `top_left`、`wrist_left`、`wrist_right` 三路图片：

```bash
export PI0_BASE_CKPT=/path/to/pi0_base
export VLA_DATASET_PATH=/path/to/lerobot_dataset
bash examples/train.sh \
  --python .venvs/pi0/bin/python \
  --config examples/configs/train/pi0.yaml \
  --gpus 0,1 --run-name pi0_task \
  -- model.state_dim=18 model.action_dim=18 \
     data.field_mapping.images.base=observation.images.top_left \
     data.field_mapping.images.left_wrist=observation.images.wrist_left \
     data.field_mapping.images.right_wrist=observation.images.wrist_right \
     data.batch_size=1 train.grad_accum_steps=2 \
     train.max_steps=10 train.save_every=5 train.log_every=1 \
     scheduler.total_steps=1000 scheduler.warmup_steps=10
```

这里全局 batch 为 `1 × 2 张卡 × 2 次累积 = 4`。同一份 YAML 可以单卡运行，只需
改为 `--gpus 0`；若要保持全局 batch 4，还需将 `train.grad_accum_steps` 改为 4，
或把 `data.batch_size` 改为 2。自动选择规则见[统一训练配置](getting_started/03_training.md#单卡和多卡共用配置)。
训练参数与 Adam 状态保留 FP32，计算默认 BF16；`model.precision` 和 `runtime.precision` 必须一致。
推理默认将参数加载为计算精度；如需保留 FP32 权重，设置
`model.inference_parameter_precision=fp32`，BF16 计算使用自动精度转换。

Pi0 使用均值/标准差归一化。省略 `data.norm_stats_path` 时，训练会按动作分块和
absolute/delta 设置计算统计量并缓存。自备统计文件必须匹配当前数据和动作语义。
启用 delta 时，用 `data.transforms.action.delta_mask` 明确每一维是否减去当前状态，
不要把绝对夹爪目标默认视为关节增量。推理会使用 Artifact 中保存的同一套统计与变换。

TensorBoard 默认记录在 `runs/pi0_task/tensorboard/`：

```bash
.venvs/pi0/bin/python -m dobot_nexus.cli.main tensorboard --logdir runs/pi0_task
```

## 保存、续训与推理

每次保存生成 `runs/pi0_task/checkpoints/step_XXXXXXXX/`，包含部署权重、配置、
归一化统计、Tokenizer，以及在 `train.resume_limit > 0` 时保留的续训状态。
续训应复用训练配置、GPU 数量、全局 batch 和 scheduler 设置；增加 `train.max_steps`
并追加 `train.resume_from=auto train.trust_resume_checkpoint=true`。
仅对自己生成或可信来源的续训文件开启信任；部署 Artifact 不需要此选项。

用训练产物推理，不需要保留初始 checkpoint：

```bash
bash examples/infer.sh \
  --python .venvs/pi0/bin/python --model pi0 --gpu 0 \
  --artifact runs/pi0_task/checkpoints/latest_deployment.txt \
  --input observation.npz --output actions.npz \
  -- model.state_dim=18 model.action_dim=18 inference.num_steps=10
```

模板默认 14 维，因此 18/24/32 维 Artifact 需显式传入匹配维度。输入格式见
[离线推理指南](getting_started/04_inference.md)：图片为 RGB uint8，state 保持训练数据
的物理单位，每个样本提供一个 prompt。输出为 `[batch, action_horizon, action_dim]`。
`inference.seed` 控制采样噪声，`num_steps` 是去噪迭代次数，不是输出动作长度。

通用 HTTP 服务使用相同 Artifact 和数据处理：将离线命令的 `--input/--output`
替换为 `--serve --host 127.0.0.1 --port 12023`。真机执行仍需按本体配置字段映射、
关节单位、动作范围和执行频率；离线功能测试不验证机器人任务成功率。

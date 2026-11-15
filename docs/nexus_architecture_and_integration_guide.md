# Nexus 架构与模型接入

本文描述当前源码中的职责与调用关系。正式策略训练使用 `BCTrainer`；BC、RECAP 条件策略
和 Memory 策略均通过 `model.target` 接入。Pi0.5 的训练和部署只使用 Torch。
在线 RL / RL_token 的 Trainer、rollout 和多优化器恢复尚未实现。

## 目录与职责

```text
src/dobot_nexus/
├── cli/                    # train / infer / serve 参数与任务入口
├── core/                   # 配置、公共输入输出、动态加载、Experiment、Artifact
├── data/                   # 通用 LeRobot reader、action transform、统计与缓存
├── training/               # BCTrainer、Runtime、EMA、seed、训练恢复
├── inference/
│   ├── policy.py           # 从 Artifact 装配模型并调用完整 action chunk 推理
│   ├── session.py          # 缓存动作块、按 execute_steps 消费
│   ├── handlers.py         # 普通预测请求、种子序列与计时
│   ├── worker.py           # 串行执行阻塞任务和会话清理
│   ├── server.py           # Nexus HTTP 接口
│   └── transport/          # 通用 WebSocket 服务/客户端、NumPy msgpack 编解码
├── algorithms/recap/       # Return、Advantage、离线 Value 训练与标注
├── modules/value/          # 可复用 Value 网络、损失与输入处理
└── models/
    ├── base.py             # 唯一公共模型与 adapter 继承接口
    ├── pi05/               # Pi0.5、Memory 与 RECAP 策略接入
    ├── pi0/                # 独立 Pi0 实现及算子快照
```

仓库外层入口：

| 目录 | 职责 |
|---|---|
| `examples/configs/{train,infer}/` | 通用配置模板 |
| `examples/recipes/` | 按任务覆盖模板的完整示例 |
| `examples/{train,infer}.sh` | 统一启动入口 |
| `examples/pi05_real_robot/` | 三个启动脚本与 `_runtime/` 下的机器人通信实现 |
| `scripts/env/` | 按模型创建环境、检查安装及可选服务依赖 |
| `data_processing/` | 独立 PKL → LeRobot 转换，保留可扩展数据字段 |
| `tests/` | CPU 模块、契约、命令入口检查 |
| `local_script/` | Git 忽略的实模、GPU 和性能实验工具 |

公共 `core/`、`training/`、`inference/` 不导入具体模型。模型可以位于本仓库，也可以由
独立 Python 包提供；只需通过 `model.target` 指向实现，不增加模型注册表。
`algorithms/` 拥有算法计算与作业编排，使用公共数据读取和 `modules/`，不导入具体 VLA 模型。
模型接入可以调用算法计算与数据辅助函数；公共 `modules/` 不反向依赖模型、算法或作业层。

## 公共协议

唯一的框架模型接口在 [`models/base.py`](../src/dobot_nexus/models/base.py)。

| 接口 | 必须实现的能力 |
|---|---|
| `BaseVLAModule.from_config(config, mode=...)` | 校验配置、构造模型、加载初始权重 |
| `build_train_dataloader(config)` | 返回模型拥有的训练 DataLoader |
| `training_step(batch)` | 返回可反向传播的 `TrainOutput`；不更新优化器 |
| `predict_action_chunk(observation, options)` | 生成并解码完整动作块 |
| `policy_spec` | 声明相机、state/action 维度和 horizon |
| `BaseDataAdapter.encode_train_batch(raw_batch, device)` | 训练字段映射、输入转换和设备传输 |
| `encode_observation(observation)` | 编码公共观测 |
| `decode_actions(actions, aux=...)` | 解码并返回 `PolicyOutput` |

`BaseVLAModule.forward()` 转发到 `training_step()`，分布式 wrapper 因此走标准
`nn.Module` 调用链。不要绕过 wrapper 直接调用被包装模型的训练方法。

[`core/contracts.py`](../src/dobot_nexus/core/contracts.py) 定义公共边界：

| 字段 | 类型/形状 | 语义 |
|---|---|---|
| `images` | `dict[str, uint8[B,H,W,3]]` | BHWC RGB |
| `state` | `float32[B,D]` | 机器人原始状态 |
| `prompt` | `list[str]` | 每个样本一个非空文本 |
| `actions` | `float32[B,H,A]` | 已解码的机器人动作块 |

BCHW、BF16、动作 padding、token 和私有 Observation 在 adapter 内处理。模型训练和
推理复用同一套 normalization、tokenizer 和动作逆变换；公共 `data/` 只提供无模型身份的
reader、数学变换、统计和缓存。

## 训练与 Artifact

```text
配置模板 + recipe + CLI 覆盖
    → build_model(mode="train")
    → model.build_train_dataloader()
    → adapter.encode_train_batch()
    → wrapped_model(batch) → TrainOutput
    → Runtime.backward / clip / optimizer / scheduler / EMA
    → ArtifactManager.save()
```

`training/bc_trainer.py` 管理优化循环；`runtime.py` 隔离 local、Accelerate、FSDP 差异；
`resume.py` 管理恢复一致性；`tracking.py` 提供仅主 rank 启用的 TensorBoard/W&B 指标记录及同步。
当前 `algorithm: bc` 对应单模型、单优化器的监督式训练流程。
RECAP 的 Value 训练与离线标注由 `algorithms/recap/cli.py` 编排，ACP 策略仍复用 BCTrainer。

schema v2 Artifact 分开保存部署和续训状态：

```text
checkpoints/
├── latest_deployment.txt
├── latest_resume.txt
└── step_XXXXXXXX/
    ├── manifest.yaml
    ├── config_resolved.yaml
    ├── resume/                         # 模型/优化器分片、Trainer、RNG、数据进度
    └── deployment/
        ├── pytorch/full_weights.pt
        ├── nexus/full_weights.pt       # 仅需要 wrapper 权重的模型保存
        ├── memory/vision.npz           # 仅 Memory 保存冻结视觉权重
        └── assets/                     # norm stats、tokenizer 等
```

Pi0.5 直接将训练参数保存为裸 Torch core 权重，并打包模型专属部署资源。
启用 EMA 时部署权重使用 EMA，续训状态保留当前训练参数及 EMA 状态。
训练和推理使用同一模型 target 与 manifest；部署应携带完整 Artifact，而非只复制 `.pt`。

模型可覆写以下 hook，公共 ArtifactManager 不按模型名分支：

| Hook | 用途 |
|---|---|
| `get_param_groups()` | 冻结、分层学习率、no-decay |
| `export_backend_state_dict()` | 导出裸 backend keys |
| `save_nexus_weights()` | 是否还需 wrapper 形状的权重，避免重复保存 |
| `export_deployment_artifacts()` | 保存模型专属部署资源 |
| `artifact_assets()` | 将配置引用的小资源打包并重定向 |
| `artifact_metadata()` | 记录模型身份、数值策略和资源信息 |

恢复约束、EMA、数据进度和保留策略见 [Pi0.5 训练指南](pi05_training.md)。

## 推理、通信和会话

`InferencePolicy` 读取 Artifact、恢复训练数据配置和资产路径，再通过同一 `model.target`
构造模型，检查输入输出规格并用锁保护完整采样。`PolicySession` 缓存已生成的动作块，
负责按 `execute_steps` 消费；它不承担 Memory 的观测历史管理。

WebSocket 链路分为三层：

| 层 | 实现位置 | 职责 |
|---|---|---|
| 通信 | `inference/transport/` | 连接、消息编码、容量限制、健康检查与客户端超时 |
| 请求执行 | `inference/worker.py`、`handlers.py` | 串行推理、取消/清理顺序；普通预测的种子与计时 |
| 模型装配 | `models/pi05/deploy/` | 根据 Artifact 选择 BC/RECAP/Memory，适配机器人观测、预热与创建 Memory 会话 |

传输层接收请求 callable 和连接上下文管理器，不认识相机字段或模型类型。
Memory 每个连接维护独立历史、种子和 reset 状态，模型权重与冻结视觉 encoder 在服务中共享。
连接的建立、推理和清理通过同一个串行 worker 执行。预热使用独立会话，避免污染正式历史。

当前真机拓扑：

```text
GPU 服务器：robot-client → WebSocket policy → Torch 模型
                 ↕ TCP：观测 / 动作 / 控制
机器人侧：  robot-server → 环境 SDK、相机、机械臂和夹爪
```

三个公开启动入口保存在 `examples/pi05_real_robot/`，Python 实现在 `_runtime/`。
机器人项目、串口/相机驱动和硬件配置属于外部环境。实际命令与执行开关见
[真机部署](../examples/pi05_real_robot/README.md)。

## Pi0.5 的内部边界

| 模块 | 职责 |
|---|---|
| `modeling_pi05.py`、`data_adapter_pi05.py` | 公共模型入口与数据适配 |
| `data/` | 字段映射、图像、tokenizer、normalization、动作编码/解码 |
| `torch_core/` | 训推共用架构、算子、数值策略；导入时不编译 |
| `training/` | 初始 checkpoint 加载、训练实例的算子编译 |
| `openpi_torch/` | 推理编译、CUDA Graph、条件投影与视觉编码加速 |
| `memory/` | 历史采样、特征、会话和 Memory 模型 |
| `modeling_recap.py`、`data_adapter_recap.py` | RECAP 条件策略的 Pi0.5 模型与观测接入 |
| `configuration_recap.py`、`data/recap.py` | RECAP 策略 Artifact 恢复、标签关联与条件 token 路由 |
| `deploy/` | Pi0.5 服务装配与机器人输入适配 |

BC、RECAP 和 Memory 共享 Pi0.5 core；RECAP 和 Memory 使用各自的 `model.target`。
`openpi_torch` 负责 Torch 推理加速；加速参数需要与 Artifact 的数值策略匹配。

Pi0 保留自己的 `training/vendor/rlinf_core/`，不导入 Pi0.5 算子。

## 接入新模型或算法

新模型先建立 `configuration_<name>.py`、`modeling_<name>.py`、
`data_adapter_<name>.py` 和 `requirements.txt`；按需增加 pipeline、算子、checkpoint
及推理加速目录。用已有实现和 [CPU fixtures](../tests/fakes.py) 对照真实接口，避免复制含未实现函数的伪代码。

```yaml
model:
  name: my_model
  target: my_package.modeling:MyModel
```

1. 明确公共相机、state/action 和 horizon，在构造时校验。
2. 用 CPU 小模型完成 adapter 往返、一次优化更新、Artifact 保存与推理回载。
3. 实现严格 checkpoint 加载，并打包 tokenizer、统计等部署资产。
4. 添加通用训练/推理 YAML，任务参数放 recipe，补充模型环境依赖。
5. 最后在本地工具中验证真实权重、GPU、分布式和目标硬件表现。

只改变单步 loss 的算法可换 `model.target` 或 `training_step()`，继续使用 BCTrainer。
未来需要 rollout、多模型、多优化器的 RL 算法，应实现具体 Trainer，并补齐 Trainer 选择、
算法配置 fingerprint、命名恢复组件、环境/数据进度和部署 actor 导出。当前 CLI 不支持
`algorithm.target` 或 `RLTokenTrainer`，不能把规划配置当作可运行示例。

RL 私有 logprob、reward、advantage 和 token mask 留在算法模块；最终部署仍输出公共动作块。
不要提前新增模型继承协议，也不要将关闭梯度的 `InferencePolicy` 用作训练 rollout 接口。

代码规则与验收清单集中在 [代码接入规则](code_integration_and_change_rules.md)，
环境命令在 [环境指南](environments.md)，测试归属和执行方式在 [测试指南](../tests/README.md)。

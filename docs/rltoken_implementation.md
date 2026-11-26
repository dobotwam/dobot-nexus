# RLToken 开发与模块验证

残差 TD3（`rlt_td3_residual_v1`）、Token-only、离线缓存/单阶段联合预热和回合定稿的当前入口，见
[残差 TD3 配方](rltoken_td3_recipe.md)。下文的 `rlt_ac_fixed_v1` / `rlt_ac_terminal_v2`
按各自显式配置使用；不同 recipe 的恢复状态不可互换。

当前实现的职责划分：`algorithms/rltoken/`
管理模型无关的算法和训练状态，`models/pi05/rltoken/` 提供首个特征模型，RLinf 管理
远程执行。真机使用单环境；仿真支持一个向量环境和批量 Rollout 推理。两个模型 GPU
Worker 分别负责 Rollout/Learner，Env Worker 可显式分配仿真与渲染 GPU。
CPU 单元测试验证算法与协议；真实 Ray 进程、多 GPU 训练闭环、数值/吞吐、
控制器时延和任务成功率分别在对应环境验收，不能互相替代。

未完成的设计与验收集中在 [TODO](../TODO.md)。设计稿中的规划
不代表已运行的能力，以下按当前代码说明模块边界和调用路径。

## 框架归属与调用路径

| 归属 | 当前入口与职责 |
| --- | --- |
| 公共契约 | `core/rl_contracts.py` 定义特征、动作坐标与身份；`models/base.py` 提供可选特征提取/执行动作编码能力；不引入第二套模型基类 |
| 公共训练入口 | `core/config.py`、`cli/train.py` 选择 Trainer；`algorithm: bc` 使用既有 BCTrainer，`algorithm.target` 加载算法 Trainer |
| 算法计算与状态 | `algorithms/rltoken/` 拥有 Token、Actor/双 Q、loss、transition、Replay、Learner、checkpoint 和同步协议 |
| 模型集成 | `models/pi05/rltoken/` 拥有 Stage 1 联合训练、冻结特征提取、Pi0.5 数据适配和控制器 codec；算法通过 target/Artifact 加载提供者 |
| 分布式执行层 | `src/dobot_nexus/runtime/ray/` 隔离 RLinf/Ray；`algorithms/rltoken/{topology,roles}.py` 声明纯 Python 拓扑与角色并委托业务对象，由 `RLTokenTrainer.run()` 延迟加载 |
| 公共部署入口 | `InferencePolicy` 加载 `RLTokenPolicy`，组合冻结提供者和 Actor；部署 Artifact 自带推理资产 |
| 共享计算组件 | `modules/` 仅承载可复用计算；现有 `modules/value/` 服务 Fusion Value，RLToken Token/Actor/Critic 当前保留在算法包 |

Stage 1 调用链为 `train CLI → BCTrainer → Pi05RLTokenModel`。同一次 VLA forward
产出 flow loss 和 prefix hidden states，Token Encoder/Decoder 使用 detach 后的 prefix
训练重建；BCTrainer 统一反向传播、更新和导出。Token 联合训练是模型训练分支，
不是 RLinf Worker，也不使用 Stage 2 Replay。

Stage 2 调用链为 `train CLI → RLTokenTrainer → RLinfExecution → 三角色 Worker`。
Env 取得观测，Rollout 用冻结提供者和当前 Actor 生成动作；Env 返回控制器确认的执行
事实；Learner 持久化 transition 并返回 ACK，再按预算更新 Actor/双 Q。
Actor 快照经独立通道发布，Rollout 在请求边界切换并返回 applied ACK。
持久 ACK 与权重 applied ACK 的含义和通道独立。

当前每个角色一个 Worker、单节点静态放置，Rollout/Learner 各占不同 GPU。协调器可让
角色 RPC 交错推进，但 Learner Worker 内接收/提交与有限次更新顺序执行；
`Learner.prefetch()` 和 checkpoint barrier 已存在，实际 Worker 尚未启用预取或独立
接收线程。向量仿真在单个 Env Worker 中推进多环境，Rollout 合批执行特征与 Actor。

## 子模块与单元测试

| 子模块 | 已实现职责 | 测试模块 |
| --- | --- | --- |
| 公共能力与入口 | 可选特征/动作编码能力、严格配置分发、单 driver 启动、轻量导入 | `test_contracts`、`test_training`，既有配置/通信依赖测试 |
| Token 表示 | 可配置 Encoder/因果 Decoder、teacher 右移、detach、padding 和 FP32 loss | `test_token` |
| Actor / Critic / loss | 固定 std 的 tanh Actor、双 Q、分块折扣、Q1 Actor 目标、BC 正则 | `test_losses` |
| Transition / Replay | 实际命令、接管参考副本、固定块准入、SQLite 持久 ACK、去重与 UTD 预算 | `test_replay` |
| Learner / resume | 独立优化器与时钟、Target Q、失败停止、预取屏障、完整恢复点 | `test_learner` |
| Actor 同步 | 版本/身份/摘要校验、双缓冲、请求边界切换、applied ACK | `test_weight_sync`、`test_workers` |
| Env / Rollout | 有限采集、观测特征复用、执行确认、超时查询、终止观测处理 | `test_env_rollout` |
| 仿真批量采集 | 向量执行、批量特征/Actor、独立 episode/cache/reset、逐样本准入与批量 ACK | `test_simulation`、`test_workers`、`test_contracts` |
| 组合策略 / Artifact | 特征与 Actor 组合、冻结参数/buffers、嵌套资产、移动后独立部署 | `test_policy` |
| Pi0.5 Stage 1 | 单次 forward 的 prefix、联合损失、完整/特征资产严格加载、绝对命令与归一化关节增量 codec | `tests/models/pi05/rltoken/test_stage1.py` |
| RLinf 执行适配 | 三组 Worker、容量为 1 的独立数据/ACK 通道、有限 RPC、同步与关闭 | `test_training`、`test_workers`；固定版本 API smoke 单独执行 |

算法测试位于 `tests/algorithms/rltoken/`，不用安装 RLinf。Worker 方法测试只替换
外部 `Worker` 基类并使用有界内存通道，验证收发和业务状态，不能代替真实 Ray 通信验证。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m unittest discover \
  -s tests/algorithms/rltoken -t . -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m unittest discover \
  -s tests/models/pi05/rltoken -t . -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m unittest discover -s tests -v
python -m compileall -q src tests
```

## Stage 1：训练可冻结的特征模型

模板：[`stage1_pi05.yaml`](../examples/configs/rltoken/stage1_pi05.yaml)。它继承普通
Pi0.5 BC 配置，仍使用 `BCTrainer`、现有数据流水线和 Artifact 保存接口。
先复制模板并填写数据集、基础权重、机器人维度以及 `action_codec.low/high`；边界未填写
时在加载权重前明确失败。`control_dt` 必须与控制器每步间隔一致。

```bash
examples/train.sh --config examples/configs/rltoken/stage1_pi05.yaml --gpus 0,1 \
  -- data.path=/path/to/dataset model.checkpoint=/path/to/base_checkpoint
```

`token.input_dim` 对应模型的 prefix hidden width；默认 Gemma 2B 为 2048。训练目标为
`token_loss + alpha_vla * flow_loss`，`alpha_vla` 必须为正。Token 的输入和监督均与
VLA 梯度断开。普通 BC 权重使用 `initialization: bc`；完整 Stage 1 权重初始化使用
`initialization: rltoken`，后者缺 Encoder 或 Decoder 均报错。已有训练状态恢复仍遵循
BCTrainer 的恢复接口。

完整 Stage 1 Artifact 保存 VLA、Encoder、Decoder。作为 Stage 2 特征提供者加载后，
显式丢弃已验证的训练专用 Decoder；后续导出的内部特征包只保存 VLA 与 Encoder。
特征身份覆盖权重、Token 配置、tokenizer、归一化、动作 codec、控制周期和数值 profile。
首版冻结提供者保留 FP32 参数，按 `model.precision` 做 autocast，传输使用独立 CPU FP32
副本。该路径没有启用 CUDA Graph 或推理编译，也不支持 Pi0.5 Memory 状态。

Pi0.5 的 `absolute_affine_v1` codec 将控制器绝对命令的声明范围映射到
`[-1,1]`。先由原 Pi0.5 pipeline 将参考动作还原到控制器命令空间，再做 RL 编码；
不能直接把 BC 的 z-score/quantile 值当作 Actor 的 tanh 坐标。实际执行越界会标记
不可编码；参考动作越界会明确失败，不做静默裁剪。

`normalized_joint_delta_v1` 用于已经归一化为 `[-1,1]` 的关节增量控制命令。
编码与解码逐位保留有限 FP32 命令，包括接近零的微小增量。
它要求 `units: normalized_joint_delta_and_gripper`，并记录控制器的
`mode: pd_joint_delta_pos`、`delta_reference: current_qpos`、各关节增量范围和
夹爪位置范围。控制器参数参与特征身份，避免在相同 8 维数值下混用不同物理语义。
此时数据变换使用 `type: absolute`，含义是原样保留数据集命令，不再次做差分。
默认拒绝越界参考动作；显式设置 `reference_bounds: clip` 时，仅将 VLA 参考命令
裁剪到控制器允许范围，然后编码参考和实际执行动作。真实反馈越界仍标记不可编码，
不会通过裁剪掩盖控制器实际执行的命令。

Pi0.5 允许配置非空的相机槽位子集。`model.io.cameras` 与
`data.field_mapping.images` 必须指向相同的核心槽位；缺失槽位使用黑图和 false mask，
训练与推理保持一致。使用双槽位训练的模型需要完整替换继承配置中的三槽位映射，
不能保留值为 `null` 的第三路键。若 BC 权重训练时已将同一手腕图像填入左右两个
有效槽位，接续 Stage 1 和推理必须保留这一映射，不能直接将其中一路改为无效。
归一化统计、tokenizer、数值 profile 和相机映射均以 BC 训练配置为准。
Pi0.5 同时接受字符串归一化配置和 `{type: quantile, version: 1}` 等 v1 Artifact
配置；未知版本拒绝加载，避免静默改变 codec。

## Stage 2：真机采集与学习

模板：[`stage2.yaml`](../examples/configs/rltoken/stage2.yaml)。默认 `C=1`，先减少
真机提前终止造成的短块排除；更改 C 时同步修改 `model.action_horizon` 和
`model.rltoken.chunk_size`，部署的 `execute_steps` 必须相同。

安装 RLinf 到训练环境，固定版本见
[`requirements-rlinf.txt`](../src/dobot_nexus/algorithms/rltoken/requirements-rlinf.txt)：

```bash
python -m pip install -r src/dobot_nexus/algorithms/rltoken/requirements-rlinf.txt
RLT_FEATURE_ARTIFACT=/path/to/stage1/artifact \
RLT_ENV_TARGET=robot_package.agent:RobotAgent \
examples/train.sh --config examples/configs/rltoken/stage2.yaml --gpus 0,1
```

`runtime.resources` 使用 RLinf Cluster 枚举的硬件编号，单节点通常对应物理 GPU 编号，
不按 driver 的 `CUDA_VISIBLE_DEVICES` 重编号。启动时检查 placement 日志中的
`visible_accelerators` 与预期设备一致；两个模型角色必须分配不同设备。RLinf 放置
控制进程可见设备，不提供对同机其他任务的资源独占。CPU Env 不加载 VLA。模板只启动
一个 driver，禁止再包一层 torchrun。首版 Learner 使用 FP32 Adam 和常量 scheduler；
拒绝继承 BC 的 FSDP、EMA、梯度累积和公共 optimizer/scheduler 配置。

数据满足 warmup、Actor 更新达到 `environment.min_actor_updates` 且对应版本已应用
后，Actor 才具备执行资格；已验收的离线初始化可在对应版本 applied ACK 后就绪。
`algorithm.policy_switch.execution_scope` 控制执行范围，`training_scope` 控制训练范围，
均支持 `all` 与 `critical_phase`，默认 `all`。启用全任务关键阶段控制时，环境观测须
携带 `control.rlt_switch_flags: bool[B]`；就绪与阶段门控独立，关键阶段的 VLA 预热数据
仍可入库。完整配置、适配器职责和边界语义见[残差 TD3 配方](rltoken_td3_recipe.md#执行阶段与训练范围policy_switch)。
GPU 空闲不会增加 UTD。每个唯一、最终准入的
transition 增加整数更新预算；成功 critic 更新扣除预算。默认 `rlt_ac_fixed_v1`
拒绝未完整执行的块；`rlt_ac_terminal_v2` 的终止前缀规则见下文“仿真批量采集”。
执行状态未知、缺 reward 或动作不可编码时保留记录但不训练。

真机设置 `environment.kind=robot`、`environment.batch_size=1`；大于 1 会在预检时报错。
机器人适配器由 `environment.target` 构造，构造参数为 `environment` 配置。
它是普通 Python 对象，必须提供：

```python
class RobotAgent:
    def __init__(self, config): ...
    def reset(self, *, timeout) -> ObservationBatch: ...
    def execute(self, command_id, commands, *, timeout) -> ExecutionFeedback: ...
    def query(self, command_id, *, timeout) -> ExecutionFeedback: ...
    def close(self): ...
```

`ExecutionFeedback` 定义在 `algorithms/rltoken/env.py`。`commands` 是经过限幅、接管
后由控制器确认的最终命令，形状为 `[1,C,A]`；短块按协议填充，`executed_steps` 标记
真实长度。`takeover[C]` 表示已执行的整步接管，`rewards[C]` 表示逐控制步奖励，
`final_observation` 必须来自 reset 前。命令 ID 的查询必须幂等；执行调用超时后 Nexus
只查询，不重新发送电机命令。适配器负责真正落实 I/O deadline，Nexus 无法取消已经
发生的物理动作。连接或状态异常需要先确认控制器状态再开启新会话。

执行入口支持 `reward_mode=per_step` 和 `episode`。回合模式先持久化 pending，再由
适配器的 `finalize_episode` / 可选 `poll_finalizations` 返回最终标签；不能先训练临时
reward 再覆盖。阶段范围排除的记录不进入 pending。具体控制器和仿真 SDK 适配保留在环境包，本仓库不自造
硬件端点或导入本地调试脚本。

## 保存、恢复和部署

输出位于 `<run_dir>/rltoken/`：

- `journal.sqlite`：全量持久事件和去重索引，Replay 的 count/bytes 上限只限制热数据。
  首版不自动清理 journal 或旧 checkpoint，需按任务长度安排磁盘空间。
- `features/`：首次初始化从当前已加载提供者导出的固定特征资产，无需原 Stage 1 目录。
- `checkpoints/step-*/`：私有 `rltoken_resume_v1`，包含 Actor/Q/Target、双优化器与
  scheduler、RNG、Replay、预算和 journal 备份；manifest 引用保留的 `features/`。
- `deployments/actor-*/`：公开 v2 Artifact，含 Actor 和独立嵌套特征包，不含 Q/Replay。
- `checkpoints/latest_resume`、`latest_deployment.txt`：只指向写完的产物；Actor 未就绪
  时仍可保存恢复点，但不发布部署指针。

```bash
examples/train.sh --config examples/configs/rltoken/stage2.yaml --gpus 0,1 \
  -- train.resume_from=auto
```

恢复同一 run 会保留 journal 中 checkpoint watermark 之后的已提交事件，只有新的
准入事件补回预算。迁移训练任务时保留整个 `rltoken/` 目录，不能只搬走 checkpoint
而丢弃其 `features/`。恢复时使用新的 learner/session epoch；机器人从新的 episode
开始，不声称恢复物理世界中途状态。

部署使用公共 `InferencePolicy`，`model.artifact` 指向 `latest_deployment.txt` 或
具体 `deployments/actor-*` 目录，`model.target` 为
`dobot_nexus.algorithms.rltoken.policy:RLTokenPolicy`。内部资产缺失或身份不符立即失败，
不会回退访问原训练目录。部署和普通 BC 导入链不要求安装 RLinf。

## 当前验证范围

实现使用 RLinf 同版本的目标函数约定和生命周期 API；Token/MLP 是 Nexus 自有的
可配置 Torch 实现，权重不与上游模块直接互换，数值等价与任务效果需另行比较。
源代码归属见 [`NOTICE.md`](../src/dobot_nexus/algorithms/rltoken/NOTICE.md)。

真实环境验收应分别记录 Ray 通信与异常清理、Actor/双 Q 参数更新、两 GPU 稳态
吞吐与数值、保存恢复、冻结评测以及真机控制时延。MetaFine 字母插槽的协议与
验收方法见[仿真评测文档](maniskill_evaluation_plan.md#metafine-corl-字母插槽任务)。
这些 smoke、性能脚本及报告按仓库约定放在 Git 忽略的 `local_script/`。CPU 单元测试
通过表示上述模块契约成立，不表示已经取得训练加速或真机任务成功。

## 仿真批量采集

模板：[`stage2_simulation.yaml`](../examples/configs/rltoken/stage2_simulation.yaml)。
`environment.batch_size` 是并行环境数；`algorithm.learning.batch_size` 是 Replay
训练 minibatch，两者独立。默认模板分别为 16 和 256。

```yaml
environment:
  kind: simulation
  batch_size: 16
  env_id: sim
  target: your_simulator.agent:VectorAgent
runtime:
  resources:
    rollout_gpu: 0
    learner_gpu: 1
    env_gpu: 0
```

每个角色仍只有一个 Worker。Env 通过一次 `execute_batch` 推进 B 个环境；Rollout
把未命中缓存的观测合为一次 `extract_rl_features`，把 Actor 请求合为一次 forward。
Actor 权重版本在整批请求内固定。解码、命令编码和 Replay 行拆分仍逐环境处理，
这些轻量操作不重复运行特征模型或 Actor。

每个环境有独立 episode、观测序号和最近两个特征包。某个环境结束只 reset 对应槽位；
其他环境的 next features 在下一次决策复用。随机参考动作在固定 batch 分组和 reset
顺序下可复现；改变 batch 或分组不承诺逐样本噪声一致。已有缓存的参考动作不会重采样。
外部特征模型若支持多环境，需返回 `RLFeatureBatch.action_contexts`：每行一个完整的
单环境 codec 上下文。算法不猜测任意 JSON 列表的 batch 维度；Pi0.5 已提供此字段。

向量仿真适配器是普通 Python 对象，实现以下方法：

```python
class VectorAgent:
    def __init__(self, config): ...
    def reset(self, env_indices, *, timeout) -> ObservationBatch: ...
    def execute_batch(self, command_ids, commands, *, timeout) -> list[ExecutionFeedback]: ...
    def query_batch(self, command_ids, *, timeout) -> list[ExecutionFeedback]: ...
    def close(self): ...
```

`reset` 只重置给定槽位，返回同顺序观测；`env_ids` 必须为 `<env_id>-<slot>`。
`execute_batch` 接收 B 个唯一 command ID 和 `[B,C,A]` 命令，按槽位返回 B 个
`ExecutionFeedback`；每行的 commands 仍为 `[1,C,A]`，rewards/takeover 为 `[C]`。
支持向量化的模拟器应通过向量执行接口推进 B 个场景。仅有独立 CPU 场景的环境也可在
每个控制步收集 B 路结果并共用一次模型推理，但这不提供 GPU 物理并行；应在评测配置
中明确后端。不能在 adapter 内把 B 个环境逐个串行跑完整 rollout。
每个环境的 `final_observation` 必须在 reset 前截取，禁止在 chunk 中途 auto-reset。
提前结束后剩余动作不计入有效步数，短块保留在 journal；默认 `rlt_ac_fixed_v1` 不准入短块。

插孔等提前成功任务使用 `algorithm.recipe: rlt_ac_terminal_v2`，可以将明确
`terminated=True` 的非空连续前缀加入 Replay。Rollout 保留决策时完整候选动作；
Critic 使用实际执行前缀加原候选后缀，TD 只累加有效奖励、关闭 bootstrap。
未执行后缀的 reward 必须为零、takeover 必须为 false；实际执行 mask 和步数不补满。
超时、截断、缺候选、执行状态不明仍拒收。`rlt_ac_fixed_v1` 保持全部短块拒收，
两种 recipe 的 Replay 分区和 checkpoint fingerprint 不兼容。

命令超时仅调用一次 `query_batch`，不重新执行向量命令。查询失败时全批记录为
unconfirmed 并停止；查询成功可以返回各槽位独立的 confirmed 状态。错误的 command ID、
批量 ACK 或观测归属会停止采集。收到整批持久 ACK 后才开始下一轮。容量为 1 的通道
现在传输一批数据，`runtime.max_message_bytes` 仍限制整个消息大小；Replay 按唯一准入
transition 逐条增加 UTD 预算，不按 batch 增加一次预算。

`chunks` 统计所有环境的动作块总数。`environment.max_chunks` 在完整批次边界检查，
最后一批最多超出 B−1 条；`train.max_steps` 继续严格限制 Critic 更新次数。
一个仿真槽位状态不明时当前向量采集器整体停止，不继续从其余槽位采样。

`env_gpu` 可与 `rollout_gpu` 共用，此时要为模型、渲染和物理状态共同预留显存；也可
指定第三张 GPU。省略或设为 null 时 Env 隐藏 CUDA，仅适合不依赖 GPU 的环境。
GPU 仿真与 RGB 渲染需显式填写 `env_gpu`。Worker 内设备编号为 `cuda:0`，不是宿主机
物理编号。外部环境 adapter 负责验证模拟器的实际设备映射。

```bash
RLT_FEATURE_ARTIFACT=/path/to/matching/features \
RLT_SIM_TARGET=your_simulator.agent:VectorAgent \
examples/train.sh --config examples/configs/rltoken/stage2_simulation.yaml --gpus 0,1 \
  -- environment.batch_size=16
```

模板继承真机的默认 state/action 维度，占位模型必须替换为与仿真控制器匹配的参数和
Artifact。ManiSkill 的 `pd_joint_delta_pos` 不能直接使用 Pi0.5 的绝对命令 codec。
外部仿真包需实现上述向量 adapter，并验证 Panda 控制器缩放与 codec 元数据一致。
诊断模型的采集检查、正式 Pi0.5/RLToken 权重的控制语义对齐和任务成功率分别验收；
诊断工具与报告放在 Git 忽略的 `local_script/`。

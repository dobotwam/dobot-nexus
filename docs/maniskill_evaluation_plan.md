# ManiSkill 仿真评测搭建与验证计划

首期目标是运行单机、单环境、可复现的视觉策略评测，输出逐 episode 结果和视频。
先通过基础环境检查，再建立 RLinf 原生基线，最后接入 Nexus `InferencePolicy`。
GPU 并行仿真和 RLToken 在线训练分别验收；单环境评测可以使用 CPU PhysX 加 GPU 渲染。

本协议对应 [RLToken 架构](rltoken_nexus_architecture.md)。环境、模型闭环与并行吞吐
分别验收，机器诊断、日志和 smoke 工具保存在 Git 忽略的 `local_script/`，
不作为生产模块依赖或通用能力声明。

## 任务选择

| 用途 | 任务与协议 | 验收范围 |
| --- | --- | --- |
| 基础环境检查 | ManiSkill 原生 `PickCube-v1`，单 Panda、随机动作 | reset、step、RGB、终止、关闭；不能据此评价模型能力 |
| RLinf benchmark 主线 | `PutOnPlateInScene25Main-v3`，`obj_set=train` | 使用 RLinf 的固定初始状态、观测处理和成功率定义建立 ID 基线 |
| RLToken 专项 | RLinf `PegInsertionSideWideClearance-v1` 的 joint observer 变体 | 验证 Pi0.5 特征、动作编码和整任务策略；与 Put-on-Plate 分开统计 |
| 泛化评测 | RLinf Put-on-Plate 的 Vision、Semantic、Execution 变体 | ID 跑通后按官方 OOD 清单执行，逐场景汇总 |

RLinf 的独立评测示例目前提供 OpenVLA-OFT 配置；Pi0.5 可以从训练配置提取
`env.eval` 和 `rollout`，设置 `runner.task_type=embodied_eval`、`runner.only_eval=True`。
提取时先解析引用的模型、precision、group size 和 chunk 配置，避免删除 `actor` 或
`algorithm` 后留下悬空插值。[RLinf 评测指南](https://rlinf.readthedocs.io/en/latest/rst_source/evaluations/guides/maniskill_ood.html)

主线优先采用本地 RLinf 配置引用的
[Pi0.5 ManiSkill SFT 权重](https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-SFT)，
先在原生 RLinf 中验收。该权重不是现成 Nexus Artifact；转换和观测对齐需单独验证。
插孔任务需要匹配 Panda、相机与控制模式的 Stage 1/Actor 权重，不能用 Put-on-Plate
或真机双臂权重直接代替。基础环境验证不需要这些权重。

## 依赖和资源

- 固定 RLinf 源码到 `7e90641c4135229db6d3426950cdb45cee6259a6`，保存实际工作树差异。
  该版本的安装脚本固定 ManiSkill `v3.0.0b22`、SAPIEN `3.0.1`；先沿用此组合。
- 使用 Python 3.11 独立环境，记录完整 lock/freeze、Torch CUDA 构建、驱动、PhysX 库
  和资产 revision。确认 CUDA 构建支持目标 GPU 架构，通过实际运行检查后再锁定版本。
- 分别验证 CUDA 计算、SAPIEN 渲染、PhysX GPU；`nvidia-smi` 能枚举设备不等于三者可用。
  `vulkaninfo` 可能因枚举其他设备失败，目标设备的实际 RGB 场景运行是必要验收项。
- 单环境首期只使用一张已验证设备，显式记录物理 GPU UUID/PCI 和进程内逻辑编号。
  渲染不能只依赖 Ray 分配或 `CUDA_VISIBLE_DEVICES` 的假设，必须实测设备映射。
- 环境、缓存、权重、视频放在指定数据盘目录。下载前估算大小、检查剩余空间；视频按
  episode 抽样保留，避免模型副本和所有环境录制占满根文件系统。

ManiSkill 的 RGB 渲染需要可工作的 Vulkan/SAPIEN 链路，headless 评测不需要桌面窗口。
[ManiSkill 安装说明](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)

## 分阶段实施与验收

| 阶段 | 工作 | 通过条件 |
| --- | --- | --- |
| P0 环境预检 | 包版本、源码任务注册、逐设备 CUDA、CPU PhysX、RGB、GPU PhysX | 明确至少一个可运行的单环境组合；各失败项保留命令、返回码和日志 |
| P1 单环境 benchmark | 补齐 RLinf 场景资产，先跑 PickCube，再跑 Put-on-Plate ID | 连续 10 个完整 episode；有限 reward、非空 RGB、正确 reset/终止、无资源泄漏；保存图片和短视频 |
| P2 原生策略基线 | 使用对应 ManiSkill 权重运行 RLinf eval-only | 固定 episode 清单、保存配置和逐 episode 结果；先抽样验收，再执行完整 ID trial 清单 |
| P3 Nexus 接入 | 观测映射、动作映射、Nexus Artifact 加载、指标记录 | 各子模块 CPU 测试通过；固定输入动作对齐；相同初始状态清单下完成闭环评测 |
| P4 并行评测 | GPU PhysX 验收后，环境数按 1、4、16、32 递增；需要跨进程时复用 RLinf Worker/Cluster | 记录推理、物理、渲染、传输耗时、显存和 episodes/s；结果无漏计或重复计数 |
| P5 RLToken 专项 | 插孔环境与模型动作语义对齐，评测冻结 Actor；在线训练另立验收 | 区分 reference、纯 Actor、reference/Actor 门控策略；评测关闭专家接管；通过后才接 Learner/Replay |

P1 的 CPU PhysX 加 GPU 渲染路径允许先推进单环境评测。GPU PhysX 不通过时，P4
保持待验收，不把单环境 smoke 的用时作为 GPU 并行吞吐结果。

## 复用边界

```text
RLinf 的任务定义和 ManiSkill 场景
        ↓ RGB / state / prompt
仿真观测适配 → Nexus InferencePolicy → 动作适配 → ManiSkill step
        ↓                                      ↓
episode 身份与 reset 状态                 reward / done / success
        └──────────── 评测记录器 ────────────────┘

需要并行时：由 RLinf Cluster / Worker 放置环境与策略进程
```

| 子模块 | 设计边界 | 验证方法 |
| --- | --- | --- |
| 环境工厂 | 复用固定版本 RLinf 注册任务、场景变体与 seed/reset 规则；延迟导入 ManiSkill | 缺依赖错误、配置校验；真实 reset/step 在本地 smoke 中验证 |
| 观测适配 | 生成 `ObservationBatch`，明确相机键、RGB uint8/BHWC、state 顺序、prompt 与 env_id；不泄漏任务真值给策略 | 用固定数组验证相机缺失、batch、dtype、维度和向量环境切片 |
| 动作适配 | 区分网络输出、归一化动作、控制器实际命令；明确单位、范围、夹爪方向和执行步数 | 边界值、往返编码、越界、NaN、chunk 截断以及夹爪方向测试 |
| 策略调用 | 统一调用现有 `InferencePolicy.predict_chunk`，由 Artifact 的 `model.target` 选择 BC 或 RLToken | fake policy 验证调用/重置；真实权重固定输入对齐在 `local_script/` 中完成 |
| 评测循环 | 显式 reset，先记录最终观测与指标再重置；单环境先行，向量化后保持各环境独立计数 | 提前成功、超时、终止在 chunk 中间、部分环境 reset、尾部不足 batch 的测试 |
| 统计与落盘 | 配置快照、逐 episode JSONL、summary JSON、抽样视频；记录 checkpoint/资产/源码摘要 | 防重、漏 episode 检测、success_once 聚合、错误与截断不混淆 |

环境适配器、评测 runner、benchmark 配置及诊断测试按仓库规则放在
`local_script/maniskill/` 或独立环境包。`src/`、`tests/` 不反向导入本地工具。
若接入确实需要修改稳定契约，相关 CPU 契约测试放在 `tests/`；模型专属处理仍归
`models/<model>/`。不把 RLinf policy wrapper、Ray 或模拟器放入模型算子层。

RLToken 执行器支持 `environment.kind=simulation`、`environment.batch_size` 和
`runtime.resources.env_gpu`。向量仿真可以批量提取特征、批量 Actor 推理，逐环境
保存 episode/cache/Replay 状态；接口与配置见[批量采集说明](rltoken_implementation.md#仿真批量采集)。
真机模式继续隐藏 Env Worker 的 CUDA。视觉仿真即使使用 CPU PhysX，也可能需要
GPU 渲染，必须显式声明 Env GPU。正式模型闭环与并行吞吐仍按 P3/P4 验收。
独立评测不需要启动 Learner、Replay 或训练权重同步，也不需要重写调度器。

## RLToken 的专项边界

Nexus 插孔接入使用 `local_script/maniskill_peg/`：环境适配器直接注册固定版本
RLinf 的插孔变体，训练入口调用 Nexus `BCTrainer` / `RLTokenTrainer`，部署与评测
调用 Nexus `InferencePolicy`。RLinf 提供任务定义及 Stage 2 的 Cluster/Worker/Channel
执行设施。任务脚本、配置、权重和诊断报告按仓库规则保留在 Git 忽略目录。

示范来源为 [RLinf 的 400 条插孔成功示范](https://huggingface.co/datasets/RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ)，
固定数据 revision `2b92d5ef3fe274d30219130133f9e34c7ab91ebf`。数据为嵌入图像的
LeRobot v2.1 Parquet；Nexus 的读取器同时支持这种格式与视频形式，动作块不跨 episode。
环境包负责下载、数据准备和任务入口；新 clone 需单独准备这些本地工具。
完整数据配置与小样本闭环配置分开，评测必须同时记录示范数量和真实更新次数。

Stage 1 对通用 Pi0.5 基座联合执行任务 BC 与 Token 重建训练。普通基座不能称为
已经完成插孔 BC；冻结参考策略成功率是后续效果比较的前置证据。
Stage 2 插孔配置使用 `rlt_ac_terminal_v2`，以保留 chunk 中途成功的奖励，详见
[动作块准入设计](rltoken_nexus_architecture.md#动作块与-replay-准入)。
当前 Nexus 采用整任务 reference warmup → Actor 路由与固定 loss 权重；
这不包含 RLinf 示例的自动关键阶段切换、专家接管和完整权重调度。

固定版本 RLinf 的 `rlt_openpi_joint` 会把插孔任务解析成
`PegInsertionSideWideClearanceObserverWideWrist-v1`：`panda_wristcam_wide` 机器人、
`3rd_view_camera` 和 `wide_hand_camera` 两路 384×384 图像、9 维 qpos、8 维动作，
控制模式为 `pd_joint_delta_pos`。参考配置为 100 Hz 仿真、10 Hz 控制、每次执行
10 步动作、最多 500 步。该协议与通用 Put-on-Plate 分开维护。

Nexus Pi0.5 RLToken adapter 提供 `normalized_joint_delta_v1`，标注控制器缩放、
夹爪语义与版本，并纳入 feature identity 和 Artifact 元数据。增量控制应使用该
codec，不能只把 `absolute_affine_v1` 的上下界填写为 `[-1,1]`。codec 仍由模型的
数据 adapter 拥有，算法不绑定 Pi0.5。
若改为绝对关节控制，必须另立 task/config/权重基线，不能声称与原 RLinf 任务等价。

先完成该 codec 和匹配权重的验证，才能评测 RLToken Actor。Stage 1 特征权重与 Actor
必须身份匹配；普通 BC 权重不能补一个未训练 Token Encoder 后冒充 Stage 1。
参考策略、Actor 和 Critic 使用相同控制周期、动作单位和固定 chunk 语义。

评测允许 episode 在 chunk 中间结束，只累计实际执行的步数，不把余下动作执行到
新 episode。训练 Replay 的固定块准入规则继续独立生效；禁止把评测记录未经检查
直接作为训练 transition。RLinf 的整任务 reference/Actor 自动切换与 Nexus 当前
warmup 后切 Actor 的规则不同，两种策略必须分别命名和报告。

## MetaFine CoRL 字母插槽任务

CoRL 数据中的 `insert_letter` 来自
[MetaFine Challenge CoRL](https://github.com/Hiangx-robotics/MetaFine-Challenge_CoRL)，
是单独注册的 ManiSkill 任务，与上述 RLinf PegInsertion benchmark 分开配置。
每个 episode 指定 C、o、R 或 L 中的一个目标字母；四种指令分别评测，不能只报告
任意字母插入成功。官方环境还检查位置、旋转、插入高度、松爪和物体静止状态。

训练数据使用 `insert_letter/mixed/lerobot`，不重复加入四个单字母数据目录。
LeRobot v3 数据提供 9 维 Panda qpos、8 维归一化关节增量命令，以及
`observation.images.base_camera`、`observation.images.hand_camera` 两路图像。
环境需显式将两路相机都设为 512×512，然后由模型预处理缩放到 224×224。
相机槽位必须遵循所选 BC 权重的训练映射：双槽位模型的第三槽位填充黑图并置为
无效；训练时将手腕图像填入两个有效槽位的模型则保留该映射。这仍然只有两个
物理视角，不得在接续训练或评测时擅自改变 mask。

数据时间戳为 30 FPS；任务控制器为 20 Hz，物理仿真为 200 Hz，因此 RL codec 的
`control_dt` 使用 0.05。Panda 手臂每步增量缩放为 ±0.1 rad，夹爪位置范围为
[-0.01, 0.04]。数据已经保存控制器归一化增量，训练变换保持原样。附带归一化统计
与所选 BC 权重须匹配；多任务 BC 权重使用其随包混合统计，不替换为单任务统计。

接入 BC 权重时先校验权重、统计和 tokenizer 摘要，核对训练来源是否包含目标任务，
并独立评测未经 Stage 1 修改的 BC 策略。保留该策略原有的执行步数、采样种子和
数值精度；若 RL 使用不同 chunk，另做相同 chunk 的基线。历史随包评测与本次实测
分别记录。更换 BC 权重后重新训练 Token 模块并建立新的特征身份，不沿用旧
Actor、Critic 或 Replay。

先验收真实模型的 Stage 1 联合更新、在线采集与 Actor/双 Q 更新、保存恢复和冻结
Artifact 评测，再扩大训练与效果实验。冻结评测保留官方 400 步上限，参考策略与
Actor 使用一致的留出 seed、控制周期和执行 chunk；记录逐字母成功率、回报、长度，
以及评测前后权重是否一致。少量工程 seed 不等同于官方隐藏测试或规划器筛选的
20-seed benchmark。环境适配、权重转换、诊断脚本、运行配置与结果保存在
`local_script/insert_letter/`，生产模块不导入它们。

## 指标与复现协议

- 主指标 `success_once`：每个 episode 曾出现一次 `info["success"]` 即记为成功。
  同时记录 `success_at_end`、reward 总和、实际步数、terminated、truncated、运行错误。
- Put-on-Plate 的论文对比保留 RLinf 的 `auto_reset=True`、
  `ignore_terminations=True`、固定 `episode_id/reset_state_id` 和 80 步时限；不能用
  首次成功立即结束的 smoke 结果代替该协议。区分 episode 指标和自动 reset 返回的数据。
- 先用 10 个固定 trial 检查链路，再根据任务 `total_num_trials` 枚举正式 ID 清单。
  报告覆盖数、成功数、成功率和置信区间；抽样结果明确标注子集，不外推完整 benchmark。
- OOD 按官方清单逐场景评测，区分 `obj_set=train/test`。每个场景独立汇总后再做
  macro average，保存 episode 清单，避免并行数改变时重复或漏掉初始状态。
- 同时记录 seed、模型采样参数、去噪步数、chunk/execute_steps、控制频率、图像预处理
  和数值 profile。seed 按 task/episode/step 确定，不依赖 worker 数量。
- 性能测试分别计时模型 forward、渲染、物理步进和传输；GPU 计时同步，初始化与 warmup
  单列；固定视频开关。CPU/GPU backend 的数值与成功率差异需要独立对照。

## 资产与验证

环境预检在独立工具中执行，显式填写解释器、可用 GPU 和报告目录，设置有限超时，
保留命令、返回码及失败原因。诊断入口和机器运行记录不作为仓库公共启动接口。

P1 下载资产前固定 Hugging Face revision，并使用独立缓存目录。
[RLinf 场景资产](https://huggingface.co/datasets/RLinf/maniskill_assets)与 ManiSkill
自带的 WidowX/Bridge 资产是两套目录；已有后者不等于前者齐全。固定版本默认从
`rlinf/envs/sim/maniskill/assets` 加载专用场景，可在独立 RLinf checkout 下连接到数据盘
资产目录，保留文件清单和 license。`MS_ASSET_DIR` 与 RLinf 的 `MANISKILL_ASSET_DIR`
含义不同，配置时分别验证实际解析路径。

正式编码接入后执行各子模块 CPU 测试，再执行仓库检查：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m unittest discover -s tests -v
python -m compileall -q src tests
```

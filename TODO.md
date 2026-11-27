# 待办事项

- [ ] **POLICY-NORM：为 Pi0 增加可配置的 state/action 归一化**

  在 Pi0 私有 pipeline 中支持 `zscore`、`quantile` 选择，让训练、推理和动作反变换
  使用同一配置与统计资产。图像像素缩放、Value min/max 和 return 缩放独立管理。

  验收：覆盖统计字段与维度检查、action encode/decode 往返、Artifact 保存和恢复。
  模型私有实现保持独立；identity、min/max 按实际需求扩展。

  入口：[Pi0 pipeline](src/dobot_nexus/models/pi0/data/pipeline.py)、
  [Pi0.5 pipeline](src/dobot_nexus/models/pi05/data/pipeline.py)。

## RLToken 架构收尾

当前已实现 Stage 1 Token/VLA 联合训练、冻结特征与动作 codec、Stage 2 Actor/双 Q、
持久 Replay、完整训练恢复、Actor 权重同步、单节点三角色 Worker、向量仿真采集和
独立部署。具体职责、配置与命令见 [RLToken 实现说明](docs/rltoken_implementation.md)。
下列条目区分实现缺口、运行验收和条件扩展；已有模块不因缺少某项真实环境验收而
重新标成“未实现”。归属遵循 [AGENTS.md](AGENTS.md)：算法状态与执行适配归
`algorithms/rltoken/`，Pi0.5 集成归 `models/pi05/rltoken/`，可复用计算才进入 `modules/`。

### RLT-in-Practice 残差 TD3 实施状态

`rlt_td3_residual_v1`（pre-clip 残差 Actor、标准 Twin-Q、合法成功行为锚定、动态 Q/BC
权重调度、单阶段联合预热、`position_bottleneck` Token 及四项可选实验开关）已接入，
运行说明见 [残差 TD3 配方](docs/rltoken_td3_recipe.md)。旧版 Dobot 双残差配方已由
`rlt_td3_residual_v1` 替代；`RLT-TD3-VALIDATION` 的 GPU 数值、分布式与任务收益验收
仍独立保留。

### 实现与验收清单

- [x] **RLT-DOBOT-RECIPE：定义 Dobot-RLT 对齐配方与配置身份**

  对齐参考为 `/share/lujunyu/Dobot-RLT` 的实验 18 严格冻结 Explicit Residual-Q，
  以及其 Token、离线预训练和 TD3 更新实现。参考入口为 `rlt/networks.py`、
  `rlt/td3.py`、`scripts/pretrain_td3.py`、`scripts/rlt_online_session.py` 和
  `docs/05_DualResidual实验.md`；实现时记录参考文件摘要并核对来源许可。
  最新双残差真机结果在参考实验记录中仍为待填，不将其视为已验证的任务收益。
  下文 `algorithms/` 与 `models/` 源码路径均相对 `src/dobot_nexus/`。

  新增独立 recipe，显式配置 Actor/Critic 类型、训练阶段、loss reduction、动作边界、
  bootstrap 策略、step/chunk 折扣单位、target smoothing、探索噪声、Replay 采样和
  更新预算。参考超参数通过专用配置提供；机器人维度、控制周期、chunk 长度和动作尺度
  由任务声明。已有 recipe 和 Artifact 按其原有语义加载，新旧训练状态不得隐式混用。

  验收：未知组合启动前失败；算法、特征、codec 或奖励语义变化使恢复身份变化；
  配方明确区分模块数值对齐、任务参数适配和独立算法消融。
  入口：`algorithms/rltoken/configuration.py`、`training.py`、`identity.py`、
  `examples/configs/rltoken/`、`docs/rltoken_integration_plan.md`。

- [x] **RLT-DOBOT-DATA：建立离线特征数据与分来源 Replay**

  通过配置的冻结 provider 缓存示范和纯 VLA 轨迹。分别保存原始 VLA reference、
  实际执行动作、监督动作及有效 mask；区分离线数据来源、在线决策来源和逐步接管。
  按实际时间索引构造 chunk 与 next observation，禁止跨 episode 或使用 reset 观测。
  缓存记录 feature identity、codec、数据版本、采样设置、奖励定义和 episode 划分，
  并记录各训练阶段实际看过的数据范围。

  支持示范/人工与 VLA/在线两个采样池及 `50/50` 配方，独立管理容量和不足时的补采
  规则，报告实际采样比例。离线导入不产生在线 UTD；仅新采集、首次定稿准入的记录
  产生在线预算。MC 标签具有来源和有效 mask，纯 VLA 轨迹的回报可用于基础 Critic；
  截断尾部不自动视为零回报。

  验收：三种动作不互相覆盖；重复导入/恢复不重复计数；同一 episode 的窗口不跨训练
  和验证集；示范不会被在线 FIFO 无意淘汰；缓存身份不一致时拒绝使用。
  入口：`algorithms/rltoken/transitions.py`、`replay.py`，新增算法私有离线数据构建入口。

- [x] **RLT-DOBOT-TD3：实现 Residual Actor 与确定性 TD3 更新**

  增加 Context MLP + LSTM chunk Actor，零初始化动作头；策略输出为 reference 加
  有界 residual，维度来自 feature spec。统一 Actor、target smoothing 和部署的动作
  限幅及 codec，支持逐维残差尺度。离线 BC 监督为可达的示范修正；联合训练默认使用
  residual L2 约束，两者分别定义。确定性 Actor loss 使用 min(Q1,Q2)，支持 Q 动作
  梯度归一化和梯度裁剪；按上游核对 MSE reduction，避免系数相同但 loss 尺度不同。

  新增 target Actor；TD target 使用其确定性动作加独立平滑噪声，再经动作限幅。
  环境探索与 target 噪声分别配置、分别使用 RNG，探索允许为零。配置 Critic:Actor
  更新比及 target 更新时机，对齐配方采用 2:1；支持显式 gamma_chunk 到 gamma_step
  的换算。稀疏终奖的 target 裁剪范围由奖励配方声明。

  上游会将示范 TD 动作投影到 reference 残差范围：对齐配方必须显式声明该训练视图、
  记录投影比例并保留原始执行动作/奖励/next state；使用真实动作或筛除范围外样本的
  版本作为独立消融，不默认为数值等价。BC 目标投影与 TD 动作投影分开管理。

  验收：零 residual 在实际 codec 中复现 reference；确定性 Actor 梯度不依赖探索
  RNG；target 网络没有梯度；限幅、终止 mask、折扣及更新时钟符合配方；范围外动作
  不被写回原始 transition。
  入口：`algorithms/rltoken/actor_critic.py`、`losses.py`、`learner.py`、`policy.py`。

- [x] **RLT-DOBOT-CRITIC：实现严格冻结的双残差 Critic**

  为两个 Q 分支分别增加基础 Q、策略偏移 O 和动作修正 H。Explicit 结构使用
  `Q = Q_base + L*tanh((O + H(delta) - H(0))/L)`；动作修正分支在 delta=0 时为零，
  策略偏移允许非零。支持 residual 输入归一化、残差正则、专家方向监督及 terminal
  RankQ，并将监督资格与各项 loss 的阶段配置分开。

  基础 Critic 阶段使用 VLA reference 和显式配置的 target smoothing 作为 bootstrap，
  Residual Critic 阶段使用 target residual policy。冻结时从优化器更新集合中排除
  Base，并冻结在线和 target 两套 Base 的参数与 buffers；Polyak 仅更新允许训练的
  分支。加载 checkpoint 后恢复冻结规则，不能依赖 state_dict 保存 requires_grad。

  验收：Base 在 Critic 更新、Actor 更新、target 同步及保存恢复后逐位不变；Actor
  阶段梯度通过 Q 的动作输入而不更新 Critic；方向监督仅使用有效监督样本，失败回合
  不自动视为专家正例。
  入口：`algorithms/rltoken/actor_critic.py`、`losses.py`、`learner.py`、`checkpoint.py`。

- [x] **RLT-DOBOT-OFFLINE：增加分阶段预训练与在线初始化**

  增加算法私有阶段 Trainer：Actor BC、Base Critic、固定 Actor 的 Residual Critic、
  offline joint TD3。阶段显式声明可训练参数、优化器、target 初始化、loss、预算和
  验证选点；离线运行不需要启动 RLinf/Ray。Actor BC 与基础 Critic 产物可分别准备，
  组装时校验特征、codec 和任务身份，再训练 Residual Critic 与离线联合策略。

  区分离线初始化新在线任务和中断恢复：前者新建在线会话与预算，并显式声明优化器/
  target 状态的继承规则；后者恢复完整阶段、采样 RNG、优化器、target 和 Replay。
  发布包含依赖特征资产的可审计阶段产物，保存 best 与 last，避免仅凭训练步数选点。

  验收：BC 阶段不更新 Critic；critic-only 不更新 Actor/target Actor；初始化和恢复
  不混淆在线预算；在相同 batch/RNG 下，恢复后的下一次更新与连续训练一致。
  入口：新增 `algorithms/rltoken/` 下离线阶段入口，复用 `learner.py`、`checkpoint.py`。

- [x] **RLT-DOBOT-TOKEN：增加冻结 VLA 的 bottleneck Token 训练**

  增加 Token-only 模式，冻结 VLA 参数并保持 eval，优化器仅管理 Encoder/Decoder；
  复用模型拥有的 prefix 提取，避免额外计算动作生成 loss。对齐 image-only bottleneck
  的 query、attention mask、层结构和重建 reduction，Encoder/Decoder 层数可分别配置。
  图像 token 选择由 Pi0.5 提供，不在算法层写死 768 个位置。

  使用 episode 级验证和 best checkpoint；增加 zero/shuffle latent 依赖诊断，
  诊断结果和后续阶段准入规则显式关联。缓存与部署使用一致的图像处理、统计资产、
  camera/prompt 和采样协议；更新 Token 后生成新 feature identity 并重建特征缓存。

  验收：Token-only 更新不改变 VLA 参数/buffers；Decoder 不读取目标 token 内容；
  导出后的 Encoder 与训练验证路径一致。表示诊断和 GPU 模型对照留在 local_script。
  入口：`algorithms/rltoken/token.py`、`models/pi05/rltoken/modeling_stage1.py`、
  `models/pi05/rltoken/features.py`、`models/pi05/rltoken/data_adapter.py`。

- [x] **RLT-DOBOT-ONLINE：对齐回合更新、Actor 就绪与恢复部署**

  结合 RLT-REWARD 接通回合定稿和审核，支持回合结束后集中更新；有效失败数据保留，
  pending 和拒绝记录不参与训练。单环境/向量采集均传递明确的探索配置。关键阶段
  起录/结束由环境适配器提供，通用训练层不包含 USB 或插销任务判断。

  Actor 就绪由离线初始化验收和对应权重的 applied ACK 决定，不能要求已预训练 Actor
  再累计固定数量的在线更新。发布版本与各阶段更新计数分别管理。明确采集停止、更新
  上限、剩余预算消费及终止原因；记录目标/实际 UTD 和采样比例。扩展算法私有恢复
  格式和部署配置，完整保存新网络、阶段、冻结状态、噪声 RNG 和预算；部署继续使用
  公共 Artifact 与配置的模型 target。

  验收：重复/乱序定稿和恢复不会重复准入；部署只使用已同步完整版本；在线步数为零
  的合格离线 Actor 可启用；保存恢复和迁移 Artifact 后确定性动作一致。
  入口：`algorithms/rltoken/training.py`、`env.py`、`simulation.py`、`rollout.py`、
  `roles.py`、`topology.py`、`weight_sync.py`、`checkpoint.py`、`policy.py`、`artifacts.py`。

- [x] **RLT-PHASE：独立控制执行阶段与训练数据范围**

  `algorithm.policy_switch` 提供 `execution_scope`、`training_scope` 和 `task_mode`。
  参考 RLinf 的 readiness/critical-phase 分离：Actor 需同时就绪且允许在当前阶段执行，
  关键阶段 VLA 预热仍可训练。单环境、向量、离线缓存和部署使用一致的布尔观测信号；
  环境适配器负责触发、锁存和逐环境 reset。阶段外命令保留执行日志但不入 Replay、
  不增加 UTD；奖励定稿不能修改阶段事实，缓存与恢复拒绝范围不匹配。

  CPU 验收覆盖混合阶段批量路由、重置、预热、缓存筛选、TD bootstrap、迟到标签、
  journal 恢复与部署配置。真实任务触发条件与训练收益在下项进行环境验收。
  入口：`algorithms/rltoken/routing.py`、[阶段配置](docs/rltoken_td3_recipe.md#执行阶段与训练范围policy_switch)。

- [ ] **RLT-TD3-VALIDATION：逐层验证对齐与任务收益**

  `rlt_td3_residual_v1` 已完成模块重构、对抗评审与 CPU 契约测试（涵盖 pre-clip BC、
  合法成功锚定、单阶段联合预热接续、四项实验开关与 `position_bottleneck` Token）。
  GPU 数值对照、多卡 Ray 闭环与真机/仿真任务成功率评测仍需独立验收；参考脚本与报告
  保留在 Git 忽略的 `local_script/`，不进入 `src/` 或 `tests/`。
  入口：`tests/algorithms/rltoken/`、`tests/models/pi05/rltoken/`、
  `docs/rltoken_td3_recipe.md`、`docs/maniskill_evaluation_plan.md`。

- [ ] **RLT-REWARD：打通延迟奖励和审核定稿链路**

  已接通 `reward_mode=episode`、`Finalization`、环境 callback、持久 ACK 与重启 pending
  恢复。尚需对接实际外部奖励/审核服务，验证延迟、断线和真实任务终止语义；
  已参与训练的记录不能被覆盖。

  验收：pending 不增加 UTD；定稿后仅首次准入增加预算；重复、乱序、重启重放不重复
  准入；未知记录和修改执行事实的定稿被拒绝。CPU 协议测试与外部奖励服务联调分别记录。
  入口：[env.py](src/dobot_nexus/algorithms/rltoken/env.py)、
  [simulation.py](src/dobot_nexus/algorithms/rltoken/simulation.py)、
  [roles.py](src/dobot_nexus/algorithms/rltoken/roles.py)、
  [replay.py](src/dobot_nexus/algorithms/rltoken/replay.py)。

- [ ] **RLT-RETENTION：约束日志和训练产物的磁盘增长**

  Replay 的 count/bytes 上限仅约束热数据。旧 checkpoint/deployment 的保留旋转
  （`train.keep_last_checkpoints` / `train.keep_last_deployments`）、run 级只读密封
  `rltoken/features`（剔除训练专用 `token_decoder.*`）以及部署产物对冻结特征的
  hardlink 复用已落地并通过 CPU 契约测试与真实 GPU 验证；SQLite 事件日志
  （`journal.sqlite`）压缩/归档与磁盘不足行为仍待补齐。

  验收：清理后当前恢复点和保留恢复点均可加载；重复 transition 不再计预算；写入或清理
  中断不产生悬空指针；不会删除仍被引用的资产；后续补齐 `journal.sqlite` 按
  checkpoint watermark 裁切且保留去重身份。磁盘压力与长时间运行测试留在本地工具中。
  入口：[replay.py](src/dobot_nexus/algorithms/rltoken/replay.py)、
  [checkpoint.py](src/dobot_nexus/algorithms/rltoken/checkpoint.py)、
  [artifacts.py](src/dobot_nexus/algorithms/rltoken/artifacts.py)。

- [ ] **RLT-INGEST：将有界预取和接收调度接入实际训练循环**

  Learner 已有 `prefetch()` 和恢复采样 RNG 的 barrier，但实际 Worker 未调用预取；
  当前 update RPC 先接收并持久化一条消息，再执行有限次更新，没有独立接收线程。
  先测量 ACK 等待与采样耗时，再确定预取深度和接收/训练重叠方式。明确 Replay 的
  单一提交 owner、SQLite 连接的线程归属、队列 count/bytes 上限和错误传播。

  验收：持续采集与更新都能推进；ACK 仅在持久提交后发出；积压有界；checkpoint
  清空未消费预取后，恢复的采样序列、更新预算和 optimizer 时钟一致；关闭不遗留工作线程。
  入口：[learner.py](src/dobot_nexus/algorithms/rltoken/learner.py)、
  [roles.py](src/dobot_nexus/algorithms/rltoken/roles.py)、
  [training.py](src/dobot_nexus/algorithms/rltoken/training.py)。

- [ ] **RLT-METRICS：补齐采集、学习和同步的运行指标**

  在现有更新数、预算、块数和 loss 基础上，记录准入/拒绝原因、pending、持久 ACK
  等待、超时、样本年龄、Actor 版本落后、队列背压，以及环境、特征推理和更新耗时。
  统一指标身份和持久输出，明确恢复后的累计值与本次会话值。

  验收：能定位“无可训练样本”“预算耗尽”“等待环境/持久化/权重应用”的停滞原因；
  指标不改变训练 RNG，不逐步传输大张量；GPU 耗时以真实设备测量验收。
  入口：[training.py](src/dobot_nexus/algorithms/rltoken/training.py)、
  [roles.py](src/dobot_nexus/algorithms/rltoken/roles.py)。

- [x] **RLT-DOCS：统一设计文档与当前实现边界**

  将设计稿中已经实现的能力整理为当前说明，将剩余设计引用到本 TODO。核对 Trainer
  选择、组合策略命名、特征包/恢复点格式、静态 GPU 放置及接收/训练并发模型；入口文档
  不再将已有在线 RL 能力描述为尚未实现。保留来源、许可证、依赖版本和有效配置名。

  验收：开发者可由 README 找到真实入口和可运行模板；设计稿不存在与实现冲突的状态
  声明或伪路径；诊断运行记录和性能报告留在 Git 忽略的 `local_script/`。
  入口：[集成方案](docs/rltoken_integration_plan.md)、
  [算法架构](docs/rltoken_nexus_architecture.md)、
  [资源管理](docs/rltoken_resource_management.md)、
  [Nexus 接入指南](docs/nexus_architecture_and_integration_guide.md)。

### 待补齐的真实环境验收

- [ ] **RLT-FAILURE：完成执行链路的故障注入和长时间运行验收**

  在已有三 Worker 执行、有限 RPC、持久 ACK、同步 barrier 和关闭逻辑上，验证角色进程
  退出、通道阻塞、ACK 丢失、checkpoint 与权重同步交错、磁盘写入失败等场景。

  验收：不重复执行控制命令，不重复累计 UTD，不发布未完成的部署；恢复覆盖 journal
  尾部事件；超时/失败后有界退出并只清理本次任务资源。真实 Ray/GPU 证据不能由 mock
  Worker 测试替代，脚本和报告放在 `local_script/`。
  入口：[runtime/ray/backend.py](src/dobot_nexus/runtime/ray/backend.py)、
  [topology.py](src/dobot_nexus/algorithms/rltoken/topology.py)、
  [training.py](src/dobot_nexus/algorithms/rltoken/training.py)、
  [weight_sync.py](src/dobot_nexus/algorithms/rltoken/weight_sync.py)。

- [ ] **RLT-ROBOT：完成具体机器人适配器与控制闭环验收**

  通用 `RobotAgent` 协议已具备 reset/execute/query/close；具体硬件实现仍由外部环境包
  提供。落实控制器最终命令、整步接管、奖励、reset 前终止观测、I/O deadline 与断线后的
  状态确认。按轴接管若成为需求，先明确与现有整步 takeover 的兼容方式。

  验收：真实执行反馈与 codec 一致；超时只查询、不重发；未知执行状态停止采集；
  人工接管、终止和重连均保留可追溯身份；控制时延单独实测。
  入口：[env.py](src/dobot_nexus/algorithms/rltoken/env.py)、
  [Pi0.5 features.py](src/dobot_nexus/models/pi05/rltoken/features.py)。

- [ ] **RLT-EVALUATION：补齐数值、训练恢复和任务效果的独立验收**

  为使用中的模型/精度/并行配置验证 Stage 1 梯度隔离、资产导出与冻结特征一致性，
  Stage 2 Actor/双 Q 的有效更新、恢复与部署一致性，以及两种 recipe 的真实终止样本。
  固定任务、数据划分、种子和控制周期比较 reference 与 Actor 的成功率、回报和成本。
  Nexus Token/MLP 与上游权重不直接互换，不能据目标函数相似宣称数值等价。

  验收：分别记录模块正确性、GPU 数值/吞吐、工程闭环与任务收益；报告失败与无收益结果；
  不把 CPU 通过或训练完成当作策略提升。脚本、冻结评测记录和报告放在 `local_script/`。
  入口：[实现说明](docs/rltoken_implementation.md)、
  [仿真评测协议](docs/maniskill_evaluation_plan.md)。

### 条件扩展

以下能力尚未提供，需要对应任务或性能测量后再实施，不作为当前单节点基线的必备条件。

- [ ] **RLT-SCALE：多机器人与多 Worker 路由**

  当前向量仿真是在一个 Env Worker 内批量推进，不等于多个 Env/Rollout Worker。
  扩展时明确 `(env, session)` 路由、跨请求合批 deadline、特征身份、Actor 版本、ACK
  归属和单一 Replay 提交顺序。若允许单槽位失败后继续采集，需要独立的隔离/reset 策略。
  验收：多来源不串观测/缓存/ACK，积压有界，重连与终止不影响其他会话。

- [ ] **RLT-PLACEMENT：扩展静态单节点资源模式**

  当前 `Cluster(num_nodes=1)`，Rollout/Learner 必须使用不同 GPU；Env 可使用 CPU 或
  显式指定仿真 GPU。单 GPU 需要重新设计模型所有权、显存预算和推理/更新调度；多节点
  需要节点放置及资产/恢复状态可达性；多 GPU Learner 需先证明更新吞吐是瓶颈。
  CPU offload、GPU 租约和动态调度均未提供。各模式须独立验证设备映射、时延与故障恢复。

- [ ] **RLT-FEATURE-PERF：加速冻结特征提取**

  当前 Pi0.5 特征路径复用 prefix/KV，但未接入推理编译或 CUDA Graph。测量后将加速
  保留在模型拥有的推理实现内，明确 batch shape、缓存、reset 与 RNG 约束，保持参考动作
  和控制语义。验收包含真实 GPU 性能与数值对齐；数值 profile 改变时更新特征身份。

- [ ] **RLT-PROVIDERS：接入其他特征模型与有状态模型**

  当前实际提供者是 Pi0.5 的无状态路径；其他模型通过现有 `model.target` 和可选能力
  接入。Memory/递归状态需要先定义 history、reset、版本及 Replay 中上下文的恢复语义。
  验收：新增提供者独立导出/部署，不静态依赖 Pi0.5；算法不猜测模型私有状态。

- [x] **RLT-SHARED-EXECUTION-P1：将 Ray/RLinf 执行机械下沉至 `src/dobot_nexus/runtime/`**

  已将固定版 RLinf + Ray 调度适配从 `algorithms/rltoken/` 剥离至顶层可选包 `runtime/`
  （纯 Python 契约 `contracts.py`/`config.py`/`sync.py`/`factory.py`、测试注入后端
  `runtime/local/`、延迟加载后端 `runtime/ray/`），RLToken 迁移为纯 Python `topology.py`
  + `roles.py` 且零 `ray`/`rlinf` 导入；保留原 YAML、恢复、权重双槽与部署契约。
  入口：`src/dobot_nexus/runtime/`、`src/dobot_nexus/algorithms/rltoken/{topology,roles,training}.py`、
  `tests/runtime/`。

- [ ] **RLT-SHARED-EXECUTION-P2：接入第二个 RL 算法验证公共 Runtime 语义**

  待第二个在线 RL 算法（如 PPO/GRPO/SAC）接入 `src/dobot_nexus/runtime/` 时，基于真实需求
  扩展多副本路由、分布式进程组与大权重同步契约，验证两个算法复用同一 Runtime 层且普通 BC、
  模型与部署继续零 Ray/RLinf 依赖。

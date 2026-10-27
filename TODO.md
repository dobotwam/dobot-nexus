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

### 待实现与文档收尾

- [ ] **RLT-REWARD：打通延迟奖励和审核定稿链路**

  Replay 已支持 pending 与 `commit(..., finalize=True)`，但训练入口只接受
  `reward_mode=per_step`，采集器和 Worker 尚无 episode 奖励/审核结果的定稿消息。
  补齐消息身份、定稿权限、持久 ACK 和恢复语义；已参与训练的记录不能被覆盖。

  验收：pending 不增加 UTD；定稿后仅首次准入增加预算；重复、乱序、重启重放不重复
  准入；未知记录和修改执行事实的定稿被拒绝。CPU 协议测试与外部奖励服务联调分别记录。
  入口：[env.py](src/dobot_nexus/algorithms/rltoken/env.py)、
  [simulation.py](src/dobot_nexus/algorithms/rltoken/simulation.py)、
  [workers_rlinf.py](src/dobot_nexus/algorithms/rltoken/workers_rlinf.py)、
  [replay.py](src/dobot_nexus/algorithms/rltoken/replay.py)。

- [ ] **RLT-RETENTION：约束日志和训练产物的磁盘增长**

  Replay 的 count/bytes 上限仅约束热数据，SQLite 事件日志和旧 checkpoint/deployment
  尚无保留策略。设计容量预算、归档/清理条件和磁盘不足行为，保留去重身份、checkpoint
  watermark 后的恢复事件、活动指针及其引用的冻结特征资产。

  验收：清理后当前恢复点和保留恢复点均可加载；重复 transition 不再计预算；写入或清理
  中断不产生悬空指针；不会删除仍被引用的资产。磁盘压力与长时间运行测试留在本地工具中。
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
  [workers_rlinf.py](src/dobot_nexus/algorithms/rltoken/workers_rlinf.py)、
  [training.py](src/dobot_nexus/algorithms/rltoken/training.py)。

- [ ] **RLT-METRICS：补齐采集、学习和同步的运行指标**

  在现有更新数、预算、块数和 loss 基础上，记录准入/拒绝原因、pending、持久 ACK
  等待、超时、样本年龄、Actor 版本落后、队列背压，以及环境、特征推理和更新耗时。
  统一指标身份和持久输出，明确恢复后的累计值与本次会话值。

  验收：能定位“无可训练样本”“预算耗尽”“等待环境/持久化/权重应用”的停滞原因；
  指标不改变训练 RNG，不逐步传输大张量；GPU 耗时以真实设备测量验收。
  入口：[training.py](src/dobot_nexus/algorithms/rltoken/training.py)、
  [workers_rlinf.py](src/dobot_nexus/algorithms/rltoken/workers_rlinf.py)。

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
  入口：[execution_rlinf.py](src/dobot_nexus/algorithms/rltoken/execution_rlinf.py)、
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

- [ ] **RLT-SHARED-EXECUTION：按实际复用需求抽取执行层**

  RLinf/Ray 目前隔离在算法私有 `execution_rlinf.py`、`workers_rlinf.py`。待第二个算法
  确认需要相同启动、放置、通道和关闭语义，再确定公共执行适配的归属；不可直接放入
  `modules/` 计算组件或扩大稳定层职责。Replay、优化器、recipe 和采集语义继续归算法。
  验收：两个算法复用明确的执行行为，普通 BC/部署仍可在没有 RLinf 的环境导入运行。

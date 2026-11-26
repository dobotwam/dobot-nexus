# RLToken 资源管理与分布式 Runtime 执行

当前 RLinf + Ray 执行适配已下沉至框架级可选运行时包 `src/dobot_nexus/runtime/ray/`（契约位于
`runtime/contracts.py`、`config.py`、`sync.py`、`factory.py`，测试注入后端位于
`runtime/local/`），由 `RLTokenTrainer.run()` 按 `runtime.launcher=rlinf` 延迟加载。
RLToken 算法包仅保留纯 Python 拓扑声明 `algorithms/rltoken/topology.py` 与角色壳
`algorithms/rltoken/roles.py`，不直接导入 `ray` 或 `rlinf`。RLinf 固定 revision 为
`7e90641c4135229db6d3426950cdb45cee6259a6`（实测验证 `ray[default]==2.56.1`）；算法、模型、
BC 与普通部署不依赖活跃 Ray 集群。启动示例见[实现说明](rltoken_implementation.md)，尚未提供的
资源模式见[TODO](../TODO.md)。

## 复用边界

| 能力 | 当前用法 |
| --- | --- |
| Cluster、Worker、WorkerGroup | 一个 driver、单节点、每角色一个 Worker |
| PackedPlacementStrategy | 将 Rollout/Learner 及可选仿真 Env 放到指定 GPU |
| NodePlacementStrategy | CPU Env 的节点放置；Worker 内显式隐藏 CUDA |
| Channel | 六条容量为 1 的独立通道，Nexus 另加消息字节上限 |
| `runtime/sync.py`（`SingleInFlightSyncCoordinator`） | 初始阻塞同步，后续单个在途收发对同步协调与版本校验（不再继承上游 `AsyncWeightSyncMixin`） |
| 上游 runner、模型工厂、FSDP policy | 不作为 Nexus 基类；Nexus 自有业务对象与 Artifact |
| DeviceLock、动态调度、offload/reload | 当前未接入 |

固定版本 API 来源为 [scheduler][scheduler]、[WorkerGroup][worker-group]、
[Channel][channel]、[placement][placement]、[同步 mixin][sync-mixin]和[异步入口][async-entry]。
安装上游包，不复制 scheduler。来源及 Apache-2.0 许可见
[NOTICE](../src/dobot_nexus/algorithms/rltoken/NOTICE.md) 和
[LICENSE](../src/dobot_nexus/algorithms/rltoken/LICENSE)。

## 进程与设备所有权

| 角色 | 设备 | 持有状态 |
| --- | --- | --- |
| RLTokenTrainer | CPU driver | 远程句柄、采集/更新进度、同步和保存边界 |
| `ManagedRoleWorker` → `EnvRole` | 真机为 CPU；仿真可指定 GPU | 外部 adapter、会话、执行反馈、transition 拼装 |
| `ManagedRoleWorker` → `RolloutRole` | rollout_gpu | 冻结 VLA/Encoder、Actor 双缓冲、特征缓存 |
| `ManagedRoleWorker` → `LearnerRole` | learner_gpu | Actor、双 Q、Target Q、双优化器、CPU Replay/journal |

`Cluster(num_nodes=1)`，Rollout 与 Learner 必须指定不同 GPU。仿真 `env_gpu` 可与
Rollout 共用或使用另一张设备，需合计模型、渲染和物理状态的显存。省略 Env GPU 时
隐藏 CUDA，仅适用于不需要 GPU 的环境。每个 GPU Worker 必须只看到一张 CUDA 设备，
内部编号固定为 `cuda:0`；环境 SDK 仍需验证实际物理/渲染设备映射。

资源编号使用 RLinf 枚举的硬件编号，单节点通常对应物理 GPU 编号，不能按 driver 的
`CUDA_VISIBLE_DEVICES` 自行重编号。启动脚本在 rlinf 模式只启动一个 Python driver，
拒绝 torchrun rank 环境和 Stage 2 FSDP；BC 的 launcher 保持独立。

设备可见性与 placement 不提供同机其他任务的资源独占、显存配额或实时优先级。
运行环境需明确可用资源与任务隔离；参见固定版本的 [Cluster][cluster]、
[节点放置][node-placement]以及[执行模式说明][execution-doc]。

## 消息、预算与并发

通道为 requests/responses、transitions/durable_acks、weights/weight_acks。
数据提交确认与权重应用确认独立，避免某条控制消息被训练数据积压阻塞。
每条通道容量为 1，单条消息还受 `runtime.max_message_bytes` 限制。
向量采集的一批数据算一条消息，Replay 则按每条 transition 独立准入和增加预算。

每个角色 `max_concurrency=1`。协调者以有限 RPC 交错驱动采集、Rollout 服务和更新；
Learner Worker 每次先取一条数据消息、持久提交并返回 ACK，再执行有限次更新。
当前没有专用接收线程，没有启用 Learner 预取；相关有界队列与采样 RNG 设计见 TODO。

Replay 的 count/bytes 限制只作用于热数据。journal、去重索引与旧 checkpoint/deployment
尚无自动清理策略，部署前需按任务长度估算磁盘占用。
UTD 由唯一准入 transition 驱动；GPU 空闲不会额外增加更新预算。

Actor 同步先生成完整不可变 CPU 快照，再校验身份、摘要、shape、dtype 与有限值，
装入空闲槽位并在请求边界切换。已应用版本返回 ACK；收到传输消息不等于应用完成。
CUDA 请求完成后才能重用退役槽位。同步协调与原子应用分别由执行适配和
`weight_sync.py` 负责，不能只靠上游 mixin 保证参数一致。

## 启动、保存与关闭

先创建 Cluster 建立 namespace，再创建 Channel 和角色 Worker。
初始化完成后阻塞同步首个 Actor 版本，随后允许采集。
保存或正常停止时，暂停新采集，服务完当前动作块的请求与持久 ACK，排空 RPC 和同步，
再保存 Learner 恢复点并按就绪条件导出部署。

执行适配先调用各角色业务 close；失败时尝试终止本任务持有的 Worker，正常路径调用
固定版本的私有 `_close()`，最后关闭 Channel WorkerGroup。私有生命周期 API 集中
封装，不能视为跨版本稳定接口或透明容错。异常、启动半途失败、通道阻塞和硬退出的
有界清理仍需在真实 Ray 环境做故障注入，验收项见 TODO 的 RLT-FAILURE。

外部 Robot Agent 必须真正落实 I/O deadline。执行超时后只查询命令状态，不重发。
无法确认执行时持久记录并停止采集；重启使用新 session/episode，不假设物理动作可回滚。
控制器状态确认和硬件恢复由环境包负责。

## 依赖与验收

安装命令（两份 pinned requirements 保持完全同步）：

```bash
python -m pip install -r src/dobot_nexus/runtime/ray/requirements.txt
# 兼容旧路径：
python -m pip install -r src/dobot_nexus/algorithms/rltoken/requirements-rlinf.txt
```

在独立运行环境记录 RLinf SHA、Ray、Torch/CUDA、Hydra/OmegaConf、驱动和 Nexus SHA。
导入成功不代表 GPU 算子、设备通信或硬件闭环已经验收。
真实进程与模型实验、诊断脚本和报告放在 Git 忽略的 `local_script/`，CPU 模块/契约测试
放在 `tests/`。设备映射、推理时延、有效采集速率、更新吞吐、版本落后、显存、磁盘与
checkpoint 暂停时间分别测量；运行指标的持续落盘仍见 TODO。

单 GPU 协作调度、多节点、多个 Env/Rollout Worker、GPU 租约与动态放置尚未提供。
当前向量仿真是在单个 Env Worker 内批量推进，不能视为多 Worker 调度。
扩容前先确定瓶颈，保持 UTD、数据语义与评测条件可比。

上游能力的补充参考：[异步 rollout][async-rollout]、[异步 Env][async-env]、
[FSDP manager][fsdp-manager]、[同步校验][sync-validation]和[集群说明][cluster-doc]。
这些参考不构成 Nexus 对 offload、资源独占或自动恢复的承诺。


[scheduler]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/scheduler/__init__.py
[worker-group]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/scheduler/worker/worker_group.py
[channel]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/scheduler/channel/channel.py
[placement]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/utils/placement.py
[node-placement]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/scheduler/placement/node.py
[cluster]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/scheduler/cluster/cluster.py
[sync-mixin]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/runners/async_weight_sync_mixin.py
[async-entry]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/examples/embodiment/train_async.py
[async-rollout]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/workers/rollout/hf/async_huggingface_worker.py
[async-env]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/workers/env/async_env_worker.py
[fsdp-manager]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py
[sync-validation]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/config.py
[execution-doc]: https://rlinf.readthedocs.io/en/latest/rst_source/concepts/execution_modes.html
[cluster-doc]: https://rlinf.readthedocs.io/en/latest/rst_source/concepts/cluster.html

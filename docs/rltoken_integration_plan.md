# RLToken 接入与来源

RLToken 使用两阶段训练：Stage 1 通过 BCTrainer 联合训练 VLA 和 Token 表示，
Stage 2 冻结特征提供者，通过 RLTokenTrainer 采集环境反馈并训练 Actor/双 Q。
运行配置与命令见[实现说明](rltoken_implementation.md)，模块关系见
[架构说明](rltoken_nexus_architecture.md)，执行约束见[资源管理](rltoken_resource_management.md)。
未完成设计与验收集中在 [TODO](../TODO.md)。

## 接入边界

| 部分 | Nexus 中的职责与入口 |
| --- | --- |
| 模型与数据 | 使用既有 `BaseVLAModule`、`BaseDataAdapter` 和 `model.target`；Pi0.5 集成位于 `models/pi05/rltoken/` |
| 表示训练 | `Pi05RLTokenModel` 复用同一次 VLA forward 的 prefix；Token 网络与重建 loss 位于 `algorithms/rltoken/token.py` |
| 在线学习 | `algorithms/rltoken/` 拥有 Actor/双 Q、loss、transition、Replay、Learner、恢复和权重同步 |
| 远程执行 | `runtime/ray/`（延迟导入 RLinf/Ray）+ `algorithms/rltoken/{topology,roles}.py` 声明并构造单节点三角色 Worker |
| 部署 | `RLTokenPolicy` 组合冻结提供者与 Actor，由公共 `InferencePolicy` 加载自包含 Artifact |
| 环境 | 外部包实现机器人或向量仿真协议，拥有控制器执行、奖励和 reset；算法包不导入本地诊断工具 |

`modules/` 只承载可复用计算。RLToken 的网络和训练状态当前留在算法包；
Pi0.5 RECAP 的共享 Value 网络不是 action-conditioned Q，不能直接替换双 Q。
具体模型通过配置和 Artifact 构造，算法不静态导入 Pi0.5。

## 算法与来源

基线 `rlt_ac_fixed_v1` 参考 RLinf 固定 revision
`7e90641c4135229db6d3426950cdb45cee6259a6` 的 Token 重建、Actor/Critic 目标和
已执行接管动作的训练参考替换约定。Nexus 的 Transformer/MLP 是可配置 Torch 实现，
权重键不与上游直接互换；CPU 契约通过不代表数值等价或任务收益。

| 参考入口 | 对应内容 |
| --- | --- |
| [Token Transformer][rlt-token]、[Pi0 特征路径][rlt-pi0] | prefix 压缩、重建及冻结特征 |
| [MLP policy][rlt-mlp]、[AC worker][rlt-ac] | 参考动作条件 Actor、双 Q、Q1 与 BC 目标 |
| [Transition][rlt-transition]、[rollout][rlt-rollout]、[route][rlt-route] | 动作来源与接管语义对照 |
| [Stage 2 配置][rlt-config]、[任务评测][rlt-eval] | 控制周期、horizon 与任务协议的来源 |
| [异步入口][rlt-async-entry]、[同步入口][rlt-sync-entry]、[runner][rlt-async-runner] | 上游执行流程参考 |
| [权重同步 mixin][rlt-sync-mixin] | 当前可选执行适配直接复用的同步协调接口 |

`rlt_ac_terminal_v2` 是 Nexus 的独立 recipe：明确终止的非空执行前缀可准入，Critic
动作使用实际前缀与原候选后缀，未执行奖励为零且不 bootstrap。它与固定块 recipe
具有不同的 Replay/恢复身份。不能在同一恢复点中隐式切换。

当前配方不含 TD3、Residual、CrossQ、RECAP 条件组合、在线更新特征模型、关键阶段
自动切换或专家规划器。需要这些能力时，应明确独立 recipe、数据语义和验收标准。

## 配置前必须明确的输入

1. 选择真实的 Stage 1 提供者及其 Artifact；Encoder 必须经过训练。
2. 固定 state/action 维度、相机槽位、动作 codec、控制周期与 `chunk_size`。
3. 由环境包声明逐控制步 reward、终止/截断、接管、实际命令和执行查询语义。
4. 指定 Replay 容量、warmup、UTD、Actor 就绪阈值及两套优化器参数。
5. 指定不同的 Rollout/Learner GPU；仿真渲染需显式配置 Env GPU。

Stage 1 模板是 [`stage1_pi05.yaml`](../examples/configs/rltoken/stage1_pi05.yaml)。
Stage 2 模板是 [`stage2.yaml`](../examples/configs/rltoken/stage2.yaml) 和
[`stage2_simulation.yaml`](../examples/configs/rltoken/stage2_simulation.yaml)。
模板中的模型资产、环境 target 和控制器维度需按任务填写。

## 执行依赖与参考范围

RLinf 通过[固定版本依赖](../src/dobot_nexus/algorithms/rltoken/requirements-rlinf.txt)
安装，Nexus 不复制 scheduler，不继承上游完整模型工厂或 FSDP RLT policy。
普通 BC 和部署不需要 RLinf。RLinf 的其他能力不自动成为 Nexus 的配置开关：
[异步 rollout][rlt-async-rollout]、[HF worker][rlt-hf-worker]、
[FSDP manager][rlt-fsdp-manager] 与[配置校验][rlt-root-config]仅用于核对复用边界。

其他固定版本参考保留为来源索引，不作为生产依赖：

- OpenVLA `c8f03f48af692657d3060c19588038c7220e9af9`：
  [动作 tokenizer][openvla-tokenizer]、[微调][openvla-finetune]、[FSDP][openvla-fsdp]、
  [数据][openvla-data]、[部署][openvla-deploy]。
- openpi-RLT `c1e40ac360185778c98cf20da2820e22d2d415e7`：
  [运行说明][openpi-rlt-runtime]、[启动][openpi-rlt-launch]、[入口][openpi-rlt-entry]、
  [配置][openpi-rlt-config]、[trainer][openpi-rlt-trainer]、
  [推理][openpi-rlt-inference]、[Replay][openpi-rlt-replay]。

来源与许可见算法包的 [NOTICE](../src/dobot_nexus/algorithms/rltoken/NOTICE.md) 和
[Apache-2.0 LICENSE](../src/dobot_nexus/algorithms/rltoken/LICENSE)。
真实模型、Ray/GPU、仿真和硬件的诊断脚本与报告保留在 Git 忽略的 `local_script/`。


[rlt-token]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/models/embodiment/modules/rlt_token_transformer.py
[rlt-pi0]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/models/embodiment/openpi/pi0.py
[rlt-eval]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/models/embodiment/openpi/tasks/eval.py
[rlt-mlp]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py
[rlt-ac]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py
[rlt-rollout]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/algorithms/rlt/rollout.py
[rlt-route]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/algorithms/rlt/route.py
[rlt-transition]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/algorithms/rlt/transition.py
[rlt-config]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml
[rlt-async-entry]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/examples/embodiment/train_async.py
[rlt-sync-entry]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/examples/embodiment/train_embodied_agent.py
[rlt-async-runner]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/runners/async_embodied_runner.py
[rlt-sync-mixin]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/runners/async_weight_sync_mixin.py
[rlt-async-rollout]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/workers/rollout/hf/async_huggingface_worker.py
[rlt-root-config]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/config.py
[rlt-hf-worker]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/workers/rollout/hf/huggingface_worker.py
[rlt-fsdp-manager]: https://github.com/RLinf/RLinf/blob/7e90641c4135229db6d3426950cdb45cee6259a6/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py
[openvla-tokenizer]: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/action_tokenizer.py
[openvla-finetune]: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/vla-scripts/finetune.py
[openvla-fsdp]: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/training/strategies/fsdp.py
[openvla-data]: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/datasets/rlds/dataset.py
[openvla-deploy]: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/vla-scripts/deploy.py
[openpi-rlt-runtime]: https://github.com/Yyshadow/openpi-RLT/blob/c1e40ac360185778c98cf20da2820e22d2d415e7/rlt_online_rl/README.md
[openpi-rlt-launch]: https://github.com/Yyshadow/openpi-RLT/blob/c1e40ac360185778c98cf20da2820e22d2d415e7/rlt_online_rl/launch/launch_machine_b.py
[openpi-rlt-entry]: https://github.com/Yyshadow/openpi-RLT/blob/c1e40ac360185778c98cf20da2820e22d2d415e7/rlt_online_rl/scripts/run_online_rl.py
[openpi-rlt-config]: https://github.com/Yyshadow/openpi-RLT/blob/c1e40ac360185778c98cf20da2820e22d2d415e7/rlt_online_rl/configs/tasks/agilex_ethernet/online_rl.yaml
[openpi-rlt-trainer]: https://github.com/Yyshadow/openpi-RLT/blob/c1e40ac360185778c98cf20da2820e22d2d415e7/rlt_online_rl/src/rlt_online_rl/trainer.py
[openpi-rlt-inference]: https://github.com/Yyshadow/openpi-RLT/blob/c1e40ac360185778c98cf20da2820e22d2d415e7/rlt_online_rl/src/rlt_online_rl/inference.py
[openpi-rlt-replay]: https://github.com/Yyshadow/openpi-RLT/blob/c1e40ac360185778c98cf20da2820e22d2d415e7/rlt_online_rl/src/rlt_online_rl/replay.py

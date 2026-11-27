# RLToken 残差 TD3 配方（`rlt_td3_residual_v1`）

`rlt_td3_residual_v1` 对齐 `RLT-in-Practice` 的核心算法设计，提供冻结 VLA 特征、
pre-clip 有界残差 Actor、标准 Twin-Q Critic、合法成功行为锚定（Success Behavior
Anchor）、动态 Q/BC 权重调度、单阶段离线联合预热（Prewarm）和在线 TD3 更新，同时
按显式配置开关支持四项研究变体及可选 Terminal RankQ。算法实现位于
`src/dobot_nexus/algorithms/rltoken/`，Pi0.5 prefix 特征与动作 codec 保持在
`src/dobot_nexus/models/pi05/rltoken/`。离线入口无需 RLinf/Ray；在线调度由已有
三角色执行适配负责。参考来源与文件摘要见 [`NOTICE.md`](../src/dobot_nexus/algorithms/rltoken/NOTICE.md)。

## 核心算法与数据语义

### 1. Pre-clip 有界残差 Actor

`ResidualActor` 为 2×256 ReLU MLP（默认无 LayerNorm），输入拼接
`[z_rl, proprio, flatten(ref.detach())]`，输出维度 `C*A`，最后一层 `Linear`
零初始化（`weight=0, bias=0`），使策略初始精确复现冻结 VLA 参考动作。前向公式：

```text
u  = ref.detach() + scale * tanh(raw) [+ exploration_std * eps]
pi = clamp(u, -1, 1)
```

- 默认 `scale=1.0`（支持标量、`[A]` 或 `[C*A]` 逐维声明）、`exploration_std=0.1`、
  `actor_update_action_noise=true`；严禁 `reference_dropout`（必须为 `0.0`）。
- 单次 `forward_with_preclip` 同时返回 `(pi, u)`，Q 分支与 BC 分支严格共享同一次
  探索噪声或潜变量采样。
- Critic Q 分支始终消费合法裁剪动作 `pi`；BC 分支默认（`bc_action_space=preclip`）
  消费裁剪前动作 `u`，避免动作在 `[-1, 1]` 边界饱和后 BC 梯度消失；`postclip` 可作
  对照开关。

### 2. 合法成功行为锚定（Success Behavior Anchor）

默认 `success_anchor=known_success_behavior`：

- 仅当当前轨迹已知成功（`episode_success=True` 且 `episode_success_known=True`）且
  步位于有效执行前缀（`valid_mask=True`）时，BC 目标使用真实执行的控制器动作
  `executed`；否则（失败、未知结局或提前终止未执行后缀）回退至 `ref.clamp(-1, 1)`。
- 严格校验选中位置 `executed` 有限且位于 `[-1, 1]`，绝不静默裁剪真实执行目标。
- BC loss 在完整 `[B, C, A]` 维度取均值，与 `RLT-in-Practice` 保持一致。
- Replay 维护独立的 `episode_outcomes` 索引：仅按 `session=(run_id, env_id, session_epoch, episode_id)`
  记录已确认且定稿的布尔结局（含 `outside_training_scope` 的定稿终块），采样时只读
  回填该回合此前未标注的 transition，不修改原始落盘记录，也不重复发放 UTD 预算。

### 3. Twin-Q Critic 与动态 Q/BC 权重调度

- `ResidualTwinQ` 为两支独立 2×256 ReLU MLP，输入 `[z_rl, proprio, flatten(actions)]`；
  默认 MSE loss 对两支 Q 取 batch 均值（`F.mse_loss(q, target.expand_as(q))`）。
- 逐步折扣 `gamma=0.99`，默认 `critic_actor_ratio=2`、`target_update_frequency=1`、
  `tau=0.005`、`grad_clip=10.0`、`actor_lr=critic_lr=1e-4`（常量学习率，不做 Q 动作
  梯度归一化）。
- `weight_schedule` 默认开启：前 `warmup_updates=20000` 步使用
  `warmup_q_weight=0.05, warmup_bc_weight=7.0`，随后在 `ramp_updates=50000` 步内按
  `progress = clamp((update_step - warmup_updates + 1) / ramp_updates, 0, 1)`
  线性过渡至 `online_q_weight=0.5, online_bc_weight=2.8`。

### 4. 四项可选实验开关与 Terminal RankQ

所有实验开关默认关闭，不影响主配方；冲突配置在构造/预检期立即报错：

| 实验开关 | 配置路径与取值 | 行为与约束 |
|---|---|---|
| Same-Sample Latent Residual | `model.rltoken.residual.latent_dim`（如 `C*A=84`）、`eval_latent_mode: sample \| zero` | 在 Actor 输入拼接 `z ~ N(0, I)`；强制 `exploration_std=0.0` 且 `actor_update_action_noise=false`；Q/BC 共享同一次 `z`，Replay 不落盘 `z`，Target Actor 独立采样 `z` |
| Q-Gated Success Anchor | `algorithm.learning.td3.success_anchor: q_gated_success` | 在 `no_grad` 下比较合法参考补齐执行动作与参考动作的 `min(Q1, Q2)`，严格 `q_exec > q_ref` 才使用执行行为作 BC 锚点（平局回退 VLA reference），Replay/Critic 不受影响 |
| Critic LN + Bootstrap Target Clip | `model.rltoken.residual.critic_layer_norm: true`、`algorithm.learning.td3.target_q_clip: [0.0, 1.0]` | Critic 每层隐藏 `Linear` 后插入 `LayerNorm(eps=1e-5)`；`target_q_clip` **仅裁剪** bootstrap `min(Q1', Q2')`，不裁剪最终 Bellman target 或当前 Q 输出 |
| Sigmoid Twin-Q + BCE | `model.rltoken.residual.critic_output: sigmoid`、`algorithm.learning.td3.critic_objective: bce` | 二者必须成对开启且 `target_q_clip=null`；当前 Critic 回归使用 raw logits + `binary_cross_entropy_with_logits`，严格校验未修改的 TD target 有限且位于 `[0, 1]`（绝不静默裁剪） |
| Terminal RankQ（辅助项） | `algorithm.learning.td3.rank_weight > 0`（默认 `0.0`） | 仅对已知且未截断样本启用；在成功终态比较记录动作与近/远扰动、随机、错配及前序状态，非终态/失败轨迹施加 `rank_failure_weight=0.1` 弱支持项 |

## 运行顺序与配置模板

三个离线入口使用普通 Python 单进程运行，无需安装 RLinf/Ray：

```bash
PYTHONPATH=src:. python -m dobot_nexus.cli.train --config examples/configs/rltoken/td3_token.yaml
PYTHONPATH=src:. python -m dobot_nexus.cli.train --config examples/configs/rltoken/td3_cache.yaml
PYTHONPATH=src:. python -m dobot_nexus.cli.train --config examples/configs/rltoken/td3_prewarm.yaml
```

### 1. Token 预训练（`td3_token.yaml`）

- 使用 `training_mode=token_only`、`mode=position_bottleneck`、`selection=full`（也可按需切换 `image_only`）。
- `position_bottleneck` 复用 `TokenEncoder` 与因果 decoder 的可学习 `position` 作为查询（不引入额外 `query` 参数，不读取目标 token 内容），对应 `RLT-in-Practice` 的 `SelectableRLTTokenTransformer(decoder_mode="bottleneck")`。既有 `teacher_forcing` 与 `bottleneck` 模式保持不变。
- 训练按 `algorithm.token.train_episodes / validation_episodes` 隔离验证，输出 `token_validation.json`（含 `zero_ratio` / `shuffle_ratio` 依赖门限和 `feature_id`）。

### 2. 离线特征缓存（`td3_cache.yaml`）

- 缓存格式升级为 `rltoken_episode_cache_v2`：严格保留显式 `Episode.success`（缺失时保持 `None`，不再由末步正奖励推断），避免旧版 v1 缓存把未知结局误当作已知成功锚点。
- 默认 `validation_fraction=0.0`（支持单回合或全量数据直接入预热池，也支持 `0 < validation_fraction < 1` 做 episode 级留出验证）。
- LeRobot reader 仍要求显式最终观测帧（`last_frame_is_final_observation=true`）及 `episode_labels` JSON。

### 3. 单阶段联合离线预热（`td3_prewarm.yaml`）

- 使用 `RLTokenPrewarmTrainer`，单 Uniform FIFO Replay（默认容量 50,000）联合更新 Critic 与 Actor（默认 `max_steps=30000`、`critic_actor_ratio=2`，即 30k Critic / 15k Actor 更新）。
- 导出 `rltoken/prewarm.pt`（格式 `rltoken_prewarm_v2`）与可续训 checkpoint（`rltoken_td3_resume_v3`）。
- 在线接续时通过 `initialize_from_prewarm` 严格校验 `spec`、`heads`、`training_scope`、`task_identity`、`gamma`、`bc_action_space`、`success_anchor`、`critic_output`、`critic_objective`、`latent_dim` 一致，并要求至少完成 1 次 Actor 更新（`actor_updates >= 1`）。
- 默认 `inherit_prewarm_optimizer=true`：继承 `actor`/`critic`/`target`/`target_actor` 权重、优化器与调度器状态以及 `update_step` 时钟（同时校验 `actor_lr`/`critic_lr` 一致），在线预算 `budget` 从 `0` 重新开始，并标记 `pretrained_ready=True`。

### 4. 在线微调与 Latent 变体（`td3_online.yaml`、`td3_latent.yaml`）

```bash
examples/train.sh --config examples/configs/rltoken/td3_online.yaml --gpus 0,1
examples/train.sh --config examples/configs/rltoken/td3_latent.yaml --gpus 0,1
```

- `td3_online.yaml` 为主配方在线模板；`td3_latent.yaml` 继承它并开启 `latent_dim=84`（匹配 `C=6, A=14`）、`exploration_std=0.0`、`actor_update_action_noise=false`。
- 恢复格式为 `rltoken_td3_resume_v3`（与旧版 `rltoken_td3_resume_v2` 及 `rlt_ac_*` 不互通），完整保存网络、双 target、优化器/调度器、双 RNG、Rollout RNG、`update_step` 及 Replay（含 `episode_outcomes`）。
- 部署仍走公共 `InferencePolicy` + `RLTokenPolicy`，仅打包冻结特征提供者与 Actor 权重。

## 执行阶段与训练范围（`policy_switch`）

`algorithm.policy_switch` 支持独立配置 `task_mode`、`execution_scope` 与 `training_scope`：

```yaml
algorithm:
  policy_switch:
    task_mode: full_task
    execution_scope: critical_phase
    training_scope: critical_phase
```

- 两个 scope 独立支持 `all` 或 `critical_phase`（未配置时均为 `all`）。
- `execution_scope` 控制哪些观测允许调用 Actor（需同时满足 Actor 就绪和权重 applied ACK）；`training_scope` 控制哪些 chunk 进入 Replay/离线缓存。
- `task_mode=full_task` 要求环境适配器在观测中提供 `observation.control = {"rlt_switch_flags": np.array([in_critical_phase], dtype=np.bool_)}`；缺失时明确报错。
- 阶段外 chunk 保留持久执行日志与 ACK，但不入训练池、不进 pending、不增 UTD；若在 `reward_mode=per_step` 下携带已确认且定稿的布尔 `episode_success`，Replay 会仅索引该回合结局以回填同回合的关键阶段样本。
- 若下一观测不允许使用 Actor，TD bootstrap 直接使用冻结 VLA reference（不叠加 Actor 残差或探索噪声）。

## 回合定稿（`reward_mode=episode`）

环境配置 `reward_mode=episode` 时，adapter 提供 `finalize_episode(env_id=..., episode_id=..., pending=..., timeout=...)`（及可选 `poll_finalizations`），返回 `Finalization(key, rewards, accepted, episode_success=None)`。

- Pending 与被拒记录不参与训练、不计入 UTD；定稿仅允许补充奖励、布尔 `episode_success` 与审核结论，重复定稿幂等，冲突或未知 key 报错。
- `update_schedule=episode_end` 在回合结束后集中消耗更新预算；`drain_on_stop=true` 在达到采集上限后继续消费剩余预算直至上限或耗尽。

## 验证边界

CPU 单元测试覆盖：`rlt_td3_residual_v1` 配置约束、pre-clip 边界梯度与单次采样共享、合法成功锚定与严格 `>` Q-gate、Critic LN + 仅 bootstrap 裁剪、Sigmoid/BCE 严格目标范围检查、Terminal RankQ、Replay 回合结局传播与跨恢复一致性、单阶段离线预热接续与完整断点恢复、`position_bottleneck` Token 路径、以及公开 `InferencePolicy` 推理部署。

CPU 契约测试通过仅证明模块计算、数据契约与生命周期语义正确，不替代 GPU 数值/吞吐验证、真实 Ray 多卡闭环或真机/仿真任务成功率验收。

# RLToken 双残差配方

`rlt_td3_dual_residual_v1` 提供冻结 VLA 特征、残差 Actor、严格冻结 Base 的双残差
Critic，以及四个离线训练阶段和在线 TD3。算法计算在 `algorithms/rltoken/`，
Pi0.5 prefix 选择和动作 codec 在 `models/pi05/rltoken/`。离线入口无需 RLinf/Ray；
在线调度仍由已有三角色执行适配负责。参考文件摘要见算法目录 `NOTICE.md`。

## 算法与数据

Actor 使用 Context MLP + LSTM，输出 `reference + rho*tanh(head)`，动作头零初始化。
`rho` 必须逐维声明，单位为 provider 的动作 codec；示例的 14 维、6 步和 0.05 尺度
仅是可编辑任务声明。所有执行动作满足 codec 的 `[-1,1]` 边界。

两个 Q 分支分别计算 `Q_base + L*tanh((O + H(delta/rho) - H(0))/L)`。
Base Critic 阶段拟合记录动作的显式投影视图，以 VLA reference + smoothing bootstrap；
Residual Critic 和 joint 阶段改用独立 target Actor。Base 的在线参数、target 参数均冻结，
不在残差优化器中，也不参与残差阶段 Polyak。这是定义基础 VLA 价值的配方选择；
不要将上游所有预训练命令中使用 target Actor 的 Base bootstrap 视为同一个实验。

TD loss 为两个 Q 的 batch MSE 之和；Base MC 项为两个 Q 的平均 MSE，只对完整纯 VLA
轨迹的有效 MC 标签启用。`gamma` 默认解释为 chunk 折扣，内部换算为 `gamma**(1/C)`。
稀疏终奖示例把 TD target 裁剪至 `[0,1]`，其他奖励必须更换任务身份并声明范围。
`td_action_view=project` 明确投影 Critic 的训练动作；`executed` 为独立消融。
投影不改变实际执行动作、reward 或 next observation。

Actor BC 使用有效监督 mask，目标为可达的示范修正。joint loss 使用确定性动作、
`min(Q1,Q2)`、batch 级 Q 动作梯度范数归一化和 residual L2；梯度裁剪为 1。
每两次 Critic 更新进行一次 Actor 更新，然后同步两套 target；critic-only 每步更新
Critic target，Actor/target Actor 保持不变。探索和 target smoothing 使用独立 RNG。

辅助项默认关闭。可分别启用 residual Q L2、成功且有有效专家监督样本的 Q 动作梯度
方向对齐、terminal RankQ。RankQ 比较同一终止成功状态下的记录动作、小/大扰动、
随机和错配动作，并可加入成功前序状态和失败轨迹的支持项；失败轨迹不参加专家方向监督。
这些系数、RankQ 类型与温度都进入恢复身份，需分别做消融。

Replay 保存原始 reference、实际执行动作、监督动作及其 mask。`origin=demo/vla/online`
与决策 `source`、逐步 takeover 分别记录。`demo_joint_50` 每个 batch 各抽一半：
示范/接管池，以及 VLA/在线池；空池时从另一池补齐，容量和字节上限独立。
只有新在线记录第一次定稿准入才增加 UTD。报告 `online_admitted`、`target_utd`、
`actual_utd`、`sample_demo_fraction` 和 TD 投影比例。

## 运行顺序

使用普通 Python 单进程启动三个离线入口。先配置 provider 的 checkpoint、统计资产、
相机映射、controller codec，以及数据路径和完整 episode 划分。

```bash
PYTHONPATH=src:. python -m dobot_nexus.cli.train --config examples/configs/rltoken/dobot_token.yaml
PYTHONPATH=src:. python -m dobot_nexus.cli.train --config examples/configs/rltoken/dobot_cache.yaml
PYTHONPATH=src:. python -m dobot_nexus.cli.train --config examples/configs/rltoken/dobot_pretrain.yaml
```

Token 使用 `training_mode=token_only`、`mode=bottleneck`、`selection=image_only`。
VLA 保持 eval 和冻结；只优化 Encoder/Decoder，不计算 flow loss。Pi0.5 根据自身
vision patch size 和相机图像尺寸选择图像 token。Decoder 只接收 latent 和 learned
query，不读取目标内容。对于前置缺失相机，首个 query 保持可见，防止 causal attention
出现全部屏蔽的行；缺失位置不参与 MSE。

Token Trainer 按 `algorithm.token.train_episodes/validation_episodes` 划分，保存
每个验证点的完整 Artifact、`best`/`last` 指针和 `resume.pt`。验证包含 zero/shuffle
latent 依赖比率，`token_validation.json` 绑定 feature ID。缓存入口默认要求两个依赖
门限通过。旧 provider 的算法消融必须显式设置 `data.require_token_acceptance=false`；
不能把这种运行记作完整表示配方复现。Token-only 优化器只含可训练参数。

Pi0.5 的 `model.rltoken.frozen_parameter_precision` 默认是 `fp32`。已有 BC 部署使用
BF16 参数时，可在 token-only 模式显式设为 `bf16`，并保持
`model.inference_parameter_precision=bf16`。冻结 VLA 使用原生 BF16 前向，Token
参数仍为 FP32；训练特征和部署特征采用同一数值 profile，profile 进入 feature ID。
接入既有权重时应先验证固定观测/噪声下的参考动作，再比较完整回合。

Token 数据默认使用模型的 `build_train_dataloader`。任务拥有其他原始记录格式时，
可通过 `algorithm.token.data_loader=module:function` 提供 `(model, config) -> loader`；
loader 需返回可索引 dataset 和 collate_fn，并遵守传入的 `data.episode_ids` 划分。
图像、状态、动作与语言预处理仍应使用模型的数据适配器；任务读取工具放在实验目录。

缓存入口可通过 `data.episode_reader=module:function` 读取 `offline_data.Episode`。
每个 Episode 提供 T 个控制器实际命令、T 个奖励和 takeover，以及 T+1 个实际观测。可单独提供监督动作/mask；时间戳存在时校验控制周期。
最后一个观测属于同一 episode，不能使用 reset 观测。缓存按各来源的完整 episode
划分训练/验证，每种来源至少两个 episode，截断尾部没有 MC 标签。
纯 VLA 回放可提供 `Episode.reference_seeds`，按原始观测索引记录生成命令时的 seed。
缓存提取对应特征时复用这些 seed，使 reference 与已执行的 VLA chunk 对齐；显式提供
此映射时必须覆盖全部需要提取特征的观测。未提供时使用缓存的确定性派生 seed。

内置 LeRobot reader 使用共享 reader，并要求数据显式保存最终观测帧：
`last_frame_is_final_observation=true`，最后一帧的 action 不当作已执行命令。
只有满足此布局的数据才能使用该选项；普通缺少最终观测的数据应由任务 reader 补齐
真实记录。`episode_labels` JSON 示例：

```json
{"0": {"origin": "demo", "terminated": true, "success": true, "accepted": true},
 "1": {"origin": "vla", "terminated": true, "success": false, "accepted": true}}
```

配置 `reward_key` 时读取逐步奖励；否则只在完整回合最后执行步生成 success 奖励。
截断回合必须提供逐步奖励。`takeover_key` 标记逐步接管；VLA 来源中包含接管的轨迹
不作为纯 VLA MC。失败示范默认只有接管位置可作 BC 监督；额外专家标签通过显式 mask 声明。缓存记录数据版本、奖励身份、折扣、chunk/stride、采样步数、seed、
特征/codec 身份、episode 划分及完整内容摘要。更新 Token 后必须重建缓存。

离线阶段使用同一个 `dobot_pretrain.yaml`，分别运行到不同目录：

| 阶段 `algorithm.pretrain.phase` | 初始化 `algorithm.pretrain.initialize` | 可更新参数 |
|---|---|---|
| `actor_bc` | `{}` | Actor |
| `base_critic` | `{}` | Base Critic |
| `residual_critic` | `{actor: /.../actor_bc/rltoken/best.pt, base: /.../base_critic/rltoken/best.pt}` | O、H |
| `offline_joint` | `{all: /.../residual_critic/rltoken/best.pt}` | Actor、O、H |

每个阶段显式设置 `train.max_steps`、`save_every`、`max_validation_loss`，保存 `best.pt`、
`last.pt`、完整 checkpoint 和依赖的 `features/`。阶段产物记录数据范围、验证 loss、
准入结论和已验收组件；缺少验收 Actor/Base 的产物不能初始化残差阶段。
Critic/joint 的选点指标为验证 Critic loss，**不是任务成功率**；joint 产物仍需配对
评测后再用于实际任务。独立组件初始化只继承权重，新建 optimizer/target；
`train.resume_from` 则恢复完整 optimizer、target、采样 RNG、阶段时钟和 Replay。
两种方式互斥。恢复离线阶段时设 `algorithm.pretrain.initialize={}`；恢复在线任务时设
`algorithm.initialize_from=null`。迁移阶段产物时保留其 `features/` 和 checkpoints；
离线续训仍需相同缓存以完成验证。

完整实验可显式使用 `algorithm.pretrain.validation_policy=diagnostic`，省略
`max_validation_loss`：仍计算验证 loss 并按它保存 best，有限 loss 的产物允许初始化
下一阶段。产物的 `validation.policy` 记录该选择；此时 `accepted` 只表示允许进入
后续实验，不代表任务效果已验收。非有限验证 loss 始终报错。默认策略
`loss_threshold` 保留显式阈值准入。最终收益必须由独立任务评测确定。

## 在线与部署

`dobot_td3.yaml` 从 `algorithm.initialize_from` 加载通过准入的阶段产物，启动新在线
会话，在线更新数和预算从零开始，target 从所加载权重初始化。预训练 Actor 在对应
版本收到 applied ACK 后可启用；无需再累计固定在线更新数。公开 Artifact 只部署
Actor 与冻结 provider，算法恢复格式为 `rltoken_td3_resume_v2`。

### 执行阶段与训练范围

`dobot_td3.yaml`（及继承它的缓存、离线预训练配置）显式选择：

```yaml
algorithm:
  policy_switch:
    task_mode: full_task
    execution_scope: critical_phase
    training_scope: critical_phase
```

两个 scope 独立支持 `all` 或 `critical_phase`。未配置时均为 `all`。
`execution_scope` 决定哪些观测可以使用 Actor，仍需满足训练就绪和权重 applied ACK；
`training_scope` 决定哪些 chunk 可以进入 RLT 的离线训练、在线 Replay 和验证数据。
该范围控制不裁剪 Stage 1 VLA/Token 表示学习数据。

在上述配置下，行为与 RLinf 的 `SimulatorRLTRoute` 阶段控制一致：

| 当前观测处于关键阶段 | Actor 就绪 | 执行动作来源 | Replay 范围资格 |
|---|---|---|---|
| 否 | 否或是 | VLA reference | 无 |
| 是 | 否 | VLA reference | 有 |
| 是 | 是 | Actor | 有 |

范围资格仍须通过真实执行、动作编码、奖励定稿及审核检查。关键阶段 VLA 预热样本可
用于训练；阶段外 chunk 仍获得持久执行日志与 ACK，但不挂起奖励标签、不增加 UTD，
也不能经延迟审核变成训练样本。人工 takeover 与阶段信号分别记录，不互相替代。

`task_mode=full_task` 要求环境适配器在 reset 及每个 post-action observation 中提供：

```python
observation.control = {"rlt_switch_flags": np.array([in_critical_phase], dtype=np.bool_)}
```

批量观测使用 bool `[B]`，顺序与 `env_ids` 相同。信号描述**这份观测开始的动作块**；
在 chunk 执行后才进入关键阶段，不会把刚完成的接近动作块补记为训练样本。
只在 chunk 边界切换策略，块内控制器接管仍通过实际动作和 takeover 反馈。
全任务阶段控制启用时，信号缺失会明确报错，不自动当作 Actor 全程可用。

环境适配器拥有任务阈值、键盘触发、锁存和逐环境 reset：例如 RLinf 仿真适配器可将
`info["rlt_switch_flags"]` 映射到上述字段；真机可在按键切换后保持 true，直到当前
episode 重置。使用 `latch_until_done` 的任务必须在适配器内只清除已结束环境的状态。
算法不内置插销距离或 USB 任务判断。`task_mode=critical_phase` 表示环境直接重置到
精细操作任务，所有观测都视为关键阶段；该选项本身不会移动机器人或裁剪仿真场景。

`all` 训练范围与阶段执行可以组合使用。若下一观测不能使用 Actor，TD bootstrap
使用原始 VLA reference，不叠加 Actor 修正或 target smoothing；真实终止语义不变。
单环境、向量采集和部署共用 `routing.py`；非 Actor 行不消耗 Actor 探索噪声。
部署 Artifact 保留同一份配置，调用方继续提供阶段信号；部署模型视为已就绪。

LeRobot 缓存的 `data.policy_switch_key` 指定 bool 阶段列，示例为 `rlt_switch_flags`，
包括最后观测帧。自定义 Episode reader 在观测的 `control` 中提供同一信号。
缓存按 chunk 起点筛选，再按 episode 隔离训练/验证，MC return 仍从实际奖励计算。
缓存、Replay、恢复 fingerprint 和离线阶段产物校验数据范围；阶段配置不同的缓存
不能直接导入，需按目标范围重建。仅含关键阶段的已裁剪数据可显式配置
`task_mode=critical_phase`，并在缓存、离线训练、在线和部署中保持一致。

环境配置 `reward_mode=episode` 时，adapter 提供：

```python
finalize_episode(env_id=..., episode_id=..., pending=tuple_of_transitions, timeout=...)
# 返回 iterable[Finalization(key, rewards, accepted, episode_success=None)]，也可以暂时返回 None。
poll_finalizations(pending=tuple_of_transitions, timeout=...)
# 可选：后续审核或重启后补交标签。
```

标签只允许补充奖励、episode 成败标签和审核结论。Learner 从 journal 取出原记录，校验身份、执行事实
和奖励形状，持久化后 ACK；重复定稿不重复计预算，冲突和未知记录被拒绝。
失败回合可 `accepted=True`，pending/拒绝记录不训练。重启将 pending 记录交还环境
adapter；环境 adapter 负责真实任务阶段和执行状态的协调。

`update_schedule=episode_end` 在回合结束后集中消耗预算；向量采集以一次完整反馈批次
为边界。`drain_on_stop=true` 达到采集上限后继续消费预算，直到更新上限、预算耗尽或
有效 Replay 不足 warmup。checkpoint 保存 Learner target/optimizer/噪声 RNG 和 rollout
探索 RNG。环境恢复会建立新执行 session，不声称能逐位重放真实机器人状态。

```bash
examples/train.sh --config examples/configs/rltoken/dobot_td3.yaml --gpus 0,1
```

GPU 编号和环境 adapter 由实际资源安排决定。该示例不是自动启动真机的指令。

## 验证边界

CPU 测试覆盖新旧配方隔离、执行事实与监督隔离、Base 冻结、阶段优化器、延迟 targets、
缓存 episode 隔离、在线预算去重、完整恢复后的下一次更新、Token-only 冻结和部署资产。
数值对照和诊断脚本保留在 Git 忽略的 `local_script/rltoken_alignment/`，不进入生产依赖。

任务验收仍需依次比较 reference 透传、小 Actor BC、固定 Actor 训练 Critic、短程 joint、
在线微调；固定场景和 reference seed，报告配对成败、动作偏离、Q 排序和实际 UTD。
本实现不把模块测试或参考项目尚未填写的实验 18 真机结果当作任务收益证明。

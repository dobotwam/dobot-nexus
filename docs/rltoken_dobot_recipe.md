# RLToken 双残差配方

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

## Token 预训练

```bash
PYTHONPATH=src:. python -m dobot_nexus.cli.train --config examples/configs/rltoken/dobot_token.yaml
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

### 执行阶段与训练范围

算法配置通过 `policy_switch` 选择执行和训练范围。

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

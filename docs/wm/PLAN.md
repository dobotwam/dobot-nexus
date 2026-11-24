# PLAN：统一框架中的 WM 接口与 Wan2.2 接入

> 设计状态：待评审；不是已实现的使用说明。
> 日期：2026-11-23。远端：B300_2，`<repo-root>`，分支：`wm`。
> 基线 HEAD：`ff67100afe73b3381b2de2b9a4ecfddc8f0a066a`，并包含现场已有未提交修改。
> 配套：[伪代码](PSEUDOCODE.md)、[依据与来源](SOURCES.md)。

## 1. 目标与本轮交付

在现有 `dobot_nexus` 目录体系中增加视频生成与环境动态建模能力，沿用 VLA 的
“配置 → 动态模型装配 → 数据适配 → Trainer → Artifact → 推理服务”调用组织。
WM 的基础类和公共协议统一放在 `core/`，以 `wm_*.py` 命名；模型、训练、数据和推理
分别放回现有职责目录。视频、潜变量和生成条件使用 WM 协议，实验与执行设施共用。

本轮交付只包括 PLAN、伪代码和来源记录；不增加可导入的实现、安装依赖、下载权重、启动训练或机器人服务。
`wm` 从远端当前工作树创建，已有修改原样保留，不自动提交或推送。上一轮本地 README 重构不在本次同步范围。

分两层建设：

1. **视频生成基础能力**：Wan2.2 的 T2V / I2V 训练适配、生成、组件保存和恢复。
2. **具身世界模型能力**：在可验证的生成基线之上，增加机器人状态、动作序列条件和未来预测评价。

普通 I2V 输出的是以图像和文本为条件的可能未来。动作条件世界模型还需学习
`p(未来观测 | 历史观测, 机器人状态, 动作序列, 任务)`，不能把基础视频模型默认视为可用于控制的动力学模型。
首版不承诺闭环规划、在线 RL、多视角联合生成或实时机器人控制。

## 2. 现有框架审阅结论

| 现有入口 | 当前约束 | WM 处理 |
|---|---|---|
| `core/config.py::validate_config` | 要求 `state_dim/action_dim/action_horizon` | 独立校验视频规格、条件方式和组件配置 |
| `models/base.py::BaseVLAModule` | `predict_action_chunk`、`policy_spec` | `core/wm_base.py` 定义 `BaseWorldModel`，Spec 放在 `core/wm_contracts.py` |
| `models/base.py::BaseDataAdapter` | 观测编码、动作解码 | `core/wm_base.py` 定义视频 adapter 契约 |
| `cli/train.py` | 直接构造 `BCTrainer` | `wm train` 分派到 `WorldModelTrainer` |
| `training/bc_trainer.py` | 一个模型、监督训练、VLA `TrainOutput` 与 Artifact | 借鉴优化循环；不直接包装成 BC 任务 |
| `inference/policy.py` | 校验 state、相机和 `[B,H,A]` 动作输出 | 独立生成入口与 `[B,T,H,W,3]` 视频输出 |
| `core/artifacts.py` | VLA manifest、完整 state dict 与部署资产 hook | `core/wm_artifacts.py` 管理 WM 多组件产物 |
| `training/runtime.py` | Local / Accelerate / FSDP；注册现有 `TrainOutput` | 共用 Runtime，补充 WM 输出与分布式契约测试 |
| `core/experiment.py` | 分布式共享 run 路径、日志、resolved config | 共用 `Experiment`，记录 WM 的 resolved config |

`AGENTS.md` 当前规定只有 VLA 的模型继承协议。实施时将规则补充为：VLA 契约保留在
`models/base.py`，WM 契约集中在 `core/wm_base.py`，每个家族只有一套基础接口。
Wan 使用 WM 契约；现有 Pi0/Pi0.5 的目录和公共行为保持兼容。

## 3. Wan2.2 路线与模型边界

官方结构核对见 [SOURCES](SOURCES.md)。下面区分官方事实与平台设计建议。

| 变体 | 官方结构要点 | 对接要求 |
|---|---|---|
| TI2V-5B | 单个 DiT，支持 T2V/I2V；Wan2.2 VAE，时间/空间压缩为 `(4,16,16)` | 首版建议；单模型 Trainer 和组件导出 |
| I2V-A14B | 高噪声、低噪声两个专家；Wan2.1 VAE，压缩为 `(4,8,8)` | 增加专家训练阶段、推理切换及双组件恢复 |
| T2V-A14B | 双专家；VAE 与 I2V-A14B 同系列；图像条件结构不同 | 复用专家调度，独立校验任务与条件 |

TI2V-5B 和 I2V-A14B 都能以首帧为条件生成后续视频，优先级差异主要是接入复杂度与资源成本。
建议 TI2V-5B 优先；如决定 A14B 优先，P3 的专家与条件支持必须提前进入首版，不能只换权重路径。
音频生成和 Animate 不纳入此次范围。

模型专属结构放在 `models/wan22/`：

- UMT5 文本编码、tokenizer、VAE、DiT，以及精度/潜变量缩放规则。
- TI2V 的首帧潜变量约束与逐 token 时间步；A14B I2V 的图像潜变量和 mask 拼接。
- 专家路由、flow 时间步、CFG、采样器，以及原生 checkpoint 格式转换。
- 模型参数和分辨率来自配置与资产 metadata；不把 5B 的潜变量维度、VAE 或条件方式写入公共 core。

官方生成类会把模型置为 eval 并关闭梯度，不能把其 `generate()` 直接用作训练接口。
建议以官方算子和权重格式为参考，在 Nexus 中拥有独立训练装配逻辑；正式实现前固定上游 commit、依赖版本和文件校验值，保留许可证与来源。

## 4. 统一目录与 WM 命名

```text
src/dobot_nexus/
├── core/
│   ├── config.py                  # 共用配置解析；保留 VLA load_config/校验入口
│   ├── import_utils.py            # 共用符号解析；保留 VLA 模型加载入口
│   ├── experiment.py              # 共用实验目录、配置记录与日志
│   ├── contracts.py               # 既有 VLA 输入输出协议
│   ├── artifacts.py               # 既有 VLA Artifact
│   ├── wm_base.py                 # BaseWorldModel、BaseWMDataAdapter
│   ├── wm_contracts.py            # VideoBatch、Request、Output、Spec
│   ├── wm_config.py               # WM 配置校验
│   ├── wm_loading.py              # model.target + WM 类型检查
│   └── wm_artifacts.py            # WM 多组件 manifest、导出与恢复
├── models/
│   ├── base.py                    # 既有 VLA 基础接口
│   ├── pi0/, pi05/                # 既有模型
│   └── wan22/
│       ├── configuration.py
│       ├── modeling.py            # 训练/生成共用模型实现
│       ├── data_adapter.py
│       ├── components.py          # DiT、VAE、text encoder 装配
│       ├── conditioning.py        # 5B、A14B、后续 action 条件
│       ├── noise_schedule.py      # 时间步、专家路由、采样器
│       └── checkpoint.py          # 模型原生格式导入与导出
├── data/
│   └── video/                     # clip manifest、LeRobot 片段、bucket、cache
├── training/
│   ├── runtime.py                 # 共用 Local / Accelerate / FSDP
│   ├── bc_trainer.py              # 既有 VLA 训练
│   ├── wm_trainer.py              # WorldModelTrainer
│   └── wm_resume.py               # WM 恢复状态、数据游标、专家计数
├── inference/
│   ├── policy.py                  # 既有 VLA 推理
│   ├── wm_generator.py            # WorldGenerator
│   ├── wm_worker.py               # 视频生成任务生命周期
│   └── wm_server.py               # WM 生成服务
└── cli/
    ├── main.py                    # 共用顶层命令分派
    ├── wm.py                      # wm 子命令分派
    ├── wm_train.py
    ├── wm_generate.py
    ├── wm_serve.py
    └── wm_prepare.py
examples/configs/wm/               # 实施后才新增可运行配置
scripts/env/create_wan22_env.sh     # 实施后才新增独立环境
tests/models/wan22/                # 模型契约测试，其余测试按职责归入现有 tests
```

命名规则：公共目录中的 WM 专属文件使用 `wm_` 前缀；具体模型以 `models/wan22/` 隔离。
`wm` 是 CLI 子命令及文档/配置分组，不新增同名顶层 Python 子系统。
`core/wm_base.py` 是 WM 基础接口的唯一位置，`core/wm_contracts.py` 是其输入输出协议的唯一位置。

依赖方向：`models/wan22` → `core/wm_base` → `core/wm_contracts`。
WM Trainer、生成器和 loader 使用上述接口；公共 core 不导入 Wan 实现，不包含 VAE 压缩率或专家阈值。
后续视频模型直接实现 `BaseWorldModel`，由 `model.target` 选择。

共用与扩展边界：

- 在 `core/config.py` 中公开领域无关的 `load_config_data()`，完成 YAML 继承、环境变量和 CLI 覆盖。
  既有 `load_config()` 继续执行 VLA 校验；`core/wm_config.py::load_wm_config()` 复用解析后执行 WM 校验。
  返回类型沿用 `Config`，以复用 `save_config()` 和 `Experiment`。
- 动态符号解析复用 `core/import_utils.py::import_from_string()`；`core/wm_loading.py` 只增加 WM 类型检查和构造，
  不复制 import 机制或增加模型注册表。
- 共用 `core/experiment.py::Experiment`、`training/runtime.py`、seed 和 tracking。
  Runtime 补充 `WMTrainOutput` 的 pytree/分布式兼容；组件保存和恢复语义分别归 WM Artifact 与 resume。
- `training/wm_trainer.py` 负责视频优化循环，`inference/wm_generator.py` 负责生成调用，
  `core/wm_artifacts.py` 负责 WM 产物。各自位于现有职责层，不复制 Experiment 或 Runtime。
- 顶层 CLI 懒加载 `cli/wm.py`，再分派到 `cli/wm_train.py` 等入口；旧命令和 VLA 导入路径保持兼容。
- `core/__init__.py` 保持轻量，不通过公共包导入自动加载 WM 基础类或 Wan 依赖。
  模型算子、条件注入和 checkpoint 格式均由 `models/wan22/` 管理。

## 5. 数据、时间与条件契约

| 边界 | 拟议格式 | 约束 |
|---|---|---|
| 视频样本 | RGB `uint8[B,T,H,W,3]` | 公共格式；模型 adapter 转为 `float[B,3,T,H,W]` |
| 时间 | `timestamps[B,T]` 秒、`fps[B]` | 真实采样间隔；VFR 必须显式重采样或拒绝 |
| 文本 | `prompts: list[str]` | 训练 prompt dropout 与生成 negative prompt 分开记录 |
| 片段身份 | dataset/episode/clip/camera、原始帧索引 | 缓存、划分与恢复可追溯 |
| 图像条件 | 单独的当前帧及条件 mask | 条件只能来自可见前缀，不能读取未来帧 |
| 可选机器人条件 | 状态、动作、时间戳、布局/单位 metadata | 基础 Wan 不支持时直接报错，不静默忽略 |
| 输出 | `uint8[B,T,H,W,3]`、实际帧时间、seed、生成配置 | 包含首帧与否由 Spec 声明；服务可落盘后返回结果引用 |

首版数据按固定 FPS 分组；模型没有 FPS 条件输入时，不混用不同物理时间尺度的片段。
首版每个样本只选择一个相机；多路相机可形成独立样本，但同一 episode 的所有相机必须进入同一 split。
不把相机拼接默认解释为多视角联合世界模型。先按 episode/场景划分训练验证集，再切片，避免相邻重叠片段泄漏。

视频片段来自两种 reader：普通视频 + 文本 manifest、LeRobot 连续片段。后者按 episode 和 timestamp
取图像，逐帧图片和 MP4 使用同一时间契约，不用动作序号代替视频时间戳。

Wan 的形状约束由模型 Spec 提供：`T = 4k+1`；5B 的像素空间需匹配 VAE 和 patch 的总步长，
A14B 使用自己的步长。首版用固定 `(T,H,W)` bucket，短片段拒绝或另分 bucket，不通过重复末帧伪造有效未来。
训练全程同一片段使用一致的空间增强；涉及状态和动作时，翻转/裁剪需核对几何含义。

机器人 20 Hz 数据不能只把导出 FPS 改成 24 就称为保持同一动作。保留真实时间或显式重采样，
记录动作在帧间区间的含义；首帧 `t0` 到下一帧 `t1` 对应的是该区间实际动作。
高频动作条件单独记录时间轴与 mask，适配器负责聚合到潜变量时间，不简单截断为 T 个动作。

缓存按 dataset 内容身份、split、原始帧范围、resize/crop、VAE/text 权重和版本、prompt、dtype 建键。
固定 encoder 时可缓存 text/latent；随机增强需选择固定离线增强并记录，或在线编码。CFG dropout 在读取缓存后采样。
不得缓存未来信息到条件输入；不得重用换 VAE/换相机后的旧潜变量。

## 6. 训练流程与分布式语义

1. `wm train` 加载 WM 配置，校验家族、组件、bucket、任务、条件和资源路径。
2. 初始化运行时，再创建所有 rank 一致的实验目录和配置快照。
3. `model.target.from_config(mode='train')` 构造可训练 DiT/adapter，冻结并固定 VAE、text encoder。
4. 建立 clip loader、优化器、学习率调度器、EMA（可选）；之后由 runtime 包装。
5. adapter 编码视频与条件；从 batch 采样 flow 噪声/时间，必须经 `wrapped_model(batch)` 进入 forward。
6. FP32 计算有效位置上的 loss，执行累积、梯度检查、裁剪、优化器更新；只有成功更新才计 step。
7. 按独立验证配置生成固定样例并保存视频、metadata 和指标；恢复训练 RNG、模块模式及 encoder 的 eval 状态。
8. 保存组件权重、优化器/调度器/EMA、每 rank RNG、bucket/sampler 游标和数据身份。

拟采用的微调目标是 rectified-flow velocity matching：

```text
z_sigma = (1 - sigma) * z_data + sigma * epsilon
velocity_target = epsilon - z_data
loss = masked_mean((velocity_pred - velocity_target)^2)
```

这是本项目拟实现的微调方案，不声称复现官方预训练配方。时间采样分布、loss weighting 与推理采样器分别配置。
默认先用明确的 uniform sigma 基线，后续评估其他分布；将 sigma 转为模型时间步时使用模型专属规则。

TI2V I2V 的已知首帧潜变量保持固定，对应 token 时间设为零，loss 排除已知区域；T2V 无该条件。
A14B I2V 沿用其条件 latent/mask 拼接方式，不套用 5B 的覆盖规则。具体 mask 与 loss 行为需做定点数值测试。

**A14B 实施策略**：先分别微调高/低噪声专家，每个训练作业只训练指定专家并从其时间范围采样，
生成时装配两份匹配产物。每个专家的训练 budget、optimizer、EMA 和 resume 独立，未微调专家也要记录身份。
后续若联合训练，所有 rank 在一个 optimizer update（含累积窗口）必须采用一致专家调度，记录每个专家的更新次数。
不能让不同 rank 随机进入不同专家分支；不能套用普通单模型恢复而遗漏另一个专家。

初期支持 local → DDP/FSDP，先验证小模型分布式，再验证真实权重。Sequence parallel 在后续阶段处理，
不把官方推理 USP/FSDP 支持等同于 Nexus 已支持相同训练组合。
有效 batch 以 clip 计：`每数据并行 rank 的 batch × DP size × accumulation`。
引入 sequence parallel 后不能将总 GPU 数直接作为 DP size。

精度、VAE 分块、activation checkpointing、LoRA/full finetune 均由显式配置控制。
建议先用冻结基础组件 + DiT LoRA 验证微调链路，再开放全参数训练；LoRA 参数应在分布式包装前注入。
冻结编码器的计算不参与 optimizer，推理 offload 不得搬动正在训练的参数。

## 7. 生成、服务与世界模型扩展

生成调用链：

```text
WM Artifact + GenerationRequest
  → WorldGenerator 恢复组件/Spec
  → 模型 adapter 编码文本、可见图像及可选动作条件
  → scheduler 创建当前请求的 sigma/timestep 序列
  → DiT 或当前噪声区间专家 + CFG
  → scheduler 更新 latent / 应用模型专属条件约束
  → VAE decode → 视频 + metadata
```

专家切换依据 scheduler 的实际模型时间步，不按“前一半/后一半采样次数”切分。
高低噪声 guidance、边界与 VAE 身份写入 Artifact，运行时不靠模型目录名推断。

`wm generate` 先支持离线生成；服务使用独立任务生命周期：提交、排队、生成、取消、结果读取。
大视频写入受控结果目录，返回 job ID/结果引用，避免阻塞 VLA 的动作请求 worker 或使用动作 chunk 消费器。
每个 job 独立 seed/generator、scheduler、输出目录和状态；worker 设队列长度、帧数/分辨率/时长上限。
取消在 denoise step 间检查，落盘完成才发布成功状态。API 不接受任意服务器文件路径或动态 `model.target`；
外部媒体采用受控上传/资产 ID，延续已有服务的认证与路径边界。

具身扩展建议单列阶段：

- 增加 action/state encoder，将候选动作按时间对齐到视频 token，通过独立条件分支/attention 注入 DiT。
- 新参数采用可回退的初始化；关闭动作分支时必须与基础视频模型一致。
- 用同一初始观测、不同动作测试条件敏感性；用置乱动作、无动作条件作为消融。
- 多步 rollout 每段只读取已知或已生成的前缀，携带显式时间和动作窗口；长时误差累积独立评估。
- 初期输出未来视频，不直接输出关节控制、不连接真机执行；规划器以后作为独立算法调用预测接口。

## 8. WM Artifact、依赖与恢复

独立 `artifact_kind: world_model`、`schema_version: 1`；VLA 的既有 schema 和 loader 行为保持兼容。
WM loader 拒绝 VLA Artifact，VLA 加载路径需对 WM kind 给出明确错误。

```text
step_XXXXXXXX/
├── manifest.yaml                  # family、variant、components、spec、source revisions
├── config_resolved.yaml
├── deployment/
│   ├── denoiser/                  # 5B 或 LoRA；A14B 分 high_noise / low_noise
│   ├── vae/
│   ├── text_encoder/
│   ├── tokenizer/
│   └── assets/                    # transform、action layout/stats、scheduler 配置
└── resume/                        # optimizer、scheduler、EMA、RNG、sampler、expert counters
```

各组件采用逻辑名 + 相对路径或明确的外部资产引用 + revision + checksum。
LoRA 产物必须同时标识基础权重、注入模块、rank/alpha 与 adapter 权重，不能声称单独 adapter 可独立部署。
提供两种显式模式：portable bundle 带齐资产；reference bundle 使用固定内容地址的共享资产，离线校验缺失即失败。
不可偷偷回退到某台机器的预训练目录；VAE/text 权重大，不套用“小文件 assets”自动复制策略。
部署分片导出，避免 A14B 两份完整 FP32 state dict 同时聚合到一个 rank 导致内存峰值。

保存采用临时目录、各 rank 写入确认、组件完整性检查、manifest 最后发布和原子更新 latest。
恢复 fingerprint 包括模型/变体、条件方式、数据划分与 bucket、VAE/text 身份、噪声采样、专家计划、
优化器/精度/LoRA、并行布局与有效 batch。首版精确续训要求相同并行布局；不宣称跨拓扑确定性恢复。

Wan 使用独立环境和可选依赖。基础包和 VLA 命令不导入 Wan、diffusers 或专属 attention 扩展。
安装时验证 B300 架构对应的 Torch/CUDA/attention kernel，而非只检查 `import torch`。

## 9. 配置与命令草案

以下是拟议配置，不是可运行 recipe。低分辨率/短片段仅用于接入检查；不能代表官方分辨率下的质量。

```yaml
schema_version: 1
family: wm
task: train
exp:
  name: wan22_ti2v5b_i2v_smoke
  output_dir: ${WM_RUN_ROOT}
  seed: 42
model:
  target: dobot_nexus.models.wan22.modeling:Wan22WorldModel
  variant: ti2v_5b
  pretrained_path: ${WAN22_BASE}
  condition_mode: image
  freeze: [vae, text_encoder]
  finetune: {method: lora, rank: 16, alpha: 16}
data:
  format: video_manifest
  manifest: ${WM_DATA_MANIFEST}
  split: train
  batch_size: 1
  bucket: {num_frames: 33, height: 384, width: 640}
  fps_policy: preserve
  camera_mode: single_view
train:
  max_steps: 10
  grad_accum_steps: 1
  resume_from: null
objective:
  type: flow_matching
  timestep_sampling: uniform_sigma
  prediction: velocity
  condition_loss: exclude_known
runtime:
  launcher: local
  compute_dtype: bf16
  loss_dtype: fp32
artifact:
  kind: world_model
  asset_mode: reference
```

A14B recipe 必须另外声明 `variant: i2v_a14b`、`expert_train: high_noise|low_noise`、
专家时间范围、双组件资产及其组合校验；不允许在上面的 5B 配置中仅替换模型名称。

拟议命令：

```text
dobot-nexus wm prepare  --config <wm-data-config>
dobot-nexus wm train    --config <wm-train-config>
dobot-nexus wm generate --artifact <wm-artifact> --request <request-json>
dobot-nexus wm serve    --artifact <wm-artifact>
```

正式学习率、有效 batch、训练长度、生成长度与 FPS，待真实数据清点和显存/吞吐验证后确定。

## 10. 实施阶段与验收

| 阶段 | 主要改动 | 必须通过的验收 |
|---|---|---|
| P0：接口定稿 | 确定优先变体、任务/数据、上游 revision；细化本 PLAN | 明确基础生成与 action-WM 边界，锁定输入输出和条件 |
| P1：WM 接口 | `core/wm_*`、`training/wm_trainer.py`、WM 生成与 CLI；复用配置解析、Experiment 和 Runtime | CPU fake model 完成一次优化、保存/恢复/生成；旧 VLA 契约与 CLI 回归通过 |
| P2a：5B 原生生成 | 固定组件与权重加载、T2V/I2V adapter、scheduler | 同权重/输入/seed 与官方参考逐层及生成结果对齐；不以“能出视频”代替验收 |
| P2b：5B 微调 | clip reader/bucket/cache、flow loss、LoRA、验证采样和续训 | 单 clip 可过拟合；无未来泄漏；冻结组件无梯度；缓存和在线编码对齐 |
| P2c：分布式 | DDP/FSDP、累计步数、分片导出 | local/多卡受控数值对齐；中断续训对齐样本顺序和成功更新次数 |
| P3：A14B | 双专家、专家独立微调、组合 Artifact、分支 CFG | 边界上下及等于边界路由正确；双组件恢复完整；跨 rank 调度一致 |
| P4：服务 | 独立生成 worker、任务状态/取消/结果接口 | 并发隔离、seed 可追溯、取消无假成功、资源上限和认证测试 |
| P5：动作条件 WM | 时间对齐、action/state 分支、预测与 rollout | action 消融/置乱、同场景不同动作、持出场景、多步漂移评估 |

CPU 测试覆盖形状/时间/mask、配置错误、依赖隔离、mock 模型梯度、Artifact 完整性和恢复契约。
GPU 测试单独记录软件版本、输入、seed、误差阈值、显存、吞吐，不把 CPU 测试视为数值或模型质量验证。
视频指标可包含重建误差、感知/时序指标，使用冻结版本的评价器；多模态未来不能只用像素误差判断。
动作条件任务还要报告任务相关预测指标和条件敏感性，视频美观不能证明物理一致性。

## 11. 资源与待定项

现场只读检查：8 张 NVIDIA B300，每卡报告 275040 MiB；2026-11-23 检查时显存使用为 0。
`/data` 约剩 320 GB（96% 已用）。该快照不是资源预留或显存适配结论。
实施前估算模型、潜变量缓存、优化器状态、保留 checkpoint 和验证视频的总空间；共享资产按内容去重，
缓存和 checkpoint 设置保留策略，确认完整产物后再回收本作业旧文件。不能因显存空闲就忽略磁盘风险。

需在正式实现/训练前定稿：优先 5B 还是 I2V-A14B；首批视频/LeRobot 数据与合法使用范围；
原生 I2V 基线和动作条件阶段的先后；训练帧数/分辨率/FPS；固定的上游 commit 与权重 revision。
本草案在这些参数确定前仍可完成 core 设计和 CPU fake model 方案审阅。

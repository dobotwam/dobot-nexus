# PSEUDOCODE：core 内的 WM 接口与 Wan2.2 调用链

> **仅用于设计评审，不可执行。** 下列类型和 helper 是拟议接口，不表示仓库已有实现。
> 省略的存储、分布式和模型底层操作由相应模块实现；关键语义在 [PLAN](PLAN.md) 中约束。
> 本轮不创建 `src/`、`tests/` 或可运行 YAML。

## 1. core 内的 WM 契约

```python
# core/wm_contracts.py —— 公共布局，不携带 Wan 的 VAE/专家常量
@dataclass
class VideoBatch:
    rgb: UInt8Tensor             # [B, T, H, W, 3]，单视角
    timestamps: FloatTensor      # [B, T]，相对时间，单位秒
    fps: FloatTensor             # [B]，与实际采样间隔一致
    prompts: list[str]
    sample_ids: list[ClipIdentity]
    valid_frames: BoolTensor     # [B, T]；首版只接收完整 bucket
    robot: RobotCondition | None # 数据/动作时间轴与 layout 必须显式声明

@dataclass
class GenerationRequest:
    prompts: list[str]
    negative_prompts: list[str]
    context_rgb: UInt8Tensor | None  # [B, Tc, H, W, 3]；基础 I2V 仅 Tc=1
    robot: RobotCondition | None
    num_frames: int                  # I2V 包含已知首帧
    height: int
    width: int
    fps: float                      # 与训练/任务时间语义匹配，非随意播放倍率
    seed: int
    sampling: GenerationOptions     # steps / solver / shift / named CFG scales

@dataclass
class WorldModelSpec:
    tasks: set[str]                  # t2v / i2v / 后续 action_i2v
    context_frames: set[int]
    temporal_stride: int
    spatial_multiple: tuple[int, int]
    condition_types: set[str]
    output_includes_context: bool
    time_policy: TimePolicy          # 固定 FPS 或明确的时间条件能力

@dataclass
class WorldModelOutput:
    rgb: UInt8Tensor                 # [B, T, H, W, 3]
    timestamps: FloatTensor
    metadata: dict                   # seed、component IDs、sampling、crop、fps

@dataclass
class WMTrainOutput:
    loss: Tensor                    # scalar，保留梯度
    metrics: dict[str, Tensor]
```

`RobotCondition` 保存 state/action、各自时间轴、有效 mask、单位和关节/夹爪布局。
基础模型未声明 `robot_action` 能力时，收到非空 robot 条件立即报错。

```python
# core/wm_base.py —— WM 模型和 adapter 基类，集中放在 core
# 仅依赖 wm_contracts，不导入具体模型
from dobot_nexus.core.wm_contracts import WorldModelSpec, WorldModelOutput, WMTrainOutput

class BaseWorldModel(nn.Module, ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, config, *, mode): ...

    @property
    @abstractmethod
    def spec(self) -> WorldModelSpec: ...

    @abstractmethod
    def build_train_dataloader(self, config): ...

    @abstractmethod
    def training_step(self, encoded_batch) -> WMTrainOutput: ...

    def forward(self, encoded_batch):
        return self.training_step(encoded_batch)

    @abstractmethod
    def generate(self, request, *, generator, cancel) -> WorldModelOutput: ...

    @abstractmethod
    def component_specs(self) -> list[ComponentSpec]: ...

    @abstractmethod
    def export_components(self, exporter) -> ComponentManifest: ...

    def train(self, mode=True):
        super().train(mode)
        self.keep_frozen_components_in_eval()
        return self

class BaseWMDataAdapter(ABC):
    def validate_video_batch(self, raw, spec): ...
    def encode_train_batch(self, raw, *, encoders, rng): ...
    def encode_request(self, request, *, encoders): ...
    def decode_video(self, latents, *, codec, metadata): ...
```

## 2. 配置、模型装配与 CLI

```python
# core/config.py —— 在原模块内分离通用解析与 VLA 校验
def load_config_data(path, overrides=None):
    resolved = read_and_merge_yaml(path)
    resolved = resolve_environment(resolved)
    resolved = apply_overrides(resolved, overrides)
    # 保留原有路径解析、extends 检查、Config 包装和 exp.config_path 记录
    return wrap_config_with_source_path(resolved, path)

def load_config(path, overrides=None, *, expected_task=None):
    config = load_config_data(path, overrides)
    validate_config(config, expected_task=expected_task)  # 原 VLA 校验
    return config

# core/wm_config.py —— 使用同一解析器，执行 WM 校验
from dobot_nexus.core.config import load_config_data

def load_wm_config(path, overrides=None, *, expected_task=None):
    config = load_config_data(path, overrides)
    validate_wm_config(config, expected_task=expected_task)
    # 校验 family=wm、任务、视频与条件字段；不要求 VLA 动作维度
    return config

# core/wm_loading.py —— 使用共用符号解析，只增加 WM 类型检查
from dobot_nexus.core.import_utils import import_from_string
from dobot_nexus.core.wm_base import BaseWorldModel

def build_world_model(config, *, mode):
    Model = import_from_string(config.model.target)  # 仅可信本地配置
    require_subclass(Model, BaseWorldModel)
    model = Model.from_config(config, mode=mode)
    require_instance(model, BaseWorldModel)
    validate_config_against_spec(config, model.spec)
    return model

# 顶层 cli/main.py 新增一个懒加载入口；原 train/infer/serve 分派不变
if command == "wm":
    from dobot_nexus.cli.wm import main as wm_main
    wm_main(remainder)

# cli/wm_train.py，由 cli/wm.py 分派
from dobot_nexus.core.wm_config import load_wm_config
from dobot_nexus.training.wm_trainer import WorldModelTrainer

config = load_wm_config(args.config, args.overrides, expected_task="train")
result = WorldModelTrainer(config).run()
print_run_result(result)
```

## 3. 片段读取与条件隔离

```python
# data/video/：先划分 episode，再生成 clip；同 episode 的不同相机不能跨 split
episodes = manifest.episodes(split=config.data.split)
clip_ids = make_contiguous_clips(
    episodes,
    length=bucket.num_frames,
    timestamp_policy="validate_or_explicit_resample",
    cross_episode=False,
)
clip = reader.read(clip_ids[i], camera=config.data.camera)
assert strictly_increasing(clip.timestamps)
assert consistent_fps(clip.timestamps, clip.fps)
assert clip.frame_count == bucket.num_frames
assert frame_shape_matches_bucket(clip)

# models/wan22/data_adapter.py
validate_shape_using_spec(clip, model.spec)
rgb = spatial_transform_consistently_across_time(clip.rgb)
video = bthwc_to_bcthw(rgb).float() / 127.5 - 1.0
# 输入 [B,T,H,W,C]，输出 [B,C,T,H,W]

with no_grad():
    target_latent = vae.encode(video)          # 训练标签可读取完整片段
    text = text_encoder(clip.prompts)
    # 条件只读公开首帧，单独编码，不从完整片段 latent 中盲取前缀
    first_rgb = video[:, :, :1] if task == "i2v" else None
    condition = wan_conditioner.encode_visible(first_rgb, output_shape=video.shape)

text = sample_prompt_dropout_after_cache(text, unconditional_text, rng)
return EncodedTrainingBatch(
    target_latent=target_latent,
    text=text,
    condition=condition,
    robot=validate_optional_robot_condition(clip.robot, model.spec),
    sample_ids=clip.ids,
)
```

`vae.encode` 的因果缓存必须按 clip 重置。target 编码和条件编码之间不共享未清理的时序缓存。
现阶段只接受完整固定 bucket；将来支持 padding 时还需明确像素 mask 到潜变量/patch mask 的映射。

## 4. Wan 组件与 flow loss

```python
# from_config：导入底层组件，不调用会关闭全部梯度的官方生成类初始化训练
variant = resolve_wan_variant(config.model.variant, pinned_component_metadata)
vae = load_vae(variant.vae_asset).eval().requires_grad_(False)
text_encoder = load_text_encoder(variant.text_asset).eval().requires_grad_(False)

if variant.is_dense:
    denoiser = load_dit(variant.dit_asset)
else:
    # 首版 A14B 训练一个指定专家；另一个只在生成装配时加载
    expert = require_choice(config.model.expert_train, {"high_noise", "low_noise"})
    denoiser = load_dit(variant.expert_asset(expert))

if config.model.finetune.method == "lora":
    denoiser.requires_grad_(False)
    inject_trainable_lora_before_distributed_wrap(denoiser, config.model.finetune)
else:
    denoiser.requires_grad_(True)

# models/wan22/modeling.py::training_step
z_data = batch.target_latent
sigma = noise_schedule.sample_training_sigma(
    batch_size=z_data.shape[0],
    rng=train_rng,
    expert_range=active_expert_range,  # dense=None；A14B 作业固定高/低范围
)
epsilon = randn_like(z_data, generator=train_rng)
z_noisy = (1 - broadcast(sigma)) * z_data + broadcast(sigma) * epsilon
velocity_target = epsilon - z_data
model_t = noise_schedule.sigma_to_model_t(sigma)

# variant-specific，独立测试每个变体；不能用统一首帧覆盖替代 A14B 条件
if variant == "ti2v_5b" and task == "i2v":
    z_noisy = keep_known_latents(z_noisy, batch.condition)
    timesteps = make_token_timesteps(model_t, known_mask=batch.condition.mask)
    kwargs = {}                                  # 已知首帧 token 的时间为零
    loss_mask = unknown_valid_latent_mask(batch)
elif variant == "i2v_a14b":
    timesteps = model_t
    kwargs = {"y": batch.condition.image_latents_and_mask}
    loss_mask = all_valid_latent_mask(batch)      # 条件在另一输入分支，非 5B 固定首帧方式
else:
    timesteps = make_native_timesteps(model_t, variant)
    kwargs = {}
    loss_mask = all_valid_latent_mask(batch)

pred = denoiser_forward(
    z_noisy, timesteps, batch.text, batch.robot, **kwargs
)
sq_error = (pred.float() - velocity_target.float()).square()
loss = masked_mean_with_correct_dp_normalization(sq_error, loss_mask)
return WMTrainOutput(loss=loss, metrics={"flow_loss": loss.detach()})
```

`masked_mean` 的分母按参与 loss 的元素数计，包含 latent channel；所有 rank 的有效元素数不同时，
按全局有效元素数归一化并补偿 DDP 梯度平均。首版固定 bucket/条件模式使每个累积窗口计数一致，
不要把不同有效长度的局部均值直接平均。更复杂的可变长度累积需按整个更新窗口统一归一化。

这套 loss 是拟议微调目标，噪声方向、sigma→t、条件 mask、首帧及 CFG dropout 均需数值测试。
不能把模型输出称作 epsilon 后直接对 epsilon 做回归而忽略官方 flow velocity 的方向。

## 5. Trainer 与断点续训

```python
# training/wm_trainer.py —— 共用 Experiment 和 Runtime
from dobot_nexus.core.experiment import Experiment
from dobot_nexus.core.wm_loading import build_world_model
from dobot_nexus.training.runtime import build_runtime

runtime = build_runtime(config)
experiment = Experiment.from_config(config, runtime=runtime)
model = build_world_model(config, mode="train")
loader = model.build_train_dataloader(config)
optimizer = build_optimizer(model.trainable_param_groups())
lr_scheduler = build_lr_scheduler(optimizer, config)
model, optimizer, loader, lr_scheduler = runtime.prepare(
    model, optimizer, loader, lr_scheduler
)
ema = maybe_create_ema(runtime.unwrap_model(model))
state = restore_or_initialize_wm_training_state(
    model, optimizer, lr_scheduler, ema, loader, runtime, config
)
# 恢复每 rank RNG、sampler/bucket 游标、专家计数；恢复后才能拉下一条样本
optimizer.zero_grad(set_to_none=True)
model.train()

while state.optimizer_steps < config.train.max_steps:
    raw = next_resumable_batch(loader, state)
    encoded = runtime.unwrap_model(model).adapter.encode_train_batch(
        raw, encoders=frozen_encoders, rng=state.data_rng
    )
    with runtime.accumulate(model):
        with runtime.autocast():
            out = model(encoded)           # 经分布式 wrapper，不能调用 unwrapped.training_step
        runtime.require_finite(out.loss, label="WM training loss")
        runtime.backward(out.loss)         # 共用 Runtime，只做一次 accumulation scaling
        if not runtime.sync_gradients:
            continue
        grad_norm = runtime.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
        runtime.require_finite(grad_norm, label="WM gradient norm")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if runtime.optimizer_step_was_skipped:
            continue
        lr_scheduler.step()
        update_ema_if_enabled(ema)
        state.optimizer_steps += 1
        state.expert_updates[active_expert] += 1

    log_metrics_on_primary_rank(out, state)
    if should_validate(state):
        with preserve_all_rng_and_module_modes():
            # 所有 rank 遵循相同 collective 顺序；不在 rank0 单独调用 FSDP forward
            distributed_validation_generate(model, fixed_requests, runtime)
    if should_save(state):
        wm_artifacts.save_atomic(
            model, optimizer, lr_scheduler, ema, loader, state, runtime
        )
```

共用 Runtime 的返回类型、pytree、状态导出和 `sync_gradients` 行为要覆盖 WM 契约测试。
`WMTrainOutput` 在现有分布式输出注册机制中兼容；组件恢复语义由 `training/wm_resume.py` 管理。
自动验证优先用单独部署副本/进程，避免模型 offload、EMA 切换和生成状态影响训练。
采用 FSDP 原地验证时，所有 rank 都必须按相同顺序参与，输出由主 rank 落盘。

## 6. 生成与双专家切换

```python
# inference/wm_generator.py
manifest = read_manifest(artifact_dir)
require(manifest.artifact_kind == "world_model")
require_supported_schema(manifest.schema_version)
verify_component_files_and_revisions(manifest)
model = build_world_model_from_manifest(manifest, mode="generate")
request = validate_request(request, spec=model.spec, resource_limits=limits)
# metadata 不允许改写模型 variant、组件身份或任意加载服务器文件
rng = new_request_generator(seed=request.seed)
result = model.generate(request, generator=rng, cancel=cancel_token)

# models/wan22/modeling.py::generate
condition = adapter.encode_request(request, encoders=frozen_encoders)
scheduler = new_scheduler_for_request(manifest.sampling_profile, request.sampling)
latents = random_latents(model.spec, request, generator=rng)
latents = conditioner.apply_initial_constraint(latents, condition)

with inference_mode():
    for step in scheduler.steps():
        cancel.raise_if_requested()
        t = step.model_t                   # 实际 scheduler 时间，不是采样步号
        if variant.is_moe:
            expert_name = (
                "high_noise"
                if t >= manifest.routing.boundary * manifest.routing.num_train_timesteps
                else "low_noise"
            )
        else:
            expert_name = "denoiser"
        dit = components.activate_for_inference(expert_name)
        inputs = conditioner.prepare_denoiser_inputs(latents, condition, t)
        v_cond = dit(**inputs, text=condition.positive_text)
        v_uncond = dit(**inputs, text=condition.negative_text)
        # 图像条件在正/负文本分支中保持一致；action CFG 的语义另行定义
        cfg = named_guidance_scale(expert_name, request.sampling)
        velocity = v_uncond + cfg * (v_cond - v_uncond)
        latents = scheduler.step(velocity, step, latents)
        latents = conditioner.apply_after_step_constraint(latents, condition)
        # 5B I2V 固定已知首帧；A14B native I2V 不套用该覆盖
    video = adapter.decode_video(latents, codec=vae, metadata=request_metadata)

require_video_matches_request(video)
return WorldModelOutput(video.rgb, video.timestamps, request_metadata)
```

A14B 两专家按名称保存 CFG scale，避免 low/high 顺序颠倒。生成中新建 scheduler 和 RNG，
不将上一请求的 step index、历史或视频缓存复用到下一请求。
不要对 5B 无条件应用 I2V mask，也不要将 A14B 图像条件作为真实未来视频编码输入。

## 7. 分组件产物保存

```python
# core/wm_artifacts.py，分布式保存操作由所有 rank 共同参与
staging = runtime.broadcast_new_staging_dir(final_step_dir)
components = export_components_sharded(
    model, directory=staging / "deployment", runtime=runtime,
    # LoRA 导出 adapter + base identity；full 导出相应组件分片
    preserve_component_names=True,
)
write_training_shards_and_rank_state(
    staging / "resume", optimizer, lr_scheduler, ema, sampler, state, runtime
)
runtime.synchronize_save_errors()          # 任一 rank 失败不发布 latest

if runtime.is_main_process:
    verify_components_and_resume_complete(staging)
    manifest = WMManifest(
        artifact_kind="world_model", schema_version=1,
        model_target=config.model.target, variant=config.model.variant,
        spec=model_spec, components=components,
        conditioning=condition_identity, sampling=sampling_identity,
        dataset_fingerprint=data_fingerprint,
        resume_fingerprint=training_fingerprint,
    )
    write_manifest_last(staging, manifest)
    atomic_publish_directory(staging, final_step_dir)
    atomic_replace_latest_pointer(final_step_dir)
runtime.barrier()
```

需要共享的 VAE/text encoder 通过有 checksum 的 reference bundle 解析；portable bundle 导出时
显式带齐。所有发布路径必须位于作业产物目录。损坏/缺失文件不能被标为可用 checkpoint。

## 8. 动作条件扩展草案

```python
# 后续 action_i2v 模型能力；原生 Wan 不声明此能力
require_robot_layout_matches_training(request.robot.layout)
action_tokens = action_encoder(
    actions=request.robot.actions,
    timestamps=request.robot.action_timestamps,
    video_timestamps=planned_frame_times,
    masks=request.robot.valid,
)
# 拟议：独立 attention / residual 分支，时间对齐至 latent token
# 初始化需保证禁用/零输出时复现基础模型；不是将 action 字符串塞入 prompt
velocity = denoiser_with_action_branch(
    noisy_latents, video_time, text_tokens, image_condition, action_tokens
)
```

验证同一当前观测下动作改变是否导致相应未来变化；同时报告无动作与置乱动作对照。
基础 I2V 复现通过后才验证该扩展，避免把条件模块错误与视频模型接入错误混在一起。

# Pi0.5 Dobot RECAP / ACP

RECAP 使用离线 Value 模型计算轨迹的 advantage，再训练带 positive/negative/base 条件的
ACP 策略。Value 使用 Fusion + state 网络，policy 使用 Torch、Nexus BCTrainer 和 Artifact。
配置中的 recipe 为 `dobot_acp_v1`，RECAP schema 为 2。

## 训练配置

以下参数对应 USB 任务的 [BC recipe](../../examples/recipes/pi05_usb_bc.yaml)、
[ACP recipe](../../examples/recipes/pi05_usb_recap.yaml) 和
[Value 配置](../../examples/configs/recap/pi05_offline.yaml)。
新任务可通过 recipe 继承通用模板并覆盖任务字段。

| 配置 | Dobot BC / ACP policy | Fusion + state Value |
|---|---|---|
| seed | 42 | 0 |
| optimizer steps | 20,000 | 20,000 |
| batch | 全局 672；示例每卡 96 × 7 卡，累积 1 | 单卡 56，累积 1 |
| AdamW peak LR | 2.5e-5 | 1e-4 |
| betas / eps | (0.9, 0.95) / 1e-8 | (0.9, 0.999) / 1e-8 |
| weight decay / grad clip | 1e-10 / 1.0 | 1e-4 / 1.0 |
| LR schedule | warmup 500，cosine 总计 20,000，末值 2.5e-6 | warmup 1,000，stable 0，decay 19,000，末值 2.5e-6 |
| warmup 起点 | peak / (500 + 1) | 2.5e-6 |
| EMA | 0.99 | 不使用 |
| worker / prefetch | 每进程 32 / 2 | 4 / PyTorch 默认 2 |
| log / save interval | 100 / 2,000 | 10 / 2,000 |
| checkpoint 保留 | `resume_limit: 2`，仅保留最近两份训练恢复状态，部署导出继续保留 | 保留 step checkpoint 和 last/best |
| 数据与模型 | absolute action，quantile，horizon 50，14 维输入/输出 | state tokens，384 图像，201-bin two-hot，全量微调两个 backbone |
| ACP / validation | 去条件概率 0.3，n-step 50，positive ratio 0.3 | 按 episode 划分，验证集 8 集 |

Policy 使用每卡一个 Torch 进程和 FSDP2。`num_workers: 32` 表示每 rank 32 个 worker，可按主机资源降低，
例如启动时覆盖 `data.num_workers=4`。batch 调整必须满足
`data.batch_size × world_size × train.grad_accum_steps = 672`。

Value 使用 `max_steps` 控制总更新次数，保存 optimizer/scheduler/RNG 以支持续训。
日志在首步、间隔步和末步输出当前指标。

## 数据与资产

输入为单个本地 LeRobot v2.1 数据集，支持三路视频或三路内嵌图像（不混用）。必须有完整
连续的 episode/frame/task 索引，每条 episode 只有一个 task，动作记录实际执行指令。
字段通过配置映射到 base、left_wrist、right_wrist。任务文本、相机身份、ROI、RGB、采样
节拍及动作单位应在训练与部署时保持一致；本模块不执行采集、时间重采样或自动裁剪。

rollout 的成功结果必须显式提供：逐帧布尔列的末帧，或完整的 JSONL：

```jsonl
{"episode_index": 0, "success": true}
{"episode_index": 1, "success": false}
```

`dataset_type=sft` 可声明全成功示范，但不会将所有帧覆盖为 positive。接管字段显式指定为
`is_human` 或数据中实际列名，已知 human 帧默认强制 positive。未记录来源时设置 null，
保留 unknown、不强制 positive；配置的接管列支持 bool/null 混合，null 保留为 unknown，
只有明确的 True 才强制 positive。配置了字段但实际缺失则报错。标签属于 chunk 起始帧，
chunk 可跨接管边界，不按整段投票。所有样本参与 flow-matching loss。

Value 使用本地 `siglip-so400m-patch14-384`、`gemma-3-270m` 和 Gemma tokenizer。

Value 训练支持通过 `torchrun --standalone --nproc-per-node=N -m
dobot_nexus.algorithms.recap.cli train-value --config ...` 使用 DDP；`train.batch_size`
为每卡 batch，有效 batch 为 `batch_size × N × grad_accum_steps`。标注仍使用单设备 CLI。
Value 的 `max_steps` 定义学习率调度长度；可用 `stop_steps` 在较早的更新步保存并结束，
不改变此前的学习率曲线。增加有效 batch 后，应按样本量折算更新次数、warmup、decay
和所选标注 checkpoint；相同样本量不保证不同 batch 的优化结果相同。
`train.cache_files` 默认 2；覆盖到不小于内嵌图像 parquet 文件数时，Value 会在 worker
启动前预读不可变 Arrow 表并共享内存，避免随机取样反复加载整张表。Policy 保持默认缓存。
多卡 checkpoint 保存各 rank 的 RNG/数据迭代状态；恢复要求相同 world size 和训练配置。

依赖延迟加载，固定使用 Transformers 4.53.2，不自动下载权重：

```bash
python3 scripts/env/create_model_env.py --model pi05
.venvs/pi05/bin/python -m pip install -r src/dobot_nexus/algorithms/recap/value/requirements.txt
```

开启 state 文本时，`meta/stats.json` 或 `meta/episodes_stats.jsonl` 必须提供 state min/max。
state 离散为 0..255、补到 32 维（补值 127），使用 Value 自身的预处理。
训练使用的 stats 随 checkpoint 保存，预测恢复该 stats。图像按 Pillow BILINEAR 直接
resize 到 384，映射到 [-1,1]。文本模板保留大小写和标点：
`Task: <task>, State: <tokens>\nValue: `。

## 执行

先用 BC 策略完成采集，再执行 Value/ACP 阶段。对应配置：

- [Dobot USB BC 训练](../../examples/recipes/pi05_usb_bc.yaml)
- [离线 value/标注](../../examples/configs/recap/pi05_offline.yaml)
- [ACP policy 训练](../../examples/recipes/pi05_usb_recap.yaml)
- [policy 推理](../../examples/configs/infer/pi05_recap.yaml)

```bash
# BC 完成后，用其部署权重采集真实执行轨迹；转换入口和图像处理要求见下文。
bash examples/train.sh --config examples/recipes/pi05_usb_bc.yaml --gpus 0,1,2,3,4,5,6
# ACP 复用这次 BC 的统计资产；请填写实际 Artifact 路径。
export RECAP_BC_NORM_STATS=/path/to/bc/artifact/deployment/assets/norm_stats.json
python -m dobot_nexus.algorithms.recap.cli returns --config examples/configs/recap/pi05_offline.yaml
python -m dobot_nexus.algorithms.recap.cli train-value --config examples/configs/recap/pi05_offline.yaml
python -m dobot_nexus.algorithms.recap.cli label --config examples/configs/recap/pi05_offline.yaml --checkpoint /path/to/value_run/step_00010000.pt
bash examples/train.sh --config examples/recipes/pi05_usb_recap.yaml --gpus 0,1,2,3,4,5,6
python -m dobot_nexus.cli.infer --config examples/configs/infer/pi05_recap.yaml --input /path/to/observation.npz
```

上例使用 Value step 10,000 生成标注。也可选择 `best.pt`、`last.pt` 或其他 step，
并记录实际用于标注的版本。
离线 CLI 支持重复 `--override KEY=VALUE`。示例 val_episodes=8，数据集不足时必须调整。
Value 训练支持单设备和 DDP、按 episode 划分验证集、AdamW + warmup-stable-decay，支持梯度累积。
默认全量微调视觉/语言 backbone，backbone BF16、fusion/head FP32，不额外启用 autocast。
可用 `train.resume_from=/path/to/step_....pt` 恢复 optimizer、scheduler、RNG、划分及数据位置；
同一配方的训练参数必须一致。`last.pt`、`best.pt` 是原子更新的 checkpoint 硬链接。

Policy 配置自动选择单卡 local 或多卡 FSDP2，通过 `examples/train.sh --gpus ...` 启动。
上述 USB recipe 默认使用七卡；两卡可使用以下覆盖保持全局 batch 672：

```bash
bash examples/train.sh --config examples/recipes/pi05_usb_recap.yaml --gpus 0,1 \
  -- data.batch_size=4 train.grad_accum_steps=84 data.num_workers=4
```

BC 使用相同覆盖。也可按显存选择其他满足有效 batch 的组合。
单卡将 GPU 列表改为 `--gpus 0`，并按显存与目标全局 batch 调整每卡 batch 和累积次数。
policy 从 Pi0.5 base Torch 权重初始化，20k optimizer steps，warmup=500，EMA=0.99。
USB BC recipe 采用相同设置。
通用 `configs/train/pi05.yaml` 和 `pi05_recap.yaml` 保存无任务绑定的默认值，
USB recipe 通过 `extends` 覆盖 batch、warmup、EMA 和 prompt。

## 训练与数据规则

- Pi0.5 `data.normalization` 可选 `zscore` / `quantile`；Dobot BC 示例和 RECAP 使用 quantile，
  RECAP 拒绝 zscore。state、状态 token、action 监督和动作反变换共用该模式及 q01/q99。
  公式为 `2*(x-q01)/(q99-q01+1e-6)-1`，不裁剪超出分位范围的值。
  模式与统计资产随 Artifact 恢复；Value 的 min/max 和 return 缩放保持独立。
- 训练图像使用本地 Torch 增强：顶部随机 95% crop、±5° rotate，单次双线性采样；
  三视角以 0.5 概率调整亮度、对比度和 hue（范围 ±0.1），饱和度不变。
  两个腕部共享每样本颜色参数，不做几何增强。推理不做训练增强。
- 模型训练 RNG 使用 Trainer 已保存的每 rank CPU/CUDA 状态；image/noise/time 派生独立流。
  actions、noise、Beta time、插值、目标和最终 MSE 保持 FP32；模型算子仍使用配置的精度。
- resolved config/Artifact 记录 `model.training_recipe=openpi_augmax041_fp32_v1`。
  更改训练配方时创建新实验，续训时保持配方一致。
  `data.global_batch_size=672` 对启动后的实际 world size 校验，不自动改累积步数。
- BC 与 RECAP 共用 Pi0.5 的 video/inline-PNG 数据入口。训练会检查 n-step=action_horizon，
  policy 与 labels 的相机/状态/动作/语言映射一致，FPS 覆盖不能冒充数据重采样。

原始 PKL→ROI/OpenCV/PNG LeRobot v2.1 的独立入口位于一级目录
[`data_processing/`](../../data_processing/README.md)，与 BC 等算法共用
`python -m data_processing.convert`。RECAP rollout 设置 `require_success: true`，
示例见 `data_processing/recap_example.yaml`。它保留原始步数和物理时间戳；
`fps: 20` 本身不会改变实测控制频率。
重建数据后，使用新的 returns/Value/labels 目录重新生成标注并训练。

外部机器人程序必须与所训练数据使用相同 ROI 和缩放。USB 示例的图像几何为顶部
`[145,30,400,400]` ROI、腕部全图、OpenCV warpAffine 等比缩放补边到 224，并转换为 RGB；
policy 不额外裁剪或增强。

## 算法

每个任务的 `T=max(episode length)`，`C=c_fail_coef*T`（默认系数 1）：

```text
target[t] = clip((-(L-1-t) - failure*C) / (T+C), -1, 0)
r[t] = target[t] - target[t+1]；末帧 r=target
A[t] = sum(r[t:t+n]) + V[t+n] - V[t]；终止后 V=0
threshold[task] = quantile(A[task], 0.7)
positive[t] = A[t] >= threshold[task]，然后覆盖已知 human 帧为 True
```

n 默认 50；阈值平局、接管覆盖可能使正比例超过 30%。全成功不等于全 positive。
Return sidecar 保存归一化 reward 和 return；value 直接以 return 为训练目标。
标注目录记录 task_max_lengths、c_fail_coef、各任务 thresholds、接管来源与文件校验和。
完成阶段禁止覆盖；换配置使用新目录。

训练对全部样本以 0.3 概率去除条件，其余根据标签添加 `Advantage: positive` 或
`Advantage: negative`。连续 advantage/value 不作为 policy loss 权重。
推理默认 base；正条件评估显式设置
`inference.recap.condition=positive`，亦支持 negative。客户端只发原 task，adapter 注入一次。
保持 policy 的严格 prompt 长度检查，超长报错；不支持任意 CFG scale。

Torch 推理使用与 BC 相同的 `torch_core` 和 `openpi_torch` 引擎；对 legacy 数值策略的
Artifact，可设置 `model.inference_backend=torch`、`runtime.compile=true`、
`runtime.compile_mode=legacy`、`runtime.cuda_graph=true`。切换 base/positive/negative
会刷新 token 输入，保持相同形状时复用 CUDA Graph。CPU 推理需关闭 compile 和 cuda_graph。

## 产物

Policy resume 保存当前训练权重和每 rank EMA shard；启用 EMA 时部署输出 Torch EMA 权重。
Artifact metadata 记录 recipe、条件、标注来源和 EMA，policy 部署不依赖 value 或原始数据。
EMA 恢复要求与已有 Nexus resume 一致的 world_size 和参数分片布局。

## 代码边界与验证

Value 网络、配置、RGB/state 预处理和分布损失位于公共层
[`modules/value/`](../../src/dobot_nexus/modules/value/README.md)，不依赖 Pi0.5、RECAP 或数据集。
算法、Value 训练/标注调度和 checkpoint 语义校验位于 `algorithms/recap/`，
通过公共 `data/lerobot/source.py` 与 `data/images.py` 读取和转换数据，不依赖具体 VLA 模型。
Pi0.5 条件策略由 `models/pi05/modeling_recap.py`、`data_adapter_recap.py`、
`configuration_recap.py` 和 `data/recap.py` 接入；归一化、图像增强、tokenizer 与动作编码
由 Pi0.5 数据管线负责。policy 使用标准 BCTrainer、InferencePolicy 和 Artifact。
公共 EMA 支持 BC 与 ACP，仅在成功的 optimizer update 后更新；
保存时先保存原始训练状态，再导出 EMA 权重，最终恢复训练参数。

训练和推理使用 `dobot_nexus.models.pi05.modeling_recap:Pi05ReCapModel`。
加载 Artifact 时也接受 `dobot_nexus.models.pi05.recap.modeling_recap:Pi05ReCapModel`，
manifest 与保存配置中的 target 必须一致。

`Pi05ReCapModel` 通过基类 `_build_adapter()` hook 一次构造 `ReCapDataAdapter`。
训练路由仅替换 batch 的 observation，保留 `model_kwargs` 等其余字段；训练与推理都由基类
调用模型主体。Value/return/advantage 标注链路不依赖 `torch_core` 的包加载或编译方式。

Value 复用主干 `training.resume` 的随机状态捕获、恢复和临时保护；checkpoint 模块只负责
将 NumPy 状态编码为可由 `weights_only=True` 加载的列表，使用 checkpoint schema 2。
训练与标注共用 annotations 中的 return 配方校验和 Value configuration 中的设备校验；
标注入口不依赖 Value 训练循环。Value 的 episode 划分、设备/DDP 和调度由算法目录负责。

sidecar 通过 dataset/episode/frame 键及摘要校验来源，源数据保持不可变。
Value checkpoint 包含网络、processor/min-max、return 尺度、优化器、调度、划分和 RNG；
policy Artifact 包含统计资产、训练配方、条件和标注来源，部署不加载 Value。

```bash
PYTHONPATH=src:. python -m unittest discover -s tests -v
python -m compileall -q src tests examples data_processing
```

正式测试覆盖 return/advantage、条件、数据入口、归一化、随机状态、Value/Policy 恢复和 EMA。
公共 Value 的网络、损失、预处理和依赖边界测试位于 `tests/modules/value/`；
RECAP 的算法、Value 续训、标注与依赖隔离测试位于 `tests/algorithms/recap/`；
Pi0.5 条件策略与配置恢复测试位于 `tests/models/pi05/test_recap.py` 和 `test_recap_config.py`。
Pi0.5 的归一化、配方、增强/flow 分别归入 `test_data_adapter.py`、`test_model.py`、
`test_training.py`；EMA 归入公共 checkpoint 测试，scheduler 归入公共训练测试。
GPU、分布式和性能实验及报告放在 Git 忽略的 `local_script/`。
完整模型训练、推理精度与真机成功率需要单独验证。

## TensorBoard 训练监控

策略与独立 Value 模板均默认开启 TensorBoard。Value 事件位于
`<train.output_dir>/tensorboard/`，安装 Value requirements 时会安装依赖。启动：

```bash
python -m dobot_nexus.cli.main tensorboard --logdir /path/to/value_run --port 6006
```

Scalars 包含 `value_loss`、`expected_value_mae`、`lr`、`grad_norm`、`step_seconds`、
`data_seconds`，启用 episode 验证时还包含 `val_loss`、`val_mae`。
只有 rank 0 写入，横轴是 optimizer step；续训复用目录并清理未保存的旧步数。
可用 `--override tracking.tensorboard.enabled=false` 关闭记录。

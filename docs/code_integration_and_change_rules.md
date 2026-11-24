# Dobot Nexus 代码接入与修改规则

本文是面向开发者和 AI Agent 的可执行规范。背景设计见
[`nexus_architecture_and_integration_guide.md`](nexus_architecture_and_integration_guide.md)；发生冲突时，
以仓库根目录 `AGENTS.md`、稳定公共接口和本文中的“必须/禁止”规则为准。

## 1. 首要目标

接入完成不等于“模型脚本可以运行”。一个模型或算法只有同时满足以下条件才算完成：

1. 能由统一 YAML 和 `model.target` 构造。
2. 能通过标准 Trainer 完成最小训练。
3. 能生成包含配置、权重和数据资产的 Artifact。
4. 能由标准 `InferencePolicy` 从 Artifact 独立恢复。
5. 最终动作严格为 `float32[B,H,A]`。
6. 不污染其他模型环境，不让模型私有依赖进入稳定公共层。

## 2. 先判断改动属于哪一层

| 变化 | 修改位置 | 默认是否修改公共层 |
|---|---|---|
| 新模型架构 | `src/dobot_nexus/models/<model>/` | 否 |
| 同一模型的新 backend | 模型自己的 backend 子目录 | 否 |
| tokenizer、action codec、数据映射 | 模型自己的 adapter/pipeline/assets | 否 |
| 单次 forward/loss 变化 | 新 `model.target` 或 `training_step()` | 否 |
| 独立算法流程与离线/在线作业 | `src/dobot_nexus/algorithms/<algorithm>/` | 否（仅在多优化器/在线 RL 需要时扩展 Trainer 路由） |
| 已被两个以上模型稳定复用的模型无关组件 | `src/dobot_nexus/modules/<component>/` | 仅提取模型无关原语，不引入模型依赖 |
| 设备、FSDP、mixed precision | Runtime | 仅增加模型无关能力 |
| 保存、恢复、部署格式 | Artifact hook 和模型私有导出器 | 仅增加模型无关能力 |

判断口诀：模型解决“怎么算”，Trainer 解决“按什么流程算和更新”，Runtime 解决“在哪里算”，
Artifact 解决“如何恢复和部署”。

## 3. 稳定公共边界

以下位置是稳定契约：

- `src/dobot_nexus/core/`
- `src/dobot_nexus/training/`
- `src/dobot_nexus/inference/`
- `src/dobot_nexus/models/base.py`

修改这些位置前，必须证明能力至少被两个模型以相同语义需要，并增加 CPU 合约测试。禁止为了单个
模型、单个 checkpoint 或单个实验扩张公共接口。

公共数据协议固定为：

| 字段 | 类型和形状 |
|---|---|
| `images` | `dict[str, uint8[B,H,W,3]]`，BHWC RGB |
| `state` | `float32[B,D]` |
| `prompt` | 每个样本一个非空字符串 |
| `actions` | `float32[B,H,A]` |

BCHW、BF16、32 维 padding、token、codebook 和 backend 私有 Observation 必须在模型 adapter
边界内转换，不能泄漏到公共协议。

## 4. 模型必须拥有的代码

推荐结构如下；没有对应能力时可以省略子目录：

```text
src/dobot_nexus/models/<model>/
├── __init__.py
├── configuration_<model>.py
├── modeling_<model>.py
├── data_adapter_<model>.py
├── requirements.txt
├── assets/
├── data/pipeline.py
├── training/checkpoint.py
├── training/compile.py          # 如需训练小算子编译，按模型实例配置
├── torch_core/                 # 训推共用的架构、算子和数值策略
└── openpi_torch/                # 模型私有推理编译/CUDA Graph
```

职责必须保持单一：

- Config：收敛用户字段、保存模型常量、在构造阶段完成静态校验。
- Adapter/Pipeline：字段映射、图像变换、normalization、tokenization、padding、动作逆变换。
- Modeling：模型构造、单步 loss、完整 action chunk 采样和 Artifact hooks。
- Checkpoint：查找文件、抽取 state dict、前缀归一化、严格加载和格式转换。
- Torch core：用普通 Python 包组织训推共用算子；导入时不编译，不引入训练平台 runtime。
- Compile/Inference：分别负责训练和推理的执行加速，不复制模型计算实现。
- Pi0.5 固定使用本地 Torch core，定制模型通过 `model.target` 接入。Pi0 使用独立的 vendor 快照。

模型必须实现 `BaseVLAModule` 的五个接口：

```python
from_config(config, *, mode)
build_train_dataloader(config)
training_step(batch) -> TrainOutput
predict_action_chunk(observation, options) -> PolicyOutput
policy_spec -> PolicySpec
```

模型 adapter 必须实现：

```python
encode_train_batch(raw_batch, device)
encode_observation(observation)
decode_actions(actions, *, aux=None) -> PolicyOutput
```

## 5. 复用、复制与依赖隔离

可以直接复用：

- 公共 contracts、LeRobot reader、`data/` 下的图像 `resize_with_pad`、action transform 与统计/归一化数学（`zscore`/`quantile`/整数安全 `delta`）。
- BCTrainer、Runtime、ArtifactManager、InferencePolicy、PolicySession。
- `inference/transport/` 的 WebSocket/NumPy wire 协议与 `worker.py` 的串行执行。
- 公共 `modules/` 下的模型无关组件：`modules/value/`、`modules/memory/`、`modules/serving/`。
- 语义完全一致的同模型 pipeline。

应当复制到模型目录：

- 模型专属或版本敏感的算子，保留来源与许可证。
- 不同模型可能独立演进的 source/checkpoint loader。
- 外部项目中只为该模型服务的轻量兼容代码。

禁止：

- 一个模型 import 另一个模型的 `torch_core/`、`vendor/`、`integration.py`、adapter、pipeline 或私有 backend。
- 创建共享 `models/pi/` 来合并 Pi0 和 Pi0.5 的相似实现。
- 在公共层或 `modules/` 中 import 具体模型或算法作业编排器。
- 在通用 `inference/transport/` 中解释模型观测、相机布局或 Memory 状态；模型无关协议适配归 `modules/serving/`，Artifact/相机列解析与冻结视觉编码器装配归模型的 `deploy/`。
- 将 Ray、RLinf worker、scheduler、registry 或 policy wrapper 放入 vendor 算子快照。
- 通过环境变量偷偷发现必需依赖；自定义路径必须由配置显式传入并校验。

代码重复在模型隔离边界上是允许的。只有两个实现已经稳定、测试证明语义相同，才允许提取真正
通用且不含模型身份的能力。

## 6. 配置规则

- 模型配置优先使用 `@dataclass(frozen=True)`。
- 用户 YAML 只暴露用户需要理解和调整的字段。
- 固定网络结构、内部 padding 维度和算子常量放在模型 Config 中。
- 同一语义只能有一个权威字段；兼容字段只在入口转换一次。
- checkpoint、维度、horizon、camera、精度和 backend 名称必须在构造阶段校验。
- 训练和推理必须使用同一个 `model.target`。
- 新增必填字段或改变字段语义时，必须升级 schema 或给出明确迁移说明。
- 部署不得依赖训练机器的绝对路径；Artifact 内使用相对路径。

## 7. 数据和动作语义

- 训练和推理必须共用同一个 pipeline，禁止分别实现 normalization/tokenizer/action decode。
- state/action 在进入模型前必须检查 batch、horizon 和 dimension。
- 只有协议明确允许时才能 padding/truncate；否则必须报错。
- normalization 统计必须在最终 action transform 之后计算。
- action encode/decode 必须具有 round-trip 测试。
- episode 尾部、token padding 和无效 step 必须携带明确 mask。
- `predict_action_chunk()` 始终返回完整 chunk；部分执行由 `PolicySession` 或环境控制。

## 8. 训练、精度和随机性

- `training_step()` 只计算模型 forward 和 loss，不得调用 optimizer、scheduler 或保存 checkpoint。
- `TrainOutput.loss` 必须可反向传播；metrics Tensor 必须 detach。
- Trainer 不得解析模型私有 batch 字段。
- persistent parameter dtype、operator compute dtype 和 Runtime mixed precision 必须分别声明并测试。
- 推理随机数必须使用 backend-local Generator。
- `SamplingOptions` 按调用传入，禁止临时修改模型成员或全局 RNG。
- noise、采样步数和输出形状必须在模型边界显式校验。

## 9. Checkpoint 和 Artifact

- 正式 checkpoint 必须使用 `strict=True` 加载。
- DDP/FSDP/compile/wrapper 前缀必须集中在 checkpoint 模块归一化。
- Artifact 的 PyTorch 权重优先保存裸 backend keys，不依赖 Nexus wrapper 类名。
- tokenizer、norm stats 和 codebook 必须作为 Artifact 资产携带并重定向 resolved config。
- `deployment/` 只包含推理必需内容；optimizer、scheduler、RNG 和 cursor 只进入 `resume/`。
- Pi0.5 训练和部署使用 Torch；Artifact 直接保存训练参数和模型专属资源。
- 模型格式差异通过 `export_backend_state_dict()`、`artifact_assets()`、
  `export_deployment_artifacts()` 等 hook 处理；ArtifactManager 中禁止出现模型名称分支。

## 10. Python、依赖和错误风格

- 使用 `from __future__ import annotations`。
- 公共返回值使用有类型的 dataclass，不返回含义不清的多层 tuple。
- 函数参数和返回值应写类型；不可控 backend 类型只在局部使用 `Any`。
- Ruff 行宽为 100，并遵循仓库 import 顺序。
- 重型或可选依赖必须 lazy import，`ImportError` 必须给出具体安装方式。
- 每个模型维护独立 `requirements.txt`，互不兼容时使用 `.venvs/<model>`。
- vendored 源码必须保留 LICENSE、NOTICE、来源 revision，并只使用相对 import。
- 错误信息必须包含字段名、期望值和实际值；禁止静默吞错或截断错误维度。

## 11. AI Agent 执行协议

Agent 修改代码时必须按以下顺序执行：

1. 阅读根目录 `AGENTS.md`、目标模块和对应测试。
2. 用搜索确认调用方、配置、文档和 Artifact 路径，不能只修改单个文件。
3. 先声明修改边界和不修改项；未经要求不扩大到公共协议。
4. 保留工作区中与任务无关的用户改动。
5. 模型接入优先在 `models/<model>/` 内闭环；发现跨模型 import 时停止并改为独立实现。
6. 同时更新示例 YAML、requirements、package data、环境检查和用户文档。
7. 先写 CPU fake/minimal-core 测试，再进行真实权重和 GPU 验证。
8. 完成后检查依赖边界和引用路径，并报告未能执行的外部验证。

Agent 禁止自行执行破坏性 Git 操作、覆盖无关改动、修改第三方 submodule 或下载未授权权重。

## 12. 测试与 PR Checklist

每个新模型或重大重写至少覆盖：

- [ ] Config 成功和失败用例。
- [ ] `model.target` 动态加载。
- [ ] camera/state/action dtype 和 shape 校验。
- [ ] Adapter 训练编码和 action round-trip。
- [ ] `training_step()` 与一个 CPU optimizer step。
- [ ] checkpoint 前缀归一化和严格加载。
- [ ] Artifact save → load → inference 闭环。
- [ ] 输出严格为 `float32[B,H,A]`。
- [ ] tokenizer/norm stats/codebook 随 Artifact 恢复。
- [ ] 模型私有依赖未进入公共层，也未跨模型 import。
- [ ] requirements、YAML、package data 和文档已更新。

基础门禁：

```bash
PYTHONPATH=src:. python -m unittest discover -s tests -v
python -m compileall -q src tests examples
ruff check src tests examples
```

`tests/` 仅保留模块与 CPU 契约测试。真实 checkpoint、GPU、分布式 smoke、FSDP/参考输出
parity、性能实验及其辅助测试统一放在 Git 忽略的 `local_script/`；正式测试不得依赖本地脚本。
`docs/` 保留架构、接入与使用说明，本地实验报告和操作记录随调试脚本保存。
测试职责与命令入口归类见 [测试指南](../tests/README.md)；安装和服务 extras 见
[环境指南](environments.md)。

# 公共 Value 组件

本目录提供独立的 PyTorch Value 网络与输入处理，不依赖 policy 模型、RECAP、
轨迹键、LeRobot 或训练调度。当前实现为 Dobot Fusion（SigLIP + Gemma3 + 查询交叉注意力）。
Value 网络是普通 `torch.nn.Module`，不增加框架模型协议。

```text
modules/value/
├── distributions.py       logits 到期望 Value
├── losses.py              相邻分桶 two-hot 交叉熵
└── fusion/
    ├── __init__.py        延迟构造与依赖检查
    ├── configuration.py   ValueModelConfig 与资产校验
    ├── modeling_fusion.py SigLIP/Gemma3 融合网络
    ├── processing.py      RGB、prompt、state tokens
    ├── requirements.txt   可选依赖
    └── NOTICE.md          来源说明

algorithms/recap/value/
├── dataset.py             公共 LeRobot reader 到 Value 样本
├── processing.py          FrameKey、return targets、设备搬运与结果包装
├── training.py            episode 划分、优化器、调度与训练循环
├── distributed.py         RECAP Value 的 DDP 运行时
└── checkpoint.py          return/data/processor 校验与续训
```

算法或模型适配器负责提供相机顺序、RGB 图像、任务文本、state 和 min/max 统计：

```python
from pathlib import Path

from dobot_nexus.modules.value.distributions import expected_value
from dobot_nexus.modules.value.fusion import build_value_model
from dobot_nexus.modules.value.fusion.configuration import ValueModelConfig
from dobot_nexus.modules.value.fusion.processing import FusionProcessor
from dobot_nexus.modules.value.losses import categorical_loss

config = ValueModelConfig(
    siglip_path=Path("assets/siglip-so400m-patch14-384"),
    gemma3_path=Path("assets/gemma-3-270m"),
    tokenizer_path=Path("assets/gemma-3-270m"),
)
model = build_value_model(config)
processor = FusionProcessor(
    config,
    camera_names=("front", "wrist"),
    state_stats=state_stats,  # {"min": [...], "max": [...]}
)
observation = processor.encode_observation(
    images=images,   # 每个样本一个 {相机名: HWC uint8 RGB} 字典
    prompts=tasks,   # 每个样本一条任务文本
    states=states,   # 每个样本一个 state 向量
)
logits = model(observation)
values = expected_value(logits)
loss, metrics = categorical_loss(logits, normalized_returns)
```

输入和网络需放在同一设备；设备选择、优化器、归一化 return 和标签规则由调用方管理。
相机顺序由调用方显式传入。RECAP 适配器固定使用 `base/left_wrist/right_wrist`，
并把顺序、state 统计和数据来源记录在自己的 checkpoint 中。

Fusion 使用 201 个 `[-1, 0]` 分桶、384 图像尺寸、200 token 上限、32 维 state 文本
和缺省填充值 127。state 超出有效统计长度或 32 维时截断。

只使用公共 Value 时，可从仓库根目录安装：

```bash
pip install -r src/dobot_nexus/modules/value/fusion/requirements.txt
```

RECAP 作业安装 `src/dobot_nexus/algorithms/recap/value/requirements.txt`，
其中包含上述依赖及 RECAP 数据读取依赖。Transformers 固定为 `4.53.2`，
模型和 tokenizer 仅从本地资产加载。Pi0.5 与 Pi0 均已完成 RECAP policy 接入；
其他新模型同样可独立调用本组件并按需完成条件化数据适配。

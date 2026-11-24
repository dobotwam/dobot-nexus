# 公共 FrameSample Memory 组件

本目录提供模型无关的 FrameSample 历史记忆原语，供 `models/pi05/memory/`、`models/pi0/memory/`
及后续需要长时历史条件化的 VLA 模型共同复用。本目录不导入 `models/`、`algorithms/` 或具体数据集读取器。

```text
modules/memory/
├── types.py           MemoryConfig、MemoryReference、MemorySnapshot、MemoryObservationBatch
├── position.py        相对时间步正弦位置编码（768 维）
├── sampling.py        历史帧索引选择与 MemorySnapshot 组装
├── torch_layers.py    MemoryTokenEncoder、MemoryAttention、MemoryModulation（纯 torch.nn.Module）
├── runtime.py         HistoryBuffer、MemorySession（按环境隔离的稠密历史与动作块会话）
└── feature_store.py   FrameFeatureStore（原子发布、SHA-256 流式校验、分块认证与每 worker 缓存预算）
```

## 职责边界

- **共享层负责**：FrameSample v1 布局校验、时间位置编码、均匀/滑窗历史采样、交叉注意力与调制算子、
  推理期单环境历史缓冲及多环境会话隔离、离线视觉特征库的原子写入与完整性读取。
- **模型侧负责**：冻结视觉编码器构建与前向提取（如 `models/<model>/memory/torch_vision.py`）、
  特征库预处理 CLI（`prepare.py`）、训练数据管线与列溯源校验（`data.py`、`feature_store.py`）、
  以及将 `MemoryTokenEncoder` / `MemoryModulation` 挂载到具体 backbone（`torch.py`、`modeling_memory.py`）。

## 典型用法

```python
from dobot_nexus.modules.memory.runtime import HistoryBuffer, MemorySession
from dobot_nexus.modules.memory.torch_layers import MemoryModulation, MemoryTokenEncoder
from dobot_nexus.modules.memory.types import MemoryConfig

config = MemoryConfig(budget=512, camera="base")
encoder_module = MemoryTokenEncoder()
modulation = MemoryModulation(rope=apply_rope_fn)
buffer = HistoryBuffer(config)
```

`MemoryAttention` 通过构造函数接收调用方传入的 `rope` 回调，并在推理时通过可选 `inference_ops`
参数接入模型专属加速算子，避免反向依赖任何具体模型实现。

`FrameFeatureStore` 默认限制每个 worker 的私有缓存预算为 512 MiB（可通过训练配置
`data.memory_cache_bytes` 调整）；超出预算的长 episode 采用流式全文件 SHA-256 校验加按帧分块认证，
避免整文件落盘暂存和重复全量读取。

# 公共 Serving 协议适配器

本目录提供模型无关的 OpenPI 与 Memory WebSocket 协议适配器，构建在公共 `inference/transport/`
传输层和 `InferencePolicy` 契约之上。任何实现 `BaseVLAModule`（提供标准 `PolicySpec`）的模型均可直接复用，
本目录不导入 `models/` 或 `algorithms/`。

```text
modules/serving/
├── openpi.py          OPENPI_TO_NEXUS_IMAGES、camera_mapping 校验、观测转换、NexusOpenPiPolicy、metadata 构造
└── memory_ws.py       MemoryConnection、MemoryWebsocketServer、configured_memory_limits
```

## 分层关系

```text
inference/transport/   纯传输层（WebSocket 帧、msgpack_numpy、连接限额、健康检查；不认识相机或模型）
modules/serving/       协议适配层（OpenPI 观测/动作转换、camera_mapping 校验、Memory 会话与 history_wire_camera）
models/<model>/deploy/ 模型装配层（从 Artifact 解析配置、相机列映射、冻结视觉 encoder 与推理加速参数）
```

## 关键契约

- **统一相机映射校验**：`NexusOpenPiPolicy`、`build_openpi_metadata`、`MemoryConnection` 与
  `MemoryWebsocketServer` 共用 `validated_camera_mapping(..., camera_names=policy.spec.camera_names)`，
  确保对外广播的 `input_schema` 与实际推理路由严格一致。
- **Memory 历史相机路由校验**：`MemoryWebsocketServer` 在握手 metadata 中广播 `history_camera` 与
  `history_wire_camera`；当历史相机经过重映射或改名时，`add_buffer` 请求必须在编码前声明匹配的
  `history_wire_camera`，防止将错误视角写入历史缓存。
- **每连接独立状态**：`MemoryWebsocketServer.open_connection()` 为每个客户端连接创建独立的
  `MemoryConnection` / `MemorySession` 并共享底层模型与视觉 encoder；预热在独立临时连接中执行，
  并在连接关闭时仅清理当前连接的历史缓冲。

## 典型用法

```python
from dobot_nexus.modules.serving.openpi import NexusOpenPiPolicy, build_openpi_metadata

adapter = NexusOpenPiPolicy(
    policy,
    num_steps=10,
    seed=42,
    camera_mapping={"top_camera": "base", "left_camera": "left_wrist", "right_camera": "right_wrist"},
)
metadata = build_openpi_metadata(policy, camera_mapping=adapter.camera_mapping)
```

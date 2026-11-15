# Dobot 通用轨迹转换

第一次使用请先阅读[LeRobot v2.1 格式与转换步骤](../docs/getting_started/02_data.md)，
其中包含 PKL 必需字段、独立环境创建、输出目录说明和数据读取检查。
本文补充标签、扩展字段和转换语义。

将 Dobot recorder 的 PKL 转成 BC、RECAP 和后续 RLT 可复用的 LeRobot v2.1 数据。
转换只负责数据格式、图像处理和已记录标签的保留；算法所需的 return、advantage、
训练样本和损失由下游负责。转换工具使用独立的数据处理环境。

从仓库根目录运行：

```bash
pip install -r data_processing/requirements.txt
python -m data_processing.convert \
  --config data_processing/example.yaml --output /path/to/new_dataset
```

先复制 [通用配置](example.yaml) 并填写来源、任务和部署使用的图像几何。
每个 `sources` 条目选一个受信 PKL 文件或包含 PKL 的目录；
顺序由配置条目及文件名决定。输入是 recorder 的逐步记录列表，左右各 7 维，state 为观测
关节/夹爪，action 为该观测后实际下发的目标，不做平滑、截断或动作错位。

目前支持 Dobot 双臂三相机的采集格式；复用范围是训练算法，并非任意机器人的输入格式。

## BC、RECAP 和 RLT 的使用方式

| 用途 | 配置 | 行为 |
| --- | --- | --- |
| BC | `require_success: false`（通用入口默认值） | 允许没有成败标签，未知结果写为 null |
| RECAP rollout | `require_success: true` | 缺少结果时拒绝转换；下游读取 `is_success`、`is_human` |
| RLT / 其他算法 | 按下游要求配置 `extra_fields` | 保留采集时记录的 reward、终止标记、策略版本等标量 |

RECAP 的 USB 图像几何和严格标签示例见 [recap_example.yaml](recap_example.yaml)，
也使用同一个通用入口：

```bash
python -m data_processing.convert \
  --config data_processing/recap_example.yaml --output /path/to/recap_dataset
```

BC 转出的未知结果不能作为 RECAP rollout 的成败标签，使用前需要补充经确认的结果。

成败读取轨迹中的 `episode_success`（布尔值或 `success`/`failure`），也可通过来源的
`success: true/false` 补充。所有已知结果必须一致；未知不会被当作失败或成功。
`is_human` 保留逐帧标记，缺失保留 null，不当作自主，也不用于推断成功。
来源配置与原记录标签冲突时拒绝转换。停止/退出导致结果未知的轨迹，在要求结果的
下游任务中必须人工补充。

离线 RECAP 可设置 `intervention_field: is_human` 读取 bool/null 混合来源，保留 unknown；
整列来源不可用时设置 `intervention_field: null`。

## 扩展采集字段

`extra_fields` 将逐帧 PKL 键映射到输出 Parquet 列，例如：

```yaml
extra_fields:
  reward: {source: reward, dtype: float32, required: true}
  terminated: {source: terminated, dtype: bool, required: true}
  truncated: {source: truncated, dtype: bool, required: true}
  behavior_log_prob: {source: log_prob, dtype: float32}
  policy_version: {source: policy_version, dtype: string}
  chunk_step: {source: chunk_step, dtype: int64}
```

输出列名是映射的 key；`source` 默认同名。支持 `bool`、`float32`、`float64`、`int64`、
`string` 标量；数值按声明类型存储，浮点转换可能发生相应精度的舍入。
`required` 默认 false，缺失或 null 保留为 null；设为 true 则缺失时失败。
非法类型、非有限数、越界数值及与基础列冲突的列名都会被拒绝。
未知数量写入 `meta/sources.jsonl`，字段配置和严格标签要求写入 `meta/conversion.json`。
未列入配置的其他原始字段不会自动写出；额外字段也不改变 state/action 的统计量。

这些字段必须确实存在于采集记录中。工具不会将成功映射为 reward，也不会把最后一帧
自动当成环境 terminated；truncated 与 terminated 分开保留。
目前提供的是 RLT 所需数据的保留能力，具体 RLT 数据适配器仍需定义 reward 的时间对齐、
next observation、bootstrap 和终止语义。当前 Dobot recorder 不保证记录上述全部字段。

## 图像、时间与统计资产

通用示例显式设置 `top_roi_xywh: null`，使用全图；RECAP USB 示例使用顶部
`[145,30,400,400]` ROI。省略该键时默认采用此 USB ROI；请按实际相机配置显式填写。
腕部使用全图，OpenCV warpAffine 等比缩放补黑边至 224，BGR→RGB，内嵌 PNG。
输出是一个新的 LeRobot v2.1 数据集，已有目录拒绝覆盖；全量转换和 Parquet 回读成功后才发布。
ROI 超边界、空轨迹、维度/时间/标签错误会终止。任务文本必须与实际训练/部署一致。

`fps: 20` 是输出索引时钟，**保留记录步数，不进行时间重采样**。
`source_timestamp`、`source_frame_index` 和 `meta/sources.jsonl` 保存原始时间、帧号、文件 SHA256、
间隔统计和 unknown 数量。10 Hz 原始数据不会因此成为 20 Hz 控制数据；采集/执行频率需实测对齐。
更改步数/频率会改变 return 和 n-step 的物理含义，需要重建数据、returns、Value 和 labels。

`meta/stats.json` 提供 Value 所需 state min/max；它不是 policy 的 q01/q99 资产。
BC pipeline 在最终 action transform 后计算 policy 统计量；ACP 显式复用 BC Artifact 的
`deployment/assets/norm_stats.json`。推理应使用所选权重自身的统计资产。

采集与真机部署由外部机器人程序负责。转换输入应为三路原始 JPEG；已裁剪的图像不能再次
套用原图 ROI。部署需使用与转换相同的 ROI、OpenCV warpAffine 等比缩放补边和 BGR→RGB；
请核对外部程序的实际处理逻辑，
不能只凭 ROI 相同认定像素处理一致。policy 推理不额外裁剪或应用训练增强。

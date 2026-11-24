# 4. 加载 Artifact 做离线推理

[新手路线](../../README.md#从哪里开始) · 上一步 [训练](03_training.md) · 下一步 [真机部署](05_deployment.md)

本篇在 GPU 主机执行，从真实数据中抽取一帧观测，加载模型生成动作并保存到 `.npz`。
10 步 smoke 产物可以检查推理链路，部署模型应根据正式训练后的任务评测选择。

## 4.1 选择训练产物

```bash
source .venvs/pi05/bin/activate
export PI05_ARTIFACT=runs/pi05_my_task_smoke/checkpoints/latest_deployment.txt
export VLA_DATASET_PATH=/path/to/datasets/my_task_v21
test -e "$PI05_ARTIFACT"
```

完成正式训练后，可把 `PI05_ARTIFACT` 改成
`runs/pi05_my_task_train/checkpoints/latest_deployment.txt`。
也可以指向具体 `step_XXXXXXXX` 目录，固定使用某次保存的模型。

跨机器部署时复制完整 `step_XXXXXXXX` Artifact，保留 `manifest.yaml`、`config_resolved.yaml`
和 `deployment/` 中所有资源。`latest_deployment.txt` 中保存的是路径，移动时应连同目标目录一起处理，
或直接使用复制后的 Artifact 目录。权重和统计、tokenizer 必须来自同一个 Artifact。

## 4.2 从数据集生成一份真实输入

公共输入使用 RGB 图像、原始状态和任务文本。下面按 Artifact 保存的字段映射读取数据集首帧，
生成 `runs/quickstart/observation.npz`。数据集只用于本步骤制作输入。

```bash
python - <<'PY'
import os
from pathlib import Path
import numpy as np
from PIL import Image
from dobot_nexus.core.artifacts import resolve_artifact
from dobot_nexus.core.config import load_config
from dobot_nexus.data.images import convert_image
from dobot_nexus.data.lerobot.source import build_source

artifact = resolve_artifact(os.environ["PI05_ARTIFACT"])
config = load_config(artifact / "config_resolved.yaml", expected_task="train")
mapping = config.data.field_mapping
source = build_source(
    data_path=os.environ["VLA_DATASET_PATH"],
    action_key=mapping.action,
    language_key=mapping.get("language"),
    action_horizon=1,
    format_version_name=config.data.get("format_version", "auto"),
    fps=config.data.get("fps"),
    tolerance_s=config.data.get("tolerance_s", 1e-4),
)
frame = source[0]
output = Path("runs/quickstart")
output.mkdir(parents=True, exist_ok=True)
payload = {}
for name, column in mapping.images.items():
    image = convert_image(frame[column])
    payload[f"image.{name}"] = image
    Image.fromarray(image).save(output / f"{name}.png")
state = np.asarray(frame[mapping.state], dtype=np.float32)
assert state.shape == (config.model.state_dim,)
assert np.isfinite(state).all()
prompt = frame.get("prompt") or config.data.get("default_prompt", "")
assert isinstance(prompt, str) and prompt.strip()
payload["state"] = state
payload["prompt"] = np.asarray([prompt])
np.savez_compressed(output / "observation.npz", **payload)
print("输入已保存", output / "observation.npz")
print("任务", prompt)
print("状态", state)
PY
```

打开生成的 `base.png`、`left_wrist.png`、`right_wrist.png`，确认画面、左右相机、颜色和裁剪正确。
自定义任务的列名来自 Artifact，公共输入键如下。

| `.npz` 键 | 单样本形状 |
|---|---|
| `image.base` / `image.left_wrist` / `image.right_wrist` | `uint8[H,W,3]` RGB |
| `state` | `float32[state_dim]`，与采集协议相同的单位 |
| `prompt` | 一个非空字符串组成的数组 |

adapter 会执行训练时对应的归一化和动作逆变换，输入应保留真实状态数值。

Pi0/Pi0.5 模板默认使用 BF16 参数和 BF16 计算。训练保存的 FP32 参数会在加载时按推理配置转换；
`model.inference_parameter_precision` 不会从 Artifact 中的训练配置自动继承。
如需保留 FP32 参数，在下面推理命令的 `--` 后追加
`model.inference_parameter_precision=fp32`。计算仍为 BF16；全 FP32 推理还需追加
`model.precision=fp32`。完整字段、默认值及适用范围见[精度参数表](../precision.md)。

## 4.3 在一张 GPU 上运行

首次检查使用 eager 推理，便于观察模型加载和数据处理。

```bash
bash examples/infer.sh \
  --model pi05 \
  --python .venvs/pi05/bin/python \
  --artifact "$PI05_ARTIFACT" \
  --input runs/quickstart/observation.npz \
  --output runs/quickstart/actions.npz \
  --gpu 0 --device cuda \
  -- runtime.compile=false runtime.cuda_graph=false inference.num_steps=10
```

成功时打印的 `shape` 应为 `[1, 50, 14]`，其中后两维由实际 Artifact 决定。
`inference.num_steps=10` 是去噪采样迭代次数，`action_horizon=50` 是输出动作长度。

```bash
python - <<'PY'
import numpy as np

with np.load("runs/quickstart/actions.npz", allow_pickle=False) as result:
    actions = result["actions"]
    assert actions.ndim == 3 and np.isfinite(actions).all()
    print("动作形状", actions.shape)
    print("第一步动作", actions[0, 0])
    print("每维最小值", actions.min(axis=(0, 1)))
    print("每维最大值", actions.max(axis=(0, 1)))
PY
```

动作已转换回训练配置对应的机器人动作空间。本路线使用绝对目标，14 维布局与采集一致。
先核对预测的量级、关节方向和夹爪范围，再进入真机联调。

## 4.4 开启推理加速

基础链路通过后，用同一观测和种子测试模板中的编译与 CUDA Graph 设置。

```bash
bash examples/infer.sh \
  --model pi05 \
  --python .venvs/pi05/bin/python \
  --artifact "$PI05_ARTIFACT" \
  --input runs/quickstart/observation.npz \
  --output runs/quickstart/actions_accelerated.npz \
  --gpu 0 --device cuda \
  -- inference.num_steps=10 inference.seed=42
```

Pi0.5 BC 推理模板默认开启 `runtime.compile=true`、`compile_mode=legacy` 和 CUDA Graph。
第 3.8 节的 18 维叠衣服 Artifact 使用 OpenPI 算子，需在命令的 `--` 后追加
`runtime.compile_mode=openpi`；同样适用于其他 `operator_numerics=openpi` 的 BC Artifact。
第一次运行包含编译与捕获耗时。Memory 使用专门的推理模板和历史输入，见[Memory 指南](../pi05_memory.md)。

## 4.5 常见问题

| 现象 | 优先检查 |
|---|---|
| 找不到 Artifact | 指针文件和它指向的目录是否一起复制，`manifest.yaml` 是否存在 |
| 相机缺失、state 维度不符 | `.npz` 公共键名和 Artifact 中的相机、维度声明 |
| 归一化或 tokenizer 文件缺失 | `deployment/assets/` 是否完整 |
| 首次请求很慢 | 是否正在加载权重、编译或捕获 CUDA Graph |
| 图片变色、视角错误 | RGB/BGR、相机映射、训练与推理 ROI |

下一步启动[真机部署的三个进程](05_deployment.md)。通用 HTTP API 另见
[HTTP 服务参考](../pi05_training.md#93-nexus-http-服务)。

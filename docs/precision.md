# 精度参数与生效范围

[README](../README.md#进阶参考) · [训练参数](pi05_training.md) · [离线推理](getting_started/04_inference.md)

Pi0、Pi0.5 策略默认使用 BF16 计算。训练保留 FP32 持久参数和 Adam 状态；
推理默认加载 BF16 参数。Pi0.5 RECAP 和 Memory 策略共用这组配置规则。

| 字段 | 生效阶段 | 默认值与行为 |
|---|---|---|
| `model.precision` | 训练、推理 | 模板为 `bf16`；字段缺失时读取 `runtime.precision`，再缺失时为 `bf16` |
| `runtime.precision` | 训练混合精度 | 训练模板为 `bf16`，必须与 `model.precision` 一致；推理显式设置了 `model.precision` 后，不会被此字段覆盖 |
| `runtime.autocast` | 训练、Pi0/Pi0.5 推理 | `auto` 保留运行时默认 AMP 策略；`"off"` 关闭外围 autocast，保留模型内部和 FSDP 的显式 dtype 转换 |
| `model.parameter_precision` | 训练 | `fp32`；Pi0/Pi0.5 训练拒绝低精度持久参数 |
| `model.inference_parameter_precision` | 推理 | 省略时跟随计算精度；显式 `fp32` 可保留 FP32 参数；不从 Artifact 的训练配置自动继承 |
| `runtime.compile_preserve_precision` | Pi0.5 推理编译 | `true`；保留中间转换的舍入边界，`legacy`/`openpi` 编译必须开启 |

训练配置：

```yaml
model:
  precision: bf16
  parameter_precision: fp32
runtime:
  precision: bf16
```

推理配置可显式选择以下组合：

| `model.precision` | `model.inference_parameter_precision` | 含义 |
|---|---|---|
| `bf16` | 省略或 `bf16` | 默认 BF16 参数与计算 |
| `bf16` | `fp32` | FP32 参数，计算通过 autocast 使用混合精度 |
| `fp32` | 省略或 `fp32` | FP32 参数与计算，适合数值诊断 |
| `fp32` | `bf16` 或 `fp16` | 不支持；配置阶段拒绝，避免模型加载后矩阵乘 dtype 不匹配 |

上表采用 `runtime.autocast: auto`。其生效规则为：Local 的 CUDA BF16/FP16
训练开启 AMP；Accelerate 非 FSDP 训练按 `runtime.precision` 选择 AMP；FSDP
训练关闭外围 AMP，使用自己的混合精度参数转换。Pi0/Pi0.5 推理在参数精度与计算精度
不同且计算精度非 FP32 时开启 AMP；默认 BF16 参数/BF16 计算已经关闭 AMP。

需要显式关闭外围 AMP 时，在本次启动的配置中设置：

```yaml
runtime:
  autocast: "off"
```

CLI 等价覆盖为 `runtime.autocast=off`。未加引号的 YAML `off` 被解析为 `false` 时也会
按关闭处理；不接受 `on`/`true`。该选项不会把模型整体改成 FP32，也不会取消模型内部的
BF16 转换或 FSDP 的 `param_dtype`。FP32 参数配 BF16 计算时，关闭 AMP 可能让部分
视觉 Attention/Linear 使用 FP32，改变动作输出、速度及显存；它不保证与 FSDP 逐算子一致。
Pi0/Pi0.5 在 `off` 下对普通 Linear 显式将计算权重/偏置转换为输入 dtype，避免 BF16
激活与 FP32 参数直接相乘报错；主参数、梯度和权重键保持原有格式。OpenPiLinear
仍按输入与参数的 dtype promotion 执行，不改为普通 Linear 的规则。

推理配置不会从 Artifact 的训练配置继承 `runtime.autocast`。修改后重新加载 policy；
模型的 `inference_metadata` 和 Pi0.5 WebSocket 元数据会报告 `autocast`、
`autocast_enabled`，训练日志报告实际 AMP 状态。自定义模型是否支持推理该选项由其实现决定。
续训时省略该字段与 `auto` 等价；改为 `off` 会改变续训兼容性指纹，应作为新的训练实验启动。

切换全 FP32 时，如果原推理 YAML 已显式指定参数精度，需同时修改两个字段：

```yaml
model:
  precision: fp32
  inference_parameter_precision: fp32
```

配置解析器也接受 FP16 及其别名；默认路线和历史验证以 BF16/FP32 为主。
计算精度描述的是模型的运算策略，归一化、采样噪声、Euler 状态和输出等边界按算子实现保留 FP32，
因此返回动作是 `float32` 并不能证明权重也是 FP32。

## Artifact 与启动入口

Artifact 中的 `config_resolved.yaml` 记录训练配置；其中的参数精度和 manifest 精度元数据
描述训练产物。加载时会恢复结构、数据变换和模型资产，推理精度由本次启动的配置决定。
仅将 `inference_parameter_precision` 写入训练 YAML 不会改变后续默认推理。

- Python：在传给 `load_policy(config)` 的推理配置中设置该字段。
- 离线/HTTP：在 `examples/infer.sh` 的 `--` 后追加 `model.inference_parameter_precision=fp32`。
- Pi0.5 WebSocket：通过 `--config` 选择包含该字段的推理 YAML。
- Memory 独立服务：在其位置参数指定的推理 YAML 中设置该字段。

已加载 policy 的参数精度持续用于该实例；修改 YAML 后需重新加载 policy。
重新启动未指定精度的服务时，按当前 profile 的默认值解析。

## 编译、Memory 和 Value

`runtime.compile` 和 `runtime.cuda_graph` 分别控制编译与独立 CUDA Graph；它们不选择权重精度。
它们可以与 `autocast: "off"` 组合使用；是否保持数值及获得加速需按模型和精度组合验证。
比较编译/Graph 与 eager 时，必须保持同一 autocast 策略。Pi0.5 Graph 缓存区分
外围 autocast 的启用状态和 dtype，防止复用不同精度上下文捕获的图。
Pi0.5 的 `compile_mode` 必须匹配 Artifact 的 `operator_numerics`：`legacy` 对应 `legacy`，
`openpi` 对应 `openpi`。18 维叠衣服教程使用 `openpi`，加速推理需显式选择相同模式。
Pi0 的 `runtime.compile` 默认关闭，只编译模型内的部分算子，不提供 Pi0.5 的数值保持模式
或独立 CUDA Graph 开关。开启后可能出现浮点融合、舍入带来的输出差异；建立严格数值基线时
使用 `runtime.compile=false`，启用前用相同输入和噪声验证任务允许的误差。
完整开关说明见 [Pi0.5 推理参数](pi05_training.md)。

Memory 的独立冻结历史视觉编码器使用固定的混合精度策略：FP32 stem/归一化参数、BF16 Transformer；
不会被策略模型的 `inference_parameter_precision` 改写，保证与历史缓存特征的生成方式一致。

RECAP Value 使用独立的 `value.backbone_precision`（`bfloat16` 或 `float32`），
融合头为 FP32，不读取策略模型的 `model.inference_parameter_precision`。

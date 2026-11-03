# RLToken 双残差配方

## Token 预训练

```bash
PYTHONPATH=src:. python -m dobot_nexus.cli.train --config examples/configs/rltoken/dobot_token.yaml
```

Token 使用 `training_mode=token_only`、`mode=bottleneck`、`selection=image_only`。
VLA 保持 eval 和冻结；只优化 Encoder/Decoder，不计算 flow loss。Pi0.5 根据自身
vision patch size 和相机图像尺寸选择图像 token。Decoder 只接收 latent 和 learned
query，不读取目标内容。对于前置缺失相机，首个 query 保持可见，防止 causal attention
出现全部屏蔽的行；缺失位置不参与 MSE。

Token Trainer 按 `algorithm.token.train_episodes/validation_episodes` 划分，保存
每个验证点的完整 Artifact、`best`/`last` 指针和 `resume.pt`。验证包含 zero/shuffle
latent 依赖比率，`token_validation.json` 绑定 feature ID。缓存入口默认要求两个依赖
门限通过。旧 provider 的算法消融必须显式设置 `data.require_token_acceptance=false`；
不能把这种运行记作完整表示配方复现。Token-only 优化器只含可训练参数。

Pi0.5 的 `model.rltoken.frozen_parameter_precision` 默认是 `fp32`。已有 BC 部署使用
BF16 参数时，可在 token-only 模式显式设为 `bf16`，并保持
`model.inference_parameter_precision=bf16`。冻结 VLA 使用原生 BF16 前向，Token
参数仍为 FP32；训练特征和部署特征采用同一数值 profile，profile 进入 feature ID。
接入既有权重时应先验证固定观测/噪声下的参考动作，再比较完整回合。

Token 数据默认使用模型的 `build_train_dataloader`。任务拥有其他原始记录格式时，
可通过 `algorithm.token.data_loader=module:function` 提供 `(model, config) -> loader`；
loader 需返回可索引 dataset 和 collate_fn，并遵守传入的 `data.episode_ids` 划分。
图像、状态、动作与语言预处理仍应使用模型的数据适配器；任务读取工具放在实验目录。

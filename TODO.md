# 待办事项

- [ ] **POLICY-NORM：为 Pi0 增加可配置的 state/action 归一化**

  在 Pi0 私有 pipeline 中支持 `zscore`、`quantile` 选择，让训练、推理和动作反变换
  使用同一配置与统计资产。图像像素缩放、Value min/max 和 return 缩放独立管理。

  验收：覆盖统计字段与维度检查、action encode/decode 往返、Artifact 保存和恢复。
  模型私有实现保持独立；identity、min/max 按实际需求扩展。

  入口：[Pi0 pipeline](src/dobot_nexus/models/pi0/data/pipeline.py)、
  [Pi0.5 pipeline](src/dobot_nexus/models/pi05/data/pipeline.py)。

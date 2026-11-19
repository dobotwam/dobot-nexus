# 真机部署

从仓库根目录选择与本体匹配的入口。训练、离线预测和模型服务模板位于 [examples](../README.md)。

| 本体 | 状态 / 动作 | 启动入口 |
| --- | --- | --- |
| [ATOM D](atom_d/README.md) | 18 / 18，双臂、双 DH 夹爪、头部 | `atom_d/run_atom_d.sh`。PC2 连接或启动本机策略，操作终端控制推理和归位。 |
| [ATOM W 0759](atom_w_0759/README.md) | 18 / 18，双臂、双 DH 夹爪、头部 | `atom_w_0759/start_atom_runtime.sh`。独立连接策略服务，执行前按机型指南完成只读验证。 |
| [Dobot 双臂 / XTrainer](dobot_dual_arm_14d/README.md) | 14 / 14，每臂六轴加一个夹爪 | 机器人端使用 `start_dobot_dual_arm_14d_server.sh`，GPU 主机使用 `start_robot_client.sh` 连接机器人与策略。 |

需要单独提供 Pi0.5 策略服务时，使用 [start_policy_server.sh](start_policy_server.sh)。机型推理配方位于 [recipes/infer](../recipes/infer)。

`atom_common/` 提供 ATOM 共用的 18 维映射、相机与控制器接口。`shared/` 保存跨机型使用的字段契约、布局和 Artifact 解析。双臂 TCP 与 Memory 历史传输代码随 `dobot_dual_arm_14d/_runtime/` 一起部署。

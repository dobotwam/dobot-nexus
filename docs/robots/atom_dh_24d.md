# ATOM W + DH 夹爪：24 维数据与部署映射

本体部署按机器人和末端执行器实现。通信层只校验向量尺寸，硬件端还负责字段顺序、
单位、执行接口与控制权限。`dobot_dual_arm_14d` 使用每臂六关节加单夹爪的布局；
ATOM W 使用每臂七关节，DH 夹爪各占一个归一化开度，需使用对应硬件适配器。

本文依据 2026-11-11 至 2026-11-13 核对的 ATOM 数采实现：`contract.py` 的
`NAMES` / `compose24`、`schema_features.json`、`hardware.py`、`recorder.py`
和 `collection_runner.py`。数据协议名为 `atom-dh-fullbody-24d-v1`。
这些字段描述当前 DH 夹爪采集协议，不用于推断旧数据或灵巧手协议。

## 24 维字段顺序

以下索引从 0 开始，`observation.state` 和 `action` 使用同一顺序。
两者分别是实时反馈和经过任务约束后最后成功发送的目标，数值来源不同。

| 索引 | 字段 | 单位 | 硬件来源 / 目标 |
|---|---|---|---|
| 0 | `left_shoulder_pitch_joint` | rad | upper[1]，左 J1 |
| 1 | `left_shoulder_roll_joint` | rad | upper[2]，左 J2 |
| 2 | `left_shoulder_yaw_joint` | rad | upper[3]，左 J3 |
| 3 | `left_elbow_pitch_joint` | rad | upper[4]，左 J4 |
| 4 | `left_elbow_roll_joint` | rad | upper[5]，左 J5 |
| 5 | `left_wrist_pitch_joint` | rad | upper[6]，左 J6 |
| 6 | `left_wrist_yaw_joint` | rad | upper[7]，左 J7 |
| 7 | `left_gripper_opening` | [0, 1] | 左 DH 夹爪开度 |
| 8 | `right_shoulder_pitch_joint` | rad | upper[8]，右 J1 |
| 9 | `right_shoulder_roll_joint` | rad | upper[9]，右 J2 |
| 10 | `right_shoulder_yaw_joint` | rad | upper[10]，右 J3 |
| 11 | `right_elbow_pitch_joint` | rad | upper[11]，右 J4 |
| 12 | `right_elbow_roll_joint` | rad | upper[12]，右 J5 |
| 13 | `right_wrist_pitch_joint` | rad | upper[13]，右 J6 |
| 14 | `right_wrist_yaw_joint` | rad | upper[14]，右 J7 |
| 15 | `right_gripper_opening` | [0, 1] | 右 DH 夹爪开度 |
| 16 | `head_yaw_joint` | rad | upper[15] |
| 17 | `head_pitch_joint` | rad | upper[16] |
| 18 | `torso_joint` | rad | upper[0]，躯干转轴 |
| 19 | `base_linear_velocity` | m/s | AMR 线速度 |
| 20 | `base_angular_velocity` | rad/s | AMR 角速度 |
| 21 | `crus_pitch_joint` | rad | lower[0] |
| 22 | `thigh_pitch_joint` | rad | lower[1] |
| 23 | `plevis_pitch_joint` | rad | lower[2] |

`plevis_pitch_joint` 沿用现有协议拼写。DH 开度 0 表示闭合，1 表示张开。
24 个元素包含 20 个旋转关节位置、2 个夹爪开度和 2 个底盘速度量。
下肢三轴共同影响腰部高度和姿态，协议中没有以米表示的独立 Z 轴高度。

## 与 SDK 顺序互转

上肢 DDS 的 `upper[17]` 顺序为躯干、左臂七轴、右臂七轴、头部两轴。
模型状态按下列方式组装，Python 切片右边界不包含在内：

```python
state = concatenate([
    upper[1:8], hands[0:1],
    upper[8:15], hands[1:2],
    upper[15:17], upper[0:1], base[0:2], lower[0:3],
])
```

若获得了各部件的控制权限，动作字段的逆映射为：

```python
upper_target = concatenate([action[18:19], action[0:7], action[8:15], action[16:18]])
hands_target = action[[7, 15]]
base_target = action[19:21]
lower_target = action[21:24]
```

这是字段映射，实际执行须分别经过上肢 DDS、DH 夹爪、AMR 和下肢接口。
17 维上肢速度/力矩、3 维下肢速度/力矩等附加观测保留 SDK 自身顺序，
不能按上表的模型向量顺序解释。

## 当前任务的执行语义

当前桌面任务先用 VR 调整头部视角，再锁定头部、躯干和下肢姿态。
录制动作中的索引 16–23 为保持目标；索引 19–20 固定为零速度。
状态中的索引 19–20 来自实际底盘反馈，移动时暂停采集。

因此这份数据支持学习双臂与夹爪任务，但没有提供主动移动底盘或调整身体的任务示范。
对应推理适配器应保留同样的身体保持策略和底盘静止约束，独立管理各部件控制权。
不能仅凭 `state_dim=action_dim=24` 将所有输出直接下发。

## 与 14 维、18 维及灵巧手的关系

- `dobot_dual_arm_14d`：每侧六关节和一个夹爪，不能使用 ATOM 的切片。
- 旧 ATOM-DH 18 维客户端：前 16 维是双臂和夹爪，16–17 是头部反馈；
  既有客户端保持头部，不执行预测的头部动作。旧叠衣服数据仅以
  `source_00`–`source_17` 命名，当前采集协议不能反向证明旧数据的物理含义。
- ATOM-DH 24 维：在已核对的 18 维布局后增加躯干、底盘两速度和下肢三轴。
  18 维 Artifact 的输入、输出和归一化仍为 18 维，需要匹配对应适配器；
  不能补零或修改握手元数据将其变成 24 维模型。
- 灵巧手：每只手的关节数、顺序和单位需按该末端协议定义，不能套用 DH 的单开度字段。

本仓库提供 Dobot 双臂 14 维硬件入口、可配置维度的策略通信桥，以及
[ATOM W 0759 的 18 维硬件示例](../../examples/real_robot/atom_w_0759/README.md)。
18 维示例默认保持头部，显式 `--head-mode policy` 才执行预测头部动作；
Home 流程会独立移动头部。该示例不控制底盘和下肢，也不接收 24 维模型输出。
本文规定 ATOM-DH 24 维字段与适配边界；完整 ATOM 24 维硬件执行入口需接入对应
SDK、控制权与身体保持逻辑后单独验证。

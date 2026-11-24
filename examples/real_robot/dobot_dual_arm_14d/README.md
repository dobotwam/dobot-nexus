# Pi0.5 真机部署

硬件入口按本体与维度命名：`start_dobot_dual_arm_14d_server.sh` 适配 Dobot 双臂，
每臂六关节加一个夹爪。策略服务与通信客户端按 Artifact 处理维度；其他本体需实现
对应的状态读取和动作执行，字段与单位见 [ATOM W + DH 24 维协议](../../../docs/robots/atom_dh_24d.md)。
XTrainer 双臂 14 维的 Pi0.5 BC 推理配方见
[pi05_xtrainer14.yaml](../../recipes/infer/pi05_xtrainer14.yaml)，硬件端使用本目录的机器人服务。
ATOM W 0759 的 18 维独立硬件入口见 [ATOM 示例](../atom_w_0759/README.md)。
该入口直接访问策略服务，使用共享的字段布局契约，不经过双臂 TCP 桥。

第一次联调请按[三进程真机部署](../../../docs/getting_started/05_deployment.md)操作。
新手指南以 BC 为例，分别说明两台机器的环境、启动顺序、dry-run 与人工确认执行。
本文保留 BC、RECAP、Memory 的完整启动参数和通信约定。

通信桥支持自定义状态和动作维度的 BC、RECAP、Memory；自带硬件驱动使用 14 维双臂布局。
对外只使用三个启动入口：

```text
机器人主机                         GPU 服务器
start_dobot_dual_arm_14d_server.sh  ← TCP →  start_robot_client.sh  ← WebSocket →  start_policy_server.sh
DobotEnvironment :12006          bridge client                       Torch policy :8000
```

GPU 主机使用 [策略启动器](../start_policy_server.sh) 加载 Artifact。此目录包含机器人服务、通信客户端和两者使用的 `_runtime/`，可单独复制到机器人。标准桥与 Memory 桥分别由 `dobot_dual_arm_14d.py` 和 `memory.py` 实现。

GPU 主机的 client 和 policy 使用同一个 Nexus 环境、分别运行进程。机器人主机需要自己的
硬件环境。模型和通信 client 在 GPU 主机运行。

## 1. GPU 主机：启动 policy

使用已有 Pi0.5 环境；需要补齐依赖时，在仓库根目录执行
`/path/to/pi05/bin/python -m pip install -e '.[pi05,websocket]'`。

```bash
POLICY_PYTHON=/path/to/pi05/bin/python \
GPU=4 POLICY_PORT=8001 \
bash examples/real_robot/start_policy_server.sh \
  /path/to/run/checkpoints/step_00002000
```

BC、RECAP、Memory 根据 Artifact 中的 `model.target` 自动选择。RECAP 可追加
`--recap-condition positive`，默认 `base`，也支持 `negative`。

模型路径支持 Artifact、run 目录、`latest_deployment.txt`/`latest.txt`，以及 Artifact 内的
`deployment/pytorch/full_weights.pt`。部署时保留 manifest、config_resolved 和 deployment
中的 Torch 权重、norm_stats、tokenizer；Memory 还需要冻结历史编码器 `vision.npz`。
部署应携带完整 Artifact 的配置和部署资产；`resume/` 仅用于继续训练。

| 参数 | 默认值 / 作用 |
|---|---|
| `GPU` / `POLICY_PORT` | `0` / `8000` |
| `NUM_STEPS` | `10`，去噪采样步数 |
| `SEED` / `FIXED_SEED` | `42` / `0`；默认逐请求递增，`FIXED_SEED=1` 固定种子 |
| `WARMUP` | `1`，启动时预热 |
| `TOKENIZER_PATH` | 默认使用 Artifact 自带 tokenizer |

默认 BF16，开启 torch.compile 和 CUDA Graph。BC/RECAP 默认编译模式为 `legacy`，
Memory 为 `openpi`；可追加 `--compile-mode openpi` 选择相应数值实现，或
`--no-torch-compile --no-cuda-graph` 关闭加速。`openpi` 是本地 Torch 数值配置名称。

```bash
curl --noproxy '*' http://127.0.0.1:8001/healthz
```

## 2. 机器人主机：启动 robot-server

在部署目录的父目录运行。配置实际硬件环境、训练任务指令、相机和裁剪区域。
两端需要安装 `msgpack` 并安全注入相同的 `NEXUS_ROBOT_TOKEN`（至少 32 字节随机密钥）。
以下是 30 Hz 按钮 Memory 的配置示例；`TOP_ROI_XYWH=none` 仅适用于训练使用全图的情况。

```bash
export ROBOT_PYTHON=/path/to/robot-env/bin/python
export ENV_SRC=/path/to/robot/environment/source
export ROBOT_BIND_HOST=ROBOT_IP
export TASK='press the yellow button, then press the red button, and then press the yellow button.'
export TOP_ROI_XYWH=none
export BRIDGE_MODE=memory
export STEP_DT=0.0333333333 CONTROL_FREQUENCY=30
export CAMERA_FPS=30 HISTORY_FPS=30
export ACTIONS_PER_RESPONSE=8
bash dobot_dual_arm_14d/start_dobot_dual_arm_14d_server.sh
```

`ENV_SRC`、`TASK`、`TOP_ROI_XYWH` 必须显式设置。裁剪也可以写成
`TOP_ROI_XYWH='X Y W H'`，应与数据采集配置一致。三路输出图像为 224×224 RGB。

硬件参数通过环境变量覆盖启动器默认值：

| 硬件 | 环境变量 |
|---|---|
| 三路相机序列号 | `TOP_CAMERA_ID`、`LEFT_CAMERA_ID`、`RIGHT_CAMERA_ID` |
| 夹爪串口 | `LEFT_GRIPPER_PORT`、`RIGHT_GRIPPER_PORT` |
| 机械臂 IP | `LEFT_ARM_IP`、`RIGHT_ARM_IP` |
| 遥操作串口 | `LEFT_TELOP_PORT`、`RIGHT_TELOP_PORT` |

默认 `ROBOT_EXECUTE=0`，采集观测并接收预测，不调用 `robot.step()`。
服务启动后默认处于 `positioning_paused`。保持机器人服务在前台交互终端运行，完成姿态准备后，
在**运行该服务的终端**输入 `model` 并按 Enter；这是服务交互命令，不是 shell 命令。
确认 `[control-state]` 日志中的 `state` 为 `model_waiting`，再启动第 3 节的 client。
客户端认证成功后变为 `model_control`。dry-run 与执行模式都需要交接。
如果开启了手动跟随，先输入 `adjust-left-on`、`adjust-right-on`，各按 Enter，
关闭两臂跟随后再输入 `model`。

使用同一组环境变量开启执行：

```bash
ROBOT_EXECUTE=1 bash dobot_dual_arm_14d/start_dobot_dual_arm_14d_server.sh
```

默认 `HOME_ON_START=1`：启用执行后，启动时确认并回 Home；设为 `0` 可保留当前位置。
默认 `CONFIRM_EACH_ACTION=1` 保留人工确认；`h` 触发 Home。Memory 默认 Home 后开启新一轮
历史，追加 `--memory-home-mode continue` 可保留历史。`MAX_JOINT_DELTA` 默认为 `0.15 rad`。

按钮数据使用 30 Hz；插接 BC/RECAP 使用 `STEP_DT=0.05 CONTROL_FREQUENCY=20`，训练指令为
`Left arm fixes the socket in place; right arm inserts the plug into the socket.`。
具体任务以对应数据集的指令和采集频率为准。

## 3. GPU 主机：启动 client

在另一个终端配置与机器人端相同的 `NEXUS_ROBOT_TOKEN`，确认机器人已进入
`model_waiting`，再连接机器人与本机 policy：

```bash
CLIENT_PYTHON=/path/to/pi05/bin/python \
BRIDGE_MODE=memory HISTORY_FPS=30 \
ROBOT_HOST=ROBOT_IP POLICY_HOST=127.0.0.1 POLICY_PORT=8001 \
ACTIONS_PER_RESPONSE=8 \
bash examples/real_robot/dobot_dual_arm_14d/start_robot_client.sh
```

- Memory 必须在 robot-server 和 client 两端都设置 `BRIDGE_MODE=memory`。
- BC/RECAP 可使用 `BRIDGE_MODE=standard`；做 BC/Memory 对照时也可统一使用 `memory`，
  BC policy 不编码历史。
- 两端 `ACTIONS_PER_RESPONSE`、Memory 的 `HISTORY_FPS` 应保持一致。
- client 默认采用机器人发送的任务；`TASK_OVERRIDE` 可显式覆盖。
- 机器人 TCP 默认端口为 `12006`，用 `ROBOT_PORT` 修改；policy 端口用 `POLICY_PORT` 修改。
- 其他参数可直接追加到启动命令；`BRIDGE_SCRIPT` 可指定自定义通信脚本。

客户端断开、任务结束或重启 robot-server 后，机器人恢复 `positioning_paused`。
重新连接前需在机器人服务终端再次输入 `model`，确认 `model_waiting` 后重启 client。
提前连接被拒绝时也按此顺序重试；Memory 重连会开始新的历史会话。

## Memory 与通信约定

Memory 在动作执行期间持续采集顶部相机图像，按训练时间网格重采样，连续上传并逐块 ACK；
每个新 episode 从帧 0 开始。切换 prompt 必须开启新 episode。连接关闭时清理该连接历史，
重连不会恢复上轮记忆；上传或推理失败不会自动重放。机器人端只保存尚未确认的图像，
policy 负责冻结视觉编码和长期历史。

policy 返回 `float32[H,A]` 的已解码绝对动作，`H`、`A` 由 Artifact 决定。
client 只截取前 N 个时间步，完整保留所有动作维度；不要再次归一化或做增量解码。
状态宽度 `S` 和动作宽度 `A` 可以不同。预热、观测和动作校验均采用 policy 元数据，
顶层维度、`policy_spec` 和输入/输出 schema 必须一致。维度不匹配或非有限值会报错，
不会自动补零、截断维度或转换动作含义。模型内部的 padding 维度不用于通信。

例如，18 维 Artifact 可以直接启动 policy；client 自动读取维度，也可以显式核对：

```bash
STATE_DIM=18 ACTION_DIM=18 \
ROBOT_HOST=ROBOT_IP POLICY_HOST=127.0.0.1 POLICY_PORT=8001 \
bash examples/real_robot/dobot_dual_arm_14d/start_robot_client.sh
```

`STATE_DIM` / `ACTION_DIM`（或 `--state-dim` / `--action-dim`）仅核对模型维度，不修改模型。
自定义 robot-server 的 hello 必须声明 `state_dim`、`action_dim`，随后发送 `state[S]`，
接收 `actions[N,A]`。例如 18 维 hello：

```python
{"type": "hello", "protocol": "dual_env_socket_v2", "state_dim": 18, "action_dim": 18}
```

复用 `_run_robot_server_locked(..., runtime_factory=...)` 时，自定义 runtime 提供
`communication_spec` 字典，并实现对应的观测和硬件动作映射。两端缺省维度的旧握手按
14 维处理；因此自定义机器人必须显式声明维度。标准与 Memory 桥接均使用同一维度约定。

自带 `RobotRuntimeEnv` 声明并执行 14 维：每臂 6 个关节和 1 个夹爪，夹爪索引为 6、13，
其余关节单位为 rad。KBD、DH24 等自定义机器人需要实现与训练数据顺序、单位一致的
硬件映射；仅设置环境变量不能改变这个 SDK 的动作布局。

通信使用 Nexus 自带的 WebSocket/msgpack 客户端，GPU client 还需要 NumPy 和 OpenCV。
TCP 使用认证 msgpack；WebSocket 默认绑定回环地址，跨机器绑定需设置 `NEXUS_POLICY_TOKEN`。
桥接客户端读取同名环境变量并发送 Bearer 认证；请求和响应均限制为 32 MiB。
服务不提供 TLS，跨不可信网络需要加密隧道或 TLS 网关。

### Robot TCP 认证

机器人与 GPU 桥接端均需安装 `msgpack`，并通过环境变量设置相同的
`NEXUS_ROBOT_TOKEN`（至少 32 字节随机密钥；可用 `python -c 'import secrets; print(secrets.token_hex(32))'`
生成后通过安全渠道分发）。密钥不要写入配置、命令参数或 Git。
Robot TCP v2 使用 msgpack、双向随机挑战认证、逐帧 HMAC 和序号校验，帧上限 32 MiB，
握手、动作回复和帧传输期限为 30 秒；已认证客户端等待下一条观测时允许人工暂停，
收到首字节后仍须在 30 秒内接收完整帧。两端必须一起更新。默认只监听 `127.0.0.1`，跨机器时明确设置
`ROBOT_BIND_HOST`。HMAC 不加密观测数据，跨不可信网络须使用 VPN 或加密隧道。

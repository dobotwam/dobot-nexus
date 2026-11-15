# 5. 三进程真机部署

[新手路线](../../README.md#从哪里开始) · 上一步 [离线推理](04_inference.md)

本篇面向 Dobot 双臂、三相机、14 维状态/动作的 Pi0.5 BC 部署，
硬件入口为 `start_dobot_dual_arm_14d_server.sh`。
准备好通过离线推理检查的正式 Artifact，以及机器人项目已验证可用的硬件环境。
其他机器人应使用匹配其动作维度、字段和控制协议的硬件适配器。
ATOM W + DH 的字段定义见 [24 维本体协议](../robots/atom_dh_24d.md)。

## 5.1 哪台机器运行哪个进程

```text
机器人主机                              GPU 主机

robot-server  ←──── TCP 12006 ────→  robot-client  ←── WebSocket 8000 ──→  policy-server
相机、状态、硬件执行                    转发观测和动作                      模型推理
```

需要三个终端，建议按 **policy-server → robot-server → robot-client** 顺序启动。

| 进程 | 机器 | Python 环境 |
|---|---|---|
| policy-server | GPU 主机 | `.venvs/pi05`，包含 WebSocket extra |
| robot-client | GPU 主机 | 同一个 `.venvs/pi05` |
| robot-server | 机器人主机 | 硬件项目的 Python 环境 |

GPU 主机保留 Nexus 仓库和完整 Artifact。机器人主机可以使用完整仓库，也可以完整复制
`examples/pi05_real_robot/`，包括 `_runtime/`。以下机器人命令采用完整仓库路径。

## 5.2 终端 A，在 GPU 主机启动策略服务

```bash
cd /path/to/dobot-nexus
export PI05_ARTIFACT=/path/to/run/checkpoints/step_00010000

POLICY_PYTHON="$PWD/.venvs/pi05/bin/python" \
GPU=0 POLICY_PORT=8000 \
bash examples/pi05_real_robot/start_policy_server.sh \
  "$PI05_ARTIFACT" --no-torch-compile --no-cuda-graph
```

首次联调使用 eager 模式。服务会加载模型并预热，保留这个终端运行。
另开终端检查服务健康状态。

```bash
curl --noproxy '*' http://127.0.0.1:8000/healthz
```

预热完成后应返回 HTTP 200。查看服务握手元数据可确认动作维度。

```bash
.venvs/pi05/bin/python - <<'PY'
from dobot_nexus.inference.transport.client import WebsocketPolicyClient

with WebsocketPolicyClient(host="127.0.0.1", port=8000) as client:
    print(client.get_server_metadata())
PY
```

记录其中的模型身份、state/action 维度和动作长度。硬件客户端要求 state/action 均为 14 维。
稳定后重启服务时可以去掉 `--no-torch-compile --no-cuda-graph`，启用默认加速。

## 5.3 终端 B，在机器人主机准备硬件

硬件项目应提供可导入的 `dobot_environment.Configuration`、`DobotEnvironment`，
以及对应机械臂、相机、夹爪和串口驱动。该环境由机器人项目维护者提供安装说明。
先核对设备 ID、IP 和串口，下面的占位内容全部替换成真实值。
机器人环境还需安装 `msgpack`。在机器人与 GPU 客户端终端，从密钥管理工具或
安全的交互输入设置相同的 `NEXUS_ROBOT_TOKEN`（至少 32 字节随机密钥）。

```bash
cd /path/to/dobot-nexus
export ROBOT_PYTHON=/path/to/robot-env/bin/python
export ENV_SRC=/path/to/robot/environment/source

PYTHONPATH="$ENV_SRC${PYTHONPATH:+:$PYTHONPATH}" "$ROBOT_PYTHON" - <<'PY'
from dobot_environment import Configuration, DobotEnvironment
print("机器人项目可导入", Configuration.__name__, DobotEnvironment.__name__)
PY
```

在同一终端设置任务、图像几何和硬件参数。

```bash
export TASK='Pick up the red cube.'
export TOP_ROI_XYWH=none
export BRIDGE_MODE=standard
export ROBOT_BIND_HOST=YOUR_ROBOT_HOST_IP
export ROBOT_PORT=12006
export ACTIONS_PER_RESPONSE=8
export STEP_DT=0.05 CONTROL_FREQUENCY=20
export CAMERA_FPS=30

export TOP_CAMERA_ID=YOUR_TOP_CAMERA_SERIAL
export LEFT_CAMERA_ID=YOUR_LEFT_CAMERA_SERIAL
export RIGHT_CAMERA_ID=YOUR_RIGHT_CAMERA_SERIAL
export LEFT_ARM_IP=YOUR_LEFT_ARM_IP
export RIGHT_ARM_IP=YOUR_RIGHT_ARM_IP
export LEFT_GRIPPER_PORT=/dev/YOUR_LEFT_GRIPPER
export RIGHT_GRIPPER_PORT=/dev/YOUR_RIGHT_GRIPPER
export LEFT_TELOP_PORT=/dev/YOUR_LEFT_TELEOP
export RIGHT_TELOP_PORT=/dev/YOUR_RIGHT_TELEOP
```

- `TASK` 使用训练任务的指令。
- 转换时 `top_roi_xywh: null` 对应 `TOP_ROI_XYWH=none`。使用裁剪时，写成
  `export TOP_ROI_XYWH='145 30 400 400'` 等实际值。
- `STEP_DT=0.05`、`CONTROL_FREQUENCY=20` 对应 20 Hz 控制数据。30 Hz 任务相应使用
  `STEP_DT=0.0333333333 CONTROL_FREQUENCY=30`。`CAMERA_FPS` 按相机支持的采集档位设置。
- `ACTIONS_PER_RESPONSE=8` 表示每次执行预测动作块的前 8 步，再获取新观测。
  机器人服务与客户端必须配置一致，该值应不超过模型的动作块长度。

按硬件项目规程完成上电、空间检查和急停准备。随后先以预测动作 dry-run 模式启动。

```bash
ROBOT_EXECUTE=0 HOME_ON_START=0 \
bash examples/pi05_real_robot/start_dobot_dual_arm_14d_server.sh
```

此模式仍会初始化硬件环境、读取相机和机器人状态，模型动作路径通过日志显示预测值。
`robot.step()` 由 `ROBOT_EXECUTE` 控制。保留终端，等待客户端连接。

## 5.4 终端 C，在 GPU 主机启动通信客户端

将 `ROBOT_HOST` 替换为机器人主机的 IP。GPU 主机需要能访问该地址的 TCP 12006 端口。

```bash
cd /path/to/dobot-nexus

CLIENT_PYTHON="$PWD/.venvs/pi05/bin/python" \
BRIDGE_MODE=standard \
ROBOT_HOST=YOUR_ROBOT_HOST_IP ROBOT_PORT=12006 \
POLICY_HOST=127.0.0.1 POLICY_PORT=8000 \
ACTIONS_PER_RESPONSE=8 \
bash examples/pi05_real_robot/start_robot_client.sh
```

客户端从机器人接收观测，将任务和图像发给策略服务，再把动作发回机器人。
确认三端日志都能看到连接、观测、预测和返回动作，机器人端处于 dry-run 模式。
任务文本默认采用机器人发送的 `TASK`。

## 5.5 切换到受控执行

确认相机映射、ROI、14 维动作顺序、rad 单位和夹爪范围正确，并检查预测动作与当前姿态的差值。
首次执行保留人工确认和默认的 `MAX_JOINT_DELTA=0.15` rad 检查。

1. 在客户端和机器人服务终端分别按 `Ctrl+C` 停止进程。
2. 机器人终端保留上面设置的环境变量，运行下面的命令。
3. GPU 主机重新执行第 5.4 节的客户端命令。

```bash
ROBOT_EXECUTE=1 HOME_ON_START=0 CONFIRM_EACH_ACTION=1 \
bash examples/pi05_real_robot/start_dobot_dual_arm_14d_server.sh
```

`HOME_ON_START=0` 采用当前姿态开始，按训练任务要求手动准备初始场景。
需要执行 Home 时，应先核对硬件默认 Home 位姿，再使用相应控制。
按照机器人终端提示确认执行，先逐步观察关节方向和夹爪响应。
发现异常时使用硬件急停并停止当前测试。

## 5.6 常见问题

| 现象 | 检查位置 |
|---|---|
| healthz 连接失败 | policy 终端是否加载完成，端口是否一致 |
| 连接机器人超时 | `ROBOT_HOST`、`ROBOT_PORT`、两机网络与防火墙 |
| 找不到 `dobot_environment` | `ROBOT_PYTHON`、`ENV_SRC` 和硬件项目安装 |
| 维度不匹配 | Artifact 是否为当前 14 维双臂任务 |
| 图像错误 | 相机序列号、公共映射、颜色顺序和 ROI |
| 动作被拦截 | 状态和动作单位、左右臂顺序、初始姿态、关节差值 |
| 只有预测输出 | 当前 `ROBOT_EXECUTE` 设置，以及终端中的人工确认提示 |

这套 TCP/WebSocket 链路用于受控机器人内网。TCP 使用带认证的 msgpack，机器人与桥接端需设置相同的 `NEXUS_ROBOT_TOKEN`。
WebSocket 服务部署要求见安全说明。
BC/RECAP/Memory 的完整参数、Home 控制和历史协议见[真机启动器参考](../../examples/pi05_real_robot/README.md)。

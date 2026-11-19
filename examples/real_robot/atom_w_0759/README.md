# ATOM W 0759 真机推理

本目录在机器人侧运行相机、上肢 DDS 和 DH 夹爪适配，通过现有 Nexus WebSocket
客户端访问策略服务。机型代码全部位于 `examples/`，模型仍使用 `src/` 的通用推理实现。
需要克隆完整仓库；单独安装 wheel 不包含这些示例和验证测试。

当前硬件实现限定为 **ATOM W / 0759 / mode 11**，使用经过版本固定的 SDK、增益表、
URDF 和 Home 姿态。它不是任意 ATOM 型号的驱动。更换本体、SDK 或末端执行器时，
需要实现对应适配并重新验证；不要通过删除哈希、身份或关节限制检查绕过不匹配。

## 数据和执行契约

状态和动作均为 18 维，旋转关节使用 rad，夹爪开度为 `[0,1]`，0 关闭、1 张开。

| 索引（从 0 开始） | 含义 | 硬件映射 |
|---|---|---|
| 0–6 | 左臂七轴 | `upper[1:8]` |
| 7 | 左 DH 夹爪 | 左夹爪归一化开度 |
| 8–14 | 右臂七轴 | `upper[8:15]` |
| 15 | 右 DH 夹爪 | 右夹爪归一化开度 |
| 16–17 | 头部 yaw、pitch | `upper[15:17]` |

躯干转轴 `upper[0]` 保持，不执行底盘和下肢动作。默认 `--head-mode hold`，
观测仍包含真实头部反馈，动作保持进入策略控制时的头部参考；`--head-mode policy`
才执行模型的头部输出。启动和结束的 Home 是独立动作，会回到配置的头部姿态；
需要全程保持当前头部时，同时设置 `--no-home-on-start --no-return-home`。
24 维数据含额外的躯干、底盘和下肢字段，不能补零、截断或更改握手冒充 18 维模型。
详见 [24 维映射说明](../../../docs/robots/atom_dh_24d.md)。

字段协议位于 [robot_contract.py](../shared/robot_contract.py)，
布局在 [atom18.json](../shared/layouts/atom18.json)。通用契约支持独立的
state/action 字段表、字段重排和相机映射；本硬件入口固定使用 `atom18`。
它不改变现有双臂 14 维 TCP v2 桥，也不要求切换该桥的认证协议。

- 使用完整预测块 H，要求 `1 <= H <= 256`，`--horizon` 只校验长度，不修改模型。
  不在客户端截取前 N 步，也不修改 Artifact。`inference.execute_steps` 不控制本客户端。
- 动作频率固定为 20 Hz，以零阶保持逐行执行；H=50 的动作块持续 2.5 秒。
  完成关节和夹爪最后一行后，才采集新观测并请求下一块；推理等待期间保持最后参考。
  20 Hz 不是策略闭环频率。训练数据频率和动作含义必须与此一致。
- 主执行路径为 SDK 原生 `q/kp/kd` 参考，`dq=0`、`tau=0`。关节超限拒绝执行；
  夹爪裁剪到物理区间。这里不使用额外 PI 或逐步参考限幅。Home 使用独立的有界轨迹控制。
- 三路相机输出 224×224 RGB、等比例缩放和居中黑边填充，不能沿用其他客户端的
  ROI/拉伸预处理而不核对训练数据。相机默认 15 FPS，预热至少 60 帧。
- 当前入口不提供 Memory 所需的稠密历史，握手时拒绝此类策略。

## 环境与资产

策略服务使用仓库的 Pi0.5 环境，安装方式见 [环境配置](../../../docs/getting_started/01_environment.md)。
准备完整的、字段顺序一致的 **18 维 Pi0.5 BC Artifact**，其中包含权重、统计资产和 tokenizer。
机器人侧使用 Linux、Python >=3.10，以及以下轻量依赖；SDK 和 RealSense 的安装版本按厂商环境提供：

```bash
python -m pip install numpy Pillow PyYAML msgpack 'websockets>=15,<16'
# 按厂商说明安装 cyclonedds、pyrealsense2 及对应系统库。
export PYTHONPATH=src:.
export ATOM_SDK_ROOT=/path/to/Atom_sdk
# 指向包含 script/sdk/DH_sdk.py 的厂商 robot_control_dds 目录。
export ATOM_GRIPPER_SDK_ROOT=/path/to/1200_sdk/py/robot_control_dds
export ATOM_POLICY_HOST=127.0.0.1
export NEXUS_POLICY_TOKEN='replace-with-the-same-random-secret-at-least-32-bytes'
```

`ATOM_SDK_ROOT` 应包含 `atom_training/`、`tools/raise_arms_close_grippers.py`、
`configs/robot.0759.motion-reviewed.json`、`configs/robot.example.json` 和 `vendor/`。
具体哈希在 [native.py](native.py)、[workers.py](workers.py) 和 [gripper_reset.py](gripper_reset.py)。
当前适配限定 PC2 `192.168.8.13`、PC1 `192.168.8.234`、DDS domain 0，
并读取 PC1 `22000` 端口的实际关节限制。策略服务地址独立配置。
SDK 和厂商代码不随仓库分发；本目录的 SDK 调用不能代替这些依赖。

检查 [cameras_0759.json](../atom_common/config/cameras_0759.json) 的序列号和左右相机对应关系。
如需更换相机，将 JSON 复制到本地并通过 `--cameras /absolute/path/cameras.json` 指定。
预处理、视角及相机命名必须匹配 Artifact 的训练数据。

## 启动策略服务

在策略机的仓库根目录执行：

```bash
export NEXUS_POLICY_TOKEN='replace-with-the-same-random-secret-at-least-32-bytes'
POLICY_PYTHON=.venvs/pi05/bin/python \
examples/real_robot/start_policy_server.sh /path/to/atom18/artifact \
  --config examples/recipes/infer/pi05_atom18.yaml
```

两台机器部署时，服务命令前加 `POLICY_HOST=0.0.0.0`，两端使用相同的
`NEXUS_POLICY_TOKEN`，并将机器人的 `ATOM_POLICY_HOST` 设为策略机可达地址。
认证直接复用当前仓库客户端，不使用下载版本的 pickle TCP 协议。

[推理 recipe](../../recipes/infer/pi05_atom18.yaml) 继承当前通用配置：参数和计算使用 BF16，
`runtime.autocast=auto`，默认开启 `compile_mode=legacy`、精度保护和 CUDA Graph。
模型内部的 FP32 运算边界由当前算子实现管理，硬件适配不覆盖它们。
这些默认值不是本机 GPU 性能或真实任务效果的验证结果。
需要调试关闭优化时，服务命令追加 `--no-torch-compile --no-cuda-graph`。

## 只读验证和执行

下列命令从机器人侧仓库根目录执行。先确保 SDK 反馈正常、夹爪已初始化且无掉物故障，
停止其他控制程序。验证会连接真实传感器和策略服务，但不创建电机/夹爪命令 writer：

```bash
python -m examples.real_robot.atom_w_0759.validate --policy-host "$ATOM_POLICY_HOST" --head-mode hold
```

它运行两次独立生命周期，每次消费至少三个完整动作块，再执行 CPU 契约测试。
验证记录绑定源码路径和内容。首次使用新目录部署时，重新运行上述只读验证。
报告保存在 `runs/atom_robot/validation/`；验证检查完整行消费、相机生命周期、发送时序和
工作进程退出，并绑定当前源码、相机、SDK 路径、头部模式、任务文本和服务声明的策略身份。
代码、运行参数或策略变化后重新执行。Artifact 应保持不可变；同路径替换权重或调整未在
服务元数据中声明的采样参数时，也必须重新验证。只读验证不能证明机械碰撞安全或任务成功率。

需要初始化夹爪时，可先检查 Home 目标，再由现场操作员执行独立的 Home 初始化：

```bash
# 只读取反馈并显示目标，不发送命令。
python -m examples.real_robot.atom_w_0759.home
# 此命令会初始化/张开夹爪并移动双臂与头部到配置的 Home。
python -m examples.real_robot.atom_w_0759.home --execute --site-clear --reset-devices
```

确认现场满足运动条件后，使用与只读验证一致的参数运行：

```bash
examples/real_robot/atom_w_0759/start_atom_runtime.sh \
  --policy-host "$ATOM_POLICY_HOST" --head-mode hold --execute --site-clear
```

默认先初始化夹爪并回 Home，正常完成后再次回 Home；信号中断、故障或未解除的阻塞不会
自动回 Home。程序尽力停止命令并交还上肢控制权，操作员仍需使用硬件急停处理危险情况。
所有入口共享主机级 `/tmp/dobot-nexus-atom-w-0759.lock`，报告目录不影响互斥。

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--sdk-root` | `ATOM_SDK_ROOT` | 经版本核对的 ATOM SDK 根目录 |
| `--gripper-sdk-root` | `ATOM_GRIPPER_SDK_ROOT` | Home 初始化所需的厂商 DH SDK 目录 |
| `--policy-host` / `--policy-port` | `127.0.0.1` / `8000` | 策略服务连接地址 |
| `--cameras` | `config/cameras_0759.json` | 三路相机身份和采集参数 |
| `--head-mode` | `hold` | `policy` 才执行模型头部动作 |
| `--horizon` | 服务声明值 | 校验完整块长度，不裁剪或改变采样长度 |
| `--action-fps` | `20` | 当前适配仅支持数据集的 20 Hz 节拍 |
| `--max-horizons` / `--max-seconds` | `6` / `120` | 到完整动作块边界结束 |
| `--task` | `Fold the T-shirt with both arms.` | 任务指令，须与权重能力匹配 |
| `--task-guards` | `monitor` | 重复/阻塞行为只记录；`stop` 启用对应停止逻辑 |
| `--home-on-start` / `--return-home` | 开启 | 可用 `--no-…` 关闭对应 Home 流程 |
| `--validation` | `runs/atom_robot/validation` | 报告和验证记录目录 |
| `--execute` / `--site-clear` | 均关闭 | 两者都提供才允许策略执行 |

`monitor` 不关闭控制权、驱动状态、关节限位、反馈时效、动作行顺序及发送超时检查。
不执行 `--execute` 的运行不会因 Home 默认值而发送动作。

## CPU 测试

```bash
PYTHONPATH=src:. python -m unittest discover -s tests/entrypoints/atom -t . -v
```

测试使用假设备，不需要 SDK、相机、GPU 或权重。完整仓库测试还覆盖现有策略服务的
18 维消息转换，需要仓库标准测试环境，见 [测试说明](../../../tests/README.md)。

模块分工：`native.py`/`sdk_session.py` 管理 DDS 和控制权；`workers.py` 隔离相机推理与
夹爪 I/O；`core.py` 负责字段映射、动作时钟和任务检查；`sensors.py` 管理相机；
`home.py`/`gripper_reset.py` 管理 Home；`validate.py`/`verify.py`/`validation.py` 管理验证记录。

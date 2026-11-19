# ATOM D 双臂叠衣推理

在配有 GPU 的 PC2 上运行一个入口。SDK 进程读取 ATOM D 上肢、双 DH 夹爪和三路 RGB 相机，Pi0.5 进程在本机 GPU 上计算动作。两者通过本机 WebSocket 通信。程序固定使用任务提示 `Fold the T-shirt with both arms.`。

## 部署到 PC2

将含 `src/` 和 `examples/` 的完整 v1.0.0 工作区放在 PC2 上，从仓库根目录执行以下命令。需要一个包含 `manifest.yaml`、权重、归一化统计和声明资源的 Nexus 部署 Artifact。模型从 Artifact 恢复，Home-start 姿态随代码提供。

| 运行环境 | 现场配置和依赖 |
| --- | --- |
| SDK 客户端 | 使用厂家安装好的 Atom Python。已验证环境为 Python 3.8、NumPy、PyYAML、msgpack、websockets 13.1、Pillow、OpenCV、pyrealsense2，以及与控制器配套的 Cyclone DDS 和 ATOM SDK。 |
| GPU 策略 | 单独使用含 CUDA Torch、Pi0.5 与 WebSocket 服务依赖的 Python。现场复用了已有的 Python 3.10、Torch 2.7.1+cu126 环境。新建策略环境的依赖见[安装指南](../getting_started/01_environment.md)。 |
| 机器人设备 | 两只 DH 夹爪、左右腕部 RealSense、DECXIN 双目顶视相机。顶视输入取左目，三路图像按比例补黑边至 224 × 224。相机编号、序列号和顶视分辨率从采集用的 `robotconfig.json` 读取。 |

启动脚本读取仓库根目录可选的 `.env.atom-d`。以下是部署在现有 PC2 时的配置示例。将变量指向实际安装位置，SDK 和权重均可复用。已有采集台或旧推理程序时，`ATOM_D_CONTROL_ROOT` 指向它们使用控制锁的原项目根目录。

```bash
export ATOM_PYTHON="$HOME/miniforge3/envs/Atom/bin/python"
export POLICY_PYTHON="$HOME/deploy-20261118/dobot-nexus/.venvs/pi05-edge/bin/python"
export ATOM_D_CONTROL_ROOT="$HOME/deploy-20261118/dobot-nexus"
```

其他可选设置包括 `ATOM_D_DRIVER`（厂家 `robot_control` 目录）、`ATOM_D_URDF`、`ATOM_D_CAMERA_CONFIG`、`ATOM_D_TOP_DEVICE`、`ATOM_D_PC1`、`CYCLONEDDS_URI` 和 `ATOM_D_HOME_CONFIG`。默认值对应已验证 PC2 的 ATOM P2 安装布局、PC1 地址 `192.168.8.234` 和 PC2 控制网接口地址 `192.168.8.13`。权重可由 `--weights`、其别名 `--artifact` 或环境变量 `PI05_ARTIFACT` 指定，命令行优先。

## 运行

**完整叠衣推理。** 在机器人已使能、衣物已铺好、现场清场后执行。在交互终端启动后，看到 `WAITING` 即可按键。通过 SSH 运行单条远程命令时，使用 `ssh -t` 分配终端。

```bash
./examples/atom_robot/run_atom_d.sh \
  --weights "$HOME/step_00004000" \
  --home-start --execute --site-clear --full-episode
```

| 按键 | 操作 |
| --- | --- |
| Enter | 在 `WAITING` 开始新一轮推理。带 `--home-start` 时，每轮先张开双夹爪、回到采集首帧姿态，再用新观测开始推理。 |
| h | 在推理中停止下发策略动作，等该轮释放 SDK 控制权后执行 Home-start。在 `WAITING` 也可直接归位。归位会张开双爪，手中的衣物会被释放。 |
| Ctrl+C | 在等待、推理或归位期间退出程序。退出时停止当前操作，不触发归位。 |

一轮结束或手动归位完成后，程序重新显示 `WAITING`。运行中按的 Enter 不会排队触发下一轮。推理或归位报错时查看该次 `TRIAL_RESULT`，程序等待下一次按键。

**只读检查。** 以下命令读取相机和 DDS，检查 Home-start 前置条件并执行一次模型预测。省略 `--execute` 时发送动作数为零。需要短程实机验证时，使用 `--home-start --execute --site-clear`，最多发送前 10 行动作。

```bash
./examples/atom_robot/run_atom_d.sh --weights "$HOME/step_00004000" --home-start
```

**录制观测回放。** 选取任一历史运行目录中的 `prediction.npz`，设置 `OBSERVATION_NPZ` 为其路径。回放直接读取文件中的 `state`、`top_camera`、`left_camera`、`right_camera`，仅启动模型链路。

```bash
./examples/atom_robot/run_atom_d.sh \
  --weights "$HOME/step_00004000" --observation-npz "$OBSERVATION_NPZ"
```

现场也可传入其他完整 Artifact 目录、`latest_deployment.txt` / `latest.txt` 指针，或 manifest 声明的权重文件。程序按顺序检查本机 8001 和 8000 端口，只复用 Artifact 路径一致且满足 18 维、50 帧及相机契约的策略。需要拉起服务时采用 eager 模式，并在本轮结束后关闭自己启动的进程。[推理配方](../../examples/recipes/infer/pi05_atom18.yaml)

## 动作和日志

- 状态和每行动作均为 18 维，顺序为左臂 7 轴、左爪开度、右臂 7 轴、右爪开度、头部 yaw/pitch。关节单位为弧度，夹爪开度 0 为关、1 为开。腰部保持，模型每次预测 50 行。
- Home-start 同时保存了示教首帧的实测关节位置和对应电机指令。初始姿态来自 `episode_000000.parquet` 第 0 帧。[Home-start 配置](../../examples/atom_robot/config/atom_d_home.json)
- 动作的名义周期为 50 ms。程序在每行的第一个伺服周期发夹爪指令，双爪夹持力均设为 100%。抓取、释放后读取新观测再规划，持衣阶段每段最多推进 16 行。推理阶段的关节参考按 PC1 范围限幅，跟踪误差写入报告。
- DH 反馈 `grip_state=3` 记录为 `GRIPPER_DROP_REPORTED`。单独收到此状态会继续运行，抓取和释放阶段仍按连续的新反馈与实测开度确认。SDK 故障、驱动告警、初始化状态变化、陈旧反馈及动作超时仍可中止执行。
- 每轮输出路径打印在 `EPISODE_FOLDER`，默认位于 `runs/atom_d_inference/trial-*/`。`result.json` 记录权重、状态、已发送行数、抓取和释放事件。初次预测保存在 `prediction.npz`，闭环预测保存在 `chunk-*.npz`，现场图像保存在同一目录。启动的本地策略服务将日志写入 `policy.log`。
- 监督进程可在该轮目录创建 `stop` 文件。完整推理最多运行 1600 行或 120 秒，并监测重复预测。`episode_frame_limit`、`wall_time_limit` 等状态表示退出原因，叠衣是否完成由现场或图像确认。

## 异步换段

完整执行默认使用 `--inference-mode async`，可在同一条命令末尾加 `--inference-mode sync` 对照现场表现。两种模式使用相同的权重、采样步数、模型精度和伺服周期。10 行短程验证和单帧回放各只请求一次策略。

在双爪目标稳定时，控制线程在当前段还剩 4 行时采集实际状态和三路图像。后台线程独占本次 WebSocket 请求，控制线程继续发送已有动作。换段时检查反馈新鲜度、实际用时、关节位移、图像变化、夹爪状态，以及新预测的前 4 行与已下发指令的一致性。通过时从预取结果第 5 行接续，按轨迹选取连续 1 至 4 行过渡。每个关节的过渡目标与上一段末指令相差最多 2°，双爪保持原开合目标。同时从段末的实际观测重新推理，使用相同种子。过渡结束再次核对已执行前缀，通过后立即接入段末观测生成的动作。抓取或释放确认后必须重新采集图像；交接检查失败时也会使用当前观测重算。重算复用本段的扩散采样种子，后续段仍按原顺序取种子。连接到较早版本的策略服务时，入口自动采用同步模式。

在 `result.json` 的 `handoffs` 中，`source=prefetch` 表示采用了预取结果，`skipped_rows` 是已由上一段执行的行数，`discarded` 给出弃用原因。`boundary_wait_ms` 统计上一段执行及夹爪确认结束至本段发送首条指令的时间，包含换段日志和图像处理耗时。一次预取交接可能产生 `prefetch` 和 `boundary` 两段记录，后者是从段末实际观测生成的动作。

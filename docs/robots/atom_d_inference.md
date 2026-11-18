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

**完整叠衣推理。** 在机器人已使能、衣物已铺好、现场清场后执行。程序先确认双夹爪张开，再回到采集首帧的双臂和头部姿态，随后依据新观测闭环执行。

```bash
./examples/atom_robot/run_atom_d.sh \
  --weights "$HOME/step_00004000" \
  --home-start --execute --site-clear --full-episode
```

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
- 终端 Ctrl-C 可中断当前运行，监督进程也可在该轮目录创建 `stop` 文件。完整推理最多运行 1600 行或 120 秒，并监测重复预测。`episode_frame_limit`、`wall_time_limit` 等状态表示退出原因，叠衣是否完成由现场或图像确认。

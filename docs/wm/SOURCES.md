# 设计依据与核对记录

核对日期：2026-11-23。本文件区分远端现状、官方实现与本项目设计建议。

## 远端仓库现状

- 目标：SSH alias `B300_2`，`<repo-root>`。
- 从 `v1.0.0` 当前 HEAD `ff67100afe73b3381b2de2b9a4ecfddc8f0a066a` 创建并切到 `wm`。
- 原有工作树有 63 个已跟踪文件的未提交修改，以及未跟踪文件；这些不属于本次 WM 设计改动。
- 审阅基于实际工作树内容，不将 HEAD 当作干净基线；保存了 355 个既有文件的内容校验用于同步后核对。
- 远端缺少 `rg`，首次清点命令在此处以 127 退出；后续使用 Python/Git 完成清点。
- SSH 配置中的 12021 端口转发提示失败，但 SSH 命令通道和项目访问成功；未修改 SSH 配置。

| 依据 | 与设计的关系 |
|---|---|
| [现有模型契约](../../src/dobot_nexus/models/base.py) | VLA 的训练、动作输出、policy spec、adapter |
| [配置校验](../../src/dobot_nexus/core/config.py) | 当前强制动作维度；需要 WM 专用校验 |
| [CLI 主入口](../../src/dobot_nexus/cli/main.py) / [训练入口](../../src/dobot_nexus/cli/train.py) | 当前命令与 BCTrainer 分派；新增 WM 懒加载入口 |
| [BCTrainer](../../src/dobot_nexus/training/bc_trainer.py) | 包装模型 forward、优化器更新、EMA 与恢复调用组织 |
| [Runtime](../../src/dobot_nexus/training/runtime.py) | Local/Accelerate/FSDP 与返回类型注册边界 |
| [Artifact](../../src/dobot_nexus/core/artifacts.py) | 部署/续训分离，可借鉴但不直接作为 WM manifest |
| [Experiment](../../src/dobot_nexus/core/experiment.py) | 分布式 run 路径与配置记录 |
| [VLA 推理](../../src/dobot_nexus/inference/policy.py) | 观测/动作校验需要与视频生成分开 |
| [开发约束](../../AGENTS.md) | 用户要求新 WM 家族，实施时相应扩展每个家族的接口边界 |

## 官方来源

主要来源是 [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)。
本地取得 `main` 的以下文件并记录 SHA-256；此次 GitHub API 限流、Git HTTP 读取 HEAD 超时，
尚未解析出上游 commit。因此这些是设计核对快照，**不是已锁定的可复现依赖版本**。
P0 必须补齐具体 commit 和权重 revision 后再实现与验收。

- [TI2V-5B 配置](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py)：VAE stride、patch、采样配置。
- [I2V-A14B 配置](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_i2v_A14B.py)：双专家、VAE 和路由边界。
- [T2V-A14B 配置](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_t2v_A14B.py)：与 I2V 不同的采样/边界配置。
- [TI2V 生成](https://github.com/Wan-Video/Wan2.2/blob/main/wan/textimage2video.py)：首帧 latent 保持、token 时间步、scheduler 与 CFG。
- [I2V 生成](https://github.com/Wan-Video/Wan2.2/blob/main/wan/image2video.py)：图像条件 mask/latent、专家选择和生成循环。
- [DiT](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py)：视频 patch、时空位置与文本 cross attention。
- [Wan2.2 VAE](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/vae2_2.py)：视频编码、潜变量缩放和时序缓存。
- [许可证](https://github.com/Wan-Video/Wan2.2/blob/main/LICENSE.txt)：实现时保留原始版权和许可证。

## 事实与设计推导的界限

**官方实现事实**：5B 与 A14B 的组件、VAE 和图像条件路径不同；A14B 用实际噪声时间路由专家；
生成脚本包含推理专用的 eval/no-grad/offload 行为。

**本项目设计建议**：现有 `core/` 内的 `wm_` 基础接口、统一视频契约、flow 微调目标、LoRA 优先、A14B 分专家微调、
组件化 Artifact、任务式生成服务及机器人动作条件分支。上述内容不是官方提供的 Nexus 接口或训练配方。

原版模型输入中没有机器人关节动作协议；动作条件预测需要新的结构、数据和训练验证。
资源占用与模型质量尚未实测，规划中的检查阶段不能写成已通过的实验结果。

## 官方文件快照校验

以下 hash 用于追溯本次阅读内容；未来实现应同时固定上游 commit。

| 文件 | SHA-256 |
|---|---|
| `LICENSE.txt` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| `wan/configs/shared_config.py` | `3ae102e029d4d0e3436ebbaa9f8fd32c10e8468ae8be07facdca2e20ec1858fe` |
| `wan/configs/wan_i2v_A14B.py` | `1cddca2dd2976df3d08c602e437a8d1aec94b33b3cb4dea3187003dbe5195cd2` |
| `wan/configs/wan_t2v_A14B.py` | `a1f3a70472aece6d7a3cfe83ed39be829e957484db23069c13c7a3c35ac07a48` |
| `wan/configs/wan_ti2v_5B.py` | `26d9e7b9c555eb0900b13751267a596556f869e7aee2d1572afc2a0a4a76f4c7` |
| `wan/image2video.py` | `53f4f302f7e9bc7bca2465abd3f7ff5f0e8fa909c7dc36bd0c49abb7918d5f68` |
| `wan/modules/model.py` | `8b39115298ca7322806c19b3165b3f435a94fe4a58f0624aec24f8e7f4997432` |
| `wan/modules/vae2_2.py` | `eab5ce4aa1ce03af2978f2a8e8364c419f5fbb8535d265ac86b0b02ddcf0c1f6` |
| `wan/text2video.py` | `2f212e1f6d7ba4408a25998091b70ee35dde8bcaaeaedd3ba03d6cdab84dc683` |
| `wan/textimage2video.py` | `228f2fabf23014ed41ec6b8cd713d5be7371bb558501c71a62a0258b491a877b` |
| `wan/utils/fm_solvers_unipc.py` | `0dec8c7ed17f6f2049275c6848113314da6ccec1c8db5bdc89df43c05c6038d9` |

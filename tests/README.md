# 测试结构与执行

`tests/` 保存 CPU 模块和契约测试，不需要真实模型权重、数据集、GPU 或机器人。

```text
tests/
├── test_config.py                  # 配置合并、环境与 CLI 覆盖
├── test_data_*.py                  # 通用数据读取、统计、转换契约
├── test_runtime.py                 # local / Accelerate / FSDP 适配契约
├── test_tracking.py                # TensorBoard/W&B、主 rank 限制、同步与 SDK 清理
├── test_training_tracking.py       # 训练步数记录、随机数隔离、训练异常清理
├── test_inference_dimensions.py    # Artifact 公共维度恢复与冲突检查
├── test_checkpoint_resume.py       # Artifact、恢复、EMA 与数据进度
├── test_seed.py                    # 训练与采样随机数
├── test_training_inference.py      # 小模型 train → Artifact → infer / session
├── test_policy_session.py          # InferencePolicy 锁内输出快照与 CUDA aux 同步契约
├── test_inference_transport.py     # WebSocket、NumPy 编码、客户端、串行 worker
├── test_repository_structure.py    # 模型公共接口、依赖隔离和可移植路径
├── data/                           # 共享 quantile/zscore/delta 归一化与 resize_with_pad 契约
├── entrypoints/                    # 环境脚本、训练/推理 launcher、真机样例 CLI
│   └── atom/                       # ATOM 字段映射、相机生命周期、假设备执行与验证契约
├── algorithms/recap/              # Return、Advantage、标签、Value 作业与恢复
├── models/
│   ├── test_memory_compatibility.py # Pi0 / Pi0.5 Memory 导出与跨版本兼容契约
│   ├── pi0/                       # Pi0 模型、数据、RECAP、Memory 与 FeatureStore 语义
│   └── pi05/                      # core、数据、训练、Torch 引擎、部署、相机映射、Memory
│       ├── test_action_boundary.py # 维度约束、投影扩展、内部维度恢复、训推数据边界
│       ├── test_deploy_cameras.py  # 训练相机列映射解析与 OpenPI / Memory 路由校验
│       ├── test_recap.py           # Pi0.5 条件策略、Artifact 和推理
│       └── test_recap_config.py    # RECAP 策略配置与依赖约束
├── modules/
│   ├── memory/                    # FrameSample Memory 类型、采样、Torch 层、会话与特征库
│   ├── serving/                   # 模型无关 OpenPI 与 Memory WebSocket 协议适配器
│   └── value/                     # Value 网络、损失、预处理与依赖隔离
└── fakes.py                        # 公共 CPU 小模型；算法和模型 fixture 随各自测试目录
```

## 环境与命令

建议使用已创建的 Pi0.5 环境，安装 WebSocket extra 以运行通信测试：

```bash
python3 scripts/env/create_model_env.py --model pi05 --extra websocket
source .venvs/pi05/bin/activate
python scripts/env/check_model_env.py --model pi05 --extra websocket
python -m pip install -e '.[tracking,tensorboard]'
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m unittest discover -s tests -v
python -m compileall -q src tests
```

只运行某个职责目录或模块：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m unittest discover -s tests/entrypoints -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m unittest \
  tests.test_tracking tests.test_training_tracking \
  tests.test_inference_dimensions tests.models.pi05.test_action_boundary -v
```

标准入口是 `unittest`，不强制安装 pytest。若使用 pytest，`pyproject.toml` 默认仅收集
`tests/`。部分用例对缺失的可选依赖会 skip；检查汇总中的 skipped 数，不能把跳过当作验证通过。
CPU 检查不能证明 GPU 数值一致、推理性能或真机成功率。
显式隐藏 GPU 可防止 checkpoint 测试中的分布式保存组件自动选择不可用的 CUDA 设备。

## 测试边界

- 通信测试使用本机临时端口和 fake policy，覆盖连接限额、客户端超时、worker 取消和清理顺序、
  推理期间健康检查；不连接运行中的 policy。
- 真机入口测试会导入样例 `_runtime/`，并复制整个样例目录测试 standard/memory 的 `--help`，
  不初始化机器人环境、不打开相机或串口、不执行动作。
- 安装脚本测试使用 `--dry-run` 和 mock，验证 Pi0.5、Pi0、服务 extras、含空格路径、环境复用和
  失败诊断，不执行 pip 安装。
- 模型私有行为留在 `models/<model>/`。Pi0.5 tokenizer、归一化和训练 stats 属于数据适配测试；
  EMA/恢复属于公共 checkpoint 测试。
- 模型测试覆盖 Artifact 加载、数值配置和 backend 校验；结构测试检查跨模型私有导入及依赖隔离。

开发用的实模 smoke、多进程训练、FSDP 精度对照、GPU benchmark、真机实验脚本/报告放在
Git 忽略的 `local_script/`。
可供用户运行的 ATOM 只读部署验证流程位于 `examples/real_robot/atom_w_0759/`，运行报告写入
Git 忽略的 `runs/atom_robot/`；它不在本仓库 CPU 测试中连接硬件。
正式源码和测试不得反向导入这些工具；共享协议的每次变更仍须增加相应 CPU 契约测试。

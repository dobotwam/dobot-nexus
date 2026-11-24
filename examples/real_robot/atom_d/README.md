# ATOM D 双臂叠衣

在 PC2 的仓库根目录运行，`--weights` 指向完整部署 Artifact。

```bash
./examples/real_robot/atom_d/run_atom_d.sh \
  --weights "$HOME/step_00004000" \
  --home-start --execute --site-clear --full-episode
```

终端显示 `WAITING` 后按 Enter 开始。带 `--home-start` 时先归位，再从新观测推理。按 `h` 停止推理并归位，按 Ctrl+C 退出。归位会张开双爪。

环境配置、只读预测、录制观测回放及异步执行参数见 [ATOM D 部署指南](../../../docs/robots/atom_d_inference.md)。

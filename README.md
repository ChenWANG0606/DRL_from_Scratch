# 从 0 开始手写强化学习

这个记录作者从零实现和整理深度强化学习算法
## 目录结构

- `algorithms/`: 按算法家族拆分后的模型实现
- `runners/`: 正式训练入口与共享训练流程
- `configs/`: 配置 dataclass 与命令行参数构建
- `utils/`: 通用工具
- `docs/theory/`: 理论学习笔记
- `docs/assets/`: 文档图片资源
- `outputs/`: TensorBoard 日志与训练结果图
- `examples/`: 独立实验脚本
- `train/`: 旧训练脚本兼容入口

## 运行方式

推荐使用新的 runner 入口：

```bash
python -m runners.policy_gradient.reinforce
python -m runners.policy_gradient.ppo
python -m runners.value_based.dqn
python -m runners.value_based.rainbow
python -m runners.actor_critic.gae
```

旧入口仍保留兼容：

```bash
python train/train_reinforce.py
python train/train_DQN.py
```

## 文档索引

- 强化学习基础：`docs/theory/rl_basics.md`
- 深度强化学习实现笔记：`docs/theory/deep_rl.md`

# 🎮 CartPole DQN - 深度强化学习项目

基于 **Gymnasium** 框架实现 **DQN (Deep Q-Network)** 算法，成功训练 CartPole 平衡控制智能体。

## 📋 项目简介

本项目实现了一个完整的强化学习训练 pipeline，包括：
- ✅ DQN 算法实现（Deep Q-Network）
- ✅ 经验回放机制（Experience Replay）
- ✅ ε-greedy 探索策略
- ✅ 训练过程可视化
- ✅ 模型保存与加载
- ✅ 智能体表现视频录制

## 🎯 任务目标

**CartPole-v1** 环境：通过向左或向右推动小车来保持杆子平衡。

- **状态空间**: 4维连续空间（小车位置、速度、杆角度、角速度）
- **动作空间**: 2个离散动作（左推/右推）
- **成功标准**: 平均奖励 ≥ 475 分（连续100个episode）

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练智能体

```bash
python train.py
```

训练完成后会自动保存：
- 模型文件: `models/dqn_cartpole_YYYYMMDD_HHMMSS.pth`
- 训练曲线: `results/training_curve_YYYYMMDD_HHMMSS.png`

### 3. 测试智能体

```bash
python test.py
```

自动加载最新训练的模型，并录制视频到 `videos/` 目录。

## 📊 训练结果

### 性能指标
- **训练回合数**: ~300-500 episodes
- **达到目标时间**: 约 5-10 分钟（CPU）
- **最终平均奖励**: 475+ 分
- **成功率**: 95%+

### 学习曲线
训练过程中的奖励和损失变化曲线保存在 `results/` 目录。

## 🧠 算法原理

### DQN (Deep Q-Network)
DQN 结合了 Q-learning 和深度神经网络：

1. **神经网络**: 3层全连接网络，输入状态，输出每个动作的Q值
2. **经验回放**: 打破样本相关性，提高学习稳定性
3. **ε-greedy策略**: 平衡探索与利用

### 核心超参数
- 学习率 (lr): 0.001
- 折扣因子 (γ): 0.99
- 探索率 (ε): 1.0 → 0.01
- 批量大小: 64
- 记忆容量: 10,000

## 📁 项目结构

```
my_cartpole_project/
├── dqn_agent.py        # DQN智能体实现
├── train.py            # 训练脚本
├── test.py             # 测试脚本
├── requirements.txt    # 依赖包
├── models/             # 保存的模型
├── results/            # 训练曲线图
└── videos/             # 录制的视频
```

## 🔧 技术栈

- **Gymnasium**: 强化学习环境标准API
- **PyTorch**: 深度学习框架
- **NumPy**: 数值计算
- **Matplotlib**: 数据可视化

## 📈 改进方向

- [ ] 实现 Double DQN 减少过估计
- [ ] 添加 Dueling DQN 架构
- [ ] 实现优先经验回放 (PER)
- [ ] 尝试其他环境（LunarLander, MountainCar）

## 📝 参考资料

- [DQN 论文](https://www.nature.com/articles/nature14236)
- [Gymnasium 文档](https://gymnasium.farama.org/)
- [CartPole 环境说明](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

## 👤 作者

**MemoryWorld**
- GitHub: [@MemoryWorld](https://github.com/MemoryWorld)
- 项目链接: [Gymnasium Fork](https://github.com/MemoryWorld/Gymnasium)

## 📄 License

MIT License

---

⭐ 如果这个项目对你有帮助，欢迎 Star！


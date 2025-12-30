"""
训练脚本 - 使用DQN算法训练CartPole智能体
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
import os
from datetime import datetime

def train_dqn(num_episodes=500, render=False):
    """训练DQN智能体"""
    
    # 创建环境
    env = gym.make("CartPole-v1", render_mode="rgb_array" if render else None)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print("=" * 60)
    print("开始训练 DQN Agent on CartPole-v1")
    print("=" * 60)
    print(f"状态空间维度: {state_size}")
    print(f"动作空间大小: {action_size}")
    print(f"训练回合数: {num_episodes}")
    print()
    
    # 创建智能体
    agent = DQNAgent(state_size, action_size)
    
    # 记录训练数据
    episode_rewards = []
    episode_losses = []
    moving_avg_rewards = []
    
    # 训练循环
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = []
        
        for step in range(500):  # 最大步数
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 学习
            loss = agent.replay()
            if loss is not None:
                episode_loss.append(loss)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # 记录数据
        episode_rewards.append(total_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)
        
        # 计算移动平均奖励
        window_size = min(100, episode + 1)
        moving_avg = np.mean(episode_rewards[-window_size:])
        moving_avg_rewards.append(moving_avg)
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {total_reward:.0f} | "
                  f"Avg Reward (100): {moving_avg:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")
        
        # 检查是否达到目标
        if moving_avg >= 475:  # CartPole-v1的奖励阈值
            print(f"\n✅ 在第 {episode + 1} 回合达到目标！")
            print(f"平均奖励: {moving_avg:.2f}")
            break
    
    env.close()
    
    # 保存模型
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/dqn_cartpole_{timestamp}.pth"
    agent.save(model_path)
    
    # 绘制训练曲线
    plot_training_results(episode_rewards, moving_avg_rewards, episode_losses)
    
    return agent, episode_rewards, moving_avg_rewards


def plot_training_results(episode_rewards, moving_avg_rewards, episode_losses):
    """绘制训练结果"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 奖励曲线
    episodes = range(1, len(episode_rewards) + 1)
    ax1.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    ax1.plot(episodes, moving_avg_rewards, color='red', linewidth=2, label='Moving Average (100)')
    ax1.axhline(y=475, color='green', linestyle='--', label='Target (475)')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('DQN Training on CartPole-v1 - Rewards', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 损失曲线
    ax2.plot(episodes, episode_losses, color='orange', alpha=0.7, label='Loss')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"results/training_curve_{timestamp}.png", dpi=300)
    print(f"\n训练曲线已保存到: results/training_curve_{timestamp}.png")
    
    # plt.show()


if __name__ == "__main__":
    # 开始训练
    agent, rewards, moving_avg = train_dqn(num_episodes=500, render=False)
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"最终平均奖励: {moving_avg[-1]:.2f}")
    print(f"最高奖励: {max(rewards):.0f}")
    print("=" * 60)


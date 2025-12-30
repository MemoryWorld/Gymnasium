"""
测试脚本 - 测试训练好的DQN智能体并录制视频
"""
import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent
import glob
import os

def test_agent(model_path=None, num_episodes=10, render=True, record_video=True):
    """测试训练好的智能体"""
    
    # 创建环境
    if record_video:
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env, 
            "videos",
            episode_trigger=lambda x: True,  # 录制所有episode
            name_prefix="dqn_cartpole"
        )
    else:
        env = gym.make("CartPole-v1", render_mode="human" if render else None)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 创建智能体
    agent = DQNAgent(state_size, action_size)
    
    # 加载模型
    if model_path is None:
        # 自动查找最新的模型
        model_files = glob.glob("models/dqn_cartpole_*.pth")
        if not model_files:
            print("❌ 没有找到训练好的模型！请先运行 train.py")
            return
        model_path = max(model_files, key=os.path.getctime)
    
    print(f"加载模型: {model_path}")
    agent.load(model_path)
    agent.epsilon = 0  # 测试时不探索
    
    print("\n" + "=" * 60)
    print("开始测试训练好的智能体")
    print("=" * 60)
    
    # 测试多个episode
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # 选择动作（不探索）
            action = agent.act(state, training=False)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward:.0f}, Steps = {steps}")
    
    env.close()
    
    # 统计结果
    print("\n" + "=" * 60)
    print("测试结果统计")
    print("=" * 60)
    print(f"平均奖励: {np.mean(total_rewards):.2f}")
    print(f"最高奖励: {np.max(total_rewards):.0f}")
    print(f"最低奖励: {np.min(total_rewards):.0f}")
    print(f"标准差: {np.std(total_rewards):.2f}")
    
    if record_video:
        print(f"\n[OK] Videos saved to videos/ directory")
    
    return total_rewards


if __name__ == "__main__":
    # 测试智能体并录制视频
    test_agent(
        model_path=None,  # 自动使用最新模型
        num_episodes=5,   # 测试5个episode
        render=False,
        record_video=True
    )


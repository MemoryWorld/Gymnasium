"""
CartPole环境测试脚本
快速了解环境的观察空间、动作空间和基本运行
"""
import gymnasium as gym
import numpy as np

# 创建CartPole环境
env = gym.make("CartPole-v1", render_mode="rgb_array")

print("=" * 50)
print("CartPole-v1 环境信息")
print("=" * 50)
print(f"观察空间: {env.observation_space}")
print(f"动作空间: {env.action_space}")
print(f"最大步数: {env.spec.max_episode_steps}")
print(f"奖励阈值: {env.spec.reward_threshold}")
print()

# 运行一个随机策略的episode
print("运行随机策略测试...")
observation, info = env.reset(seed=42)
print(f"初始观察: {observation}")

total_reward = 0
step = 0

for _ in range(500):
    # 随机选择动作
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    total_reward += reward
    step += 1
    
    if terminated or truncated:
        print(f"\nEpisode结束！")
        print(f"总步数: {step}")
        print(f"总奖励: {total_reward}")
        print(f"最终观察: {observation}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        break

env.close()

print("\n" + "=" * 50)
print("测试完成！随机策略的表现很差，需要智能算法！")
print("=" * 50)


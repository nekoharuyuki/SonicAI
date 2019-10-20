import gym
import os
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# ログフォルダの生成
log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)

# 環境の生成
env = gym.make('CartPole-v1')
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

# モデルの生成
model = PPO2(MlpPolicy, env, verbose=1)

# モデルの学習
model.learn(total_timesteps=10000)

# モデルのテスト
state = env.reset()
for i in range(200):
    env.render()
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    if done:
        break
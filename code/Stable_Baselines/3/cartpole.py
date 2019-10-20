import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# 環境の生成
env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

# モデルの生成
model = PPO2(MlpPolicy, env, verbose=1)

# モデルの学習
model.learn(total_timesteps=10000)

# モデルの保存
model.save("sample")

# モデルの削除
del model


# モデルの読み込み
model = PPO2.load("sample")

# モデルのテスト
state = env.reset()
for i in range(200):
    env.render()
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    if done:
        break
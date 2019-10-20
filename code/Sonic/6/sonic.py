import retro
import os
import time
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from baselines.common.retro_wrappers import *
from stable_baselines.bench import Monitor
from util import CustomRewardAndDoneEnv, callback, log_dir
from stable_baselines.common import set_global_seeds

# 環境の生成
env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
env = SonicDiscretizer(env) # 行動空間を離散空間に変換
env = StochasticFrameSkip(env, n=4, stickprob=0.25) # スティッキーフレームスキップ
env = Downsample(env, 2) # ダウンサンプリング
env = Rgb2gray(env) # グレースケール
env = FrameStack(env, 4) # フレームスタック
env = ScaledFloatFrame(env) # 状態の正規化
env = TimeLimit(env, max_episode_steps=4500) # 5分タイムアウト
env = CustomRewardAndDoneEnv(env) # カスタム報酬関数・完了条件
env = Monitor(env, log_dir, allow_early_resets=True)
print('状態空間: ', env.observation_space)
print('行動空間: ', env.action_space)

# シードの指定
env.seed(0)
set_global_seeds(0)

# ベクトル環境の生成
env = DummyVecEnv([lambda: env])

# モデルの生成
#model = PPO2(policy=CnnPolicy, env=env, verbose=0, learning_rate=0.000025)

# モデルの読み込み
model = PPO2.load('logs/best_model.pkl', env=env, verbose=0)

# モデルの学習
print('train...')
#model.learn(total_timesteps=20000000, callback=callback)

# モデルのテスト
print('test...')
state = env.reset()
while True:
   env.render()
   time.sleep(1.0/20.0)
   action, _ = model.predict(state)
   state, reward, done, info = env.step(action)
   if done:
      env.reset()

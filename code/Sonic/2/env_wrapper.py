import retro
import os
from baselines.common.retro_wrappers import *

# 環境の生成
env = retro.make(game='SonicTheHedgehog-Genesis')
env = SonicDiscretizer(env) # 行動空間を離散空間に変換
env = StochasticFrameSkip(env, n=4, stickprob=0.25) # スティッキーフレームスキップ
env = Downsample(env, 2) # ダウンサンプリング
env = Rgb2gray(env) # グレースケール
env = FrameStack(env, 4) # フレームスタック
env = ScaledFloatFrame(env) # 状態の正規化
env = TimeLimit(env, max_episode_steps=4500) # 5分タイムアウト
print('状態空間: ', env.observation_space)
print('行動空間: ', env.action_space)

# テスト
state = env.reset()
while True:
    env.render()
    state, reward, done, info = env.step(env.action_space.sample())
    if done:
        env.reset()
import gym
import os
import numpy as np
import datetime
import pytz
from stable_baselines.results_plotter import load_results, ts2xy

# パラメータ
GOAL_X = 9600
REWARD_SCALE = 0.1


# ログフォルダの生成
log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)


# CustomRewardAndDoneラッパー
class CustomRewardAndDoneEnv(gym.Wrapper):
    # 初期化
    def __init__(self, env):
        super(CustomRewardAndDoneEnv, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    # リセット
    def reset(self, **kwargs):
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    # ステップ
    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        # 完了条件のカスタマイズ
        if info['lives'] == 2 or info['x'] > GOAL_X:
            done = True

        # 報酬関数のカスタマイズ(設定1)
        # rew = info['x'] - self._cur_x
        # self._cur_x = info['x']

        # 報酬関数のカスタマイズ(設定2)
        self._cur_x = info['x']
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)

        # スケールの調整
        rew *= REWARD_SCALE
        return obs, rew, done, info

# コールバック
best_mean_reward = -np.inf
nupdates = 1
def callback(_locals, _globals):
    global nupdates
    global best_mean_reward
    # print('callback:', nupdates)

    # 10更新毎
    if (nupdates + 1) % 10 == 0:
        # 平均エピソード長、平均報酬の取得
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            # 最近10件の平均報酬
            mean_reward = np.mean(y[-10:])

            # 平均報酬がベスト報酬以上の時はモデルを保存
            update_model = mean_reward > best_mean_reward
            if update_model:
                best_mean_reward = mean_reward
                _locals['self'].save(log_dir + 'best_model.pkl')

            # ログ
            print("time: {}, nupdates: {}, mean: {:.2f}, best_mean: {:.2f}, model_update: {}".format(
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')),
                nupdates, mean_reward/REWARD_SCALE, best_mean_reward/REWARD_SCALE, update_model))

    nupdates += 1
    return True
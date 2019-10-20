import gym

# 定数
NUM_EPISODES = 10 # 学習するエピソード数
MAX_STEPS = 500 # 1エピソードの最大ステップ数

# 環境の生成
env = gym.make('MountainCar-v0')

# 学習ループ
for episode in range(NUM_EPISODES):
    # 環境のリセット
    state = env.reset()

    # 1エピソードのループ
    for step in range(MAX_STEPS):
        # 環境の描画
        env.render()

        # 行動の取得
        action = env.action_space.sample()

        # 1ステップの実行
        state, reward, done, info = env.step(action)

        # エピソード完了時
        if done:
            print("episode:{} step:{}".format(episode, step+1))
            break
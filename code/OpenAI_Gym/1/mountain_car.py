import gym

# 環境の生成
env = gym.make("MountainCar-v0")
env.reset()

# ランダム行動
for _ in range(2000):
    env.render()
    env.step(env.action_space.sample())
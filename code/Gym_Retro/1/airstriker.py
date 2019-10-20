import retro

# 環境の生成
env = retro.make(game='Airstriker-Genesis', state='Level1')
env.reset()

# ランダム行動
for _ in range(2000):
    env.render()
    env.step(env.action_space.sample())
import gym
import gym_my_env

env = gym.make('my-env-v0')

s = env.reset()
for _ in range(10):
    env.render()
    a = env.action_space.sample()
    s, r, d, _ = env.step(a)

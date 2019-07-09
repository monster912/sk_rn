import gym

env = gym.make('Pendulum-v0')
# env = gym.make('CartPole-v0')
# env = gym.make('Seaquest-v4')

env.reset()

for _ in range(1000):
    
    env.step(env.action_space.sample()) # take a random action
    env.render()
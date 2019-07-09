import random
import numpy as np
import gym
from gym import spaces

delta = np.array([[0,-1],[0,+1],[-1,0],[+1,0]]) # UP, DOWN, LEFT, RIGHT

class MyEnv(gym.Env):

    def __init__(self):
        self.low = 0
        self.high = 3
        self.observation_space = spaces.Box(low=self.low, high=self.high, shape=(2,), dtype=np.int8)
        self.action_space = spaces.Discrete(4)

    def step(self, action):
        assert self.action_space.contains(action)
        new = self.state + delta[action]
        if self.observation_space.contains(new):
            self.state = new
        return self.state, 0, False, None

    def reset(self):
        x = random.randrange(self.high - self.low + 1) + self.low
        y = random.randrange(self.high - self.low + 1) + self.low
        self.state = np.array([x,y])
        return self.state

    def render(self):
        print(self.state)

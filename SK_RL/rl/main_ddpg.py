# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import random
import time

# Environment
import gym
import gym_maze 

# Agent
import agents


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    return None


if __name__ == '__main__':
    set_seed(0)

    # Load environment
    print('Environment: Pendulum-v0')
    env = gym.make('Pendulum-v0')

    # Load agent
    print('Agent: ddpg')
    agent = agents.load("ddpg/agent.py").Agent(env)

    # start learning and testing
    agent.learn()
    agent.test()

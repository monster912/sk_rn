# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import os
from agents.agent import AbstractAgent
from agents.common.input import observation_dim
from agents.common.replay_buffer import ReplayBuffer
from agents.dqn.DQN import DQN



# parameter setting 
train_step = 3500000
test_step = 10000

minibatch_size = 50
pre_train_step = 3
training_interval = 4
target_update_period = 10000

class Agent(AbstractAgent):

    def __init__(self, env):
        super(Agent, self).__init__(env)
        print("DQN Agent")

        self.action_dim = env.action_space.n
        self.obs_dim = observation_dim(env.observation_space)
            
        self.model = DQN(self.obs_dim, self.action_dim)

        self.replay_buffer = ReplayBuffer(minibatch_size=minibatch_size)
        
    def learn(self):
        print("Start train for {} steps".format(train_step))
        global_step = 0
        episode_num = 0

        while global_step < train_step:
            episode_num += 1

            obs = self.env.reset()  # Reset environment

            total_reward = 0
            done = False

            while (not done and global_step < train_step):

                global_step += 1

                action = self.get_action(obs, global_step)

                obs_next, reward, done, _ = self.env.step(action)

                self.train_agent(obs, action, reward, obs_next, done, global_step)

                # GUI
                self.env.render()

                obs = obs_next
                total_reward += reward

                if global_step % 10000 == 0:
                    print(global_step)

        self.model.save_network()

    def test(self, global_step=0):
        print("Start test for {} steps".format(test_step))

        global_step = 0
        episode_num = 0
        total_reward = 0

        while global_step < test_step:

            episode_num += 1

            obs = self.env.reset()  # Reset environment
            done = False

            while (not done and global_step < test_step):

                global_step += 1

                action = self.get_action(obs, global_step, False)

                obs_next, reward, done, _ = self.env.step(action)

                # GUI
                self.env.render()

                obs = obs_next
                total_reward += reward

            print("[ train_ep: {}, total reward: {} ]".format(episode_num, total_reward))
            total_reward = 0

    def get_action(self, obs, global_step, train=True):

        eps_min = 0.1
        eps_max = 1.0
        eps_decay_steps = train_step
        epsilon = max(eps_min, eps_max - (eps_max - eps_min)*global_step/eps_decay_steps)

        if train and np.random.rand(1) < epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.model.get_action(obs)

        return action

    def train_agent(self, obs, action, reward, obs_next, done, global_step):

        state = self.model.preprocess_observation(obs)
        state_next = self.model.preprocess_observation(obs_next)

        self.replay_buffer.add_to_memory((state, action, reward, state_next, done))

        if len(self.replay_buffer.replay_memory) < minibatch_size * pre_train_step:
            return None

        if global_step % training_interval == 0:
            minibatch = self.replay_buffer.sample_from_memory()
            s, a, r, s_, done = map(np.array, zip(*minibatch))
            self.model.train_network(s, a, r, s_, done, global_step)

        if global_step % target_update_period == 0:
            self.model.update_target()
            
        return


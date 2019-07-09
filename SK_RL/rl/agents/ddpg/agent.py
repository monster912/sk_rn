# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import tensorflow as tf
from agents.agent import AbstractAgent
from agents.common.input import observation_dim
from agents.common.input import action_dim
from agents.common.replay_buffer import ReplayBuffer
from agents.ddpg.DDPG_Network import DDPG


#### HYPER PARAMETERS ####
train_step = 5000
test_step = 1000

minibatch_size = 32
pre_train_step = 3
max_step_per_episode = 200

mu = 0
theta = 0.15
sigma = 0.2

class Agent(AbstractAgent):

    def __init__(self, env):
        super(Agent, self).__init__(env)
        print("DDPG Agent")

        self.action_dim = action_dim(env.action_space) ### KH: for continuous action task
        self.obs_dim = observation_dim(env.observation_space)
        self.action_max = env.action_space.high ### KH: DDPG action bound
        self.action_min = env.action_space.low  ### KH: DDPG action bound
        self.model = self.set_model()
        self.replay_buffer = ReplayBuffer(minibatch_size=minibatch_size) 

    def set_model(self):
        # model can be q-table or q-network
            
        model = DDPG(self.obs_dim, self.action_dim, self.action_max, self.action_min)       

        return model

    def learn(self):
        print("Start Learn")

        global_step = 0
        episode_num = 0

        while global_step < train_step:

            episode_num += 1
            step_in_ep = 0

            obs = self.env.reset()  # Reset environment
            total_reward = 0
            done = False
            self.noise = np.zeros(self.action_dim)

            while (not done and step_in_ep < max_step_per_episode and global_step < train_step): ### KH: reset every 200 steps

                global_step += 1
                step_in_ep += 1

                action = self.get_action(obs, global_step)

                obs_next, reward, done, _ = self.env.step(action)

                self.train_agent(obs, action, reward, obs_next, done, global_step)

                # GUI
                self.env.render()

                obs = obs_next
                total_reward += reward

            print("[ train_ep: {}, total reward: {} ]".format(episode_num, total_reward)) ### KH: train result

    def test(self, global_step=0):
        print("Test step: {}".format(global_step))

        global_step = 0
        episode_num = 0
        total_reward = 0

        while global_step < test_step:

            episode_num += 1
            step_in_ep = 0

            obs = self.env.reset()  # Reset environment
            total_reward = 0 ### KH: Added missing
            done = False

            while (not done and step_in_ep < max_step_per_episode and global_step < test_step): ### KH: reset every 200 steps

                global_step += 1
                step_in_ep += 1

                action = self.get_action(obs, global_step, False)

                obs_next, reward, done, _ = self.env.step(action)

                # GUI
                self.env.render()

                obs = obs_next
                total_reward += reward

            print("[ test_ep: {}, total reward: {} ]".format(episode_num, total_reward)) ### KH: test result
       

    def get_action(self, obs, global_step, train=True):
        # 최적의 액션 선택 + Exploration (Epsilon greedy)                                   

        action = self.model.choose_action(obs)

        if train:
            scale = 1 - global_step / train_step
            self.noise = self.ou_noise(self.noise)
            action = action + self.noise * (self.action_max - self.action_min)/2 * scale
            action = np.maximum(action, self.action_min)
            action = np.minimum(action, self.action_max)
            
        return action

    def train_agent(self, obs, action, reward, obs_next, done, step):

        self.replay_buffer.add_to_memory((obs, action, reward, obs_next, done))

        if len(self.replay_buffer.replay_memory) < minibatch_size * pre_train_step:
            return None

        minibatch = self.replay_buffer.sample_from_memory()
        s, a, r, ns, d = map(np.array, zip(*minibatch)) 

        self.model.train_network(s, a, r, ns, d, step)

        return None

    def ou_noise(self, x):
        return x + theta * (mu-x) + sigma * np.random.randn(self.action_dim) 

# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
from agents.agent import AbstractAgent
from agents.common.input import observation_dim
import time


class Agent(AbstractAgent):

    def __init__(self, env):
        super(Agent, self).__init__(env)
        print("Q-Learning Agent is created")

        self.train_step = 5000
        self.test_step = 60

        # hyper parameter setting 
        self.df = .99   # discount factor
        self.lr = 0.1   # learning rate

        # Environment information
        self.action_dim = env.action_space.n
        self.obs_dim = np.power(int(env.observation_space.high[0]+1),2)

        # Make Q-table
        self.q_table = np.zeros([self.obs_dim, self.action_dim])


    def learn(self):
        print("Start train for {} steps".format(self.train_step))
        global_step = 0

        while global_step < self.train_step:
            obs_v = self.env.reset()  # Reset environment
            obs = self.vec2scalar(obs_v)

            done = False
            while (not done and global_step < self.train_step):
                global_step += 1

                action = self.get_action(obs, global_step)

                obs_v_next, reward, done, _ = self.env.step(action)
                obs_next = self.vec2scalar(obs_v_next)

                self.train_agent(obs, action, reward, obs_next, done)
                
                # GUI    
                self.env.render()

                obs = obs_next

    def test(self, global_step=0):
        print("Start test for {} steps".format(self.test_step))

        global_step = 0
        episode_num = 0

        while global_step < self.test_step:
            episode_num += 1
            step_in_ep = 0
            total_reward = 0
            done = False

            obs_v = self.env.reset()  # Reset environment
            obs = self.vec2scalar(obs_v)

            while (not done and global_step < self.test_step):
                global_step += 1
                step_in_ep += 1

                action = self.get_action(obs, global_step, False)

                obs_next, reward, done, _ = self.env.step(action)
                obs_next =  self.vec2scalar(obs_next)

                # GUI    
                time.sleep(0.05)
                self.env.render()

                obs = obs_next
                total_reward += reward
            
            if done:
                print("[ test_ep: {}, total reward: {} ]".format(episode_num, total_reward))

    def get_action(self, obs, global_step, train=True):

        # Fill out this part

        random_action = self.env.action_space.sample()
        return random_action

    def train_agent(self, obs, action, reward, obs_next, done):

        # Fill out this part
        return None

    def vec2scalar(self, obs):
        # Flatten obs: 
        # obs [2,3] --> state_flatten 17 (= 2+5*3) where size of map is 5 by 5
        ret = int(obs[1]*(self.env.observation_space.high[0]+1) + obs[0])  
        return ret

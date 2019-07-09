# coding=utf8

import numpy as np
import tensorflow as tf
import config

FLAGS = config.flags.FLAGS

#### HYPER PARAMETERS ####
gamma = 0.99  # reward discount factor

h_critic = 16
h_actor = 16

lr_critic = 3e-3  # learning rate for the critic
lr_actor = 1e-3   # learning rate for the actor

tau = 1e-2  # soft target update rate

file_name = FLAGS.agent + FLAGS.env
save_file = "./results/nn/" + file_name
load_file = "./results/nn/" + file_name + '-5000'

class DDPG:
    def __init__(self,state_dim, action_dim, action_max, action_min):

        tf.reset_default_graph()
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_max = float(action_max)
        self.action_min = float(action_min)

        self.state_ph = tf.placeholder(dtype=tf.float32, shape = [None, self.state_dim])
        self.reward_ph = tf.placeholder(dtype=tf.float32, shape = [None])
        self.next_state_ph = tf.placeholder(dtype=tf.float32, shape = [None, self.state_dim])
        self.done_ph = tf.placeholder(dtype=tf.float32, shape=[None])

        with tf.variable_scope('actor'):
            self.action = self.generate_actor_network(self.state_ph, True)
        with tf.variable_scope('target_actor'):
            self.target_action = self.generate_actor_network(self.next_state_ph, False)
        with tf.variable_scope('critic'):
            self.qvalue = self.generate_critic_network(self.state_ph, self.action, True)
        with tf.variable_scope('target_critic'):
            self.target_qvalue = self.generate_critic_network(self.next_state_ph, self.target_action, False)

        self.a_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
        self.ta_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor')
        self.c_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
        self.tc_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic')

        q_target = tf.expand_dims(self.reward_ph, 1) + gamma * self.target_qvalue * (1 - tf.expand_dims(self.done_ph, 1))
        td_errors = q_target - self.qvalue
        critic_loss = tf.reduce_mean(tf.square(td_errors))
        self.train_critic = tf.train.AdamOptimizer(lr_critic).minimize(critic_loss, var_list = self.c_params)
        
        actor_loss = - tf.reduce_mean(self.qvalue)
        self.train_actor = tf.train.AdamOptimizer(lr_actor).minimize(actor_loss, var_list = self.a_params)

        self.soft_target_update = [[tf.assign(ta, (1-tau) * ta + tau * a), tf.assign(tc, (1-tau) * tc + tau * c)]
                                    for a, ta, c, tc in zip(self.a_params, self.ta_params, self.c_params, self.tc_params)]

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if not FLAGS.train:
            self.saver.restore(self.sess, load_file)
        

    def choose_action(self, state):
        return self.sess.run(self.action, feed_dict = {self.state_ph: state[None]})[0]

    def train_network(self, state, action, reward, next_state, done, step):

        self.sess.run(self.train_critic, feed_dict = {self.state_ph: state,
                                                      self.action: action ,
                                                      self.reward_ph: reward,
                                                      self.next_state_ph: next_state,
                                                      self.done_ph: done})
        self.sess.run(self.train_actor, feed_dict = {self.state_ph: state})
        self.sess.run(self.soft_target_update)

        if step % 1000 == 0:
            self.saver.save(self.sess, save_file, step)

    def generate_critic_network(self, state, action, trainable):

        hidden1 = tf.layers.dense(tf.concat([state,action], axis=1), h_critic, activation=tf.nn.relu, trainable=trainable)
        hidden2 = tf.layers.dense(hidden1, h_critic, activation=tf.nn.relu, trainable=trainable)
        hidden3 = tf.layers.dense(hidden2, h_critic, activation=tf.nn.relu, trainable=trainable)

        qvalue = tf.layers.dense(hidden3, 1, trainable=trainable)

        return qvalue

    def generate_actor_network(self, state, trainable):

        hidden1 = tf.layers.dense(state, h_actor, activation=tf.nn.relu, trainable=trainable)
        hidden2 = tf.layers.dense(hidden1, h_actor, activation=tf.nn.relu, trainable=trainable)
        hidden3 = tf.layers.dense(hidden2, h_actor, activation=tf.nn.relu, trainable=trainable)
        

        non_scaled_action = tf.layers.dense(hidden3, self.action_dim, activation=tf.nn.sigmoid, trainable=trainable)
        action = non_scaled_action * (self.action_max - self.action_min) + self.action_min

        return action

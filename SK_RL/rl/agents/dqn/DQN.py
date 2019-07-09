# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import numpy as np

#### HYPER PARAMETERS ####
learning_rate = 0.001
gamma = .99

n_hidden_in = 64 * 22 * 19
n_hidden = 512

class DQN:

    def __init__(self, state_dim, action_dim):

        tf.reset_default_graph()
        self.sess = tf.Session()

        self.color = 210 + 164 + 74  # 잠수함 색깔
        self.height_to = 183         # params for crop the view
        self.height_from = 7
        self.width_to = 160          
        self.width_from = 8
        self.input_channels = 1      # gray 변환 이후 channel 수
        
        input_height = self.height_to - self.height_from
        input_width = self.width_to - self.width_from

        self.action_dim = action_dim 

        self.state_ph = tf.placeholder(tf.float32, shape=[None, input_height, input_width, self.input_channels])
        self.action_ph = tf.placeholder(tf.int32, shape=[None])
        self.q_target_ph = tf.placeholder(tf.float32, shape=[None, 1])

        input_layer = self.state_ph / 128.0   # Scale pixel intensities to the [-1.0, 1.0] range
        
        with tf.variable_scope("online") as scope:
            self.online_network = self.generate_network(input_layer)

        with tf.variable_scope("target") as scope:
            self.target_network = self.generate_network(input_layer)

        q_value = tf.reduce_sum(self.online_network * tf.one_hot(self.action_ph, self.action_dim),
                                axis=1, keepdims=True)
        error = tf.abs(self.q_target_ph - q_value)
        loss = tf.reduce_mean(tf.square(error))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_online_network = optimizer.minimize(loss)

        o_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='online')
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
        self.update_target_network = [tf.assign(t, o) for o, t in zip(o_params, t_params)]
        
        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

        return


    def save_network(self):
        self.saver.save(self.sess, "./my_dqn.ckpt")

    def generate_network(self, input_layer):

        # Convolution layer
        prev_layer = tf.layers.conv2d(input_layer, 32, (8, 8), 4, "same", activation=tf.nn.relu)
        prev_layer = tf.layers.conv2d(prev_layer, 64, (4, 4), 2, "same", activation=tf.nn.relu)
        prev_layer = tf.layers.conv2d(prev_layer, 64, (3, 3), 1, "same", activation=tf.nn.relu)

        # Flatten the result of the last convolution layer for the fully connected layer
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])

        # Fully connected layer
        hidden = tf.layers.dense(last_conv_layer_flat, n_hidden, activation=tf.nn.relu)
        output = tf.layers.dense(hidden, self.action_dim)

        return output
    
    def preprocess_observation(self, obs):
        img = obs[self.height_from:self.height_to, self.width_from:self.width_to, :]  # crop
        img = img.sum(axis=2)  # gray
        img[img==self.color] = 0  # contrast improvement 
        img = (img // 3 - 128).astype(np.int8)  # normalize
        return img.reshape(self.height_to - self.height_from, self.width_to - self.width_from, self.input_channels)

    def get_action(self, obs):

        state = self.preprocess_observation(obs)
        q_values = self.online_network.eval(session=self.sess, feed_dict={self.state_ph: [state]})

        return np.argmax(q_values)

    def train_network(self, state, action, reward, state_next, done, global_step):

        next_q_values = self.target_network.eval(session=self.sess, feed_dict={self.state_ph: state_next})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True).squeeze()
        
        q_target = (reward + (1.0 - done) * gamma * max_next_q_values).reshape((-1, 1))

        self.sess.run(self.train_online_network, 
                        feed_dict={self.state_ph: state, self.action_ph: action, self.q_target_ph: q_target})        
        
        return

    def update_target(self):

        self.sess.run(self.update_target_network)        

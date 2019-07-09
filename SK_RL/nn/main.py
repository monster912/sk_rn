# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import time


learning_rate = 0.01
train_step = 10000

class NeuralNetwork:
    def __init__(self):

        tf.reset_default_graph()
        self.sess = tf.Session()

        self.input_dim = 1
        self.output_dim = 1
        self.X = tf.placeholder(tf.float32, [None, self.input_dim])
        self.Y = tf.placeholder(tf.float32, [None, self.output_dim])   

        # Generate neural network
        self.network = self.generate_network(self.X)

        # Make flow
        error = tf.square(self.Y - self.network)
        self.loss = tf.reduce_mean(error)   

        optimizer = tf.train.AdamOptimizer()
        self.training_op = optimizer.minimize(self.loss)

        # Initialize
        self.sess.run(tf.global_variables_initializer())

    def generate_network(self, X):
        n_hidden1 = 32
        n_hidden2 = 32

        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu)
        output = tf.layers.dense(hidden2, self.output_dim)

        return output

    def train_network(self, x_data, y_data):
        for step in range (train_step):
            self.sess.run(self.training_op, feed_dict={self.X: x_data, self.Y: y_data})

            if step % 1000 == 0:
                print("[{} step]\tloss: {}".format(step, self.sess.run(self.loss, feed_dict={self.X: x_data, self.Y: y_data})))

    def test_network(self, x_data, y_data):

        result, loss = self.sess.run([self.network, self.loss], feed_dict={self.X: x_data, self.Y: y_data})
        print("\nLoss: ", loss)

        t = np.arange(0.0, 2*np.pi, 0.01)
        s = np.sin(t)
        plt.plot(t, s)

        plt.plot(x_data, result, 'ro')
        plt.show()

if __name__ == '__main__':
    # Generate training data 
    x_train_data = np.random.random([500, 1]) * 2 * np.pi
    y_train_data = np.sin(x_train_data)
    # Generat test data
    x_test_data = np.random.random([50, 1]) * 2 * np.pi
    y_test_data = np.sin(x_test_data)

    nn = NeuralNetwork()
    nn.train_network(x_train_data, y_train_data)
    nn.test_network(x_test_data, y_test_data)

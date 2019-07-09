# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

flags = tf.flags

flags.DEFINE_string("env", "CartPole-v0", "Name of environment")
# flags.DEFINE_string("env", "maze-sample-5x5-v0", "Name of environment")  # Q-learning
# flags.DEFINE_string("env", "Pendulum-v0", "Name of environment")  # For DDPG

flags.DEFINE_string("agent", "random_agent", "Name of agent")
# flags.DEFINE_string("agent", "qlearn", "Name of agent")
# flags.DEFINE_string("agent", "ddpg", "Name of agent")
flags.DEFINE_boolean("train", True, "Train or test")

flags.DEFINE_integer("train_step", 2000, "Number of training step")
flags.DEFINE_integer("test_step", 100, "Number of testing step")
flags.DEFINE_integer("seed", 0, "Random seed number")

flags.DEFINE_string("folder", "default", "Folder name for result files")
flags.DEFINE_boolean("gui", True, "Enable GUI")


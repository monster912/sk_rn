import tensorflow as tf
from gym.spaces import Discrete, Box

def observation_dim(ob_space):

    if isinstance(ob_space, Discrete):
        return ob_space.n

    elif isinstance(ob_space, Box):
        return ob_space.shape[0]

    else:
        raise NotImplementedError

def action_dim(a_space): ### KH: for continuous action task

    if isinstance(a_space, Discrete):
        return a_space.n

    elif isinstance(a_space, Box):
        return a_space.shape[0]

    else:
        raise NotImplementedError
# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from abc import ABCMeta, abstractmethod


class AbstractAgent(object):

    __metaclass__ = ABCMeta

    def __init__(self, env):
        self.env = env

    @abstractmethod    
    def learn(self):
        pass
    
    @abstractmethod     
    def test(self):
        pass
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.compat.v1.keras.layers import LeakyReLU
import math
import random
import numpy as np

class DQN(Model):
    """
    Deep neural network that represents an agent.
    """
    def __init__(self,input_size,num_actions,layer_size,depth):
        super(DQN, self).__init__()
        self.layers = []
        self.layers.append(Dense(layer_size, input_shape=(input_size,)))
        for _ in range(depth):
            self.layers.append(Dense(layer_size,layer_size))
        self.layers.append(Dense(layer_size,num_actions))


    def forward(self, x):
        for i in range(0,len(self.layers) - 1):
            x = self.layers[i](x)
            x = LeakyReLU(x)
        return self.layers[len(self.layers) - 1](x)

def select_action(state, policy, model, num_actions, EPS_START, EPS_END, EPS_DECAY, steps_done, alpha, beta):
    """
    Selects whether the next action is choosen by our model or randomly
    """
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY).data.max(1)[1].view(1, 1)
    if sample <= eps_threshold:
        z = np.zeros(num_actions)
        z[random.randint(0,num_actions - 1)] = 1
        return z

    Q = model(state)
    pi0 = policy(state)
    V = tf.log((tf.pow(pi0, alpha) * tf.exp(beta * Q)).sum(1)) / beta
    pi_i = tf.pow(pi0, alpha) * tf.exp(beta * (Q - V))
    pi_i = tf.max(tf.zeros_like(pi_i) + 1e-15, pi_i)
    action = tf.random.categorical(pi_i,1)
    return action

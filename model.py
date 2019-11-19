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



def select_action(state,model_policy , distilled_policy):

    #TODO: may need to do formatting for the state
    # Run the policy
    probs = model_policy(state)

    # Obtain the most probable action for the policy
    m = tf.random.categorical(probs)
    action =  m.sample()
    model_policy.saved_actions.append(m.log_prob(action))


    # Run distilled policy
    probs0 = distilled_policy(state)

    # Obtain the most probably action for the distilled policy
    m = tf.random.categorical(probs0)
    action_tmp =  m.sample()
    distilled_policy.saved_actions.append(m.log_prob(action_tmp))

    # Return the most probable action for the policy
    return action

def finish_episode(policy, distilled, opt_policy, opt_distilled, alpha, beta, gamma):
    reward_losses = []
    distilled_losses = []
    entropy_losses = []

    alpha_const = tf.constant(alpha)
    beta_const = tf.constant(beta)

    discounts = [gamma ** i for i in reversed(range(len(policy.rewards[::-1])))]

    for log_prob_i, log_prob_0, d, r in zip(policy.saved_actions, distilled.saved_actions,
        discounts, rewards):
        reward_losses.append(-d * tf.constant(r))
        distill_losses.append(-((d*alpha_const)/beta_const) * log_prob_0)
        entropy_losses.append((d/beta)*log_prob_i)

    loss = tf.stack(reward_losses).sum() + tf.stack(entropy_losses.sum()) + torch.stack(distill_losses).sum()
    return loss
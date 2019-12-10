import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from envs.gridworld_env import gridworld_env

### Classes ###

class Policy(Model):

	def __init__(self, input_size, num_actions, layer_size=64, depth=2):
		self.input_size = input_size
		self.num_actions = num_actions
		self.layer_size = layer_size
		self.depth = depth

		self.saved_actions = []
		self.rewards = []
		self._build_graph()

	def _build_graph(self):
		self.fc_layers = []
		for _ in range(self.depth):
			self.fc_layers = Dense(self.layer_size, activation='relu')
		self.softmax_layer = Dense(self.num_actions, activation='sigmoid')

	def call(self, inputs):
		x = inputs
		for i in range(self.depth):
			x = self.fc_layers[i](x)
		action_probs = self.sofmax_layer(x)
		return action_probs


class DistralTrainer:

	def __init__(self, envs, alpha=0.5, beta=0.005, gamma=0.8,
		max_num_steps_per_episode=10, learning_rate=0.001, log_interval=100):
		pass

### Helper Functions ###

def distral_loss(policy, distilled, opt_policy, opt_distilled, alpha, beta, gamma)

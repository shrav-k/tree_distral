from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from envs.gridworld_env import GridworldEnv

tf.keras.backend.set_floatx('float64')

### Classes ###

class FullyConnected(Model):

	def __init__(self, input_size, num_actions, layer_size, depth):
		super(FullyConnected, self).__init__()
		self.input_size = input_size
		self.num_actions = num_actions
		self.layer_size = layer_size
		self.depth = depth

		self._build_graph()

	def _build_graph(self):
		self.fc_layers = []
		for _ in range(self.depth):
			self.fc_layers.append(Dense(self.layer_size, activation='relu'))
		self.softmax_layer = Dense(self.num_actions, activation='sigmoid')

	def call(self, inputs):
		#Reshape input if only one dimension
		x = tf.reshape(inputs, [-1] + [self.input_size]) if len(inputs.shape) == 1  else inputs

		for i in range(self.depth):
			x = self.fc_layers[i](x)
		action_probs = self.softmax_layer(x)
		return action_probs

class Distral2ColTrainer:

	def __init__(self, envs, alpha=0.5, beta=0.005, gamma=0.8,
		learning_rate=0.001, layer_size=64, depth=2):
		self.envs = envs
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
		self.learning_rate = learning_rate
		self.layer_size = layer_size
		self.depth = depth

		self.input_size = envs[0].observation_space.shape[0]
		self.num_actions = envs[0].action_space.n
		self.num_tasks = len(envs)

		self.f_networks, self.f_opt = self.build_f()
		self.h_networks, self.h_opt = self.build_h()

	def make_network(self):
		return FullyConnected(self.input_size, self.num_actions, self.layer_size, self.depth)

	def build_f(self):
		f_networks = [self.make_network() for _ in range(self.num_tasks)]
		f_opt = [Adam(self.learning_rate) for _ in range(self.num_tasks)]
		return f_networks, f_opt

	def build_h(self):
		h_networks = [self.make_network()]
		h_opt = [Adam(self.learning_rate)]
		return h_networks, h_opt

	def get_f(self, task_num):
		return self.f_networks[task_num], self.f_opt[task_num]

	def get_h(self, task_num):
		return self.h_networks[0], self.h_opt[0]

	def eval_policy(self, task_num, state):
		f, _ = self.get_f(task_num)
		h, _ = self.get_h(task_num)
		action_probs = [tf.math.exp(self.alpha * h(state)[0][action] + self.beta * f(state)[0][action]) for action in range(self.num_actions)]
		action_probs = action_probs / tf.reduce_sum(action_probs)
		return action_probs

	def eval_distilled(self, task_num, state):
		h, _ = self.get_h(task_num)
		action_probs = [tf.math.exp(self.alpha * h(state)[0][action]) for action in range(self.num_actions)]
		action_probs = action_probs / tf.reduce_sum(action_probs)
		return action_probs

	def run_episode(self, task_num, env, max_num_steps_per_episode):
		total_reward = 0
		duration = 0
		state = env.reset()

		with tf.GradientTape() as h_tape, tf.GradientTape() as f_tape:
			discount = 1
			reward_loss = 0
			distilled_loss = 0
			entropy_loss = 0 

			for t in range(max_num_steps_per_episode):
				policy_probs = self.eval_policy(task_num, state)
				distilled_probs = self.eval_distilled(task_num, state)

				action = tf.random.categorical([policy_probs], 1)
				action = int(action) #Cast to integer

				policy_log_prob = tf.math.log(policy_probs[action])
				distilled_log_prob = tf.math.log(distilled_probs[action])

				next_state, reward, done, _ = env.step(action)

				reward_loss += -discount * reward
				distilled_loss += -discount * self.alpha / self.beta * distilled_log_prob
				entropy_loss += discount / self.beta * policy_log_prob

				state = next_state
				discount *= self.gamma
				total_reward += reward
				duration += 1

				if done:
					break

			loss = reward_loss + distilled_loss + entropy_loss

		f, f_opt = self.get_f(task_num)
		h, h_opt = self.get_h(task_num)

		f_gradients = f_tape.gradient(loss, f.trainable_variables)
		f_opt.apply_gradients(zip(f_gradients, f.trainable_variables))

		h_gradients = h_tape.gradient(loss, h.trainable_variables)
		h_opt.apply_gradients(zip(h_gradients, h.trainable_variables))

		return loss, total_reward, duration

	def train_policy_gradient(self, num_episodes=1000, max_num_steps_per_episode=100):
		episode_rewards = [[] for i in range(num_episodes)]
		episode_durations = [[] for i in range(num_episodes)]

		for ep_num in range(num_episodes):
			for i, env in enumerate(self.envs):
				loss, total_reward, duration = self.run_episode(i, env, max_num_steps_per_episode)

				print(i, loss, total_reward)
				episode_rewards[ep_num].append(total_reward)
				episode_durations[ep_num].append(duration)

		return episode_rewards, episode_durations

## Run Model ##
def train_distral2col(envs=[GridworldEnv(5), GridworldEnv(4)]):
	distral_trainer = Distral2ColTrainer(envs)
	episode_rewards, episode_durations = distral_trainer.train()
	return episode_rewards, episode_durations

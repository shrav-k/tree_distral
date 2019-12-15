from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from envs.gridworld_env import GridworldEnv

tf.keras.backend.set_floatx('float64')

### Base Classes ###

class Policy(Model):

	def __init__(self, input_size, num_actions, layer_size, depth):
		super(Policy, self).__init__()
		self.input_size = input_size
		self.num_actions = num_actions
		self.layer_size = layer_size
		self.depth = depth

		self._build_graph()

	def _build_graph(self):
		self.fc_layers = []
		for _ in range(self.depth):
			self.fc_layers.append(Dense(self.layer_size, activation='relu'))
		self.softmax_layer = Dense(self.num_actions, activation='softmax')

	def call(self, inputs):
		#Reshape input if only one dimension
		x = tf.reshape(inputs, [-1] + [self.input_size]) if (len(inputs.shape) == 1 or len(inputs.shape) == 3) else inputs

		for i in range(self.depth):
			x = self.fc_layers[i](x)
		action_probs = self.softmax_layer(x)
		return action_probs


class BaseDistralTrainer:

	def __init__(self, params):
		self.envs = params['envs']
		self.alpha = params['alpha']
		self.beta = params['beta']
		self.gamma = params['gamma']
		self.learning_rate = params['learning_rate']
		self.layer_size = params['layer_size']
		self.depth = params['depth']


		try:
			self.input_size = self.envs[0].observation_space.shape[0]
		except:
			self.input_size = np.prod(self.envs[0].observation_space['image'].shape)

		self.num_actions = self.envs[0].action_space.n
		self.num_tasks = len(self.envs)

		self.distilled, self.distilled_opt = self.make_distilled()
		self.policies, self.policies_opt = self.make_policies()

	def make_policy(self):
		return Policy(self.input_size, self.num_actions, self.layer_size, self.depth)

	def make_distilled(self):
		raise NotImplementedError

	def make_policies(self):
		raise NotImplementedError

	def get_distilled(self, policy_num):
		raise NotImplementedError

	def get_policies(self, policy_num):
		raise NotImplementedError

	def run_episode(self, policy_num, env, max_num_steps_per_episode):
		total_reward = 0
		duration = 0

		state = env.reset()

		if isinstance(state, dict):
			state = state['image']

		distilled, distilled_opt = self.get_distilled(policy_num)
		policy, policy_opt = self.get_policies(policy_num)

		with tf.GradientTape() as policy_tape, tf.GradientTape() as distilled_tape:
			discount = 1
			reward_loss = 0
			distilled_loss = 0
			entropy_loss = 0 

			for t in range(max_num_steps_per_episode):
				policy_probs = tf.math.log(policy(state))
				distilled_probs = tf.math.log(distilled(state))

				action = tf.random.categorical(policy_probs, 1)
				action = int(action) #Cast to integer

				#distilled_action = tf.random.categorical(distilled_probs, 1)
				#distilled_action = int(distilled_action)

				try:
					policy_log_prob = policy_probs[0][action]
					distilled_log_prob = distilled_probs[0][action]
				except:
					print("Broken action: " + str(action))

				#print("Policy Prob: " +  str(policy_probs[0][action]))
				#print("Policy log prob: " + str(policy_log_prob))
				#print("distill Prob: " + str(distilled_probs[0][action]))
				#print("distill log prob: " + str(distilled_log_prob))


				next_state, reward, done, _ = env.step(action)

				if isinstance(next_state, dict):
					next_state = next_state['image']

				reward_loss += -discount * reward
				#distilled_loss += -discount * self.alpha / self.beta * distilled_log_prob
				#entropy_loss += (discount / self.beta) * policy_log_prob

				distilled_loss += -discount * .01 * distilled_log_prob
				entropy_loss += discount  * .01 * policy_log_prob

				state = next_state
				discount *= self.gamma
				total_reward += reward
				duration += 1

				if done:
					break

			loss = reward_loss + distilled_loss + entropy_loss

		policy_gradients = policy_tape.gradient(loss, policy.trainable_variables)
		policy_opt.apply_gradients(zip(policy_gradients, policy.trainable_variables))

		distilled_gradients = distilled_tape.gradient(loss, distilled.trainable_variables)
		distilled_opt.apply_gradients(zip(distilled_gradients, distilled.trainable_variables))

		return loss, total_reward, duration
		

## Specific Classes ##

class RegularDistralTrainer(BaseDistralTrainer):
	def __init__(self, params):
		super(RegularDistralTrainer, self).__init__(params)

	def make_distilled(self):
		return self.make_policy(), Adam(self.learning_rate)

	def make_policies(self):
		return [self.make_policy() for _ in range(self.num_tasks)], [Adam(self.learning_rate) for _ in range(self.num_tasks)]

	def get_distilled(self, policy_num):
		return self.distilled, self.distilled_opt

	def get_policies(self, policy_num):
		return self.policies[policy_num], self.policies_opt[policy_num]

	def train(self, num_episodes=1000, max_num_steps_per_episode=100):
		episode_rewards = [[] for i in range(num_episodes)]
		episode_durations = [[] for i in range(num_episodes)]

		for ep_num in range(num_episodes):
			for i, env in enumerate(self.envs):
				loss, total_reward, duration = self.run_episode(i, env, max_num_steps_per_episode)

				print(i, loss, total_reward)
				episode_rewards[ep_num].append(total_reward)
				episode_durations[ep_num].append(duration)

		return episode_rewards, episode_durations

class HeirarchicalDistralTrainer(BaseDistralTrainer):
	def __init__(self,params):

		self.num_distilled = params['num_distilled']
		self.set_parent_interval = params['set_parent_interval']
		self.c = params['c']

		super(HeirarchicalDistralTrainer, self).__init__(params)

		self.distilled_training_steps = np.zeros((self.num_distilled, self.num_tasks))
		self.policy_parent_map = {}	
		
	def make_distilled(self):
		return [self.make_policy() for _ in range(self.num_distilled)], [Adam(self.learning_rate) for _ in range(self.num_distilled)]

	def make_policies(self):
		return [self.make_policy() for _ in range(self.num_tasks)], [Adam(self.learning_rate) for _ in range(self.num_tasks)]

	def get_distilled(self, policy_num):
		index = self.policy_parent_map[policy_num]
		return self.distilled[index], self.distilled_opt[index]

	def get_policies(self, policy_num):
		return self.policies[policy_num], self.policies_opt[policy_num]

	def get_distilled_parent(self, policy_num, env, max_num_steps_per_episode):
		state = env.reset()

		#Extra check for new mazes
		if isinstance(state, dict):
			state = state['image']

		policy, policy_opt = self.get_policies(policy_num)
		policy_probs_list = []
		distilled_probs_list = [[] for i in range(self.num_distilled)]

		for t in range(max_num_steps_per_episode):
			policy_probs_list.append(policy(state)[0])
			for i in range(self.num_distilled):
				distilled_probs_list[i].append(self.distilled[i](state)[0])

			action = tf.random.categorical([policy_probs_list[t]], 1)
			action = int(action) #Cast to integer
			next_state, reward, done, _ = env.step(action)

			if isinstance(next_state, dict):
				next_state = next_state['image']

			if done:
				break

		policy_probs_list = np.array(policy_probs_list)
		distilled_probs_list = np.array(distilled_probs_list)
		best_loss = float("inf")
		best_parent = -1

		for i in range(self.num_distilled):
			kl_terms = [sum(p / d) for p, d in zip(policy_probs_list, distilled_probs_list[i])]
			if self.distilled_training_steps[i][policy_num] != 0:
				training_term = np.sqrt(np.sum(self.distilled_training_steps[:,policy_num]) / self.distilled_training_steps[i][policy_num])
			else:
				training_term = 0
			loss = sum(kl_terms) - self.c * training_term
			if loss < best_loss:
				best_loss = loss
				best_parent = i

		return best_parent

	def train(self, num_episodes=1000, max_num_steps_per_episode=100):
		episode_rewards = [[] for i in range(num_episodes)]
		episode_durations = [[] for i in range(num_episodes)]

		for ep_num in range(num_episodes):
			for i, env in enumerate(self.envs):
				if ep_num % self.set_parent_interval == 0:
					distilled_parent = self.get_distilled_parent(i, env, max_num_steps_per_episode)
					self.policy_parent_map[i] = distilled_parent
					print("distilled parent ", distilled_parent, " for policy ", i)
					self.distilled_training_steps[distilled_parent] += self.set_parent_interval
				loss, total_reward, duration = self.run_episode(i, env, max_num_steps_per_episode)

				print(i, loss, total_reward)
				episode_rewards[ep_num].append(total_reward)
				episode_durations[ep_num].append(duration)

		return episode_rewards, episode_durations

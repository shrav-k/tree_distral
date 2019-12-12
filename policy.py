import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from envs.gridworld_env import GridworldEnv

### Base Classes ###

class Policy(Model):

	def __init__(self, input_size, num_actions, layer_size, depth):
		super(Policy, self).__init__()
		self.input_size = input_size
		self.num_actions = num_actions
		self.layer_size = layer_size
		self.depth = depth

		self.saved_actions = []
		self.saved_rewards = []
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

	def clear(self):
		self.saved_actions.clear()
		self.saved_rewards.clear()


class BaseDistralTrainer:

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

	def train(self, num_episodes=1000, max_num_steps_per_episode=100, save=False):
		if save:
			episode_rewards = [[] for i in range(num_episodes)]
			episode_durations = [[] for i in range(num_episodes)]

		for ep_num in range(num_episodes):
			for i, env in enumerate(self.envs):
				if save:	
					total_reward = 0
					duration = 0

				state = env.reset()
				distilled, distilled_opt = self.get_distilled(i)
				policy, policy_opt = self.get_policies(i)

				with tf.GradientTape() as policy_tape, tf.GradientTape() as distilled_tape:
					discount = 1
					reward_loss = 0
					distilled_loss = 0
					entropy_loss = 0 

					for t in range(max_num_steps_per_episode):
						policy_probs = policy(state)
						distilled_probs = distilled(state)

						action = tf.math.argmax(policy_probs, axis=1) # Change to sample from distribution

						#Cast to integer
						action = int(action)

						policy_log_prob = tf.math.log(policy_probs[0][action])
						distilled_log_prob = tf.math.log(distilled_probs[0][action])


						next_state, reward, done, _ = env.step(action)

						reward_loss += -discount * reward
						distilled_loss += -discount * self.alpha / self.beta * distilled_log_prob
						entropy_loss += discount / self.beta * policy_log_prob

						discount *= self.gamma
						if save:
							total_reward += reward
							duration += 1

						if done:
							break

					loss = reward_loss + distilled_loss + entropy_loss
					if ep_num % 10 == 0:
						print(i, loss, total_reward)

				policy_gradients = policy_tape.gradient(loss, policy.trainable_variables)
				policy_opt.apply_gradients(zip(policy_gradients, policy.trainable_variables))

				distilled_gradients = distilled_tape.gradient(loss, distilled.trainable_variables)
				distilled_opt.apply_gradients(zip(distilled_gradients, distilled.trainable_variables))

				if save:
					episode_rewards[ep_num].append(total_reward)
					episode_durations[ep_num].append(duration)

		if save:
			return episode_rewards, episode_durations

## Specific Classes ##

class RegularDistralTrainer(BaseDistralTrainer):
	def __init__(self, envs, alpha=0.5, beta=0.005, gamma=0.8,
		learning_rate=0.001, layer_size=64, depth=2):
		super(RegularDistralTrainer, self).__init__(envs, alpha, beta, gamma, learning_rate, layer_size, depth)

	def make_distilled(self):
		return self.make_policy(), Adam(self.learning_rate)

	def make_policies(self):
		return [self.make_policy() for _ in range(self.num_tasks)], [Adam(self.learning_rate) for _ in range(self.num_tasks)]

	def get_distilled(self, policy_num):
		return self.distilled, self.distilled_opt

	def get_policies(self, policy_num):
		return self.policies[policy_num], self.policies_opt[policy_num]

class HeirarchicalDistralTrainer(BaseDistralTrainer):
	def __init__(self, envs, alpha=0.5, beta=0.005, gamma=0.8,
		learning_rate=0.001, layer_size=64, depth=2):
		super(HeirarchicalDistralTrainer, self).__init__(envs, alpha, beta, gamma, learning_rate, layer_size, depth)

	def make_distilled(self):
		raise NotImplementedError

	def make_policies(self):
		raise NotImplementedError

	def get_distilled(self, policy_num):
		raise NotImplementedError

	def get_policies(self, policy_num):
		raise NotImplementedError

### Run Model ###
def train_distral(envs=[GridworldEnv(5), GridworldEnv(4)]):
	distral_trainer = RegularDistralTrainer(envs)
	episode_rewards, episode_durations = distral_trainer.train(save=True)
	print(episode_rewards, episode_durations)

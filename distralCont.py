from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from envs.gridworld_env import GridworldEnv
import random
import copy
import gym

tf.keras.backend.set_floatx('float64')


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        print(size)
        print(type(size))
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class Policy(Model):
    def __init__(self, input_size, num_actions, layer_size, depth,seed = 3):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.seed = random.seed(seed)
        self.noise = OUNoise(num_actions,self.seed)
        self.num_actions = num_actions
        self.layer_size = layer_size
        self.depth = depth
        self._build_graph()

    def _build_graph(self):
        self.fc_layers = []
        for _ in range(self.depth):
            self.fc_layers.append(Dense(self.layer_size, activation='relu'))
        self.tanh_layer = Dense(self.num_actions, activation='tanh')

    def call(self, inputs):
        # Reshape input if only one dimension
        x = tf.reshape(inputs, [-1] + [self.input_size]) if len(inputs.shape) == 1  else inputs

        for i in range(self.depth):
            x = self.fc_layers[i](x)
        action_probs = self.tanh_layer(x)
        return action_probs


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
        self.num_actions = envs[0].action_space
        self.num_tasks = len(envs)

        self.distilled, self.distilled_opt = self.make_distilled()
        self.policies, self.policies_opt = self.make_policies()

    def make_policy(self,seed):
        return Policy(self.input_size, self.num_actions, self.layer_size, self.depth,seed)

    def make_distilled(self):
        raise NotImplementedError

    def make_policies(self):
        raise NotImplementedError

    def get_distilled(self, policy_num):
        raise NotImplementedError

    def get_policies(self, policy_num):
        raise NotImplementedError

    def run_episode(self, policy_num, env, max_num_steps_per_episode,add_noise = True):
        total_reward = 0
        duration = 0

        state = env.reset()
        distilled, distilled_opt = self.get_distilled(policy_num)
        policy, policy_opt = self.get_policies(policy_num)

        with tf.GradientTape() as policy_tape, tf.GradientTape() as distilled_tape:
            discount = 1
            reward_loss = 0
            distilled_loss = 0

            for t in range(max_num_steps_per_episode):
                policy_out = policy(state)
                distilled_out = distilled(state)

                # action = tf.random.categorical(policy_probs, 1)
                # action = int(action) #Cast to integer

                # policy_log_prob = tf.math.log(policy_probs[0][action])
                # distilled_log_prob = tf.math.log(distilled_probs[0][action])
                if add_noise:
                    policy_out += policy.noise.sample()
                    policy_out = np.clip(policy_out,-1,1)

                next_state, reward, done, _ = env.step(policy_out)

                reward_loss += -discount * reward
                distilled_loss += -discount * self.alpha / self.beta * tf.keras.losses.MSE(policy_out, distilled_out)
                #add += discount / self.beta * policy_log_prob

                state = next_state
                discount *= self.gamma
                total_reward += reward
                duration += 1

                if done:
                    break

            loss = reward_loss + distilled_loss

        policy_gradients = policy_tape.gradient(loss, policy.trainable_variables)
        policy_opt.apply_gradients(zip(policy_gradients, policy.trainable_variables))

        distilled_gradients = distilled_tape.gradient(loss, distilled.trainable_variables)
        distilled_opt.apply_gradients(zip(distilled_gradients, distilled.trainable_variables))

        return loss, total_reward, duration

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


## Specific Classes ##

class RegularDistralTrainer(BaseDistralTrainer):
    def __init__(self, envs, alpha=0.5, beta=0.005, gamma=0.8,
                 learning_rate=0.001, layer_size=64, depth=2):
        super(RegularDistralTrainer, self).__init__(envs, alpha, beta, gamma, learning_rate, layer_size, depth)

    def make_distilled(self):
        return self.make_policy(888), Adam(self.learning_rate)

    def make_policies(self):
        return [self.make_policy(seed = i) for i in range(self.num_tasks)], [Adam(self.learning_rate) for _ in
                                                                     range(self.num_tasks)]

    def get_distilled(self, policy_num):
        return self.distilled, self.distilled_opt

    def get_policies(self, policy_num):
        return self.policies[policy_num], self.policies_opt[policy_num]




### Run Model ###
def train_distral(envs=[gym.make('Humanoid-v2'),gym.make('Humanoid-v2'),gym.make('Humanoid-v2')]):
    distral_trainer = RegularDistralTrainer(envs)
    episode_rewards, episode_durations = distral_trainer.train()
    return episode_rewards, episode_durations

train_distral()
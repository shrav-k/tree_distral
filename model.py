from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.compat.v1.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
import math
import random
import numpy as np

class DQN(Model):
    """
    Deep neural network that represents an agent.
    """
    saved_actions = []
    rewards = []
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
    
    with tf.GradientTape() as tape:
        loss = tf.stack(reward_losses).sum() + tf.stack(entropy_losses.sum()) + tf.stack(distill_losses).sum()

    policy_gradients = tape.gradient(loss, policy.trainable_variables)
    opt_policy.apply_gradients(zip(policy_gradients, policy.trainable_variables))

    distilled_gradients = tape.gradient(loss, distilled.trainable_variables)
    opt_distilled.apply_gradients(zip(distilled_gradients, distilled.trainable_variables))

def trainDistral(file_name="Distral_1col", list_of_envs=[GridworldEnv(5), GridworldEnv(4)], batch_size=128,
                     gamma=0.80, alpha=0.5,
                     beta=0.005, num_episodes=1000,
                     max_num_steps_per_episode=10, learning_rate=0.001,
                     memory_replay_size=10000, memory_policy_size=1000,log_interval = 100):


    # Specify Environment conditions
    input_size = list_of_envs[0].observation_space.shape[0]
    num_actions = list_of_envs[0].action_space.n
    tasks = len(list_of_envs)

    # Define our set of policies, including distilled one
    models = [DQN(input_size, num_actions,64,2) for _ in range(tasks + 1)]
    optimizers = [Adam(learning_rate) for model in models]

    # Store the total rewards
    episode_rewards = [[] for i in range(num_episodes)]
    episode_duration = [[] for i in range(num_episodes)]

    for i_episode in range(num_episodes):

        # For each one of the envs
        for i_env, env in enumerate(list_of_envs):

            # Initialize state of envs
            state = env.reset()

            # Store total reward per environment per episode
            total_reward = 0

            # Store duration of each episode per env
            duration = 0

            for t in range(max_num_steps_per_episode):

                # Run our policy
                action = select_action(state, models[i_env + 1], models[0])

                next_state, reward, done, _ = env.step(action)
                models[i_env + 1].rewards.append(reward)
                total_reward += reward
                duration += 1

                if done:
                    break

                # Update state
                state = next_state

            episode_rewards[i_episode].append(total_reward)
            episode_duration[i_episode].append(duration)

            # Distill for each environment
            finish_episode(models[i_env + 1], models[0], optimizers[i_env + 1], optimizers[0], alpha, beta, gamma)

        if i_episode % log_interval == 0:
            for i in range(tasks):
                print('Episode: {}\tEnv: {}\tDuration: {}\tTotal Reward: {:.2f}'.format(
                    i_episode, i, episode_duration[i_episode][i], episode_rewards[i_episode][i]))

    np.save(file_name + '-distral0-rewards', episode_rewards)
    np.save(file_name + '-distral0-duration', episode_duration)

    print('Completed')

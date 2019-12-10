from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import math
import random
import numpy as np
from envs.gridworld_env import GridworldEnv


class DQN(torch.nn.Module):
    """
    Deep neural network that represents an agent.
    """
    saved_actions = []
    rewards = []

    def __init__(self, input_size, num_actions, layer_size = 64):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(input_size, layer_size)
        self.l2 = nn.Linear(layer_size, num_actions)



    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)


def select_action(state, model_policy, distilled_policy):
    # TODO: may need to do formatting for the state
    #Reformat state
    state = torch.from_numpy(state).float()

    # Run the policy
    probs = model_policy(state)

    # Obtain the most probable action for the policy
    m = Categorical(probs)
    action = m.sample()
    model_policy.saved_actions.append(m.log_prob(action))

    # Run distilled policy
    probs0 = distilled_policy(state)

    # Obtain the most probably action for the distilled policy
    m = Categorical(probs0)
    #action_tmp = m.sample()
    distilled_policy.saved_actions.append(m.log_prob(probs0[action]))

    # Return the most probable action for the policy
    return action


def finish_episode(policy, distilled, opt_policy, opt_distilled, alpha, beta, gamma):
    ### Calculate loss function according to Equation 1

    ## Store three type of losses
    reward_losses = []
    distill_losses = []
    entropy_losses = []

    # Give format
    alpha = torch.Tensor([alpha])
    beta = torch.Tensor([beta])

    # Retrive distilled policy actions
    distill_actions = distilled.saved_actions

    # Retrieve policy actions and rewards
    policy_actions = policy.saved_actions
    rewards = policy.rewards

    # Obtain discounts
    R = 1.
    discounts = []
    for r in policy.rewards[::-1]:
        R *= gamma
        discounts.insert(0, R)

    discounts = torch.Tensor(discounts)
    # print(discounts)

    for log_prob_i, log_prob_0, d, r in zip(policy_actions, distill_actions, discounts, rewards):
        reward_losses.append(-d * torch.Tensor([r]))
        distill_losses.append(-((d * alpha) / beta) * log_prob_0)
        entropy_losses.append((d / beta) * log_prob_i)


    # Perform optimization step
    opt_policy.zero_grad()
    opt_distilled.zero_grad()

    loss = torch.stack(reward_losses).sum() + torch.stack(entropy_losses).sum() + torch.stack(distill_losses).sum()

    loss.backward(retain_graph=True)

    opt_policy.step()
    opt_distilled.step()

    # Clean memory
    del policy.rewards[:]
    del policy.saved_actions[:]
    del policy.saved_actions[:]

def trainDistral(file_name="Distral_1col", list_of_envs=[GridworldEnv(5), GridworldEnv(4)], batch_size=128,
                 gamma=0.80, alpha=0.5,
                 beta=0.005, num_episodes=1000,
                 max_num_steps_per_episode=10, learning_rate=0.001,
                 memory_replay_size=10000, memory_policy_size=1000, log_interval=100):
    # Specify Environment conditions
    input_size = list_of_envs[0].observation_space.shape[0]
    num_actions = list_of_envs[0].action_space.n
    tasks = len(list_of_envs)

    # Define our set of policies, including distilled one
    models = torch.nn.ModuleList([DQN(input_size, num_actions,layer_size=32) for _ in range(tasks + 1)])
    optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in models]

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

                next_state, reward, done, _ = env.step(action.item())
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

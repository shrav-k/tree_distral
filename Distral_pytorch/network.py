import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory_replay import Transition
from itertools import count
from torch.distributions import Categorical

class DQN(nn.Module):
    """
    Deep neural network with represents an agent.
    """
    def __init__(self, input_size, num_actions):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_size, 50)
        self.linear2 = nn.Linear(50, 50)
        self.head = nn.Linear(50, num_actions)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        return self.head(x)

class PolicyNetwork(nn.Module):
    """
    Deep neural network which represents policy network.
    """
    def __init__(self, input_size, num_actions):
        super(PolicyNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, 50)
        self.linear2 = nn.Linear(50, 50)
        self.head = nn.Linear(50, num_actions)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        return F.softmax(self.head(x))

def select_action(state, policy, model, alpha, beta):
    """
    Selects whether the next action is choosen by our model or randomly
    """

    #TODO: Make sure that I am formatting the state correctly
    Q = model(state)
    pi0 = policy(state)
    # print(pi0.data.numpy())
    V = torch.log((torch.pow(pi0, alpha) * torch.exp(beta * Q)).sum(1) ) / beta

    pi_i = torch.pow(pi0, alpha) * torch.exp(beta * (Q - V))
    m = Categorical(pi_i)

    #TODO: Probably need to fix this
    action = m.sample()
    return action

def optimize_policy(policy, optimizer, memories, batch_size,
                    num_envs, gamma):
    loss = 0
    for i_env in range(num_envs):
        size_to_sample = np.minimum(batch_size, len(memories[i_env]))
        transitions = memories[i_env].policy_sample(size_to_sample)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.tensor(torch.cat(batch.state))
        time_batch = torch.tensor(torch.cat(batch.time))
        actions = np.array([action.numpy() for action in batch.action])

        cur_loss = (torch.pow(torch.tensor([gamma]), time_batch) * torch.log(policy(state_batch)[:, actions])).sum()
        loss -= cur_loss

    optimizer.zero_grad()
    loss.backward()

    # for param in policy.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()

def optimize_model(policy, model, optimizer, memory, batch_size, alpha, beta, gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))


    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = torch.tensor(torch.cat([s for s in batch.next_state if s is not None]))

    #TODO: This is going to need a finer look(especially action_batch)
    state_batch = torch.tensor(torch.cat(batch.state))
    action_batch = torch.tensor(torch.stack(batch.action)).view(-1,1)
    reward_batch = torch.tensor(torch.cat(batch.reward))

    #TODO: Make sure that this is working correctly
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = model(state_batch)
    state_action_values = state_action_values.gather(1, action_batch)


    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.tensor(torch.zeros(batch_size))
    next_state_values[non_final_mask] = torch.log((torch.pow(policy(non_final_next_states), alpha) * (torch.exp(beta * model(non_final_next_states)) + 1e-16)).sum(1)) / beta

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = F.mse_loss(state_action_values, expected_state_action_values)


    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in model.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()

def kl_divergence(policy, model, memory, batch_size, alpha, beta):
    if len(memory) < batch_size:
        return 0
    transitions = memory.sample(batch_size)
    kl_loss = 0
    for t in transitions:
        state = t[0]

        Q = model(state)
        pi0 = policy(state)
        # print(pi0.data.numpy())
        V = torch.log((torch.pow(pi0, alpha) * torch.exp(beta * Q)).sum(1) ) / beta

        pi_i = torch.pow(pi0, alpha) * torch.exp(beta * (Q - V))
        temp = sum([prob_i / prob_0 for prob_i, prob_0 in zip(pi_i, pi0)])
        kl_loss += torch.sum(temp)
    return kl_loss / batch_size

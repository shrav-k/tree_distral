import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from envs.gridworld_env import GridworldEnv
from memory_replay import Transition


### Classes ###

class DQN(Model):

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
		self.head = Dense(self.num_actions)

	def call(self, inputs):
		#Reshape input if only one dimension
		x = tf.reshape(inputs, [-1] + [self.input_size]) if len(inputs.shape) == 1  else inputs

		for i in range(self.depth):
			x = self.fc_layers[i](x)
		action_probs = self.head(x)
		return action_probs

	def clear(self):
		self.saved_actions.clear()
		self.saved_rewards.clear()



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


def select_action(state, policy, model, num_actions,EPS_START, EPS_END, EPS_DECAY, steps_done, alpha, beta):

    Q = model(state)
    pi0 = policy(state)

    #TODO: Make sure this correct
    V = tf.math.log((tf.math.pow(pi0, alpha) * tf.math.exp(beta * Q)).sum(1)) / beta

    pi_i = tf.math.pow(pi0, alpha) * tf.math.exp(beta * (Q - V))

    #TODO: Make sure correct(may need to wrap pi_i in list)
    action = tf.random.categorical(pi_i,1)

    return action

#TODO: ADD gradient tape
def optimize_policy(policy, optimizer, memories, batch_size,num_envs, gamma):
    loss = 0
    for i_env in range(num_envs):
        size_to_sample = np.minimum(batch_size, len(memories[i_env]))
        transitions = memories[i_env].policy_sample(size_to_sample)
        batch = Transition(*zip(*transitions))

        #TODO: MAKE SURE CORRECT
        state_batch = tf.concat(batch.state)
        time_batch = tf.concat(batch.time)

        #TODO: POSSIBLE PROBLEMS HERE
        actions = np.array([action.numpy()[0][0] for action in batch.action])

        cur_loss = (tf.math.pow(tf.constant(gamma), time_batch) * tf.math.log(policy(state_batch)[:, actions])).sum()
        loss -= cur_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def optimize_model(policy, model, optimizer, memory, batch_size, alpha, beta, gamma):


    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)


    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)

    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(batch_size).type(Tensor))
    next_state_values[non_final_mask] = torch.log(
        (torch.pow(policy(non_final_next_states), alpha)
         * (torch.exp(beta * model(non_final_next_states)) + 1e-16)).sum(1)) / beta
    try:
        print(next_state_values)
        np.isnan(next_state_values.sum().data[0])
    except Exception:
        pass
        # print("next_state_values:", next_state_values)
        # print(policy(non_final_next_states))
        # print(torch.exp(beta * model(non_final_next_states)))
        # print(model(non_final_next_states))
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
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

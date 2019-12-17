import torch.optim as optim
import torch
import math
import numpy as np
from memory_replay import ReplayMemory, Transition
from network import DQN, select_action, optimize_model, optimize_policy, PolicyNetwork, kl_divergence
import sys
sys.path.append('../')
from envs.gridworld_env import GridworldEnv
import gym
import gym_minigrid


def trainD(file_name="Distral_1col", list_of_envs=[GridworldEnv(5),
            GridworldEnv(4), GridworldEnv(6)], batch_size=128, gamma=0.999, alpha=0.5,
            beta=.5, eps_start=0.9, eps_end=0.05, eps_decay=5, num_episodes=200,
            max_num_steps_per_episode=500, learning_rate=0.001,
            memory_replay_size=10000, memory_policy_size=1000,
            num_policies=2, c=1, episode_interval=10):
    """
    Soft Q-learning training routine. Retuns rewards and durations logs.
    Plot environment screen
    """
    num_actions = list_of_envs[0].action_space.n
    try:
        input_size = list_of_envs[0].observation_space.shape[0]
    except:
        input_size = np.prod(list_of_envs[0].observation_space['image'].shape)

    num_envs = len(list_of_envs)
    # policy = PolicyNetwork(input_size, num_actions)
    policies = [PolicyNetwork(input_size, num_actions) for _ in range(num_policies)]
    models = [DQN(input_size,num_actions) for _ in range(0, num_envs)]   ### Add torch.nn.ModuleList (?)
    memories = [ReplayMemory(memory_replay_size, memory_policy_size) for _ in range(0, num_envs)]

    policy_episodes = np.ones((len(models), len(policies))) * episode_interval
    policy_map = [i * num_policies // len(models) for i in range(len(models))]

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")
        policy.cuda()
        for model in models:
            model.cuda()

    optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in models]
    # policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    policy_optimizers = [optim.Adam(policy.parameters(), lr=learning_rate) for policy in policies]

    episode_durations = [[] for _ in range(num_envs)]
    episode_rewards = [[] for _ in range(num_envs)]

    #The current total reward for the environment
    current_env_reward = [0 for _ in range(num_envs)]


    steps_done = np.zeros(num_envs)
    episodes_done = np.zeros(num_envs)
    current_time = np.zeros(num_envs)

    # Initialize environments
    states = []
    for env in list_of_envs:
        #Format = tensor([[-0.7778, -0.5000,  0.0000]])
        state =  env.reset()

        #If from the gym mini grid environment
        if isinstance(state,dict):
            state = state['image']

        state = torch.from_numpy(state).type(torch.FloatTensor).view(-1,input_size)
        states.append(state)

    while np.min(episodes_done) < num_episodes:

        # Optimization is given by alterating minimization scheme:
        #   1. do the step for each env
        #   2. do one optimization step for each env using "soft-q-learning".
        #   3. do one optimization step for the policy

        for i_env, env in enumerate(list_of_envs):
            policy = policies[policy_map[i_env]]
            policy_optimizer = policy_optimizers[policy_map[i_env]]

            action = select_action(states[i_env], policy, models[i_env], alpha, beta)
            action = action.data[0]

            steps_done[i_env] += 1
            current_time[i_env] += 1
            next_state_tmp, reward, done, _ = env.step(action)

            #adjust current reward of environment
            current_env_reward[i_env] += reward

            reward = torch.tensor([reward]).type(torch.FloatTensor)

            # Observe new state
            # If from the gym mini grid environment
            if isinstance(next_state_tmp, dict):
                next_state_tmp = next_state_tmp['image']

            next_state = torch.from_numpy( next_state_tmp ).type(torch.FloatTensor).view(-1,input_size)


            if done:
                next_state = None

            # Store the transition in memory
            time = torch.tensor([current_time[i_env]])
            memories[i_env].push(states[i_env], action, next_state, reward, time)

            # Perform one step of the optimization (on the target network)
            optimize_model(policy, models[i_env], optimizers[i_env], memories[i_env], batch_size, alpha, beta, gamma)

            # Update state
            states[i_env] = next_state

            #TODO: make sure this is the correct way to handle max_num_steps_per_episode
            # Check if agent reached target
            if done or current_time[i_env] >= max_num_steps_per_episode:
                print("ENV:", i_env, "iter:", episodes_done[i_env],
                    "\treward:", "{0:0.3f}".format(current_env_reward[i_env]) ,
                    "\tit:", current_time[i_env], "\texp_factor:", eps_end +
                    (eps_start - eps_end) * math.exp(-1. * episodes_done[i_env] / eps_decay))

                reset_state = env.reset()
                if isinstance(reset_state, dict):
                    reset_state = reset_state['image']

                states[i_env] = torch.from_numpy( reset_state ).type(torch.FloatTensor).view(-1,input_size)
                episodes_done[i_env] += 1
                episode_durations[i_env].append(current_time[i_env])
                current_time[i_env] = 0
                episode_rewards[i_env].append(current_env_reward[i_env])
                current_env_reward[i_env] = 0
                #TODO: Put back for original environment
                #episode_rewards[i_env].append(env.episode_total_reward)

                if episodes_done[i_env] % episode_interval == 0:
                    best_loss = float("inf")
                    best_policy = -1
                    for i in range(len(policies)):
                        kl_loss = kl_divergence(policies[i], models[i_env], memories[i_env], batch_size, alpha, beta)
                        exploration_loss = c * np.sqrt(np.sum(policy_episodes[i_env]) / policy_episodes[i_env][i])
                        loss = kl_loss - exploration_loss
                        if loss < best_loss:
                            print(loss, i)
                            best_loss = loss
                            best_policy = i
                    policy_map[i_env] = best_policy
                    policy_episodes[i_env][best_policy] += episode_interval
                    print(policy_map)
                    print(policy_episodes)

        optimize_policy(policy, policy_optimizer, memories, batch_size,num_envs, gamma)

    np.save(file_name + '-distral-2col-rewards', episode_rewards)
    np.save(file_name + '-distral-2col-durations', episode_durations)

    return models, policy, episode_rewards, episode_durations

def empty_room(size = 6, wrapper = None,n = 4):
    if size != 5 and size != 6:
        print("Invalid Size")
        return
    env_type = "MiniGrid-Empty-Random-" + str(size)+ "x" + str(size) +"-v0"
    if wrapper is None:
        envs = [gym.make(env_type) for _ in range(n)]
    else:
        envs = [wrapper(gym.make(env_type)) for _ in range(n)]
    return envs



trainD(list_of_envs=empty_room(n=3))
#trainD()
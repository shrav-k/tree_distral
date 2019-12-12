import policy
import gym
import gym_maze
#import gym_sokoban

env_n = 2
environment = "maze"

if environment == 'maze':
    envs = [gym.make("maze-random-10x10-plus-v0") for i in range(env_n)]
policy.train_distral(envs)

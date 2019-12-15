import policy
import gym
import gym_minigrid
from gym_minigrid.wrappers import FlatObsWrapper,RGBImgObsWrapper, OneHotPartialObsWrapper
import numpy as np
from time import gmtime, strftime

#https://github.com/maximecb/gym-minigrid?fbclid=IwAR2FTwWfbP5W-VWNJ8b13jvyzK09bbyINaISvswWJgqZlyORr-4raZWYess

#Size can be 5 or 6
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

def two_room(wrapper = None,n = 4):
    if wrapper is None:
        envs = [gym.make("MiniGrid-MultiRoom-N2-S4-v0") for _ in range(n)]
    else:
        envs = [wrapper(gym.make("MiniGrid-MultiRoom-N2-S4-v0")) for _ in range(n)]
    return envs

def four_rooms(wrapper = None,n = 4):
    if wrapper is None:
        envs = [gym.make("MiniGrid-FourRooms-v0") for _ in range(n)]
    else:
        envs = [wrapper(gym.make("MiniGrid-FourRooms-v0")) for _ in range(n)]
    return envs

def unlock(wrapper = None,n = 4):
    if wrapper is None:
        envs = [gym.make("MiniGrid-Unlock-v0") for _ in range(n)]
    else:
        envs = [wrapper(gym.make("MiniGrid-Unlock-v0")) for _ in range(n)]
    return envs

def unlock_pick_up(wrapper = None,n = 4):
    if wrapper is None:
        envs = [gym.make("MiniGrid-UnlockPickup-v0") for _ in range(n)]
    else:
        envs = [wrapper(gym.make("MiniGrid-UnlockPickup-v0")) for _ in range(n)]
    return envs

def simple_crossing_env(wrapper = None,n = 4):
    if wrapper is None:
        envs = [gym.make("MiniGrid-SimpleCrossingS9N3-v0") for _ in range(n)]
    else:
        envs = [wrapper(gym.make("MiniGrid-SimpleCrossingS9N3-v0")) for _ in range(n)]
    return envs

def lava_crossing_env(wrapper = None,n = 4):
    if wrapper is None:
        envs = [gym.make("MiniGrid-LavaCrossingS9N2-v0") for _ in range(n)]
    else:
        envs = [wrapper(gym.make("MiniGrid-LavaCrossingS9N2-v0")) for _ in range(n)]
    return envs

def lava_gap(wrapper = None,n = 4):
    if wrapper is None:
        envs = [gym.make("MiniGrid-LavaGapS7-v0") for _ in range(n)]
    else:
        envs = [wrapper(gym.make("MiniGrid-LavaGapS7-v0")) for _ in range(n)]
    return envs

cKL = 1
cEnt = 1


alpha = cKL / (cKL + cEnt)
beta = 1 / (cKL + cEnt)

print("alpha: " + str(alpha))
print("beta: " + str(beta))


params = {
"tree" : False,
"envs" : empty_room(),
"alpha" : alpha,
"beta" : beta,
"gamma" : 0.8,
"learning_rate" : 0.001,
"layer_size" : 64,
"depth" : 2,
"num_distilled" : 2,
"set_parent_interval" : 10,
"c" : 0.5,
}




def run_experiment(params,num_episodes = 10000, max_episode = 10):
    if(params['tree']):
        distral_trainer = policy.HeirarchicalDistralTrainer(params)
        episode_rewards, episode_durations = distral_trainer.train(num_episodes,max_episode)
        np.save(str(params) + strftime("%Y-%m-%d %H:%M:%S", gmtime()),episode_rewards)
        return episode_rewards, episode_durations
    else:
        tree_trainer = policy.RegularDistralTrainer(params)
        episode_rewards, episode_durations = tree_trainer.train(num_episodes,max_episode)
        np.save(str(params) + strftime("%Y-%m-%d %H:%M:%S", gmtime()),episode_rewards)
        return episode_rewards, episode_durations


run_experiment(params)




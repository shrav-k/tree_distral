from trainingDistral2col0 import trainD
import gym
import gym_minigrid

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

batch_size = 128
gamma = 0.999
alpha = 0.5
beta = .5
alpha = cKL / (cKL + cEnt)
beta = 1 / (cKL + cEnt)
num_episodes = 200,
max_num_steps_per_episode = 500
learning_rate = 0.001,
memory_replay_size = 10000
memory_policy_size = 1000



def run_experiment():
    #envs = empty_room(n=4)
    #trainD(list_of_envs=envs)
    trainD()

run_experiment()
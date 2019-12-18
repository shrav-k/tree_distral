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
#alpha = 0.5
#beta = 0.5
alpha = cKL / (cKL + cEnt)
beta = 1 / (cKL + cEnt)
num_episodes = 200,
max_num_steps_per_episode = 500
learning_rate = 0.001,
memory_replay_size = 10000
memory_policy_size = 1000


#RUN 4 - 9 for the grid world runs


def run_experiment():
    lrs = [.001,.005,.01]
    gammas = [0.999,0.99,0.95]
    cs = [(1,1),(1,2),(2,1)]
    for lr in lrs:
        for g in gammas:
            for c in cs:
                alpha = c[0] / (c[0] + c[1])
                beta = 1 / (c[0] + c[1])
                trainD(alpha = alpha,beta = beta,learning_rate = lr,gamma = g)

def run_empty_room():
    lrs = [.001, .005, .01]
    gammas = [0.999, 0.99, 0.95]
    cs = [(1, 1), (1, 2), (2, 1)]
    for lr in lrs:
        for g in gammas:
            for c in cs:
                alpha = c[0] / (c[0] + c[1])
                beta = 1 / (c[0] + c[1])
                trainD(file_name = 'empty_room_data/',list_of_envs = empty_room(),alpha=alpha, beta=beta, learning_rate=lr, gamma=g)


run_experiment()
run_empty_room()
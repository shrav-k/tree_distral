import policy
import gym
import gym_minigrid
from gym_minigrid.wrappers import FlatObsWrapper,RGBImgObsWrapper, OneHotPartialObsWrapper

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
    policy.train_distral(envs)

def two_room(wrapper = None,n = 4):
    if wrapper is None:
        envs = [gym.make("MiniGrid-MultiRoom-N2-S4-v0") for _ in range(n)]
    else:
        envs = [wrapper(gym.make("MiniGrid-MultiRoom-N2-S4-v0")) for _ in range(n)]
    policy.train_distral(envs)

def four_rooms(wrapper = None,n = 4):
    if wrapper is None:
        envs = [gym.make("MiniGrid-FourRooms-v0") for _ in range(n)]
    else:
        envs = [wrapper(gym.make("MiniGrid-FourRooms-v0")) for _ in range(n)]
    policy.train_distral(envs)

def unlock(wrapper = None,n = 4):
    if wrapper is None:
        envs = [gym.make("MiniGrid-Unlock-v0") for _ in range(n)]
    else:
        envs = [wrapper(gym.make("MiniGrid-Unlock-v0")) for _ in range(n)]
    policy.train_distral(envs)

def unlock_pick_up(wrapper = None,n = 4):
    if wrapper is None:
        envs = [gym.make("MiniGrid-UnlockPickup-v0") for _ in range(n)]
    else:
        envs = [wrapper(gym.make("MiniGrid-UnlockPickup-v0")) for _ in range(n)]
    policy.train_distral(envs)

def simple_crossing_env(wrapper = None,n = 4):
    if wrapper is None:
        envs = [gym.make("MiniGrid-SimpleCrossingS9N3-v0") for _ in range(n)]
    else:
        envs = [wrapper(gym.make("MiniGrid-SimpleCrossingS9N3-v0")) for _ in range(n)]
    policy.train_distral(envs)

def lava_crossing_env(wrapper = None,n = 4):
    if wrapper is None:
        envs = [gym.make("MiniGrid-LavaCrossingS9N2-v0") for _ in range(n)]
    else:
        envs = [wrapper(gym.make("MiniGrid-LavaCrossingS9N2-v0")) for _ in range(n)]
    policy.train_distral(envs)

def lava_gap(wrapper = None,n = 4):
    if wrapper is None:
        envs = [gym.make("MiniGrid-LavaGapS7-v0") for _ in range(n)]
    else:
        envs = [wrapper(gym.make("MiniGrid-LavaGapS7-v0")) for _ in range(n)]
    policy.train_distral(envs)

empty_room()


'''
class ConvPolicy(Model):

	def __init__(self, input_size, num_actions):
		super(Policy, self).__init__()
		self.input_size = input_size
		self.num_actions = num_actions

		self._build_graph()

	def _build_graph(self):
		self.fc_layers = []
		for _ in range(self.depth):
			self.fc_layers.append(Dense(self.layer_size, activation='relu'))
		self.softmax_layer = Dense(self.num_actions, activation='sigmoid')

	def call(self, inputs):
		#Reshape input if only one dimension
		#TODO: Change this
		x = tf.reshape(inputs, [-1] + [self.input_size]) if (len(inputs.shape) == 1 or len(inputs.shape) == 3) else inputs

		for i in range(self.depth):
			x = self.fc_layers[i](x)
		action_probs = self.softmax_layer(x)
		return action_probs

'''
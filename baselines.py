import gym
import gym_maze
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN, A2C


env = gym.make("maze-random-10x10-plus-v0")


def DQNbaseline():
    model = DQN(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=30000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

    env.close()


def A2C():
    model = A2C(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=30000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

    env.close()
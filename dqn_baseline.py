import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN



env = gym.make('CartPole-v1')


model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=30000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

env.close()
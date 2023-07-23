import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO
import random
import os

logdir = 'logdir'
models_dir = 'models/PPO'
TIMESTEPS = 10000

# Create necessary directories if they do not exist
if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env = gym.make('LunarLander-v2')  
env.reset()

# Create the model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
model.learn(total_timesteps=10000, reset_num_timesteps=False, tb_log_name="PPO")

""" for i in range(30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
     """

""" episodes = 10 """

""" for ep in range(episodes):
	obs = env.reset()
	done = False
	while not done:
		env.render()
		obs, rewards, done, info, _  = env.step(env.action_space.sample())
		print(rewards)
		 """
  
env.close()
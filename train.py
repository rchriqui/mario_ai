import gymnasium as gym
from stable_baselines3 import PPO, A2C
import os
from stable_baselines3.common.callbacks import BaseCallback
import warnings
warnings.filterwarnings("ignore")

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)
    
callback = TrainAndLoggingCallback(check_freq=10000, save_path=logdir)

env = gym.make("ALE/DonkeyKong-v5")  # Replace with a valid Gym environment
env.reset()

#model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
iters = 0

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

for i in range(20):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    
    
""" model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
model.save(models_dir) """

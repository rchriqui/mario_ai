import gymnasium as gym
from stable_baselines3 import A2C
import random

env = gym.make("CartPole-v1", render_mode="human")  # continuous: LunarLanderContinuous-v2
env.reset()

#model = A2C('MlpPolicy', env, verbose=1)
#model.learn(total_timesteps=1000)

episodes = 10

for episode in range(1, episodes + 1):
    state=env.reset()
    done=False
    score=0
    
    while not done:
        action=random.choice([0,1])
        
        _, reward, done, _, _ =env.step(action)
        score+=reward
        env.render()
    print('Episode:{} Score:{}'.format(episode, score))
env.close()
        
        
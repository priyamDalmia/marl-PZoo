import gym
import numpy as np
import time
env = gym.make("CarRacing-v0")
done = False
env.reset()
while not done:
    action = env.action_space.sample()
    obs, r, done, info = env.step(action)
    env.render()
    time.sleep(0.2)

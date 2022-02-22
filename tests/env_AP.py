import os
import sys
import gym
from pettingzoo.magent import adversarial_pursuit_v3

if __name__=="__main__":
    env = adversarial_pursuit_v3.env(map_size=7, minipal_mode=False, tag_penalty=-0.2, 
            max_cycles=500, extra_features=False)
    
    agent_ids = env.agents
    
    # Take arguments via parser.
    epochs = 1000
    learning_rate = 0.001
    gamma = 0.9
    training = False

    for ep in range(epcohs):
        # reset env; reset control variables.
        env.reset()
        done = False
        ep_score = 0

        while not done:
            observation reward, done, info = env.last()


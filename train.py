import numpy as np
import os 
import sys
from pettingzoo.mpe import tiger_deer_v3


# create a loop to initailize the agents.
def initialize_agents(env):


    pass

if __name__=="__main__":

    episodes = 1000
    learning_rate = 0.01 
    max_iter = 1000

    # Initialize the environment.
    env = tiger_deer.env(map_size=10, minimap_mode=False, extra_features=False)
    
    
    # game varialbes 
    agents = env.agents
    game_agents = initialize_agents()
    
    for ep in episodes:
        env.reset()


        for agent in env.agent_iter():
            observation, reward, done , info = env.last()
            
            # take a random action 
            action = 0
            
            # step forward
            env.step(aciton)

        # an episode ends when the iter loop exits


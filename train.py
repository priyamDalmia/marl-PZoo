import numpy as np
import os 
import sys
from pettingzoo.magent import tiger_deer_v3
from agents.agent import RandomAgent

# create a loop to initailize the agents.
def initialize_agents(agent_ids, env):
    action_type = "discrete"
    agents_list = {}
    breakpoint()
    for _id in agent_ids:
        action_space = env.action_space(_id)
        observation_space = env.observation_space(_id)
        if _id.startswith("deer"):
            # creating deer agents 
            agents_list[_id] = RandomAgent(_id, action_space, observation_space, action_type)
        else:
            # creating tiger agents 
            agents_list[_id] = RandomAgent(_id, action_space, observation_space, action_type)
    
    return agents_list

if __name__=="__main__":

    episodes = 1000
    learning_rate = 0.01 
    max_iter = 1000

    # Initialize the environment.
    env = tiger_deer_v3.env(map_size=10, minimap_mode=False, extra_features=False)
    
    breakpoint() 
    # game variables 
    agents = env.possible_agents
    game_agents = initialize_agents(agents, env)
    
    for ep in episodes:
        env.reset()
        for _id in env.agent_iter():
            observation, reward, done , info = env.last()
            action = game_agents[_id].get_action(observation)
            
            # step forward
            env.step(aciton)

        # an episode ends when the iter loop exits
    

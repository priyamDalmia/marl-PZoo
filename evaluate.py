import numpy as np
import os 
import sys
from pettingzoo.magent import tiger_deer_v3
from agents.agent import Agent

# create a loop to initailize the agents.
def initialize_agents(agent_ids, env):
    action_type = "discrete"
    agents_list = {}
    for _id in agent_ids:
        action_space = env.action_space(_id).n
        observation_space = env.observation_space(_id).shape
        if _id.startswith("deer"):
            # creating deer agents 
            agents_list[_id] = Agent(_id, action_space, observation_space, action_type)
        else:
            # creating tiger agents 
            agents_list[_id] = Agent(_id, action_space, observation_space, action_type)
    
    return agents_list

def log_results(rewards_history, steps_history):
    pass

if __name__=="__main__":

    episodes = 1000
    learning_rate = 0.01 
    max_iter = 1000
    training = False

    # Initialize the environment.
    env = tiger_deer_v3.env(map_size=10, minimap_mode=False, extra_features=False)
    
    # game variables 
    agents = env.possible_agents
    game_agents = initialize_agents(agents, env)
    
    breakpoint() 
    for ep in range(episodes):
        breakpoint()

        env.reset()
        rewards_history = {_id: 0 for _id in agents} 
        steps_history = {_id: 0 for _id in agents}
        # Play a game.
        for _id in env.agent_iter():
            agent_i = game_agents[_id]
            observation, reward, done, info = env.last()
            
            if done: 
                action = None
                env.step(action)
                continue

            action = agent_i.get_action(observation)

            # step forward
            env.step(action)

            # Storing transitions
            next_state = env.observe(_id)
            reward = env.rewards[_id]
            if training:
                agent_i.store_transition(observation, action, reward, next_state, done)
                
            # update historeis 
            rewards_history[_id] += reward
            steps_history[_id] += 1

            env.render()

        # an episode ends when the iter loop exits
        if ep % 100 == 0:
            pass
        
        if training:
            pass
  

import os
import sys
import logging
import numpy as np
import re

import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_tag_v2
import time 


SCORE_HIST = []

# The Simple-Tag environment. Adversaries (red) chase and collide with prey (green) while avoiding  
# obstacles. Adversaries/Prey recieve +-10 points for each collision.
# AEC: env > adver_i > adver_n > good_1 > good_n > env
# Observation Space: [self_vel, self_pos, landmark_rel_position, other_agent_pos, other_agent_rel_vel]
# Action Space: [no_action, move_left, move_right, move down, move_up]
# Negative penalty for leaving the map area. 

# Returns a random action.
def random_action(action_space = 5):
    return np.random.randint(action_space)

# Initializies the policies for all agents. Agents of the same type share a trained policy.
def initialize_policies(agents, env, load_checkpnt, gamma, learning_rate):
    agent_policies = {}

    for id_, agent in enumerate(agents):
        input_shape = env.observation_spaces[agent].shape[0]
        output_shape = env.action_spaces[agent].n

        if re.search("^agent", agent):
            policy = agent_prey(input_shape, output_shape, load_checkpnt, gamma, learning_rate)
        else:
            policy = agent_pred(input_shape, output_shape, load_checkpnt, gamma, learning_rate)    
        agent_policies[agent] = policy

    return agent_policies

if __name__=="__main__":
    
    n_good = 3
    n_adversaries = 2
    n_obstacles = 2 
    
    epochs = 100
    learning_rate = 0.01
    gamma = 0.9
    training = False
    load_checkpnt = False

    # TODO: Call parser here.
    
    env = simple_tag_v2.env(num_good=n_good, num_adversaries=n_adversaries, max_cycles=25,
            num_obstacles=n_obstacles)

    # BREAKPoint HERE
    breakpoint()
    
    env.reset()
    agent_policies = initialize_policies(env.agents, env, load_checkpnt, gamma, learning_rate)
    
    for i in range(epochs):
        score = []
        done = False 
        env.reset()
        step = 0
        for agent in env.agent_iter():
            
            observation, reward, dones, infos = env.last()
                       
            action = random_action()
            
            env.step(action)

            if render:
                env.render()

            if training: 
                agent_id = ids[agent]
                agent_policies[agent_id].learn()
            # env.render()
            step +=1 
            if done:
                breakpoint()

               break

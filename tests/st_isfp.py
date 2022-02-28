import os
import sys
import logging
import numpy as np
import re
import time
import pdb
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_tag_v2
from pettingzoo.utils import save_observation

sys.path.append(os.getcwd())
from common.ReplayBuffer import ReplayBuffer
from agents import agent_Indp


# Initialize the policies for all the agents. ( + paramerter sharing )
# modify function to use custom type of agents and policies.
def init_agents(env):
    policies = {}
    for id in env.agents:
        input_shape = env.observation_space(id).shape
        output_shape = env.action_space(id).n
        mem = ReplayBuffer(input_shape, output_shape, 100000, 64)
        policy = agent_Indp.agent_indp_dqn(input_shape, output_shape, 
                learning_rate = 0.01, alpha=0.9, gamma =0.1, memory = mem)
        policies[id] = policy
    return policies

def init_rewards(agent_list):
    rewards = {}
    for agent in agent_list:
        rewards[agent] = 0
    return rewards

if __name__ == "__main__":

    n_good = 1
    n_adversaries = 1
    n_obstacles = 2


    epochs = 10000
    learning_rate = 0.01
    gamma = 0.9
    training = True
    load_checkpoint = False
    all_agents = {}

    env = simple_tag_v2.env(num_good = n_good, num_adversaries = n_adversaries, max_cycles = 25,num_obstacles = n_obstacles)
    env.reset()
    all_agents = init_agents(env)
    reward_history = np.zeros(len(env.agents))

    for epoch in range(epochs):
        tic1 = time.perf_counter()
        done = False
        env.reset()
        steps = 0  
        all_rewards = init_rewards(env.agents)

        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            if done:
                env.step(None)
                continue
            
            action = all_agents[agent].get_action(observation)
            env.step(int(action))
            all_rewards[agent]+=reward
        
            next_state = env.observe(agent)

            # Storing Transitions
            reward_t = env.rewards[agent]
            all_agents[agent].store_transition(observation, action, next_state, reward_t, done)

            if epoch > 10 and training:
                tic = time.perf_counter()
                all_agents[agent].learn()
                toc = time.perf_counter() - tic


        # Stack episode values to reward history
        reward_history = np.vstack([reward_history, list(all_rewards.values())])
        toc1 = time.perf_counter() - tic1

        if epoch % 100 == 0:
            print(f"Episodes: {epoch}, Average Rewards: {np.mean(reward_history, axis = 0)}")

import numpy as np
import os
import sys
from pettingzoo.mpe import simple_tag_v2

sys.path.append(os.getcwd())
from agents import *

if __name__ == "__main__":
    
    n_good = 1 
    n_adversaries = 1
    n_obstacles = 2

    epochs = 10000
    learning_rate = 0.01
    save_checkpoints = True
    all_agents = []

    for episode in n_episodes:
        env.reset()
        done = False
        env.reset()
        steps = 0
        all_rewards = init_rewards(env.agents)
        
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            if done:
                en.step(None)
                continue

            action = all_agents[agent]get_action(observation)
            env.step(int(action))

            all_rewards[agent] +== reward

            next_state = env.observe(agent)

            # Storing transitions 
            reward_t = env.rewards[agnets]
            all_agents[agent].store_transition(observation, action, next_state, reward_t, done)
            
            if epoch > 10 and training:
                tic = time.pref_counter()
                all_agnets[agents].learn()
                toc = time.pref_counter()

        # Stack episodes values to history

        if epoch % 100 == 0:
            print(f"Episodes ..")


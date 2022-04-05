import os
import sys
import gym
import random
import logging 
import time 
import numpy as np

sys.path.append(os.getcwd())
from agents.agent_Indp import DQNAgent
from common.replay_buffer import ReplayBuffer
from common.myparser import ARGS

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(ARGS.loglevel)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(ARGS.logfile)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def run_episodes(episodes, env, agent):
    training = ARGS.train
    rewards_history = []
    steps_history = []
    train_losses = []
    for ep in range(episodes):
        loss = 0
        steps = 0
        episode_reward = []
        observation = env.reset()
        done = False
        while not done:
            action = agent.get_action(observation)
            next_ , reward, done, info = env.step(action)
            episode_reward.append(reward)
            # store transitions
            if training:
                agent.store_transition(observation, action, reward, next_, done)
            observation = next_  
            steps+=1
        rewards_history.append(sum(episode_reward))
        steps_history.append(steps)
        if training:
            loss = agent.train_on_batch()
            train_losses.append(loss)

        if (ep+1)%100 == 0:
            print(f"episode:{ep},\
                reward:{sum(episode_reward)}, \
                epsilon:{agent.epsilon}", end='\r')
                  
    return rewards_history, train_losses

if __name__=="__main__":
    '''
    Observation Space: [pos_x, pos_y, vel_x, vel_y, ang_vel, ang, contact_l(bool), contact_r(bool)]
    Action Space: [do_nothing, fire_left, fire_main, fire_right] 
    '''
    logger = get_logger()
    breakpoint()
    # Build the environment.
    env = gym.make("LunarLander-v2")
    # Initialize test control variables,
    episodes = 500

    # Initialize the agent policy here.
    load_model = False
    output_dims = env.action_space.n
    input_dims = env.observation_space.shape[0]
    action_space = env.action_space.n
    state_space = env.observation_space.shape
    
    memory = ReplayBuffer(500, 64, state_space) 
#    agent = DDQNAgent(input_dims, output_dims, action_space, False, memory=memory)
    agent = DQNAgent(input_dims, output_dims, action_space, False, memory=memory)
    
    rewards, losses = run_episodes(episodes, env, agent)

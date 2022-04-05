import gym
import os 
import numpy as np
import sys 
import pdb


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(os.getcwd())
from common.replay_buffer import ReplayBuffer
from agents.agent_dqn import Agent

if __name__=="__main__":
    env = gym.make('LunarLander-v2')
    episodes = 500

    load_model = False
    output_dims = env.action_space.n
    input_dims = env.observation_space.shape
    input_space = env.action_space.n
    state_space = env.observation_space.shape
    
    memory = ReplayBuffer(100000, 64, state_space)
    agent = Agent(input_dims, 
            output_dims, 
            input_space, 
            load_model, 
            memory=memory, 
            lr=0.003)
    scores = []
    eps_history = []
    breakpoint()
    for i in range(episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            try:
                action = agent.get_action(observation)
                next_, reward, done, info = env.step(action)
                score += reward
                agent.store_transition(observation, action, reward, next_, done)
                observation = next_
                agent.train_on_batch()
            except Exception as e:
                print(e)
                breakpoint()
        eps_history.append(agent.epsilon)
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print(f"\r episodes: {i}, scores: {score}, avg_score: {avg_score}, epsilon: {agent.epsilon}")

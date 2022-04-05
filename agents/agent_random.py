import os
import sys
import random
import numpy as np
from agents.agent import Agent

class RandomAgent(Agent):
    '''
    Agent that returns a random action at each turn.
    '''
    def __init__(self, _id, action_space, observation_space, action_type, **kwargs):
        super(RandomAgent, self).__init__(_id, action_space, observation_space, action_type)
    
    @abstractmethod
    def get_action(self, observation):
        return np.random.choice(self.action_space)

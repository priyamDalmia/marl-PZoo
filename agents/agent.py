import os 
import numpy as np

class Agent:
    '''
    Parent class for creating agents for the environment.  
    Override the get_action() method for higher functionality.
    '''
    def __init__(self, _id, observation_space, action_space, action_type):
        self._id = _id
        self.observation_space = observation_space
        self.action_space = action_space 
        self.action_type = action_type

    def get_action(self, observation):
        return np.random.choice(self.action_space)

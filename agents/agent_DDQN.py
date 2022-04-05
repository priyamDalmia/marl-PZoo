import os 
import numpy as np

from agents.agnet import Agent

class DDQNAgent(Agent):
    def __init__(self, _id, action_space, observation_space, agent_type, **kwargs):
        super(self, DDQNAgent).__init__(_id, action_space, observation_space, agent_type)
        self.network = None
        self.memory = None

    def get_action(self, observation):
        return None

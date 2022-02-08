from pettingzoo.magent import battlefield_v3
import os
import sys
import numpy as np
import tensorflow as tf 
from tensorflow import keras

class Agent():
    def __init__(self, id, input_dims, output_dims):
        self.id = id
        self.input_dims = input_dims
        self.output_dims = output_dims
    
    def get_action(self, observation):
        pass

if __name__=="__main__":
    env = battlefield_v3.env()
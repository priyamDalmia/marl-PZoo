import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Dense, Flatten


class ActorCritic():
    def __init__(self, input_dims, output_dims, gamma, learning_rate):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.gamma = gamma 
        self.lr = learning_rate
    
    def get_action(self, observation):
        pass

    def learn(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

        

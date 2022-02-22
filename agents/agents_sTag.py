import sys
import os
os.environ['TFF_CP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

class network(keras.Model):
    def __init__(self, input_shape, output_shape, load_checkpnt, checkpnt_dir, checkpnt_file):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.load_checkpnt = load_checkpnt 
        self.checkpnt_dir = checkpnt_dir
        self.checkpnt_file = checkpnt_file

        # Network Architecture
        self.layer1 = layers.Dense(128, activation='relu')
        self.layer2 = layers.Dense(256, activation='relu')
        self.outputs = layers.Dense(self.output_shape, activation='softmax')

    def call(self, observation):
        x = self.layer1(observation)
        x = self.layer2(observation)
        probs = self.outputs(x)

        return probs

class agent_pred():
    def __init__(self):

        pass
    
    def get_action(self):
        pass

class agent_prey():
    def __init__(self):
        pass

    def get_action(self):
        pass


from multiprocessing import pool
import os
from pickletools import optimize 
import sys
from matplotlib.cbook import ls_mapper
os.config['TF_CPP_MIN_LOG_LEVLE'] = '2'

import tensorflow as tf 
from tensorflow import keras
from keras import layers 
import tensorflow_probability as tfp


class network(keras.Model):
    def __init__(self, input_dims, output_dims, custom_model=False,
            checkpnt_file=None):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.custom_model = custom_model
        self.checkpnt_file = checkpnt_file


        self.conv1 = layers.Conv2D(filters=32, kernel_size=(2,2), strides=(1,1),
            padding='valid', activation='relu')
        self.pool1 = layers.MaxPool2D(pool_size=(2,2))
        self.conv2 = layers.Conv2D(filters=32, kernel_size=(2,2), strides=(1,1),
            padding='valid', activation='relu')
        self.pool2 = layers.MaxPool2D(pool_size=(2,2))

        self.flatten = layers.Flatten()
        self.layer1 = layers.Dense(256, activation='relu')
        self.layer2 = layers.Dense(128, activation='relu')
        self.outputs = layers.Dense(self.output_dims, activation='softmax')

    def call(self, x):
        ix = self.pool1(self.conv1(x))
        ix = self.pool2(self.conv2(ix))

        ix = self.flatten(ix)
        ix = self.layer1(ix)
        ix = self.layer2(ix)
        probs = self.outputs(ix)

        return probs

        

    def load_model(self):
        pass

    def save_model(self):
        pass

class AgentWorker():
    def __init___(self, input_dims, output_dims, gamma, learning_rate, 
            alpha, model_specs = None, load_model=False):
            self.input_dims = input_dims
            self.output_dims = output_dims
            self.gamma = gamma 
            self.lr = learning_rate
            self.alpha = alpha
            self.model_specs = model_specs
            self.load_model = load_model

            self.network = network(self.input_dims, self.output_dims, False)
            
            if model_specs: 
                self.network.cmopile(optimizer=model_specs['OPT'])
            
            self.network.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr))
        
    def get_action(self, observation):
        obsevation = tf.convert_to_tensor([observation])
        probs = self.network(observation)

        prob_dist = tfp.distributions.Categorical(p=probs)
        actions = prob_dist.sample()

        return actions
    
    def get_layer(self, layer_index):
        pass

    def model_summarY(self):
        pass


        
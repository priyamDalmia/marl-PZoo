import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pdb
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import numpy as np

class Network(keras.Model):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        
        initializer = tf.keras.initializers.HeNormal()   
        self.layer1 = layers.Dense(512, activation='relu', kernel_initializer=initializer)
        self.layer2 = layers.Dense(512, activation='relu', kernel_initializer=initializer)
        self.layer3 = layers.Dense(512, activation='relu', kernel_initializer=initializer)
 
        self.outputs = layers.Dense(self.output_dims, activation=None)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.outputs(x)

class DQNAgent():
    def __init__(self, input_dims, output_dims, input_space, load_model, **kwargs):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.input_space = input_space 
        self.load_model = load_model 
        self.lr = kwargs["lr"]
        self.gamma = 0.95
        self.memory = kwargs["memory"]
        self.network = None

        self.epsilon = 0.95
        self.epsilon_dec = 1e-3
        self.epsilon_min = 0.005

        self.losses = []
        self.train_step = 0

        if self.load_model:
            pass
        else:
            if isinstance(self.input_dims, int):
                self.network = Network(self.input_dims, self.output_dims)
                self.critic = Network(self.input_dims, self.output_dims)
            else:
                pass
        
        self.network.compile(optimizer=Adam(learning_rate = self.lr),
                loss=tf.keras.losses.MeanSquaredError())
        self.critic.compile(optimizer=Adam(learning_rate = self.lr),
                loss=tf.keras.losses.MeanSquaredError())

    def get_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.input_space)
        observation = tf.convert_to_tensor([observation], dtype=tf.float32)
        action_values = self.network(observation)
        action = np.argmax(action_values)
        return action 

    def train_on_batch(self):
        if self.memory.counter < self.memory.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample_batch()
        
        values_t = self.network(states)
        values_next = self.network(next_states)
        
        targets = np.copy(values_t)
        batch_index = np.arange(self.memory.batch_size, dtype=np.int32)
        
        targets[batch_index, actions] = rewards + \
                self.gamma * np.max(values_next, axis=1) * dones

        self.network.train_on_batch(states, targets)

        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon >\
                self.epsilon_min else self.epsilon_min
        self.train_step += 1
        
        return
    
    def store_transition(self, state, action, reward, next_, done):
        self.memory.store_transition(state, action, reward, next_, done)
    
    def save_model(self):
        #self.network.save(filepath)
        pass
    
    def load_mdoel(self):
    #    self.load_model(filepath)
        pass


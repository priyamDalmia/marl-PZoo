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

        self.layer1 = layers.Dense(128, activation='relu')
        self.layer2 = layers.Dense(128, activation='relu')
        self.outputs = layers.Dense(self.output_dims, activation=None)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return self.outputs(x)

class agent_indp_dqn():
    def __init__(self, input_dims, output_dims, learning_rate, gamma, alpha, 
            load_model = False, memory=None):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.network = None
        self.memory = memory
        self.epsilon = 0.9
        self.action_space = [i for i in range(output_dims)]
        
        if load_model:
            pass
        else:
            self.network = Network(input_dims, output_dims)
        self.network.compile(optimizer=Adam(learning_rate = learning_rate),
                loss = "mean_squared_error")

    def get_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            observation = tf.convert_to_tensor([observation])
            action_values = self.network(observation)
            action = np.argmax(action_values)
        return action 

    def learn(self):
        if self.memory.counter < self.memory.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        # CHECK: Modify values types explicilty
        #states = tf.convert_to_tensor(states)
        #actions = tf.convert_to_tensor(actions)
        #rewards = tf.convert_to_tensor(rewards)
        #next_states = tf.convert_to_tensor(next_states)
        

        values_t = self.network(states)
        values_next = self.network(next_states)
        
        targets = np.copy(values_t)
        batch_index = np.arange(self.memory.batch_size, dtype=np.int32)
        
        targets[batch_index, actions] = rewards + \
                self.gamma * np.max(values_next, axis=1) * dones

        self.network.train_on_batch(states, targets)
        
        self.epsilon = self.epsilon * 0.95
        return
    
    def store_transition(self, s, a, n, r, done):
        self.memory.insert(s, a, n, r, done)
    
    def summary(self, observation):
        print("THIS MODEL!")

    def save_model(self):
        #self.network.save(filepath)
        pass
    
    def load_mdoel(self):
    #    self.load_model(filepath)
        pass


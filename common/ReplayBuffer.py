import os 
import numpy as np
import sys
import random


# Simple Uniform Replaybuffer.

class ReplayBuffer():
    def __init__(self, state_size, action_size, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.states = np.zeros((self.buffer_size, *state_size), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size), dtype=np.int32)
        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.next_states = np.zeros((self.buffer_size, * state_size), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size), dtype=np.bool)
        self.counter = 0

    
    def insert(self, s, a, n, r, done):
        index = self.counter % self.buffer_size
        self.states[index] = s
        self.actions[index] = a
        self.next_states[index] = n
        self.rewards[index] = r
        self.dones[index] = done
        
        self.counter +=1

    def sample(self):

        max_index = min(self.counter, self.buffer_size)
        batch = np.random.choice(max_index, self.batch_size, replace = False)

        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        next_states = self.next_states[batch]
        dones = self.dones[batch]

        return states, actions, rewards, next_states, dones






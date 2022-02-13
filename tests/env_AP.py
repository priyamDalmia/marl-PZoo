import os
import sys
import gym
from pettingzoo.magent import adversarial_pursuit_v3

if __name__=="__main__":
    env = adversarial_pursuit_v3.env(map_size=7, minipal_mode=False, tag_penalty=-0.2, 
            max_cycles=500, extra_features=False)
    

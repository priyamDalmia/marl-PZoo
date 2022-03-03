import os
import sys
import numpy as np
from pettingzoo.mpe import simple_adeversary_v2
from commons import parser


'''
One adversary competing against N good agents. 


args :
    N = number of good agent and obstacles 
    max_cycles = num of frames each agents gets 
    countinous_actions = boolean (False)
'''


if __name__ == "__main__":


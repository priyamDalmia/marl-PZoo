# marl-PZoo

This repo contains implementations of MARL algorithms tested on the following environments.
1. MpE
2. Magent

Check respective folders for model details. 

## Getting Started. 

Install the pettingzoo library using this command. This will install all including dependencies too: 

`
pip install pettingzoo[all]
`

All agents are built using Tensorflow 2.8,0 (Python >= 3.5.0)

Use evaluate command to test a trained policy. All policies can be found in the policies folder. Exact model details can be found in the log files under the same name.

`
python3 evaluate.py --policy=<policy-name> --env=<env-name> --render=False --num-episodes=<100>
`

Environment specific parameters can be adjusted directly in the tester.py files. 


## Environment Class: MPE

See [] for more info.

A simple set of non-graphical communication tasks. The three environmets define an intuitive progression to three levels of a ciriculla.

Any agents implemented in the higher elvel, will be able to solve any of the lower level environments. Implementation details found in their respective folders.


### Simple Tag 

An implementation of the Predator-Prey Environment. 

Algortihms implemented: 
\1. Independent Learners. [paper]
\2. DQNs with stabalized ER. [paper]

### Simple Speaker Listner

An implementtaion of the Referential Game Environment. 

Algortihms implemented:
\1. RIAL. [paper]

### Simple World Comm

A combination of the two environments above. 

Prey must now communicate to avoid adversaries, hide in forests and collect food. Predators must now communicate with the leader (fully observable) to tag prey.



## Environment Class: MAgnet

A set of configurable massive ppo agents. Most Enviroments here represent a class of multi-goal or multi-taks hetrogenous agnet environments.


## Usage 


## Citation 

Not published.

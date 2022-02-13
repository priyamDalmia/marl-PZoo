# Solutions to the *Adversial Pursuit* environment from the pettingZoo library.

More Information at: 
* [pettting zoo site](https://www.pettingzoo.ml/magent/adversarial_pursuit)
* [magnet home site](https://github.com/geek-ai/MAgent)

## Environment Concepts Overview. 



### Observation Space
Every agent's local observation. (NOT to be confused with the state view, which encapsulates the information of the entire environment.)
* Predator : [10 X 10 X n\_channels] map. 
* Prey : [9 X 9 X n\_channels] map.
##### Channels
1. obstacles/boundary
2. my\_team\_presence
3. my\_team\_hp
4. other\_team\_presence
5. other \_team\_hp

* extra features mode includes more channels (+20)

### Actions 
* Predator : [ do\_nothing, move\_4, tag\_8]
* Prey : [do\_nothing, move\_8]
  
### Rewards
* Predator : [ +1.0: correct tagging, -0.2: incorrect tagging]
* Prey : [ -1.0: for getting tagged] 

### State Space
if map\_size = 45, then a [45 X 45 X n\_channels] map. 

##### Channels 
1. obstacles
2. prey\_presense
3. prey\_hp
4. predator\_presense
5. predator\_hp

* extra feature mode includes more channels

## Methods
1. Independent Learners. 
Training: Centralized
Execution: Decentralized.

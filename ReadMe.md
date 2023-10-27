# GachiNet: Reinforcement Learning for Agar.io Multi-Agent System

G - "Gameplay" (referring to the game mechanics and experience)

A - "Agents" (referring to the intelligent entities in the game)

C - "Cooperation" (emphasizing the collaborative interactions between agents)

H - "Hybrid" (referring to a combination of different approaches or techniques)

I - "Intelligence" (highlighting the use of AI and machine learning techniques)

Our main goal is to customize the current environment to further implement different architectures and approaches to reinforcement learning.

It was based on [work](https://github.com/buoyancy99/PyAgar) and augmented with [pipeline](https://github.com/staghuntrpg/agar)  

However, training was implemented only on the agent's motion action space without using 2 features-dividing and leaving a stock of points in the form of food. 

## Requirements
python 3.7

## How to install
1. Install all necessary libraries\
`pip3 install -r requirements.txt`

2. Install baselines\
`python agar/baselines/setup.py install`

## Useful info
    res_obs_dict the main dictionary that contains the observation space for each agent in the game. 
    It has 2 dict keys, "obs_keys" and "metadata".

    obs_keys is the dictionary that contains the exact observation space for each agent in the game. 
            0: "self" 
            1: "food"
            2: "virus"
            3: "script_agent"
            4: "outside"


    res_obs_dict["metadata"]  is the dictionary that contains the exact observation about each agent in the game itself.

    There are 4 keys in the metadata dictionary, and each key corresponds to an agent. 
        "is_killed": indicates if the agent is killed or not
        "position_x": indicates the x coordinate of the agent
        "position_y": indicates the y coordinate of the agent
        "last_action": indicates the last action of the agent such as angle and speed
3. Useful scripts:
* `stochastic_full_obs_space.ipynb` - for running stochastic policy approach learning
* `behavior_cloning.ipynb` - for behavior running clonning (dataset is necessary in data folder by default)
* `dataset_creator.py` - for saving one episode

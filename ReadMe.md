# GacniNet: Reinforcement Learning for Agar.io Multi-Agent System

G - "Gameplay" (referring to the game mechanics and experience)

A - "Agents" (referring to the intelligent entities in the game)

C - "Cooperation" (emphasizing the collaborative interactions between agents)

H - "Hybrid" (referring to a combination of different approaches or techniques)

I - "Intelligence" (highlighting the use of AI and machine learning techniques)

Our main goal is to customize the current environment to further implement different architectures and approaches to reinforcement learning.

It was based on [work](https://github.com/buoyancy99/PyAgar) and augmented with [pipeline](https://github.com/staghuntrpg/agar)  

However, training was implemented only on the agent's motion action space without using 2 features-dividing and leaving a stock of points in the form of food. 

## How to install
1. Make sure that you have pillow gym pyglet in another way install them

\```python 
pip install pillow gym pyglet==1.5.27
\```

2. Install baselines 
\```python 
python baselines/setup.py install
\```
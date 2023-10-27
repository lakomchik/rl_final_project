# pyenv import
from pyglet.window import key
import numpy as np
from agar.Env import AgarEnv
import time

render = True
num_agents = 1
from agar.Config import Config

# baselines 3 import
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class Args:
    def __init__(self):

        self.num_controlled_agent = num_agents
        self.num_processes = 64
        self.action_repeat = 1
        self.total_step = 1e8
        self.r_alpha = 0.1
        self.r_beta = 0.1
        self.seed = 42
        self.gamma = 0.99
        self.eval = True
        

env = AgarEnv(Args())
# env.seed(0)

step = 1
window = None
action = np.zeros((num_agents, 3))


config = Config()


# def on_mouse_motion(x, y, dx, dy):
#     action[0][0] = (x / config.serverViewBaseX - 0.5) * 2
#     action[0][1] = (y / config.serverViewBaseY - 0.5) * 2
#     # for i in range(1, num_agents):
#     #     action[i][0] = action[0][0] + np.random.normal(0, 0.1)
#     #     action[i][1] = action[0][1] + np.random.normal(0, 0.1)

# def on_key_press(k, modifiers):

#     if k == key.SPACE:
#         action[0][2] = 1
#     else:
#         action[0][2] = 0

def start_action():
    action[0][0] = 0.1
    action[0][1] = 0.1
    action[0][2] = 0

obs2rl = np.zeros(26)
start = time.time()
ca = 1000
for episode in range(1):
    observation = env.reset()
    while ca:
        ca -= 1
        time.sleep(0.04)
        if step % 40 == 0:
            print("step", step)
            print(step / (time.time() - start))
        if render:
            env.render(0, render_player=True)
            if not window:
                window = env.viewer.window
                start_action()
                # window.on_key_press = on_key_press
                # window.on_mouse_motion = on_mouse_motion
        a = action.reshape(-1)
        observations, rewards, done, info, new_obs,observations_short = env.step(a)
       
        # Flatten the list observations_short
        flat_observations = [value for sublist in observations_short for value in (sublist if isinstance(sublist, list) else [sublist])]

        # Copy the flattened values to the NumPy array obs2rl
        obs2rl[:len(flat_observations)] = flat_observations

        print(rewards)
        print(obs2rl)
        # print(observations["t0"].shape)
        step += 1
env.close()


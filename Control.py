from pyglet.window import key
import numpy as np
from agar.Env import AgarEnv
import time

render = True
num_agents = 1


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


def on_mouse_motion(x, y, dx, dy):
    action[0][0] = (x / 1920 - 0.5) * 2
    action[0][1] = (y / 1080 - 0.5) * 2
    for i in range(1, num_agents):
        action[i][0] = action[0][0] + np.random.normal(0, 0.1)
        action[i][1] = action[0][1] + np.random.normal(0, 0.1)


def on_key_press(k, modifiers):

    if k == key.SPACE:
        action[0][2] = 1
    else:
        action[0][2] = 0


start = time.time()
ca = 200
for episode in range(1):
    observation = env.reset()
    while ca:
        ca -= 1
        time.sleep(0.02)
        if step % 40 == 0:
            print("step", step)
            print(step / (time.time() - start))
        if render:
            env.render(0, render_player=True)
            if not window:
                window = env.viewer.window
                window.on_key_press = on_key_press
                window.on_mouse_motion = on_mouse_motion
        a = action.reshape(-1)
        observations, rewards, done, info = env.step(a)
        # print(step, rewards)
        print(observations["t0"].shape)
        action[0][2] = 0
        step += 1
env.close()

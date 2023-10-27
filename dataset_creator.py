from pyglet.window import key
import numpy as np
from agar.Env import AgarEnv
import time
import cv2
import os

render = True
num_agents = 1
from agar.Config import Config


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


# env.seed(0)

step = 1

action = np.zeros((num_agents, 3))


config = None


def on_mouse_motion(x, y, dx, dy):
    action[0][0] = (x / config.serverViewBaseX - 0.5) * 2
    action[0][1] = (y / config.serverViewBaseY - 0.5) * 2
    action[0][2] = 0
    # for i in range(1, num_agents):
    #     action[i][0] = action[0][0] + np.random.normal(0, 0.1)
    #     action[i][1] = action[0][1] + np.random.normal(0, 0.1)


def on_key_press(k, modifiers):
    if k == key.SPACE and action[0][2] == 0:
        action[0][2] = 1
    else:
        action[0][2] = 0


window = None
obs_dim = 578
history_length = 5


def capture_episode(env, data_folder, num_iterations):
    history = []
    for _ in range(history_length):
        history.append(np.zeros((obs_dim,)))
    global window, action
    arr_folder = os.path.join(data_folder, "arr")
    os.makedirs(arr_folder, exist_ok=True)
    image_fodler = os.path.join(data_folder, "image")
    os.makedirs(image_fodler, exist_ok=True)
    start = time.time()
    observation = env.reset()
    env.step(action.reshape(-1))
    env.render(0, mode="rgb_array", render_player=False)
    time.sleep(1.0)

    i = 0
    while i < num_iterations:
        time.sleep(0.05)
        if step % 40 == 0:
            print("step", step)
            print(step / (time.time() - start))
        image = env.render(0, mode="rgb_array", render_player=False)
        print(image.shape)
        if render:
            env.render(0, render_player=True)
            if not window:
                window = env.viewer.window
                window.on_key_press = on_key_press
                window.on_mouse_motion = on_mouse_motion
        a = action.reshape(-1)
        cv2.imwrite(os.path.join(image_fodler, f"{i}.jpg"), image)
        observations, rewards, done, info, new_obs = env.step(a)
        res_array = np.array(
            [
                {
                    "observation": observations["t0"],
                    "action": a,
                    "reward": rewards,
                    "new_obs": new_obs["t0"],
                    "history": history,
                }
            ]
        )
        history.append(observations["t0"])
        history.pop(0)
        np.save(os.path.join(arr_folder, f"{i}.npy"), res_array)
        # print(step, rewards)
        # print(rewards)
        # print(observations["t0"].shape)
        i += 1
        print(rewards)
    env.close()
    time.sleep(5)


def capture_episodes(data_folder, start_num, end_num, num_iterations):
    global config
    for i in range(start_num, end_num):
        del config
        config = Config()
        env = AgarEnv(Args())
        capture_episode(env, os.path.join(data_folder, f"episode_{i}"), num_iterations)


if __name__ == "__main__":
    config = Config()
    env = AgarEnv(Args())
    i = 9
    capture_episode(env, os.path.join("data", f"episode_{i}"), 1000)
    env.close()

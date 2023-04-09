import os
import random

import gymnasium as gym
from gymnasium.envs.toy_text.taxi import TaxiEnv
import  PIL.Image
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

if os.environ.get("DISPLAY") is None:
    os.environ["DISPLAY"] = "localhost:0.0"

action_map = {
    0: "SOUTH",
    1: "NORTH",
    2: "EAST",
    3: "WEST",
    4: "PICKUP PASSENGER",
    5: "DROP OFF PASSENGER"
}

# def display_environment(arr, action, rewards) -> None:
#     """
#     show current state of the environment
#     via matplotlib
#     """

#     img = PIL.Image.fromarray(arr)
#     plt.clf()
#     plt.imshow(img)
#     plt.axis("off")
#     plt.title(f"{action_map[action]} ({rewards})")
#     plt.show(block=False)
#     plt.pause(0.1)


# epochs = 0
# penalties = rewards = 0
# frames = []
# done = False
# max_iter = 1000

# count = 0
# plt.axis("off")
# with gym.make("Taxi-v3", render_mode="rgb_array").env as env:
#     env: TaxiEnv
#     env.reset()
#     while not done and count < max_iter:
#         count += 1
#         action = env.action_space.sample()
#         state, reward, done, *info = env.step(action)
        
#         if reward == -10:
#             penalties += 1  # must have done illegal action
        
#         rewards += reward
#         display_environment(env.render(), action, rewards)

#         epochs += 1


####################
# DO Q LEARNING
####################

q_table = np.zeros(shape=(500, 6))
episodes = 6000
alpha = 0.01
gamma = 0.99
epsilon = 0.01
rewards = []
log = False

with gym.make("Taxi-v3", render_mode="rgb_array").env as env:
    env: TaxiEnv
    for i in range(episodes):
        print(i)
        reward = 0
        done = False
        s_current, info = env.reset()
        counter = 0
        if log:
            print(f"Starting episode {i + 1}")
        while not done:

            # 1) make our epsilon-greedy action choice
            mod_mask = np.where(info["action_mask"] == 1)[0]
            q_current = q_table[s_current, mod_mask].squeeze()
            rnd = random.choices([0, 1], [1 - epsilon, epsilon])[0]
            if not q_current.shape:
                # if only one choice then pick this regardless
                idx = 0
            elif rnd == 0:
                # with prob 1 - epsilon we take the greedy action
                idx = int(np.argmax(q_current))
            else:
                # with prob epsilon we take any action 
                idx = int(random.choices(range(len(mod_mask)))[0])
            a_current = mod_mask[idx]

            # 2) take one step forward using our selected action
            s_next, r_current, done, _, info = env.step(a_current)
            reward += r_current

            # 3) make q update based on our TD error (TD(1))
            td_error = r_current + gamma * np.max(q_table[s_next, :]) - q_table[s_current, int(a_current)]
            q_table[s_current, a_current] += alpha * td_error
            
            # 4) roll forward to next action
            s_current = s_next
            counter += 1
            if log and counter % 20 == 0:
                print(action_map[a_current])
        
        # record reward once terminated
        if log:
            print(f"Episode {i + 1} terminated in {counter} steps with reward {reward}")
        rewards.append((reward, counter))

# print(rewards)
df = pd.DataFrame(rewards, columns=["Reward", "Steps"])
np.save("results.npy", df.values)
df["Steps"].plot(kind="line")
plt.show(block=False)
plt.pause(4)
plt.close()
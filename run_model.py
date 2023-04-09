import os
import argparse
from typing import Dict
import random

import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.taxi import TaxiEnv


def _parse_args() -> Dict[str, str]:

    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type="str", help="File containing parameters")
    parser.add_argument("--mode", type="str", choices=["greedy", "random"], 
                        default="greedy", help="File containing parameters")
    
    return vars(parser.parse_args())




def run_model(args: Dict[str, str]) -> None:

    filepath: str = args["params"]
    mode: str = args["mode"]

    if not filepath and mode != "random":
        raise ValueError("Parameter file must be provided unless using random mode")
    
    if filepath is not None and not os.path.isfile(filepath):
        raise ValueError(f"Cannot find parameter file {filepath}")
    else:
        with open(filepath) as infile:
            q_vals = np.load(infile)

    with gym.make("Taxi-v3", render_mode="rgb-array").env as env:
        env: TaxiEnv
        done = False
        reward = 0
        s_current, info = env.reset()
        while not done:
            
            mod_mask = np.where(info["action_mask"] == 1)[0]
            if mode == "random":
                idx = random.choices(range(len(mod_mask)))[0]
            elif mode == "greedy":
                q_vals_masked = q_vals[s_current, mod_mask]
                if not q_vals_masked.shape:
                    idx = 0
                else:
                    idx = np.argmax(q_vals_masked)
            
            a_current = mod_mask[idx]
            s_current, r, done, _, info = env.step(a_current)
            reward += r


if __name__ == "__main__":

    cli_args = _parse_args()
    run_model(cli_args)

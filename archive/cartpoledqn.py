# debug
from pprint import pprint
import code

# common
import argparse
import numpy as np
import torch

# environment
from custom_minigrid import CustomMinigridEnv, CustomOneHotMinigridEnv

# ray rllib
import ray
from ray import tune
from ray.rllib.agents.dqn.dqn import DEFAULT_CONFIG, DQNTrainer

from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.agents.trainer_template import build_trainer

if __name__ == "__main__":
    ray.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000)
    args = parser.parse_args()
    tune.run(
        HERRainbowTrainer,
        config={
            "env": "CartPole-v1",
            "num_workers": 1,
            #  "num_gpus": 1,
        },
        stop={
            "timesteps_total": args.steps,
        },
        checkpoint_at_end=True
    )

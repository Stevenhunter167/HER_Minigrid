# debug
from pprint import pprint
import code

import argparse

import numpy as np
import torch

import ray
from ray import tune
from ray.rllib.agents.dqn.dqn import DEFAULT_CONFIG
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.agents.trainer_template import build_trainer

# Trainer config using Rainbow DQN with HER
HER_RAINBOW_DQN_CONFIG = DEFAULT_CONFIG.copy()
HER_RAINBOW_DQN_CONFIG.update(
    {
        # Hindsight Experience Replay
        "batch_mode": "complete_episodes",  # postprocess with full trajectory
        "num_her_traj": 6,                  # number of new trajectories sampled using HER
        # Rainbow DQN Config
        "n_step": 1,                        # n_step TD
        "noisy": True,                      # noisy network
        "num_atoms": 1,                     # number of distributional buckets
        "v_min": -10.0, 
        "v_max": 10.0
    }
)


HERRainbowTrainer = build_trainer(
    name="HER_RainbowDQN",
    default_policy=DQNTorchPolicy.with_updates(
        postprocess_fn=postprocess_with_HER
    ),
    default_config=HER_RAINBOW_DQN_CONFIG
)


if __name__ == "__main__":
    ray.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000000)
    args = parser.parse_args()
    tune.run(
        HERRainbowTrainer,
        config={
            "env": "CartPole-v1",
            "num_workers": 1,
            "num_gpus": 1,
        },
        stop={
            "timesteps_total": args.steps,
        },
        checkpoint_at_end=True
    )

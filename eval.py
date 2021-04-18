#!/usr/bin/env python
"""Example of running inference on a policy. Copy this file for your use case.
To try this out run:
    $ python cartpole_local_serving.py
"""

import os
import time
import ray
from custom_minigrid import CustomOneHotMinigridEnv
from train import HERRainbowTrainer, HER_RAINBOW_DQN_CONFIG
import code

HER_RAINBOW_DQN_CONFIG.update(
    {
        "explore": False
    }
)

CHECKPOINT_FILE = "last_checkpoint_{}.out"

if __name__ == "__main__":
    ray.init()
    # env = CustomOneHotMinigridEnv
    trainer = HERRainbowTrainer(HER_RAINBOW_DQN_CONFIG, CustomOneHotMinigridEnv)

    checkpoint_path = CHECKPOINT_FILE.format(
        "~/ray_results/HER_RainbowDQN_16x16_2021-04-18_00-48-05/HER_RainbowDQN_16x16_CustomOneHotMinigridEnv_68ba5_00000_0_2021-04-18_00-48-05/checkpoint_19846"
    )

    # Attempt to restore from checkpoint if possible.
    if os.path.exists(checkpoint_path):
        checkpoint_path = open(checkpoint_path).read()
        print("Restoring from checkpoint path", checkpoint_path)
        trainer.restore(checkpoint_path)

    print(f"{trainer.compute_action=}")

    # Serving and training loop
    env = trainer.env_creator({})
    state = trainer.get_policy().get_initial_state()
    obs = env.reset()
    while True:
        action, state, info_trainer = trainer.compute_action(
            obs, state=state, full_fetch=True)
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.1)
        if done:
            obs = env.reset()
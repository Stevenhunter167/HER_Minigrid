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
from ray.rllib.agents.dqn.dqn import DEFAULT_CONFIG
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.agents.trainer_template import build_trainer

# Policy Network
from policy import CustomModel

# Hindsight Experience Replay postprocess_fn builder
from postprocess_HER_DQN import build_DQN_HER_postprocess_fn, SamplingStrategy


# define HER sampling strategy for this environment
class MiniGridSamplingStrategy(SamplingStrategy):

    def __init__(self, policy, sample_batch):
        super().__init__(policy, sample_batch)

    def sample_strategy(self, original_traj):
        import numpy as np
        try:
            orig = original_traj.copy()
            traj_length = original_traj['obs'].shape[0]                     # trajectory length
            goal_step = np.random.randint(min(traj_length, 7))              # sample from this trajectory
            for entry in original_traj:                                     # 
                original_traj[entry] = original_traj[entry][:goal_step + 1]     # cut trajectory
            
            # change goal condition and rewards
            agent_location = original_traj['new_obs'][goal_step][:,:,10]    # get agent ending position
            for obs_i in range(original_traj['obs'].shape[0]):              # for each timestep
                original_traj['obs'][obs_i][:,:,8] = agent_location         # set new goal condition
            for obs_i in range(original_traj['new_obs'].shape[0]):          # for each timestep
                original_traj['new_obs'][obs_i][:,:,8] = agent_location     # set new goal condition
            original_traj['dones'][-1] = True                               # set last one to done
            original_traj['rewards'] = np.array(original_traj['rewards'], dtype=np.float)
            original_traj['rewards'][-1] = \
                1 - 0.9 * goal_step / self.policy.config['horizon']         # change reward
            # code.interact(local=locals())
        except:
            pass
        return original_traj

# build postprocess_fn using SamplingStrategy
postprocess_with_HER = build_DQN_HER_postprocess_fn(MiniGridSamplingStrategy)


# Trainer config using Rainbow DQN with HER
HER_RAINBOW_DQN_CONFIG = DEFAULT_CONFIG.copy()
HER_RAINBOW_DQN_CONFIG.update(
    {
        # Common
        "framework": "torch",
        "num_gpus": 1,
        # Model
        "model": {
            "dim": 6,
            "conv_filters": [
                [32, [6, 6], 1]
            ],
            "conv_activation": "relu",
            "fcnet_hiddens": [8,8],
            "max_seq_len": 100
        },
        # Hindsight Experience Replay
        "use_HER": True,
        "horizon": 100,                     # max episode length
        "batch_mode": "complete_episodes",  # postprocess with full trajectory
        "num_HER_traj": 10,                 # number of new trajectories sampled using HER
        # Rainbow DQN Config
        "n_step": 1,                        # n_step TD
        # "noisy": True,                      # noisy network
        # "num_atoms": 1,                     # number of distributional buckets
        # "v_min": -10.0, 
        # "v_max": 10.0
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
    parser.add_argument("--steps", type=int, default=10_000_000)
    args = parser.parse_args()
    tune.run(
        HERRainbowTrainer,
        config={
            "env": CustomOneHotMinigridEnv,
            "num_workers": 1,
            #  "num_gpus": 1,
        },
        stop={
            "timesteps_total": args.steps,
        },
        checkpoint_at_end=True
    )

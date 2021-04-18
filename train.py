# debug
from pprint import pprint
import code

# common
import argparse
import numpy as np
import torch

# environment
from custom_minigrid import CustomMinigridEnv, CustomOneHotMinigridEnv

# ray
import ray
from ray import tune

# rllib DQN
from ray.rllib.agents.dqn.dqn import DEFAULT_CONFIG                 # DQN config
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy    # DQN base policy

# Hindsight Experience Replay postprocess function builder
from postprocess_HER import build_HER_postprocess_fn, SamplingStrategy


# step 1: define environment specific HER sampling strategy
class MiniGridSamplingStrategy(SamplingStrategy):

    def __init__(self, policy, sample_batch):
        super().__init__(policy, sample_batch)

    def sample_strategy(self, original_traj):
        """ sample trajectory for minigrid """
        # cut trajectory
        traj_length = original_traj['obs'].shape[0]                     # trajectory length
        goal_step = np.random.randint(1, min(traj_length+1, 30))        # sample from this trajectory
        for entry in original_traj:                                     # 
            original_traj[entry] = original_traj[entry][:goal_step]     # cut trajectory
        
        # change goal condition and rewards
        agent_location = original_traj['new_obs'][goal_step - 1][:,:,1] # get agent ending position
        for obs_i in range(original_traj['obs'].shape[0]):              # for each timestep
            original_traj['obs'][obs_i][:,:,0] = agent_location         # set new goal condition
        for obs_i in range(original_traj['new_obs'].shape[0]):          # for each timestep
            original_traj['new_obs'][obs_i][:,:,0] = agent_location     # set new goal condition
        original_traj['dones'][-1] = True                               # set last one to done
        original_traj['rewards'] = np.array(original_traj['rewards'], dtype=np.float)
        original_traj['rewards'][-1] = \
            1 - 0.9 * goal_step / self.policy.config['horizon']         # recalculate reward
        return original_traj


# step 2: build postprocess_fn using SamplingStrategy and algorithm original postprocess_fn
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio
postprocess_with_HER = build_HER_postprocess_fn(MiniGridSamplingStrategy, postprocess_nstep_and_prio)


# step 3: Trainer config using Rainbow DQN with HER configs
HER_RAINBOW_DQN_CONFIG = DEFAULT_CONFIG.copy()
HER_RAINBOW_DQN_CONFIG.update(
    {
        # ================ Common ================
        "framework": "torch",
        "num_gpus": 1,

        # ============ Neural Network ============
        "model": {
            "dim": 16,
            "conv_filters": [
                # [16, [6, 6], 1]
                [256, [16, 16], 1]
            ],
            "conv_activation": "relu",
            "fcnet_hiddens": [512, 128, 32],
            # "fcnet_hiddens": [32,16,16,8],
            "max_seq_len": 5
        },

        # ===== Hindsight Experience Replay =====
        "batch_mode": "complete_episodes",  # postprocess with full trajectory
        "use_HER": True,
        "horizon": 100,                     # max episode length
        "num_HER_traj": 5,                  # number of new trajectories sampled using HER
        # note: the right amount of trajectories gives good 
        # training signal without overfitting the policy network

        # ============= Rainbow DQN =============
        "timesteps_per_iteration": 100,
        # "lr": 1e-4,
        # "grad_clip": 0.1,
        "train_batch_size": 32,             # batch size
        "num_sgd_iter": 20,                 # number of sgd steps per batch
        "n_step": 1,                        # n_step TD
        # "noisy": True,                      # noisy network
        # "num_atoms": 1,                     # number of distributional buckets
        # "v_min": -10.0, 
        # "v_max": 10.0

        # ============= Exploration =============
        "explore": True,
        "exploration_config": {
            # Exploration sub-class by name or full path to module+class
            # (e.g. “ray.rllib.utils.exploration.epsilon_greedy.EpsilonGreedy”)
            "type": "EpsilonGreedy",
            # Parameters for the Exploration class' constructor:
            # "initial_epsilon": 1.0,
            # "final_epsilon": 0.1,
            # "epsilon_timesteps": 800_000,  # Timesteps over which to anneal epsilon.
        },
    }
)


# step 3: build policy with HER postprocess function
HERRainbowPolicy = DQNTorchPolicy.with_updates(
    postprocess_fn=postprocess_with_HER,
    get_default_config=lambda:HER_RAINBOW_DQN_CONFIG
)


# step 4: build off-policy HER trainer using off-policy execution plan
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.dqn.dqn import validate_config, execution_plan
HERRainbowTrainer = build_trainer(
    name=f"{'' if HER_RAINBOW_DQN_CONFIG['use_HER'] else 'NO'}HER_RainbowDQN_16x16",
    default_policy=HERRainbowPolicy,
    default_config=HER_RAINBOW_DQN_CONFIG,
    validate_config=validate_config,
    execution_plan=execution_plan
)


if __name__ == "__main__":
    ray.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000_000)
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

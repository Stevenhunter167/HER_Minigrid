import argparse

import ray
import tensorflow as tf
from ray import tune
from ray.rllib.agents.dqn import dqn
from ray.rllib.agents.dqn import dqn_tf_policy
from ray.rllib.agents.trainer_template import build_trainer

DEFAULT_CONFIG = dqn.DEFAULT_CONFIG
DEFAULT_CONFIG["dueling"] = False
DEFAULT_CONFIG["double_q"] = False


def compute_q_values(_policy, model, obs):
    model_out, _ = model({"obs": obs}, [], None)
    q, _, _ = model.get_q_value_distributions(model_out)
    return q


def select_q_values(q, actions, num_actions):
    one_hot_selection = tf.one_hot(actions, num_actions)
    return tf.reduce_sum(q * one_hot_selection, axis=1)


class QLoss(object):
    def __init__(self, policy, train_batch):
        num_actions = policy.action_space.n
        gamma = policy.config["gamma"]

        obs = train_batch["obs"]
        actions = tf.cast(train_batch["actions"], tf.int32)
        rewards = train_batch["rewards"]
        next_obs = train_batch["new_obs"]
        dones = tf.cast(train_batch["dones"], tf.float32)

        all_q = compute_q_values(policy, policy.q_model, obs)
        selected_q = select_q_values(all_q, actions, num_actions)

        all_next_q = compute_q_values(policy, policy.target_q_model, next_obs)
        selected_next_q = select_q_values(all_next_q, tf.argmax(all_next_q, axis=1), num_actions)

        y = rewards + gamma * (1. - dones) * selected_next_q
        self.td_error = (tf.stop_gradient(y) - selected_q)
        self.loss = tf.reduce_mean(self.td_error ** 2)

        self.stats = {
            "mean_q": tf.reduce_mean(selected_q),
            "min_q": tf.reduce_min(selected_q),
            "max_q": tf.reduce_max(selected_q),
            "mean_td_error": tf.reduce_mean(self.td_error),
        }


def build_q_losses(policy, model, dist_class, train_batch):
    dqn_tf_policy.build_q_losses(policy, model, dist_class, train_batch)
    policy.q_loss = QLoss(policy, train_batch)
    return policy.q_loss.loss


DQN_Trainer = build_trainer(
    name="DQN_Trainer1",
    default_policy=dqn_tf_policy.DQNTFPolicy.with_updates(
        loss_fn=build_q_losses),
    default_config=DEFAULT_CONFIG)

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=200000)

from custom_minigrid import CustomMinigridEnv

if __name__ == "__main__":
    ray.init()
    args = parser.parse_args()
    tune.run(
        DQN_Trainer,
        config={
            "env": "Breakout-v0",
            "num_workers": 2,
        },
        stop={
            "timesteps_total": args.steps,
        },
        checkpoint_at_end=True
    )

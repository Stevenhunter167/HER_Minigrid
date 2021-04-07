from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio

class SamplingStrategy:

    """ Defines the sampling strategy for Hindsight Experience Replay """

    def __init__(self, policy, sample_batch):
        self.policy = policy
        self.sample_batch = sample_batch # original trajectory
        self.buffer_size = policy.config['buffer_size']
        self.aug_eps_id = sample_batch['eps_id']
    
    def sample_strategy(self, original_traj):
        raise NotImplementedError("Implement this method in subclass")

    def sample_trajectory(self):
        # create a copy of sampled trajectory
        augmented_batch = self.sample_batch.copy()
        self.aug_eps_id += self.buffer_size
        # assign new episode id
        augmented_batch['eps_id'] = self.aug_eps_id
        # sample new trajectory using strategy
        augmented_batch = self.sample_strategy(augmented_batch)
        return augmented_batch

def build_DQN_HER_postprocess_fn(SamplingStrategy):

    def postprocess_with_HER(policy, sample_batch, _other_agent_batches=None, _episode=None):
        """
            postprocess the sampled batch, inject modified trajectory with modified goal condition
        """
        
        # Hindsight Experience Replay trajectory augmentation
        if type(sample_batch) is SampleBatch and policy.config['use_HER'] and sample_batch['obs'].shape[0] > 0:
            # init list of new trajectories
            augmented_trajs = [sample_batch]
            # init HER sampling strategy
            her_sampler = SamplingStrategy(policy, sample_batch)
            # sample n new trajectories using sampling strategy
            for i in range(policy.config['num_HER_traj']):
                augmented_trajs.append(her_sampler.sample_trajectory())
            # concatenate sampled trajectories
            sample_batch = SampleBatch.concat_samples(augmented_trajs)

        # RLlib Original DQN postprocess_fn Implementation
        sample_batch = postprocess_nstep_and_prio(policy, sample_batch, _other_agent_batches, _episode)

        return sample_batch
    
    return postprocess_with_HER
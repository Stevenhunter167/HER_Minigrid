import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch

class SamplingStrategy:

    """ Defines the sampling strategy for Hindsight Experience Replay """

    def __init__(self, policy, sample_batch):
        self.policy = policy
        self.sample_batch = sample_batch # original trajectory
        self.buffer_size = policy.config['buffer_size']
        self.aug_eps_id = sample_batch['eps_id'][0]
    
    def sample_strategy(self, original_traj):
        raise NotImplementedError("Implement this method in subclass")

    def sample_trajectory(self):
        """ trajectory injection """
        # create a copy of sampled trajectory
        augmented_batch = self.sample_batch.copy()
        # increment eps_id
        self.aug_eps_id += self.buffer_size
        # assign new episode id
        augmented_batch['eps_id'] = np.array([self.aug_eps_id for _ in range(augmented_batch['eps_id'].shape[0])])
        # code.interact(local=locals())
        # sample new trajectory using strategy
        augmented_batch = self.sample_strategy(augmented_batch)
        return augmented_batch

def build_HER_postprocess_fn(SamplingStrategy, postprocess_fn):

    def postprocess_with_HER(policy, sample_batch, _other_agent_batches=None, _episode=None):
        """
            postprocess the sampled batch, inject modified trajectory with modified goal condition
        """
        import numpy as np
        # Hindsight Experience Replay trajectory augmentation
        if (type(sample_batch) is SampleBatch) and (policy.config['use_HER']) and (sample_batch['obs'].shape[0] > 0):
            # init list of new trajectories
            augmented_trajs = [sample_batch]
            # init HER sampling strategy
            her_sampler = SamplingStrategy(policy, sample_batch)
            # sample n new trajectories using sampling strategy
            for i in range(policy.config['num_HER_traj']):
                augmented_trajs.append(her_sampler.sample_trajectory())
            # concatenate sampled trajectories
            sample_batch = SampleBatch.concat_samples(augmented_trajs)

        # Original postprocess_fn Implementation
        sample_batch = postprocess_fn(policy, sample_batch, _other_agent_batches, _episode)
        # code.interact(local=locals())
        return sample_batch
    
    return postprocess_with_HER
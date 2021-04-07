import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from gym.spaces import Space

class CustomModel(DQNTorchModel, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        DQNTorchModel.__init__(self, 
            Space(shape=(obs_space.shape[0]*obs_space.shape[1]*4,), dtype=np.float), 
            action_space, 
            num_outputs, 
            model_config, 
            name)
        nn.Module.__init__(self)

        self.convfilter = nn.Sequential(
            nn.Conv2d(21, 8, kernel_size=3, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, padding=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(obs_space.shape[0]*obs_space.shape[1]*4, 32),
            nn.ReLU(),
        )

    @classmethod
    def with_updates(self, *args, **kwargs):
        DQNTorchModel.with_updates(self, *args, **kwargs)

    def forward(self, input_dict, state, seq_lens):
        input_dict_copy = input_dict.copy()
        input_dict_copy['obs'] = self.net(input_dict_copy["obs"])
        return DQNTorchModel.forward(self, input_dict_copy, state, seq_lens)

ModelCatalog.register_custom_model("CustomModel", CustomModel)
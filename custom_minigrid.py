import numpy as np
from gym_minigrid.wrappers import *

class CustomMinigridEnv:
    def __init__(self, env_config):
        self.env = ImgObsWrapper(FullyObsWrapper(gym.make('MiniGrid-Empty-16x16-v0')))
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return env.reset()
    def step(self, action):
        return self.env.step(action)
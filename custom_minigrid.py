import numpy as np
import matplotlib.pyplot as plt
from gym_minigrid.wrappers import *
from gym_minigrid.minigrid import Goal
from gym import spaces, Env

def obs_to_str(obs):
        res = ''
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                cell = obs[i][j]
                # cell representation
                cell_repr = '.'
                # wall
                if cell[0] == 2:
                    cell_repr = 'X'
                if cell[0] == 10:
                    cell_repr = 'A'
                res += cell_repr
            res += '\n'
        return res

def onehot2d(arr, discrete_settings):
        res = np.zeros((*arr.shape, discrete_settings), dtype=np.float)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                k = arr[i,j]
                res[i,j,k] = 1
        return res
    
def fullonehot2d(arr, settingslist):
    res = []
    arr = np.split(arr, 3, axis=2)
    for c in range(3):
        res.append(onehot2d(np.squeeze(arr[c], axis=2), settingslist[c]))
    return np.concatenate(res, axis=2)

class CustomMinigridEnv(Env):
    def __init__(self, env_config):
        super().__init__()
        self.env = ImgObsWrapper(FullyObsWrapper(gym.make('MiniGrid-Empty-Random-6x6-v0')))
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return (obs), reward, done, info


class CustomOneHotMinigridEnv(Env):
    def __init__(self, env_config):
        super().__init__()
        self.env = ImgObsWrapper(FullyObsWrapper(gym.make('MiniGrid-Empty-Random-6x6-v0')))
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(low=0, high=1, shape=
                    (self.env.observation_space.shape[0], 
                     self.env.observation_space.shape[1], 
                     21), dtype=np.float)
        self.settingslist = [11, 6, 4]
    def reset(self):
        # Place a goal square in the bottom-right corner fullonehot2dfullonehot2d
        return fullonehot2d(self.env.reset(), self.settingslist)
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return fullonehot2d(obs, self.settingslist), reward, done, info

if __name__ == '__main__':

    env = CustomMinigridEnv(None)
    obs = env.reset()
    done = False

    # COLOR_TO_IDX = {
    #     0  : (255,0,0),
    #     1  : (0,255,0),
    #     2  : (0,0,255),
    #     3  : (255,0,255),
    #     4  : (0,255,255),
    #     5  : (155,155,155)
    # }
    # for i in range(10):
    #     env.reset()
        # env.env.render()
        

    while not done:
        print(type(obs))
        print(obs)
        # print(obs[:,:,10] + obs[:,:,8])
        print(env.observation_space)
        # env.env.render()
        action = env.action_space.sample()
        print(action)
        obs, reward, done, info = env.step(action)
        print("reward", reward)
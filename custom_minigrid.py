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


# custom minigrid env: random goal
from gym_minigrid.envs.empty import EmptyEnv, Goal, Grid
class EmptyEnvRandGoal(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(agent_start_pos=(1,1), **kwargs)
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        # Place a goal square in random location
        self.goal_pos = (
            np.random.randint(1, self.width-1),
            np.random.randint(1, self.height-1)
        )
        while self.goal_pos == (1,1):
            self.goal_pos = (
                np.random.randint(1, self.width-1),
                np.random.randint(1, self.height-1)
            )
        self.put_obj(
            Goal(), 
            *self.goal_pos
        )
        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        self.mission = "get to the green goal square"


class CustomRGBMinigrid(EmptyEnvRandGoal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_env = RGBImgObsWrapper(self)
    def reset(self):
        return self.base_env.reset()['image']
    def step(self, action):
        obs, reward, done, info = self.base_env.step(action)
        return obs['image'], reward, done, info


# onehot full obs wrapper
class CustomOneHotMinigridEnv(Env):
    def __init__(self, env_config):
        super().__init__()
        # self.base_env = gym.make('MiniGrid-Empty-Random-6x6-v0')
        self.base_env = EmptyEnvRandGoal(size=16)
        self.env = ImgObsWrapper(FullyObsWrapper(self.base_env))
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=
                    (self.env.observation_space.shape[0], 
                     self.env.observation_space.shape[1], 
                     6),
                     dtype=np.float)
        self.settingslist = [11, 6, 4]
    def reset(self):
        # Place a goal square in the bottom-right corner fullonehot2dfullonehot2d
        obs = fullonehot2d(self.env.reset(), self.settingslist)
        obs = np.concatenate([obs[:,:,8:9], obs[:,:,10:11], obs[:,:,17:21]], axis=2)
        self.goal = obs[:,:,0:1]
        # print(self.goal.shape)
        return obs
    def render(self):
        self.env.render()
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = fullonehot2d(obs, self.settingslist)
        if np.sum(obs[:,:,18:21]) == 0: 
            obs[:,:,17] = obs[:,:,10]
        else: 
            obs[:,:,17] = obs[:,:,17] * 0
        obs = np.concatenate([self.goal, obs[:,:,10:11], obs[:,:,17:21]], axis=2)
        return obs, reward, done, info

if __name__ == '__main__':

    env = CustomOneHotMinigridEnv(None)
    # env = FullyObsWrapper(EmptyEnvRandGoal(size=4))#ImgObsWrapper
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
        
    import code
    import matplotlib.pyplot as plt
    while not done:
        print(type(obs))
        print(obs)

        print("agent\n", obs[:,:,1])
        print("goal \n", obs[:,:,0])

        # print(obs[:,:,10] + obs[:,:,8])
        print(env.observation_space)
        env.render()
        # img=obs['image']
        code.interact(local=locals())
        action = env.action_space.sample()
        print(action)
        obs, reward, done, info = env.step(action)
        print("reward", reward)
    print("agent\n", obs[:,:,1])
    print("goal \n", obs[:,:,0])
import numpy as np

from gym_minigrid.wrappers import *

env = ImgFullyObsWrapper(gym.make('MiniGrid-Empty-16x16-v0'))

obs = env.reset()
done = False
while not done:
    # def obs_to_str(obs):
    #     res = ''
    #     for i in range(obs.shape[0]):
    #         for j in range(obs.shape[1]):
    #             cell = obs[i][j]
    #             # cell representation
    #             cell_repr = '.'
    #             # wall
    #             if cell[0] == 2:
    #                 cell_repr = 'X'
    #             if cell[0] == 10:
    #                 cell_repr = 'A'
                
    #             res += cell_repr
    #         res += '\n'
    #     return res
    # print(obs_to_str(obs))
    print(obs)
    print(env.observation_space)
    action = int(input('action: '))
    print(action)
    obs, reward, done, info = env.step(action)
    print("reward", reward)
    import code
    import numpy as np
    code.interact(local=locals())
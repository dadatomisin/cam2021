from gym.envs.registration import register

register(id='MiniGrid-v0',
    entry_point='MiniGridEnv.envs:MiniGridEnv'
)
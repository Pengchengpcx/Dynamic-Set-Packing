from gym.envs.registration import register

register(
    id='Dynamic_setpacking-v0',
    entry_point='set_packing.model:Set_packing',
    max_episode_steps=100000
)

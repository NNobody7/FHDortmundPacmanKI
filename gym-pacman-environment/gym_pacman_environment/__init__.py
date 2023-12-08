from gym.envs.registration import register

register(
    id='pacman-python-v0',
    entry_point='gym_pacman_environment.envs:PacmanEnv',
)
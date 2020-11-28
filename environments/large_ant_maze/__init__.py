from gym.envs.registration import register

register(
    id='AntObstacles-v0',
    entry_point='ant_maze.ant_maze:AntObstaclesBigEnv',
    max_episode_steps=3000,
)

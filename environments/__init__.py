from gym.envs.registration import register

# Mujoco
# ----------------------------------------

# - randomised reward functions

register(
    'AntDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntDir2D-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDir2DEnv',
            'max_episode_steps': 200},
    max_episode_steps=200,
)

register(
    'AntGoal-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal:AntGoalEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_dir:HalfCheetahDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahVel-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_vel:HalfCheetahVelEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

# - randomised dynamics

register(
    id='Walker2DRandParams-v0',
    entry_point='environments.mujoco.rand_param_envs.walker2d_rand_params:Walker2DRandParamsEnv',
    max_episode_steps=200
)

register(
    id='HopperRandParams-v0',
    entry_point='environments.mujoco.rand_param_envs.hopper_rand_params:HopperRandParamsEnv',
    max_episode_steps=200
)

# Mujoco // Oracles
# ----------------------------------------

register(
    'HalfCheetahDirOracle-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_dir:HalfCheetahDirOracleEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahVelOracle-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_vel:HalfCheetahRandVelOracleEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntGoalOracle-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal:AntGoalOracleEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntDirOracle-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDirOracleEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    id='Walker2DRandParamsOracle-v0',
    entry_point='environments.mujoco.rand_param_envs.walker2d_rand_params:Walker2DRandParamsOracleEnv',
    max_episode_steps=200
)

register(
    id='HopperRandParamsOracle-v0',
    entry_point='environments.mujoco.rand_param_envs.hopper_rand_params:HopperRandParamsOracleEnv',
    max_episode_steps=200
)

# # 2D Navigation
# # ----------------------------------------
#
register(
    'PointEnv-v0',
    entry_point='environments.navigation.point_robot:PointEnv',
    kwargs={'max_episode_steps': 100},
    max_episode_steps=100,
)

register(
    'SparsePointEnv-v0',
    entry_point='environments.navigation.point_robot:SparsePointEnv',
    kwargs={'max_episode_steps': 100},
    max_episode_steps=100,
)
#
register(
    'PointEnvOracle-v0',
    entry_point='environments.navigation.point_robot:PointEnvOracle',
    kwargs={'max_episode_steps': 100},
    max_episode_steps=100,
)
register(
    'SparsePointEnvOracle-v0',
    entry_point='environments.navigation.point_robot:SparsePointEnvOracle',
    kwargs={'max_episode_steps': 100},
    max_episode_steps=100,
)

#
# # GridWorld
# # ----------------------------------------

register(
    'GridNavi-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'num_cells': 5, 'num_steps': 15},
)

# Oracles

register(
    'GridNaviOracle-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'oracle': True},
)

# Belief Oracle

register(
    'GridNaviBeliefOracle-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'belief_oracle': True},
)

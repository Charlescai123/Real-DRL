"""Configuration for Go2 Trot Env"""
import time

from ml_collections import ConfigDict
from src.configs.training import ppo, ddpg, sac
from src.envs import go2_wild_env
import torch
import numpy as np


def get_env_config():
    """Config for Environment"""

    config = ConfigDict()

    # Observation: [dis2goal, yaw_deviation, height, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
    config.observation_lb = np.array([0., 0., 0., -3.14, -3.14, -3.14, -1., -1., -1., -3.14, -3.14, -3.14])
    config.observation_ub = np.array([10., 3.14, 0.6, 3.14, 3.14, 3.14, 1., 1., 1., 3.14, 3.14, 3.14])
    # config.observation_lb = np.array([0., -3.14, -3.14, -3.14, -1., -1., -1., -3.14, -3.14, -3.14])
    # config.observation_ub = np.array([0.6, 3.14, 3.14, 3.14, 1., 1., 1., 3.14, 3.14, 3.14])

    # Action: [acc_vx, acc_vy, acc_vz, acc_wx, acc_wy, acc_wz]
    config.action_lb = np.array([-10., -10., -10., -20., -20., -20.])
    config.action_ub = np.array([10., 10., 10., 20., 20., 20.])
    # config.action_lb = np.array([-2., -2., -2., -3., -3., -3.])
    # config.action_ub = np.array([2., 2., 2., 3., 3., 3.])
    # config.action_lb = np.array([-5., -5., -5., -10., -10., -10.])
    # config.action_ub = np.array([5., 5., 5., 10., 10., 10.])

    # DRL-based gamma (0 < gamma < 1)
    config.gamma = 0.35
    config.obs_dim = config.observation_lb.shape[0]
    config.act_dim = config.action_lb.shape[0]
    config.episode_length_s = 120.
    config.env_dt = 0.01
    # config.env_dt = 0.002
    config.motor_strength_ratios = 1.
    config.motor_torque_delay_steps = 5
    config.use_yaw_feedback = False

    # Self-Learning Space (height, roll, pitch, yaw, vx, vy, vz, wx, wy, wz)
    # learning_space = [0.1, 0.3, 0.3, np.inf, 0.1, np.inf, np.inf, np.inf, np.inf, 1.]
    learning_space = np.array([0.08, 0.15, 0.15, np.inf, 0.2, np.inf, np.inf, np.inf, np.inf, 1.])
    # learning_space = [0.01, 0.03, 0.03, np.inf, 0.035, np.inf, np.inf, np.inf, np.inf, 0.1]

    # Trigger
    trigger_config = ConfigDict()
    trigger_config.trigger_type = 1  # 0 -> self-trigger | 1 -> event-trigger
    trigger_config.tau = 10
    config.trigger = trigger_config

    # PHY-Teacher
    phy_teacher_config = ConfigDict()
    phy_teacher_config.chi = 0.15
    phy_teacher_config.enable = False
    phy_teacher_config.correct = True
    phy_teacher_config.epsilon = 1
    phy_teacher_config.patch_interval = 5
    phy_teacher_config.apply_realtime_patch = True
    phy_teacher_config.learning_space = learning_space
    phy_teacher_config.trigger = trigger_config
    config.phy_teacher = phy_teacher_config

    # Matlab Engine
    matlab_engine_config = ConfigDict()
    cvx_toolbox_config = ConfigDict()
    matlab_engine_config.stdout = False
    matlab_engine_config.stderr = False
    matlab_engine_config.working_path = "src/phy_teacher/matlab/"
    cvx_toolbox_config.setup = False
    cvx_toolbox_config.relative_path = "./cvx"
    matlab_engine_config.cvx_toolbox = cvx_toolbox_config
    config.phy_teacher.matlab_engine = matlab_engine_config

    # Gait config
    gait_config = ConfigDict()
    gait_config.stepping_frequency = 2
    gait_config.initial_offset = np.array([0., 0.5, 0.5, 0.], dtype=np.float32) * (2 * np.pi)
    gait_config.swing_ratio = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    config.gait = gait_config

    # Stance controller
    config.base_position_kp = np.array([0., 0., 25.]) * 2
    config.base_position_kd = np.array([5., 5., 5.]) * 2
    config.base_orientation_kp = np.array([25., 25., 0.]) * 2
    config.base_orientation_kd = np.array([5., 5., 5.]) * 2
    config.qp_foot_friction_coef = 0.7
    config.qp_weight_ddq = np.diag([1., 1., 10., 10., 10., 1.])
    config.qp_body_inertia = np.array([[0.1585, 0.0001, -0.0155],
                                       [0.0001, 0.4686, 0.],
                                       [-0.0155, 0., 0.5245]]),
    config.use_full_qp = False
    config.clip_grf_in_sim = True
    config.foot_friction = 0.7  # 0.7
    config.desired_vx = 0.4  # vx
    config.desired_pz = 0.3  # height
    config.desired_wz = 0.  # wz
    config.clip_wz = [-0.7, 0.7]  # clip desired_wz

    # Swing controller
    config.swing_foot_height = 0.13
    config.swing_foot_landing_clearance = 0.02

    # Termination condition
    config.terminate_on_destination_reach = True
    config.terminate_on_body_contact = False
    config.terminate_on_dense_body_contact = True
    config.terminate_on_limb_contact = False
    config.terminate_on_yaw_deviation = False
    config.terminate_on_out_of_terrain = True
    config.terminate_on_timeout = True
    config.terrain_region = [[42, 62], [-7, 7]]
    config.yaw_deviation_threshold = np.pi * 0.75
    config.use_penetrating_contact = False

    # Check Fall condition
    config.check_fail_on_roll = 5 * np.pi / 9
    config.check_fail_on_pitch = np.pi / 2
    config.check_fail_on_height = 0.15

    # Reward
    reward_config = ConfigDict()
    reward_config.scales = {
        # 'upright': 0.02,
        # 'contact_consistency': 0.008,
        # 'foot_slipping': 0.032,
        # 'foot_clearance': 0.008,
        # 'out_of_bound_action': 0.01,
        # 'knee_contact': 5,
        # 'stepping_freq': 0.008,
        # 'com_distance_to_goal_squared': 0.016,
        # 'jerky_action': -1,
        # 'alive': 10,
        'fall_down': 1000000,
        # 'forward_speed': 0.1,
        # 'lin_vel_z': -2,
        'body_contact': 50,
        'energy_consumption': 0.002,
        'lin_vel_tracking': 1,
        'ang_vel_tracking': 1,
        'orientation_tracking': 1,
        'height_tracking': 5,
        'lyapunov': 10,
        'reach_time': 0.1,
        'distance_to_wp': 50,  # Distance to waypoint
        'reach_wp': 5000,  # Reach waypoint
        'reach_goal': 10000,  # Reach destination
        # 'com_height': 0.01,
    }
    reward_config.only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
    reward_config.tracking_sigma = 0.2  # tracking reward = exp(-error^2/sigma)
    reward_config.soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
    reward_config.soft_dof_vel_limit = 1.
    reward_config.soft_torque_limit = 1.
    reward_config.base_height_target = 1.
    reward_config.max_contact_force = 100.  # forces above this value are penalized
    config.reward = reward_config

    config.clip_negative_reward = False
    config.normalize_reward_by_phase = True

    config.terminal_rewards = []
    config.clip_negative_terminal_reward = False
    return config


def get_config():
    """Main entrance for the parsing the config"""
    config = ConfigDict()
    # config.training = ppo.get_training_config()  # Use PPO
    config.training = ddpg.get_training_config()  # Use DDPG
    # config.training = sac.get_training_config()  # Use SAC
    config.env_class = go2_wild_env.Go2WildExploreEnv
    config.environment = get_env_config()
    return config

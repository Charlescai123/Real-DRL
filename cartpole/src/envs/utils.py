import math
import os
import enum
import logging
from typing import Dict

import numpy as np
from gym.utils import seeding


def gym_seed(seed=None):
    np_random, seed = seeding.np_random(seed)
    return np_random


def is_unsafe(error_state: np.ndarray, learning_space: Dict) -> bool:
    """
    Judge the system is safe or not according to the learning space
    """
    state_map = {
        'x': error_state[0],
        'x_dot': error_state[1],
        'theta': error_state[2],
        'theta_dot': error_state[3],
    }
    for k, v in learning_space.items():
        lb, ub = v
        if state_map[k] >= ub or state_map[k] <= lb:
            return True

    return False


def energy_value(state: np.ndarray, p_mat: np.ndarray) -> int:
    """
    Get system energy value represented by s^T @ P @ s
    """
    energy = state.transpose() @ p_mat @ state
    return energy


def get_discrete_Ad_Bd(Ac: np.ndarray, Bc: np.ndarray, T: float):
    """
    Get the discrete form of matrices Ac and Bc given the sample period T
    """
    Ad = Ac * T + np.eye(4)
    # Bd = Bc * T
    Bd = Bc
    return Ad, Bd


def get_lyapunov_reward(error_state, p_matrix):
    """Lyapunov-like reward"""
    err_state = np.asarray(error_state)
    err_state = np.expand_dims(err_state, axis=0)
    Lya1 = np.matmul(err_state, p_matrix)
    Lya = np.matmul(Lya1, np.transpose(err_state))
    return Lya.item()


def get_unknown_distribution(a=None, b=None):
    rng = np.random.default_rng(seed=0)

    if a is None:
        a = 11 * np.random.random(1)[0]  # [0, 11]

    if b is None:
        b = 11 * np.random.random(1)[0]  # [0, 11]

    uu1 = -rng.beta(a, b) + rng.beta(a, b)
    uu1 *= 2.5  # [-0.5, 0.5]

    return uu1


def state2observations(state):
    x, x_dot, theta, theta_dot, failed = state
    observations = [x, x_dot, math.sin(theta), math.cos(theta), theta_dot]
    return observations, failed


def observations2state(observations, failed):
    x, x_dot, s_theta, c_theta, theta_dot = observations[:5]
    state = [x, x_dot, np.arctan2(s_theta, c_theta), theta_dot, failed]
    return state

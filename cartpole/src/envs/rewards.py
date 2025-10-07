import numpy as np
from omegaconf import OmegaConf

from src.envs.utils import get_lyapunov_reward
from src.utils.physical_design import MATRIX_P, MATRIX_A, MATRIX_B, F
from src.utils.utils import ActionMode


class Rewards:
    def __init__(self, trainer_env, reward_cfg):
        self._env = trainer_env
        self._config = OmegaConf.to_container(reward_cfg, resolve=True)

    def compute_reward(self):
        """The final output reward"""
        tot_rew = 0.0
        reward_info = {}

        for rew_fn_name, scale in self._config.items():
            reward_fn = getattr(self, rew_fn_name + '_reward')
            reward_item = scale * reward_fn()
            reward_info[rew_fn_name] = reward_item  # Update the dict
            tot_rew += reward_item

        return tot_rew, reward_info

    def teacher_discount_reward(self):
        """Penalty for using teacher"""
        teacher_discount_rwd = 0.0
        if self._env.trigger.action_mode == ActionMode.TEACHER:
            teacher_discount_rwd = -self._config['teacher_discount']
        return teacher_discount_rwd

    def action_reward(self):
        """Penalty for the action"""
        return -self._config['action'] * self._env.cartpole.ut * self._env.cartpole.ut

    def crash_reward(self):
        """Penalty for the crash"""
        return -self._config['crash']

    def lyapunov_reward(self):
        """Lyapunov-like reward in UCB form"""
        error_state_last = np.array(self._env.cartpole.last_state[:4]) - np.array(self._env.cartpole.params.set_point)
        error_state = np.array(self._env.cartpole.state[:4]) - np.array(self._env.cartpole.params.set_point)

        lyapunov_reward_current = get_lyapunov_reward(error_state_last, MATRIX_P)
        lyapunov_reward_next = get_lyapunov_reward(error_state, MATRIX_P)
        lyapunov_reward = lyapunov_reward_current - lyapunov_reward_next
        return self._config['lyapunov'] * lyapunov_reward

    def phydrl_reward(self):
        """Lyapunov-like reward in PhyDRL form"""
        MATRIX_F = np.expand_dims(F, axis=0)
        MATRIX_Abar = MATRIX_A + MATRIX_B.reshape(4, 1) @ MATRIX_F

        error_state_last = np.array(self._env.cartpole.last_state[:4]) - np.array(self._env.cartpole.params.set_point)
        error_state = np.array(self._env.cartpole.state[:4]) - np.array(self._env.cartpole.params.set_point)
        lyapunov_reward_next = get_lyapunov_reward(error_state, MATRIX_P)
        lyapunov_reward_current_aux = get_lyapunov_reward(error_state_last, MATRIX_Abar)
        lyapunov_reward_aux = lyapunov_reward_current_aux - lyapunov_reward_next
        return self._config['phydrl'] * lyapunov_reward_aux

    def distance_reward(self):
        """Distance reward of cartpole"""
        return self._config['distance'] * self._env.cartpole.get_distance_score()

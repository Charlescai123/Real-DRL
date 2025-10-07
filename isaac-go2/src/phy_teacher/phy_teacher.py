import logging
import sys
import copy
import time

import ml_collections
import numpy as np
import cvxpy as cp
from isaacgym.torch_utils import to_torch
import torch

from src.phy_teacher.lmi_solver import LMISolver
from src.phy_teacher.mat_engine import MatEngine
from src.physical_design import MATRIX_P
from src.utils.utils import energy_value_2d, check_safety, TriggerType, check_trigger_type

np.set_printoptions(suppress=True)


class PHYTeacher:
    """PHY-Teacher monitors the safety-critical systems in all envs"""

    def __init__(self, num_envs, teacher_cfg: ml_collections.ConfigDict(), device):
        self._device = device
        self._num_envs = num_envs

        # Matlab Engine
        self.mat_engine = MatEngine(cfg=teacher_cfg.matlab_engine)

        # Teacher Configure
        self.chi = torch.full((self._num_envs,), teacher_cfg.chi, dtype=torch.float32, device=device)
        self.epsilon = torch.full((self._num_envs,), teacher_cfg.epsilon, dtype=torch.float32, device=device)
        self.teacher_enable = torch.full((self._num_envs,), teacher_cfg.enable, dtype=torch.bool, device=device)
        self.teacher_correct = torch.full((self._num_envs,), teacher_cfg.correct, dtype=torch.bool, device=device)
        self.learning_space = to_torch(teacher_cfg.learning_space, dtype=torch.float32, device=device)
        self.tau = torch.full((self._num_envs,), teacher_cfg.trigger.tau, dtype=torch.float32, device=device)
        self.trigger_type = check_trigger_type(teacher_cfg.trigger.trigger_type)
        self.apply_rt_patch = teacher_cfg.apply_realtime_patch
        self.patch_interval = teacher_cfg.patch_interval

        # HAC Runtime
        self._plant_state = torch.zeros((self._num_envs, 12), dtype=torch.float32, device=device)
        self._last_plant_state = torch.zeros((self._num_envs, 12), dtype=torch.float32, device=device)
        self._patch_activate = torch.full((self._num_envs,), False, dtype=torch.bool, device=device)
        self._teacher_activate = torch.full((self._num_envs,), False, dtype=torch.bool, device=device)
        self._patch_center = torch.zeros((self._num_envs, 12), dtype=torch.float32, device=device)
        self._center_update = torch.full((self._num_envs,), True, dtype=torch.bool, device=device)
        self._dwell_step = torch.zeros(self._num_envs, dtype=torch.float32, device=device)
        self.patch_interval = torch.full((self._num_envs,), self.patch_interval, dtype=torch.int64, device=device)
        self.apply_realtime_patch = torch.full((self._num_envs,), self.apply_rt_patch, dtype=torch.bool, device=device)
        self.lmi_solver = LMISolver()

        # Patch kp and kd
        self._default_kp_runtime = torch.diag(to_torch([0., 0., 50., 50., 50., 0.], device=device))
        self._default_kd_runtime = torch.diag(to_torch([10., 10., 10., 10., 10., 10.], device=device))
        self._default_kp = to_torch([[-0., -0., -0., -0., -0., -0.],
                                     [-0., -0., -0., -0., -0., -0.],
                                     [-0., -0., 94., 0., -0., 0.],
                                     [-0., -0., - 0., 91., 0, 0],
                                     [-0., -0., 0., 0, 91, 0],
                                     [-0., -0., 0., 0, -0., 93]], device=device)
        self._default_kd = to_torch([[8., 0., 0., -0., 0., 0.],
                                     [0., 8., -0., 0., -0, 0.],
                                     [0., -0., 18., 0., -0., 0.],
                                     [0., 0., -0., 18, 0., 0.],
                                     [-0., 0., 0., -0., 18., -0.],
                                     [0., 0., 0., 0., -0., 18.]], device=device)

        self._patch_kp = torch.stack([to_torch(self._default_kp, device=self._device)] * self._num_envs, dim=0)
        self._patch_kd = torch.stack([to_torch(self._default_kd, device=self._device)] * self._num_envs, dim=0)

        self.action_counter = torch.zeros(self._num_envs, dtype=torch.int, device=device)

    def update(self, error_state: torch.Tensor):
        """
        Update real-time plant and corresponding patch center if state is unsafe (error_state is 2d)
        """
        # print(f"error_state: {error_state}")
        self._last_plant_state = self._plant_state  # Record last plant state

        self._plant_state = error_state
        energy_2d = energy_value_2d(state=error_state[:, 2:], p_mat=to_torch(MATRIX_P, device=self._device))
        unsafe = check_safety(error_state, self.learning_space)

        # Find objects that need to be deactivated according to trigger type
        if self.trigger_type == TriggerType.SELF:
            to_deactivate = (self._dwell_step >= self.tau) & self._teacher_activate
        elif self.trigger_type == TriggerType.EVENT:
            to_deactivate = (~unsafe) & self._teacher_activate
        else:
            raise RuntimeError("Unknown trigger type. Check the configure please")

        if torch.any(to_deactivate):
            indices = torch.argwhere(to_deactivate)
            for idx in indices:
                logging.info(f"The system {idx} status is inside the learning space, deactivate HA-Teacher")
            self._teacher_activate[to_deactivate] = False

        # Find objects that need to be activated
        to_activate = check_safety(error_state, self.learning_space) & (~self._teacher_activate)

        if torch.any(to_activate):
            indices = torch.argwhere(to_activate)
            for idx in indices:
                if self.trigger_type == TriggerType.SELF:
                    self._dwell_step[idx] = 0
                self._teacher_activate[idx] = True  # Activate teacher

                if torch.any(self.teacher_correct):     # Teaching-to-learn
                    self._patch_activate[idx] = True  # Patch activate flag
                    self._patch_center[tuple(idx)] = self._plant_state[tuple(idx)] * self.chi[tuple(idx)]
                    logging.info(
                        f"Activate HA-Teacher at {int(idx)} with new patch center: {self._patch_center[tuple(idx)]}")
                else:       # Without Teaching-to-learn
                    self._patch_center[tuple(idx)] = self._plant_state[tuple(idx)] * 0.

        return energy_2d

    def get_action(self):
        """
        Get updated teacher action during real-time
        """
        self.action_counter += 1

        teacher_actions = torch.zeros((self._num_envs, 6), dtype=torch.float32, device=self._device)
        prev_teacher_actions = torch.zeros((self._num_envs, 6), dtype=torch.float32, device=self._device)
        dwell_flags = torch.full((self._num_envs,), False, dtype=torch.bool, device=self._device)

        # If All HA-Teacher disabled
        if not torch.any(self.teacher_enable):
            logging.info("All HA-teachers are disabled")
            return teacher_actions, dwell_flags

        # If All HA-Teacher deactivated
        if not torch.any(self._teacher_activate):
            logging.info("All HA-teachers are deactivated")
            return teacher_actions, dwell_flags

        # Find the object that needs to be patched
        # to_patch = self.apply_realtime_patch & (self.action_counter % self.patch_interval == 0)
        to_patch = self.apply_realtime_patch & self._patch_activate
        if torch.any(to_patch):
            indices = torch.argwhere(to_patch)

            for idx in indices:
                if torch.any(self.teacher_correct):  # Teaching-to-learn
                    logging.info(f"Applying realtime patch at index {int(idx)}")
                    self.realtime_patch(int(idx))

                else:       # Without teaching-to-learn
                    self._patch_kp[idx, :], self._patch_kd[idx, :] = self._default_kp_runtime, self._default_kd_runtime

                self._patch_activate[int(idx)] = False

        # print(f"kp: {self._patch_kp}")
        # print(f"kd: {self._patch_kd}")
        # Do not turn on
        # time.sleep(0.02)

        # Find objects with PHY-Teacher enabled and activated
        teacher_alive = self.teacher_enable & self._teacher_activate
        indices = torch.argwhere(teacher_alive)
        if indices.size == 0:
            raise RuntimeError("teacher_alive contains no True values, indices is empty. Check the code please")

        kp_mul_term = (self._plant_state[indices, :6] - self._patch_center[indices, :6])
        kd_mul_term = (self._plant_state[indices, 6:] - self._patch_center[indices, 6:])

        teacher_actions[indices.flatten()] = torch.squeeze(self._patch_kp[indices] @ kp_mul_term.unsqueeze(-1) +
                                                           self._patch_kd[indices] @ kd_mul_term.unsqueeze(-1))

        ################################  Action Compensation  ###################################
        # prev_kp_mul_term = (self._last_plant_state[indices, :6] - self._patch_center[indices, :6])
        # prev_kd_mul_term = (self._last_plant_state[indices, 6:] - self._patch_center[indices, 6:])
        #
        # prev_teacher_actions[indices.flatten()] = torch.squeeze(
        #     self._patch_kp[indices] @ prev_kp_mul_term.unsqueeze(-1) +
        #     self._patch_kd[indices] @ prev_kd_mul_term.unsqueeze(-1))
        #
        # error_state = self._plant_state - self._patch_center  # current error state
        # error_state_last = self._last_plant_state - self._patch_center  # Use last time error_state
        # error_state = error_state[indices.flatten(), 6:]
        # error_state_last = error_state_last[indices.flatten(), 6:]
        #
        # Compute the compensation action
        # error_state_predict = error_state_last - prev_teacher_actions[indices.flatten()]
        # prediction_difference = error_state - error_state_predict
        # compensation_action = prediction_difference
        #
        # Terminal action of teacher
        # teacher_actions[indices.flatten()] += compensation_action * 1
        #
        ##########################################################################################

        # Dwell time for self-trigger
        if self.trigger_type == TriggerType.SELF:
            dwell_flags[indices] = True  # Set dwell flag for them
            assert torch.all(self._dwell_step <= self.tau)
            for idx in indices:
                self._dwell_step[idx] += 1
                logging.info(f"HA-Teacher {int(idx)} runs for dwell time: "
                             f"{int(self._dwell_step[idx])}/{int(self.tau[idx])}")

        return teacher_actions, dwell_flags

    def realtime_patch(self, idx):
        """Give real-time patch for the plant[idx]"""
        # import pdb
        # pdb.set_trace()
        tracking_err = self._plant_state[idx, 2:]
        # res, lmi_solved = self.lmi_solver.patch_lmi(tracking_err=tracking_err.cpu(), device=self._device)
        F_kp, F_kd, lmi_solved = self.mat_engine.patch_lmi(tracking_err=tracking_err.cpu())
        # print(f"F_kp: {F_kp}")
        # print(f"F_kd: {F_kd}")

        # If LMIs solved successfully
        if lmi_solved:
            self._patch_kp[idx, :], self._patch_kd[idx, :] = (to_torch(F_kp, device=self.device),
                                                              to_torch(F_kd, device=self.device))
        # Use default value if failed to solve LMIs
        else:
            print(f"Failed to solve LMI, use default patch instead.")
            self._patch_kp[idx, :], self._patch_kd[idx, :] = self._default_kp, self._default_kd

        # self._patch_kp[idx, :], self._patch_kd[idx, :] = self._default_kp, self._default_kd

    def get_As_Bs_by_state(self, plant_state: torch.Tensor):
        """
        Update the physical knowledge matrices A(s) and B(s) in real-time based on the current state
        """
        N = plant_state.shape[0]

        roll = plant_state[:, 3]
        pitch = plant_state[:, 4]
        yaw = plant_state[:, 5]

        cos_r = torch.cos(roll)
        sin_r = torch.sin(roll)
        cos_p = torch.cos(pitch)
        sin_p = torch.sin(pitch)
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)

        # Rotation matrix (N x 3 x 3)
        Rx = torch.zeros((N, 3, 3), device=self.device)
        Rx[:, 0, 0] = 1
        Rx[:, 1, 1] = cos_r
        Rx[:, 1, 2] = -sin_r
        Rx[:, 2, 1] = sin_r
        Rx[:, 2, 2] = cos_r

        Ry = torch.zeros((N, 3, 3), device=self.device)
        Ry[:, 0, 0] = cos_p
        Ry[:, 0, 2] = sin_p
        Ry[:, 1, 1] = 1
        Ry[:, 2, 0] = -sin_p
        Ry[:, 2, 2] = cos_p

        Rz = torch.zeros((N, 3, 3), device=self.device)
        Rz[:, 0, 0] = cos_y
        Rz[:, 0, 1] = -sin_y
        Rz[:, 1, 0] = sin_y
        Rz[:, 1, 1] = cos_y
        Rz[:, 2, 2] = 1

        Rzy = torch.bmm(Rz, Ry)  # [N x 3 x 3]
        Rzyx = torch.bmm(Rzy, Rx)  # [N x 3 x 3]

        # As [N, 10, 10]
        As = torch.zeros((N, 10, 10), device=self.device)
        As[:, 0, 6] = 1
        As[:, 1:4, 7:10] = Rzyx

        # Bs [N, 10, 6]
        Bs = torch.zeros((N, 10, 6), device=self.device)
        Bs[:, 4:, :] = torch.eye(6, device=self.device).unsqueeze(0).repeat(N, 1, 1)

        return As, Bs

    def get_discrete_Ad_Bd(self, As: torch.Tensor, Bs: torch.Tensor, T: float):
        """
        Convert batch of continuous-time system matrices (As, Bs) to discrete-time (Ad, Bd).

        Args:
            As: [N, 4, 4] continuous-time A matrices
            Bs: [N, 4, m] continuous-time B matrices
            T: sample period

        Returns:
            Ad_batch: [N, 4, 4] discrete-time A matrices
            Bd_batch: [N, 4, m] discrete-time B matrices
        """
        N, n, _ = As.shape
        eye = np.eye(n)[None, :, :]  # shape: [1, 4, 4]
        Ad = T * As + eye  # broadcasting eye to [N, 4, 4]
        Bd = Bs  # or Bd = T * Bc_batch if needed
        return Ad, Bd

    @property
    def device(self):
        return self._device

    @property
    def plant_state(self):
        return self._plant_state

    @property
    def patch_center(self):
        return self._patch_center

    @property
    def patch_gain(self):
        return self._patch_kp, self._patch_kd

    @property
    def dwell_step(self):
        return self._dwell_step

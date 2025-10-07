import logging
import time

import ml_collections
import torch
# import numpy as np

from isaacgym.torch_utils import to_torch
from src.utils.utils import ActionMode, energy_value_2d, check_safety, TriggerType, check_trigger_type
from src.physical_design import MATRIX_P


# torch.set_printoptions(sci_mode=False)


class Trigger:

    def __init__(self, num_envs, trigger_cfg: ml_collections.ConfigDict(), device: str = "cuda"):
        self._device = device
        self._num_envs = num_envs
        self._plant_action = torch.zeros((self._num_envs, 6), dtype=torch.float32, device=device)
        self._action_mode = torch.full((self._num_envs,), ActionMode.STUDENT.value, dtype=torch.int64, device=device)
        self._trigger_type = check_trigger_type(trigger_cfg.trigger_type)
        self._tau = torch.full((self._num_envs,), trigger_cfg.tau, dtype=torch.float32, device=device)
        self._dwell_step = torch.zeros(self._num_envs, dtype=torch.float32, device=device)

        # self._default_epsilon = 1  # Default epsilon
        self._last_action_mode = None

    def get_terminal_action(self,
                            stu_action: torch.Tensor,
                            tea_action: torch.Tensor,
                            plant_state: torch.Tensor,
                            learning_space: torch.Tensor,
                            dwell_flag=None):
        """Given the system state and envelope boundary (epsilon), analyze the current safety status
        and return which action (hp/ha) to switch for control"""

        if self._trigger_type == TriggerType.SELF:
            terminal_stance_ddq, action_mode = self.self_trig_action(
                stu_action=stu_action,
                tea_action=tea_action,
                plant_state=plant_state,
                learning_space=learning_space,
                dwell_flag=dwell_flag
            )
        elif self._trigger_type == TriggerType.EVENT:
            terminal_stance_ddq, action_mode = self.event_trig_action(
                stu_action=stu_action,
                tea_action=tea_action,
                plant_state=plant_state,
                learning_space=learning_space,
            )
        else:
            raise RuntimeError(f"Unknown trigger type {self._trigger_type}")

        return terminal_stance_ddq, action_mode

    def self_trig_action(self,
                         stu_action: torch.Tensor,
                         tea_action: torch.Tensor,
                         plant_state: torch.Tensor,
                         learning_space: torch.Tensor,
                         dwell_flag=None):
        """Given the system state and envelope boundary (epsilon), analyze the current safety status
        and return which action (hp/ha) to switch for control"""

        if dwell_flag is None:
            dwell_flag = torch.full((self._num_envs,), False, dtype=torch.bool, device=self._device)

        terminal_stance_ddq = torch.zeros((self._num_envs, 6), dtype=torch.float32, device=self._device)
        action_mode = torch.full((self._num_envs,), ActionMode.UNCERTAIN.value, dtype=torch.int64, device=self._device)

        self._last_action_mode = self._action_mode

        # Obtain all energies
        energy_2d = energy_value_2d(plant_state[:, 2:], to_torch(MATRIX_P, device=self._device))

        # Check current safety status
        is_unsafe = check_safety(error_state=plant_state, learning_space=learning_space)

        for i, energy in enumerate(energy_2d):

            # Display current system status based on energy
            if is_unsafe[i].item():
                logging.info(f"current system {i} is unsafe")
            else:
                logging.info(f"current system {i} is safe")

            # When Teacher disabled or deactivated
            if not torch.any(tea_action[i]) and bool(dwell_flag[i]) is False:
                logging.info("HA-Teacher is deactivated, use HP-Student's action instead")
                self._action_mode[i] = ActionMode.STUDENT.value
                self._plant_action[i] = stu_action[i]

                terminal_stance_ddq[i] = stu_action[i]
                action_mode[i] = ActionMode.STUDENT.value
                continue

            # Teacher activated
            if self._last_action_mode[i] == ActionMode.TEACHER.value:

                # Teacher Dwell time
                if dwell_flag[i]:
                    if tea_action[i] is None:
                        raise RuntimeError(f"Unrecognized HA-Teacher action {tea_action[i]} from {i} for dwelling")
                    else:
                        logging.info("Continue HA-Teacher action in dwell time")
                        self._action_mode[i] = ActionMode.TEACHER.value
                        self._plant_action[i] = tea_action[i]

                        terminal_stance_ddq[i] = tea_action[i]
                        action_mode[i] = ActionMode.TEACHER.value

                # Switch back to HPC
                else:
                    self._action_mode[i] = ActionMode.STUDENT.value
                    self._plant_action[i] = stu_action[i]
                    logging.info(f"Max HA-Teacher dwell time achieved, switch back to HP-Student control")

                    terminal_stance_ddq[i] = stu_action[i]
                    action_mode[i] = ActionMode.STUDENT.value

            elif self._last_action_mode[i] == ActionMode.STUDENT.value:

                # Inside safety subset
                if not is_unsafe[i].item():
                    self._action_mode[i] = ActionMode.STUDENT.value
                    self._plant_action[i] = stu_action[i]
                    logging.info(f"Continue HP-Student action")

                    terminal_stance_ddq[i] = stu_action[i]
                    action_mode[i] = ActionMode.STUDENT.value

                # Outside safety envelope (bounded by epsilon)
                else:
                    logging.info(f"Switch to HA-Teacher action for safety concern")
                    self._action_mode[i] = ActionMode.TEACHER.value
                    self._plant_action[i] = tea_action[i]

                    terminal_stance_ddq[i] = tea_action[i]
                    action_mode[i] = ActionMode.TEACHER.value
            else:
                raise RuntimeError(f"Unrecognized last action mode: {self._last_action_mode[i]} for {i}")

        return terminal_stance_ddq, action_mode

    def event_trig_action(self,
                          stu_action: torch.Tensor,
                          tea_action: torch.Tensor,
                          plant_state: torch.Tensor,
                          learning_space: torch.Tensor):
        """Given the system state and envelope boundary (epsilon), analyze the current safety status
        and return which action (hp/ha) to switch for control"""

        terminal_stance_ddq = torch.zeros((self._num_envs, 6), dtype=torch.float32, device=self._device)
        action_mode = torch.full((self._num_envs,), ActionMode.UNCERTAIN.value, dtype=torch.int64, device=self._device)

        self._last_action_mode = self._action_mode

        # Obtain all energies
        energy_2d = energy_value_2d(plant_state[:, 2:], to_torch(MATRIX_P, device=self._device))

        # Check current safety status
        is_unsafe = check_safety(error_state=plant_state, learning_space=learning_space)

        for i, energy in enumerate(energy_2d):

            # Display current system status based on energy
            if is_unsafe[i].item():
                logging.info(f"current system {i} is unsafe")
            else:
                logging.info(f"current system {i} is safe")

            # When Teacher disabled or deactivated
            if not torch.any(tea_action[i]):
                logging.info("HA-Teacher is deactivated, use HP-Student's action instead")
                self._action_mode[i] = ActionMode.STUDENT.value
                self._plant_action[i] = stu_action[i]

                terminal_stance_ddq[i] = stu_action[i]
                action_mode[i] = ActionMode.STUDENT.value
                continue

            # Inside the Learning Space
            if not torch.any(is_unsafe[i]):

                # Teacher Dwell time
                if self._last_action_mode[i] == ActionMode.STUDENT.value:
                    logging.info(f"System is in learning space, continue HP-Student control")

                # Switch back to HP-Student Control
                else:
                    logging.info(f"System is back to learning space, switch to HP-Student control")

                self._action_mode[i] = ActionMode.STUDENT.value
                self._plant_action[i] = stu_action[i]
                terminal_stance_ddq[i] = stu_action[i]
                action_mode[i] = ActionMode.STUDENT.value

            # Outside the Learning Space
            else:

                logging.info(f"System is outside learning space, switch to HA-Teacher control for safety concern")
                self._action_mode[i] = ActionMode.TEACHER.value
                self._plant_action[i] = tea_action[i]
                terminal_stance_ddq[i] = tea_action[i]
                action_mode[i] = ActionMode.TEACHER.value

        return terminal_stance_ddq, action_mode

    @property
    def device(self):
        return self._device

    @property
    def plant_action(self):
        return self._plant_action

    @property
    def action_mode(self):
        return self._action_mode

    @property
    def last_action_mode(self):
        return self._last_action_mode

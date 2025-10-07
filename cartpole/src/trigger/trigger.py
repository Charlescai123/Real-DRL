import logging
import numpy as np

from src.envs.utils import is_unsafe
from src.logger.utils import logger
from src.utils.utils import ActionMode, TriggerType

np.set_printoptions(suppress=True)


class Trigger:

    def __init__(self, config):

        self.trigger_type = TriggerType(config.trigger_type)
        self.tau = config.tau

        # Real time status
        self._plant_action = 0
        self._action_mode = ActionMode.TEACHER
        self._last_action_mode = None

    def reset(self, state, learning_space):
        self._plant_action = 0
        # energy = energy_value(state=state, p_mat=MATRIX_P)
        unsafe = is_unsafe(error_state=state, learning_space=learning_space)
        self._action_mode = ActionMode.TEACHER if unsafe else ActionMode.STUDENT
        # print(f"energy is: {energy}")
        print(f"action mode: {self._action_mode}")
        self._last_action_mode = None

    def get_terminal_action(self, student_action, teacher_action, plant_state, learning_space, dwell_flag=None):
        """Given the system state and learning space boundary, analyze the current safety status
        and return which action (hp/ha) to switch for control"""

        if self.trigger_type == TriggerType.SELF:  # Self-trigger
            terminal_action, action_mode = self.self_trig_action(
                student_action=student_action,
                teacher_action=teacher_action,
                plant_state=plant_state,
                learning_space=learning_space,
                dwell_flag=dwell_flag
            )
        elif self.trigger_type == TriggerType.EVENT:  # Event-trigger
            terminal_action, action_mode = self.event_trig_action(
                student_action=student_action,
                teacher_action=teacher_action,
                plant_state=plant_state,
                learning_space=learning_space,
            )
        else:
            raise RuntimeError(f"Unknown trigger type {self.trigger_type}")

        return terminal_action, action_mode

    def self_trig_action(self, student_action, teacher_action, plant_state, learning_space, dwell_flag=None):
        """Self-triggered"""

        self._last_action_mode = self._action_mode

        # When Teacher deactivated
        if teacher_action is None:
            logger.debug("PHY-Teacher deactivated, use DRL-Student's action instead")
            self._action_mode = ActionMode.STUDENT
            self._plant_action = student_action
            return self._plant_action, self._action_mode

        # Check current safety status
        is_unsafe_flag = is_unsafe(error_state=plant_state, learning_space=learning_space)

        # Inside learning space
        if not is_unsafe_flag:
            logger.debug(f"current system is safe")

            # Teacher already activated
            if self._last_action_mode == ActionMode.TEACHER:

                # Teacher Dwell time
                if dwell_flag is True:
                    if teacher_action is None:
                        raise RuntimeError(f"Unrecognized PHY-Teacher action {teacher_action} for dwelling")
                    else:
                        logger.debug("PHY-Teacher action continues in dwell time")
                        self._action_mode = ActionMode.TEACHER
                        self._plant_action = teacher_action
                        return teacher_action, ActionMode.TEACHER

                # Switch back to Student
                else:
                    self._action_mode = ActionMode.STUDENT
                    self._plant_action = student_action
                    logger.debug(f"Max dwell time achieved, switch back to DRL-Student control")
                    return student_action, ActionMode.STUDENT
            else:
                self._action_mode = ActionMode.STUDENT
                self._plant_action = student_action
                logger.debug(f"Continue DRL-Student action")
                return student_action, ActionMode.STUDENT

        # Outside the learning space
        else:
            logger.debug(f"current system is unsafe")
            logger.debug(f"Use PHY-Teacher action for safety concern")
            self._action_mode = ActionMode.TEACHER
            self._plant_action = teacher_action
            return teacher_action, ActionMode.TEACHER

    def event_trig_action(self, student_action, teacher_action, plant_state, learning_space):
        """Event-triggered"""
        self._last_action_mode = self._action_mode

        # When Teacher deactivated
        if teacher_action is None:
            logger.debug("PHY-Teacher deactivated, use DRL-Student's action instead")
            self._action_mode = ActionMode.STUDENT
            self._plant_action = student_action
            return self._plant_action, self._action_mode

        # Check current safety status
        is_unsafe_flag = is_unsafe(error_state=plant_state, learning_space=learning_space)

        # Inside the learning space
        if not is_unsafe_flag:
            logging.debug(f"current system is safe")

            # Teacher Dwell time
            if self._last_action_mode == ActionMode.STUDENT:
                logging.debug(f"System is in learning space, continue DRL-Student control")

            # Switch back to DRL-Student Control
            else:
                logging.debug(f"System is back to learning space, switch to DRL-Student control")

            self._action_mode = ActionMode.STUDENT
            self._plant_action = student_action

        # Outside the learning space
        else:
            logging.debug(f"current system is unsafe")
            logging.debug(f"System is outside learning space, switch to PHY-Teacher control for safety concern")
            self._action_mode = ActionMode.TEACHER
            self._plant_action = teacher_action

        return self._plant_action, self._action_mode

    @property
    def plant_action(self):
        return self._plant_action

    @property
    def action_mode(self):
        return self._action_mode

    @property
    def last_action_mode(self):
        return self._last_action_mode

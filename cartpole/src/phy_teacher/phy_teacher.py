import numpy as np

from src.utils.physical_design import F_Simplex
from src.phy_teacher.mat_engine import MatEngine
from src.envs.utils import is_unsafe, get_discrete_Ad_Bd
from src.logger.utils import logger
from src.utils.utils import TriggerType

np.set_printoptions(suppress=True)


class PHYTeacher:
    def __init__(self, teacher_cfg, cartpole_cfg):

        # Matlab Engine
        self.mat_engine = MatEngine(cfg=teacher_cfg.matlab_engine)

        # Teacher Configuration
        self.chi = teacher_cfg.chi
        self.epsilon = teacher_cfg.epsilon
        self.max_dwell_steps = teacher_cfg.tau

        self.trigger_type = TriggerType(teacher_cfg.trigger_type)
        self.teacher_enable = teacher_cfg.teacher_enable
        self.teacher_correct = teacher_cfg.teacher_correct
        self.learning_space = teacher_cfg.learning_space
        self.force_compensation = teacher_cfg.force_compensation

        # Cart-Pole Configuration
        self.mc = cartpole_cfg.mass_cart
        self.mp = cartpole_cfg.mass_pole
        self.g = cartpole_cfg.gravity
        self.l = cartpole_cfg.length_pole / 2
        self.freq = cartpole_cfg.frequency

        # Real-time status
        self._plant_state = None
        self._teacher_activate = False
        self._patch_update = False
        self._patch_center = np.array([0., 0., 0., 0.])
        self._patch_gain = F_Simplex  # F_simplex
        self._dwell_step = 0  # Dwell step
        self._activation_cnt = 0
        self._last_plant_state = np.array([0., 0., 0., 0.])  # Record last time plant state

    def reset(self, state):
        self._plant_state = state
        self._teacher_activate = False
        self._patch_update = False
        self._patch_center = np.array([0., 0., 0., 0.])
        self._patch_gain = F_Simplex  # F_simplex
        self._dwell_step = 0
        self._activation_cnt = 0  # Count of teacher activate times
        self._last_plant_state = np.array([0., 0., 0., 0.])  # Record last time plant state

    def update(self, state: np.ndarray):
        """
        Update real-time plant state and corresponding patch center if state is unsafe
        """
        self._last_plant_state = self._plant_state  # Update last plant state
        self._plant_state = state  # Update current plant state
        unsafe = is_unsafe(state, self.learning_space)  # is unsafe or not

        # When HA-Teacher activated
        if self._teacher_activate:

            # Find objects that need to be deactivated according to trigger type
            if self.trigger_type == TriggerType.SELF:
                if self._dwell_step >= self.max_dwell_steps and self._teacher_activate:
                    logger.debug(f"Self-Trigger: Reaching maximum dwell steps, deactivate HA-Teacher")
                    self._teacher_activate = False

            elif self.trigger_type == TriggerType.EVENT:
                if not unsafe and self._teacher_activate:
                    logger.debug(f"Event-Trigger: System status back to learning space, deactivate HA-Teacher")
                    self._teacher_activate = False
            else:
                raise RuntimeError("Unknown trigger type. Check the configure please")

        # When HA-Teacher deactivated
        else:
            # Outside learning space
            if unsafe:
                self._dwell_step = 0
                self._teacher_activate = True  # Activate teacher
                self._patch_update = True  # Activate patch gain
                self._activation_cnt += 1  # Activation count plus 1
                self._patch_center = self._plant_state * self.chi  # Update patch center
                logger.debug(f"Activate HA-Teacher and updated patch center is: {self._patch_center}")

    def get_action(self):
        """
        Get updated teacher action during real-time
        """
        approximated_freq = 150

        # If teacher is disabled or deactivated
        if self.teacher_enable is False or self._teacher_activate is False:
            return None, False

        # Moment to update the patch gain
        if self._patch_update:
            self._patch_update = False

            As, Bs = self.get_As_Bs_by_state(state=self._plant_state)
            self.Ak, self.Bk = get_discrete_Ad_Bd(Ac=As, Bc=Bs, T=1 / approximated_freq)

            # Call Matlab Engine for patch gain (F_hat)
            F_hat, t_min = self.mat_engine.system_patch(Ak=self.Ak, Bk=self.Bk, state=self._plant_state)

            # F_hat, t_min = self.mat_engine.system_patch_origin(Ak=Ak, Bk=Bk, chi=self.chi)
            if t_min > 0:
                print(f"LMI has no solution, use last updated patch")
                # import pdb
                # pdb.set_trace()
            else:
                self._patch_gain = np.asarray(F_hat).squeeze() * approximated_freq

        # State error form
        # error_state = self._last_plant_state - self._patch_center    # Use last time error_state
        error_state = self._plant_state - self._patch_center  # current error state
        error_state_last = self._last_plant_state - self._patch_center  # Use last time error_state

        # redundancy_term = self._patch_center - Ak @ self._patch_center
        # v1 = np.squeeze(redundancy_term[1] / Bk[1])
        # v2 = np.squeeze(redundancy_term[3] / Bk[3])
        # v = np.linalg.pinv(self.Bk).squeeze() @ (np.eye(4) - self.Ak) @ sbar_star
        # v = (v1 + v2) / 2

        # Action from HA-Teacher
        teacher_action = self._patch_gain @ error_state

        if self.force_compensation:
            ################################  Action Compensation  ###################################
            ## Compute the compensation action
            error_state_predict = self.Ak @ np.expand_dims(error_state_last,
                                                           axis=0).T + (self.Bk * teacher_action) / approximated_freq
            prediction_difference = np.expand_dims(error_state, axis=0).T - error_state_predict

            Bk = (1 / approximated_freq) * self.Bk

            aux_difference = (Bk.transpose()) @ prediction_difference
            aux_Bk = (Bk.transpose()) @ Bk

            plambda = 0.001
            compensation_action = np.linalg.inv(aux_Bk + plambda) @ aux_difference
            # print(f"prediction_difference: {prediction_difference}")
            # print(f"error_state_predict: {error_state_predict}")
            # print(f"compensation_action: {compensation_action}")

            # squeeze to one dimension
            compensation_action = np.squeeze(compensation_action).item() * 1.0

            ## Terminal action of teacher
            teacher_action = teacher_action + compensation_action * 1.0
            # print(f"teacher_action: {teacher_action}")
            ##########################################################################################

        logger.debug(f"patch gain: {self._patch_gain}")
        logger.debug(f"self._plant_state: {self._plant_state}")
        logger.debug(f"self._patch_center: {self._patch_center}")
        logger.debug(f"Generated teacher action: {teacher_action}")

        unsafe = is_unsafe(self._plant_state, self.learning_space)

        # Self Trigger
        if self.trigger_type == TriggerType.SELF:
            assert self._dwell_step <= self.max_dwell_steps

            # Increment one dwell step
            self._dwell_step += 1
            logger.debug(f"HA-Teacher runs for dwell time: {self._dwell_step}/{self.max_dwell_steps}")

            return teacher_action, True

        # Event Trigger
        elif self.trigger_type == TriggerType.EVENT:
            if unsafe:
                return teacher_action, True
            else:
                return teacher_action, False

        # Unknown Trigger type
        else:
            raise Exception(f"Unknown trigger type: {TriggerType.UNKNOWN}")

    def get_As_Bs_by_state(self, state: np.ndarray):
        """
        Update the physical knowledge matrices A(s) and B(s) in real-time based on the current state
        """
        x = state[0]
        x_dot = state[1]
        theta = state[2]
        theta_dot = state[3]

        As = np.zeros((4, 4))
        As[0][1] = 1
        As[2][3] = 1

        mc = self.mc
        mp = self.mp
        g = self.g
        l = self.l

        term = 4 / 3 * (mc + mp) - mp * np.cos(theta) * np.cos(theta)

        As[1][2] = -mp * g * np.sin(theta) * np.cos(theta) / (theta * term)
        As[1][3] = 4 / 3 * mp * l * np.sin(theta) * theta_dot / term
        As[3][2] = g * np.sin(theta) * (mc + mp) / (l * theta * term)
        As[3][3] = -mp * np.sin(theta) * np.cos(theta) * theta_dot / term

        Bs = np.zeros((4, 1))
        Bs[1] = 4 / 3 / term
        Bs[3] = -np.cos(theta) / (l * term)

        return As, Bs

    @property
    def plant_state(self):
        return self._plant_state

    @property
    def patch_center(self):
        return self._patch_center

    @property
    def patch_gain(self):
        return self._patch_gain

    @property
    def dwell_step(self):
        return self._dwell_step

    @property
    def activation_cnt(self):
        return self._activation_cnt

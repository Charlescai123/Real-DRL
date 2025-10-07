import numpy as np
from src.physical_design import KD, KP


class PHYTeacher:
    def __init__(self, initial_ref_point=None):

        self.kp = KP
        self.kd = KD
        self.ref_state = initial_ref_point

        self.patch_kp = np.zeros_like(self.kp)
        self.patch_kd = np.zeros_like(self.kd)
        self.patch_ready = False

    def set_ref_point(self, ref_point):
        self.ref_state = ref_point

    def get_action(self, robot_state, patch_center):
        if patch_center is None:
            self.patch_ready = False

        if self.patch_ready:
            action = np.squeeze(self.patch_kp @ (robot_state[:6] - patch_center[:6]) * -1
                                + self.patch_kd @ (robot_state[6:] - patch_center[6:]) * -1)
        else:
            action = np.squeeze(self.kp @ (robot_state[:6] - self.ref_state[:6]) * -1
                                + self.kd @ (robot_state[6:] - self.ref_state[6:]) * -1)
        return action

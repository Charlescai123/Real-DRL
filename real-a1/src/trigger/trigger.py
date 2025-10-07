import copy
import pickle
import numpy as np
import dataclasses

np.set_printoptions(suppress=True)
import time


@dataclasses.dataclass
class TriggerConfig:
    # Simulation Settings
    chi: float = 0.25
    eta: float = 1.2
    tau: int = 30
    beta: float = 0.9
    kappa: float = 0.02
    epsilon: float = 0.6


class Trigger:

    def __init__(self, params: TriggerConfig, redis_connection, initial_P, ref_state):
        self.params = params
        self.student_activate = True
        self._P_matrix = initial_P
        self._dwell_time = 0
        self._max_dwell_steps = self.params.tau
        self._refer_state = ref_state
        self.patch_center = None
        self.patch_information = None
        self.redis_connection = redis_connection
        self.redis_params = self.redis_connection.params

    def trigger_monitor(self, robot_state, tracking_error):
        energy_value = self.calculate_energy_value(tracking_error[2:])
        if energy_value > self.params.epsilon:
            self.student_activate = False
            trigger_state = robot_state
            if self.patch_center is None:
                self.patch_center = self.params.chi * tracking_error + self._refer_state
                self.patch_information = (copy.deepcopy(trigger_state),
                                          copy.deepcopy(self.patch_center), True)
                self.send_patch_update_command(self.patch_information)
                current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                print(f"[{current_time}]: Patch condition is triggered, send command to update patch")

        if not self.student_activate:
            self._dwell_time += 1

            if self._dwell_time > self._max_dwell_steps:
                self.student_activate = True
                self._dwell_time = 0
                # reset the patch center to the reference state
                self.patch_center = None

        return self.student_activate, self.patch_center

    def send_patch_update_command(self, patch_information):
        patch_info_pack = pickle.dumps(patch_information)
        self.redis_connection.publish(channel=self.redis_params.ch_edge_patch_update, message=patch_info_pack)

    def calculate_energy_value(self, state_vector):
        state_vector = state_vector.reshape(-1, 1)
        value = state_vector.transpose() @ self._P_matrix @ state_vector
        return value

import numpy as np
import cvxpy as cp
import pickle
import time
from src.physical_design import MATRIX_P


def system_patch(roll, pitch, yaw):
    """
     Computes the patch gain with roll pitch yaw.

     Args:
       roll: Roll angle (rad).
       pitch: Pitch angle (rad).
       yaw: Yaw angle (rad).

     Returns:
       F_kp: Proportional feedback gain matrix.
       F_kd: Derivative feedback gain matrix.
     """

    Rzyx = np.array([[np.cos(yaw) / np.cos(pitch), np.sin(yaw) / np.cos(pitch), 0],
                     [-np.sin(yaw), np.cos(yaw), 0],
                     [np.cos(yaw) * np.tan(pitch), np.sin(yaw) * np.tan(pitch), 1]])


    bP = np.array([[140.6434, 0, 0, 0, 0, 0, 5.3276, 0, 0, 0],
                   [0, 134.7596, 0, 0, 0, 0, 0, 6.6219, 0, 0],
                   [0, 0, 134.7596, 0, 0, 0, 0, 0, 6.622, 0],
                   [0, 0, 0, 49.641, 0, 0, 0, 0, 0, 6.8662],
                   [0, 0, 0, 0, 11.1111, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 3.3058, 0, 0, 0, 0],
                   [5.3276, 0, 0, 0, 0, 0, 3.6008, 0, 0, 0],
                   [0, 6.6219, 0, 0, 0, 0, 0, 3.6394, 0, 0],
                   [0, 0, 6.622, 0, 0, 0, 0, 0, 3.6394, 0],
                   [0, 0, 0, 6.8662, 0, 0, 0, 0, 0, 4.3232]])

    # Sampling period
    T = 1 / 30  # work in 25 to 30

    # System matrices (continuous-time)
    aA = np.zeros((10, 10))
    aA[0, 6] = 1
    aA[1:4, 7:10] = Rzyx
    aB = np.zeros((10, 6))
    aB[4:, :] = np.eye(6)

    # System matrices (discrete-time)
    B = aB * T
    A = np.eye(10) + T * aA

    alpha = 0.8
    kappa = 0.01
    chi = 0.2
    # gamma = 1
    # hd = 0.000
    gamma1 = 1
    gamma2 = 1  # 1

    b1 = 1 / 0.15  # height  0.15
    b2 = 1 / 0.35  # velocity 0.3

    D = np.array([[b1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, b2, 0, 0, 0, 0, 0]])
    c1 = 1 / 45
    c2 = 1 / 70
    C = np.array([[c1, 0, 0, 0, 0, 0],
                  [0, c1, 0, 0, 0, 0],
                  [0, 0, c1, 0, 0, 0],
                  [0, 0, 0, c2, 0, 0],
                  [0, 0, 0, 0, c2, 0],
                  [0, 0, 0, 0, 0, c2]])

    Q = cp.Variable((10, 10), PSD=True)
    T = cp.Variable((6, 6), PSD=True)
    R = cp.Variable((6, 10))
    mu = cp.Variable((1, 1))

    constraints = [cp.bmat([[(alpha - kappa * (1 + (1 / gamma2))) * Q, Q @ A.T + R.T @ B.T],
                            [A @ Q + B @ R, Q / (1 + gamma2)]]) >> 0,
                   cp.bmat([[Q, R.T],
                            [R, T]]) >> 0,
                   (1 - chi * gamma1) * mu - (1 - (2 * chi) + (chi / gamma1)) >> 0,
                   Q - mu * np.linalg.inv(bP) >> 0,
                   np.identity(2) - D @ Q @ D.transpose() >> 0,
                   np.identity(6) - C @ T @ C.transpose() >> 0,
                   # T - hd * np.identity(6) >> 0,
                   mu - 1.0 >> 0,
                   ]

    # Define problem and objective
    problem = cp.Problem(cp.Minimize(0), constraints)

    # Solve the problem
    problem.solve(solver=cp.CVXOPT)

    # Extract optimal values
    # Check if the problem is solved successfully
    if problem.status == 'optimal':
        print("Optimization successful.")
    else:
        print("Optimization failed.")

    optimal_Q = Q.value
    optimal_R = R.value

    P = np.linalg.inv(optimal_Q)

    # Compute aF
    aF = np.round(aB @ optimal_R @ P, 0)
    Fb2 = aF[6:10, 0:4]
    # Compute F_kp
    F_kp = -np.block([[np.zeros((2, 6))], [np.zeros((4, 2)), Fb2]])
    # Compute F_kd
    F_kd = -aF[4:10, 4:10]

    if np.all(np.linalg.eigvals(P) > 0):
        print("LMIs feasible")
    else:
        print("LMIs infeasible")

    return F_kp, F_kd


class PatchUpdater:
    def __init__(self, redis_connection):
        self.redis_connection = redis_connection
        self.redis_params = self.redis_connection.params

        # Teacher Configure
        self.p_mat = MATRIX_P
        self.patch_subscriber = self.redis_connection.subscribe(channel=self.redis_params.ch_edge_patch_update)
        self.patch_kp = None
        self.patch_kd = None

    def listen_and_update(self):
        while True:
            patch_info_pack= self.patch_subscriber.parse_response()[2]
            patch_info = pickle.loads(patch_info_pack)
            trigger_state, patch_center, is_update = patch_info
            if is_update:
                roll, pitch, yaw = trigger_state[3:6]
                F_kp, F_kd = self.patch_compute(roll, pitch, yaw)
                self.send_new_patch((F_kp, F_kd))
                current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                print(f"[{current_time}]: New patch is computed and sent")

    def send_new_patch(self, patch_gain):
        patch_gain_pack = pickle.dumps(patch_gain)
        self.redis_connection.publish(channel=self.redis_params.ch_edge_patch_gain, message=patch_gain_pack)

    @staticmethod
    def patch_compute(roll, pitch, yaw):
        F_kp, F_kd = system_patch(roll, pitch, yaw)
        return F_kp, F_kd

    def run(self):
        self.listen_and_update()
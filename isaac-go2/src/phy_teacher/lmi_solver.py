import logging
import sys
import time

import asyncio
import numpy as np
import cvxpy as cp
# import cvxpy as cp
from typing import Tuple, Any, Optional
import multiprocessing as mp

from isaacgym.torch_utils import to_torch
import torch

from omegaconf import DictConfig
from numpy import linalg as LA
import matplotlib.pyplot as plt
from collections import deque
from numpy.linalg import pinv
from torch import Tensor


class LMISolver:
    def __init__(self):
        pass

    @staticmethod
    # @tf.function
    def patch_lmi(tracking_err, device="cuda:0") -> Tuple[Optional[Tuple[Any, Any]], bool]:
        """
         Computes the patch gain with roll pitch yaw.

         Args:
           tracking_err: error of [height, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
           device: device to torch tensor

         Returns:
           [F_kp: Proportional feedback gain matrix.
           F_kd]: Derivative feedback gain matrix.
           flag: solve successfully or not
         """

        roll = tracking_err[1]
        pitch = tracking_err[2]
        yaw = tracking_err[3]

        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        Rzyx = Rz.dot(Ry.dot(Rx))
        # print(f"Rzyx: {Rzyx}")

        # Rzyx = np.array([[np.cos(yaw) / np.cos(pitch), np.sin(yaw) / np.cos(pitch), 0],
        #                  [-np.sin(yaw), np.cos(yaw), 0],
        #                  [np.cos(yaw) * np.tan(pitch), np.sin(yaw) * np.tan(pitch), 1]])

        # Sampling period
        T = 1 / 20  # work in 25 to 30

        # System matrices (continuous-time)
        aA = np.zeros((10, 10))
        aA[0, 6] = 1
        aA[1:4, 7:10] = Rzyx
        aB = np.zeros((10, 6))
        aB[4:, :] = np.eye(6)

        # System matrices (discrete-time)
        B = aB * T
        A = np.eye(10) + T * aA

        alpha = 0.9
        hd = 1e-10
        phi = 0.15

        cc = 0.6
        b1 = 1 / 0.8  # yaw
        b2 = 1 / (1.0 * cc)  # height
        b3 = 1 / 1.5  # velocity
        b4 = 1 / 1

        D = np.matrix([[b2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       # [0, 0, b4, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, b3, 0, 0, 0, 0, 0],
                       [0, 0, 0, b1, 0, 0, 0, 0, 0, 0]])
        c1 = 1 / 25
        c2 = 1 / 50
        C = np.matrix([[c1, 0, 0, 0, 0, 0],
                       [0, c1, 0, 0, 0, 0],
                       [0, 0, c1, 0, 0, 0],
                       [0, 0, 0, c2, 0, 0],
                       [0, 0, 0, 0, c2, 0],
                       [0, 0, 0, 0, 0, c2]])

        Z = np.diag(tracking_err)

        Q = cp.Variable((10, 10), PSD=True)
        T = cp.Variable((6, 6), PSD=True)
        R = cp.Variable((6, 10))

        constraints = [cp.bmat([[alpha * Q, Q @ A.T + R.T @ B.T],
                                [A @ Q + B @ R, Q / (1 + phi)]]) >> 0,
                       cp.bmat([[Q, R.T],
                                [R, T]]) >> 0,
                       Q - 10 * Z @ Z >> 0,
                       np.identity(3) - D @ Q @ D.transpose() >> 0,
                       np.identity(6) - C @ T @ C.transpose() >> 0,
                       T - hd * np.identity(6) >> 0,
                       ]

        # Define problem and objective
        problem = cp.Problem(cp.Minimize(0), constraints)

        # Solve the problem
        problem.solve(solver=cp.CVXOPT)

        # Extract optimal values
        # Check if the problem is solved successfully
        if problem.status == 'optimal':
            logging.info("Optimization successful.")

            optimal_Q = Q.value
            optimal_R = R.value

            # print(optimal_Q)
            # print(optimal_R)

            P = np.linalg.inv(optimal_Q)

            # Compute aF
            aF = np.round(aB @ optimal_R @ P, 0)
            Fb2 = aF[6:10, 0:4]

            # Compute F_kp
            F_kp = -np.block([
                [np.zeros((2, 6))],
                [np.zeros((4, 2)), Fb2]])
            # Compute F_kd
            F_kd = -aF[4:10, 4:10]

            # print(f"Solved F_kp is: {F_kp}")
            # print(f"Solved F_kd is: {F_kd}")

            # Check if the problem is solved successfully
            if np.all(np.linalg.eigvals(P) > 0):
                logging.info("LMIs feasible")
            else:
                print("LMIs infeasible")

            res = (to_torch(F_kp, device=device), to_torch(F_kd, device=device))
            is_solved = True

        # Failed to solve LMIs
        else:
            print(f"tracking_err: {tracking_err}")
            print("Optimization failed.")
            res = None
            is_solved = False

        return res, is_solved

    @staticmethod
    # @tf.function
    def patch_lmi_old(roll, pitch, yaw, device="cuda:0"):
        """
         Computes the patch gain with roll pitch yaw.

         Args:
           roll: Roll angle (rad).
           pitch: Pitch angle (rad).
           yaw: Yaw angle (rad).
           device: device to torch tensor

         Returns:
           F_kp: Proportional feedback gain matrix.
           F_kd: Derivative feedback gain matrix.
         """

        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        Rzyx = Rz.dot(Ry.dot(Rx))
        # print(f"Rzyx: {Rzyx}")

        # Rzyx = np.array([[np.cos(yaw) / np.cos(pitch), np.sin(yaw) / np.cos(pitch), 0],
        #                  [-np.sin(yaw), np.cos(yaw), 0],
        #                  [np.cos(yaw) * np.tan(pitch), np.sin(yaw) * np.tan(pitch), 1]])

        # Sampling period
        T = 1 / 22  # work in 25 to 30

        # System matrices (continuous-time)
        aA = np.zeros((10, 10))
        aA[0, 6] = 1
        aA[1:4, 7:10] = Rzyx
        aB = np.zeros((10, 6))
        aB[4:, :] = np.eye(6)

        # System matrices (discrete-time)
        B = aB * T
        A = np.eye(10) + T * aA

        alpha = 0.9
        kappa = 0.01
        gamma = 1  # 1
        mu = 1e-6

        b1 = 1 / 0.15  # height  0.15
        b2 = 1 / 0.35  # velocity 0.35
        # b3 = 1 / 0.1   # yaw 0.1
        b4 = 1 / 1.  # yaw rate 1

        D = np.array([[b1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, b2, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, b4]])
        c1 = 1 / 5
        c2 = 1 / 10
        C = np.array([[c1, 0, 0, 0, 0, 0],
                      [0, c1, 0, 0, 0, 0],
                      [0, 0, c1, 0, 0, 0],
                      [0, 0, 0, c2, 0, 0],
                      [0, 0, 0, 0, c2, 0],
                      [0, 0, 0, 0, 0, c2]])

        Q = cp.Variable((10, 10), PSD=True)
        T = cp.Variable((6, 6), PSD=True)
        R = cp.Variable((6, 10))

        constraints = [cp.bmat([[(alpha - kappa * (1 + (1 / gamma))) * Q, Q @ A.T + R.T @ B.T],
                                [A @ Q + B @ R, Q / (1 + gamma)]]) >> 0,
                       cp.bmat([[Q, R.T],
                                [R, T]]) >> 0,
                       np.identity(3) - D @ Q @ D.transpose() >> 0,
                       np.identity(6) - C @ T @ C.transpose() >> 0,
                       Q - mu * np.identity(10) >> 0,
                       T - mu * np.identity(6) >> 0,
                       ]

        # Define problem and objective
        problem = cp.Problem(cp.Minimize(0), constraints)

        # Solve the problem
        problem.solve(solver=cp.CVXOPT)

        # Extract optimal values
        # Check if the problem is solved successfully
        if problem.status == 'optimal':
            logging.info("Optimization successful.")
        else:
            print("Optimization failed.")

        optimal_Q = Q.value
        optimal_R = R.value

        P = np.linalg.inv(optimal_Q)

        # Compute aF
        aF = np.round(aB @ optimal_R @ P, 0)
        Fb2 = aF[6:10, 0:4]

        # Compute F_kp
        F_kp = -np.block([
            [np.zeros((2, 6))],
            [np.zeros((4, 2)), Fb2]])
        # Compute F_kd
        F_kd = -aF[4:10, 4:10]

        # print(f"Solved F_kp is: {F_kp}")
        # print(f"Solved F_kd is: {F_kd}")

        # Check if the problem is solved successfully
        if np.all(np.linalg.eigvals(P) > 0):
            logging.info("LMIs feasible")
        else:
            print("LMIs infeasible")

        return (to_torch(F_kp, device=device),
                to_torch(F_kd, device=device))


if __name__ == '__main__':
    As = np.array([[0, 1, 0, 0],
                   [0, 0, -1.42281786576776, 0.182898194776782],
                   [0, 0, 0, 1],
                   [0, 0, 25.1798795199119, 0.385056459685276]])

    Bs = np.array([[0,
                    0.970107410065162,
                    0,
                    -2.04237185222105]])

    Ak = np.array([[1, 0.0100000000000000, 0, 0],
                   [0, 1, -0.0142281786576776, 0.00182898194776782],
                   [0, 0, 1, 0.0100000000000000],
                   [0, 0, 0.251798795199119, 0.996149435403147]])

    Bk = np.array([[0,
                    0.00970107410065163,
                    0,
                    -0.0204237185222105]])

    sd = np.array([[0.234343490000000,
                    0,
                    -0.226448960000000,
                    0]])
    roll, pitch, yaw = 0.0, 0.0, -0.000

    # zz = np.array([0.1, 0.05, 0.1, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.16]) * 1.0
    # zz = np.array([-0.0158, -0.0417, -0.1517, 0.0032, 0.2703, -0.1057, 0.0472, 0.3559, -0.2925, -0.6624])   # Raw data

    zz = np.array([0.0073, -0.4821, -0.1257, 0.0309, 0.1311, 0.2719, 0.0342, 0.4687, -0.3929, -0.6300])

    # zz = np.array([-0.0158, -0.0417, -0.1517, 0.0032, 0.2703, -0.1057, 0.0472, 0.3559, -0.2925, -0.6624])  # Raw data

    # zz = np.array([0.01, 0.00, -0.01, -0.00, 0.4, 0.03, 0.03, -0.08, -0.08, 0.08])
    (F_kp, F_kd), _ = LMISolver().patch_lmi(zz)
    # phy_teacher = HATeacher()
    # K = phy_teacher.feedback_law(0, 0, 0)
    # print(K)

    # testN = 100
    # s = time.time()
    # for i in range(testN):
    #     F_kp, F_kd = HATeacher.system_patch(0., 0., 0.)
    # e = time.time()
    # duration = (e - s) / testN
    print(f"F_kp is: {F_kp}")
    print(f"F_kd is: {F_kd}")
    # print(f"time: {duration}")

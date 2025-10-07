"""A mpc based controller framework."""

import enum
import numpy as np
from typing import Tuple, Any
from typing import Mapping
from omegaconf import DictConfig

from envs.robot.unitree_a1.motors import MotorCommand
from envs.robot.gait_scheduler import offset_gait_scheduler
from envs.robot.state_estimator import com_velocity_estimator
from envs.robot.mpc_controller import stance_leg_controller_mpc, stance_leg_controller_quadprog, swing_leg_controller


class ControllerMode(enum.Enum):
    DOWN = 1
    STAND = 2
    WALK = 3
    TERMINATE = 4


class GaitType(enum.Enum):
    CRAWL = 1
    TROT = 2
    FLYTROT = 3


class LocomotionController(object):

    def __init__(
            self,
            robot: Any,
            desired_speed: Tuple[float, float] = [0., 0.],
            desired_twisting_speed: float = 0.,
            desired_com_height: float = 0.,
            mpc_body_mass: float = 110 / 9.8,
            mpc_body_inertia: Tuple[float, float, float, float, float, float, float, float, float] = (
                    0.07335, 0, 0, 0, 0.25068, 0, 0, 0, 0.25447),
            gait_config: DictConfig = None,
            vel_estimator_config: DictConfig = None,
            swing_config: DictConfig = None,
            stance_config: DictConfig = None,
            logdir: str = 'logs/',
    ):
        """Initializes the class.

        Args:
          robot: A robot instance. (Provides sensor input and kinematics)
          ddpg_agent: An agents instance used to get PhyDRL action (WBC will get mpc action if None)
          desired_speed: desired CoM speed in x-y plane.
          desired_twisting_speed: desired CoM rotating speed in z direction.
          desired_com_height: The standing height of CoM of the robot.
          mpc_body_mass: The total mass of the robot.
          mpc_body_inertia: The inertia matrix in the body principle frame. We assume the body principle
                            coordinate frame has x-forward and z-up.
          swing_params: Parameters for swing leg controller.
          stance_params: Parameters for stance leg controller.
        """
        self._robot = robot
        self._gait_scheduler = None
        self._velocity_estimator = None

        # Parameter config
        self._gait_config = gait_config
        self._swing_config = swing_config
        self._stance_config = stance_config
        self._vel_estimator_config = vel_estimator_config

        # Logs
        self._logs = []
        self._logdir = logdir

        # Desired v/w
        self._desired_speed = desired_speed
        self._desired_twisting_speed = desired_twisting_speed

        # MPC parameters
        self._desired_com_height = desired_com_height
        self._mpc_body_mass = mpc_body_mass
        self._mpc_body_inertia = np.asarray(mpc_body_inertia)

        self._time_since_reset = 0

        self._mode = ControllerMode.WALK
        self._desired_gait = GaitType.TROT


        self._robot_state = np.zeros(12)
        vx = self._desired_speed[0]
        vy = self._desired_speed[1]
        wz = self._desired_twisting_speed
        pz = self._desired_com_height

        self._ref_point = np.array([0., 0., pz,  # p
                                    0., 0., 0.,  # rpy
                                    vx, vy, 0.,  # v
                                    0., 0., wz])  # rpy_dot

        self.beta_distribution_noise = np.random.beta(a=0.5, b=0.5, size=6) * 0.5
        self.setup_controllers(robot)

    def setup_controllers(self, robot):
        print("Setting up the whole body controller...")
        self._clock = lambda: robot.time_since_reset

        # Gait Generator
        init_gait_phase = np.array(self._gait_config.init_gait_phase)
        gait_param_tuple = tuple(self._gait_config.gait_parameter_tuple)
        self._gait_scheduler = offset_gait_scheduler.OffsetGaitScheduler(
            robot=robot,
            init_phase=init_gait_phase,
            gait_parameters=gait_param_tuple
        )

        # State Estimator
        window_size = self._vel_estimator_config.window_size
        ground_normal_window_size = self._vel_estimator_config.ground_normal_window_size
        self._velocity_estimator = com_velocity_estimator.COMVelocityEstimator(
            robot=robot,
            velocity_window_size=window_size,
            ground_normal_window_size=ground_normal_window_size
        )

        # Swing Leg Controller
        self._swing_controller = \
            swing_leg_controller.RaibertSwingLegController(
                robot=robot,
                gait_scheduler=self._gait_scheduler,
                state_estimator=self._velocity_estimator,
                desired_speed=self._desired_speed,
                desired_twisting_speed=self._desired_twisting_speed,
                desired_com_height=self._desired_com_height,
                swing_params=self._swing_config
            )

        # Stance Leg Controller
        if self._stance_config.qp_solver == 'quadprog':
            self._stance_controller = \
                stance_leg_controller_quadprog.TorqueStanceLegController(
                    robot=robot,
                    gait_scheduler=self._gait_scheduler,
                    state_estimator=self._velocity_estimator,
                    desired_speed=self._desired_speed,
                    desired_twisting_speed=self._desired_twisting_speed,
                    desired_com_height=self._desired_com_height,
                    body_mass=self._mpc_body_mass,
                    body_inertia=self._mpc_body_inertia,
                    stance_params=self._stance_config
                )

        elif self._stance_config.qp_solver == 'qpOASES':
            self._stance_controller = \
                stance_leg_controller_mpc.TorqueStanceLegController(
                    robot=robot,
                    gait_scheduler=self._gait_scheduler,
                    state_estimator=self._velocity_estimator,
                    desired_speed=self._desired_speed,
                    desired_twisting_speed=self._desired_twisting_speed,
                    desired_com_height=self._desired_com_height,
                    body_mass=self._mpc_body_mass,
                    body_inertia=self._mpc_body_inertia,
                    stance_params=self._stance_config
                )

        else:
            raise RuntimeError("Unspecified objective function for stance controller")

        print("MPC controller setup complete.")


    @property
    def swing_leg_controller(self):
        return self._swing_controller

    @property
    def stance_leg_controller(self):
        return self._stance_controller

    @property
    def gait_scheduler(self):
        return self._gait_scheduler

    @property
    def state_estimator(self):
        return self._velocity_estimator

    @property
    def time_since_reset(self):
        return self._time_since_reset

    def reset_controllers(self):
        self._reset_time = self._clock()
        self._time_since_reset = 0
        self._gait_scheduler.reset()
        self._velocity_estimator.reset(self._time_since_reset)
        self._swing_controller.reset(self._time_since_reset)
        self._stance_controller.reset(self._time_since_reset)

    def update(self):
        self._time_since_reset = self._clock() - self._reset_time
        self._gait_scheduler.update()
        self._velocity_estimator.update(self._gait_scheduler.desired_leg_states)
        self._swing_controller.update(self._time_since_reset)
        self._stance_controller.update(self._time_since_reset)
        self._robot_state = self.state_vector

    def get_action(self, action=None, student_on=True, action_mag=1):
        swing_action = self._swing_controller.get_action()
        # normalized action is used for residual policy learning
        if action is not None:
            if student_on:
                normalized_action = action / action_mag
                ddq = action + self._stance_controller.get_model_action()
            else:
                ddq = action
                normalized_action = (action - self._stance_controller.get_model_action()) / action_mag
        else:
            ddq = self._stance_controller.get_model_action() # use default mpc
            normalized_action = 0

        stance_action, _ = self.stance_leg_controller.map_ddq_to_action(ddq=ddq)
        motor_action = self.get_motor_action(swing_action=swing_action, stance_action=stance_action)
        # qp_sol = None
        return motor_action, normalized_action

    def get_motor_action(self,
                         swing_action: Mapping[int, MotorCommand],
                         stance_action: Mapping[int, MotorCommand]
                         ):
        """Convert swing/stance leg action to motor action"""
        actions = []
        for joint_id in range(12): # 12 motors
            if joint_id in swing_action:
                actions.append(swing_action[joint_id])
            else:
                assert joint_id in stance_action
                actions.append(stance_action[joint_id])

        motor_action = MotorCommand(
            desired_position=[action.desired_position for action in actions],
            kp=[action.kp for action in actions],
            desired_velocity=[action.desired_velocity for action in actions],
            kd=[action.kd for action in actions],
            desired_torque=[
                action.desired_torque for action in actions
            ])

        return motor_action


    @property
    def state_vector(self):
        com_position = self.state_estimator.com_position_in_ground_frame
        com_velocity = self.state_estimator.com_velocity_in_body_frame
        com_roll_pitch_yaw = np.array(
            self._robot.pybullet_client.getEulerFromQuaternion(
                self.state_estimator.com_orientation_quaternion_in_ground_frame))
        com_roll_pitch_yaw_rate = self._robot.base_angular_velocity_in_body_frame

        state_vector = np.hstack((com_position, com_roll_pitch_yaw, com_velocity, com_roll_pitch_yaw_rate))
        return state_vector

    @property
    def robot_state(self):
        return self._robot_state

    @property
    def tracking_error(self):
        return self._robot_state - self.ref_point

    @property
    def ref_point(self):
        return self._ref_point

    def set_gait_parameters(self, gait_parameters):
        raise NotImplementedError()

    def set_qp_weight(self, qp_weight):
        raise NotImplementedError()

    def set_mpc_mass(self, mpc_mass):
        raise NotImplementedError()

    def set_mpc_inertia(self, mpc_inertia):
        raise NotImplementedError()

    def set_mpc_foot_friction(self, mpc_foot_friction):
        raise NotImplementedError()

    def set_foot_landing_clearance(self, foot_landing_clearance):
        raise NotImplementedError()
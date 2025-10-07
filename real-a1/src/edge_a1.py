import time
import signal
import sys
import copy
import numpy as np
import threading
import pickle

from src.edge_control import EdgeControl, EdgeControlConfig


def signal_handler(signal, frame):
    global run
    print("Safe exiting")
    run = False


run = True
signal.signal(signal.SIGINT, signal_handler)


class A1EdgeControl(EdgeControl):

    def __init__(self, edge_config: EdgeControlConfig, robot, mpc_controller, teacher, trigger, agent_a, agent_b,
                 redis_connection, real_world_robot=False):
        super().__init__(edge_config, agent_a, agent_b, redis_connection)
        self.steps_since_calibration = 0
        self.params = edge_config
        self.robot = robot
        self.mpc_controller = mpc_controller
        self.safety_trigger = trigger
        self.teacher = teacher
        self.action_magnitude = np.array(self.params.ControlParams.action_magnitude)
        self.real_world_robot = real_world_robot

        self.t6 = threading.Thread(target=self.subscribe_patch_gain)
        # warm up
        for _ in range(10):
            agent_a.get_action(np.zeros(agent_a.observation_shape))
            agent_b.get_action(np.zeros(agent_b.observation_shape))

    def generate_action(self):
        while True:
            if self.params.AgentParams.add_actions_observations:
                action_list_dim = self.params.AgentParams.n_action_step * self.agent_a.action_shape
                action_observations = np.zeros(shape=action_list_dim)
            else:
                action_observations = []

            while not self.robot.normal_mode:
                self.reset_control()

            t0 = time.perf_counter()
            time_out_counter = 0

            self.mpc_controller.reset_controllers()

            while self.robot.normal_mode:
                self.step += 1
                self.steps_since_calibration += 1
                self.mpc_controller.update()

                # the state is from an estimator from the mpc controller
                # com_position, com_roll_pitch_yaw, com_velocity, com_roll_pitch_yaw_rate
                robot_state = np.array(self.mpc_controller.robot_state)
                tracking_error = np.array(self.mpc_controller.tracking_error)
                observations = np.hstack(
                    (tracking_error, action_observations))  # option 1: include action to compensate for delay
                # option 2: include absolute state to avoid the ambiguity of the tracking error

                agent = self.agent_a if self.agent_a_active else self.agent_b
                student_activate, patch_center = self.safety_trigger.trigger_monitor(robot_state, tracking_error)

                if student_activate:
                    action = agent.get_action(tracking_error)
                    applied_action = action * self.params.ControlParams.action_magnitude
                else:
                    applied_action = self.teacher.get_action(robot_state, patch_center)

                motor_action, normalized_action = self.mpc_controller.get_action(action=applied_action,
                                                                                 student_on=student_activate,
                                                                                 action_mag=self.action_magnitude)
                self.robot.step(action=motor_action)

                failed = self.if_termination(robot_state)

                if failed:
                    self.robot.normal_mode = False  # set to false to reset

                normal_mode = copy.deepcopy(self.robot.normal_mode)
                self.edge_trajectory = [observations, self.last_action, failed, normal_mode, self.step,
                                        student_activate]

                self.last_action = normalized_action

                if self.params.AgentParams.add_actions_observations:
                    action_observations = np.hstack((action_observations, normalized_action))[
                                          self.agent_a.action_shape:]

                if self.trajectory_sending_condition.acquire(False):
                    self.trajectory_sending_condition.notify_all()
                    self.trajectory_sending_condition.release()

                one_loop_time = time.perf_counter() - t0

                if one_loop_time < self.sample_period:
                    time.sleep(self.sample_period - one_loop_time)
                    time_out_counter = 0
                else:
                    time_out_counter += 1

                if time_out_counter >= 10:
                    t0 = time.perf_counter()
                    print("TIMEOUT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    time_out_counter = 0
                else:
                    t0 = t0 + self.sample_period

                if run is False:
                    sys.exit("Safe exiting...")

    def reset_control(self):
        self.robot.reset()  # double check, reset to what position in real world.
        self.safety_trigger.student_activate = True
        self.teacher.patch_ready = False
        self.robot.normal_mode = True
        if self.real_world_robot:
            sys.exit("Safe exiting to reset...")  # this is for real world as we don't have a reset controller

    def set_normal_mode(self, normal_mode):
        self.robot.normal_mode = False

    def if_termination(self, robot_state):
        # todo implement termination condition
        return False

    def subscribe_patch_gain(self):  # consider move this to the edge control loop
        while True:
            patch_gain_pack = self.patch_gain_subscriber.parse_response()[2]
            patch_gain = pickle.loads(patch_gain_pack)
            self.teacher.patch_kp, self.teacher.patch_kd = patch_gain
            self.teacher.patch_ready = True

    def run(self):
        self.t2.daemon = True
        self.t3.daemon = True
        self.t4.daemon = True
        self.t5.daemon = True
        self.t6.daemon = True

        self.t2.start()
        self.t3.start()
        self.t4.start()
        self.t5.start()
        self.t6.start()

        self.generate_action()

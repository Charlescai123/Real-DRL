import pickle
import threading
import time

import dataclasses
from src.student.DDPG import DDPGAgent
from src.redis import RedisConfig, RedisConnection
from src.student.DDPG import DDPGConfig as AgentConfig
from job.job_config import JobConfig


@dataclasses.dataclass
class ControlConfig:
    frequency: int = 200  # hz
    initialize_from_cloud: bool = False
    action_magnitude: list = dataclasses.field(default_factory=lambda: [4, 4, 2, 8, 8, 4])


@dataclasses.dataclass
class EdgeControlConfig:
    RedisParams: RedisConfig = dataclasses.field(default_factory=RedisConfig)
    AgentParams: AgentConfig = dataclasses.field(default_factory=AgentConfig)
    ControlParams: ControlConfig = dataclasses.field(default_factory=ControlConfig)
    JobParams: JobConfig = dataclasses.field(default_factory=JobConfig)


class EdgeControl:
    def __init__(self, params: EdgeControlConfig, agent_a: DDPGAgent, agent_b: DDPGAgent,
                 redis_connection: RedisConnection):
        self.params = params
        self.redis_connection = redis_connection

        self.weights_subscriber = self.redis_connection.subscribe(channel=self.params.RedisParams.ch_edge_weights)
        self.training_mode_subscriber = self.redis_connection.subscribe(channel=self.params.RedisParams.ch_edge_mode)
        self.plant_reset_subscriber = self.redis_connection.subscribe(channel=self.params.RedisParams.ch_plant_reset)
        self.patch_gain_subscriber = self.redis_connection.subscribe(channel=self.params.RedisParams.ch_edge_patch_gain)

        self.agent_a_active = True  # True: agent_a is controller, False: agent_b is controller

        # todo double check this control frequency
        self.control_frequency = self.params.ControlParams.frequency
        self.sample_period = 1. / self.control_frequency

        self.agent_a = agent_a
        self.agent_b = agent_b

        self.t2 = threading.Thread(target=self.update_weights)
        self.t3 = threading.Thread(target=self.receive_mode)
        self.t4 = threading.Thread(target=self.receive_reset_command)
        self.t5 = threading.Thread(target=self.loop_sending_edge_trajectory)

        self.trajectory_sending_condition = threading.Condition()
        self.training = True if self.params.JobParams.run_mode == "train" else False
        self.step = 0
        self.last_action = [0] * agent_a.action_shape
        # observation, action, failed, operation_mode, step
        self.edge_trajectory = [[0.] * agent_a.observation_shape, [0] * agent_a.action_shape, 0, 0, 0]

        if params.ControlParams.initialize_from_cloud:
            print("waiting for weights from cloud")
            self.ini_weights_from_cloud(self.agent_a, self.agent_b)
        else:
            print("initialize weights from scratch")

    def receives_weights(self):
        weights_pack = self.weights_subscriber.parse_response()[2]
        weights = pickle.loads(weights_pack)
        return weights

    def send_ready_update(self, ready):
        ready_pack = pickle.dumps(ready)
        self.redis_connection.publish(channel=self.params.RedisParams.ch_edge_ready_update, message=ready_pack)

    def send_edge_trajectory(self, edge_trajectory):
        """send trajectory from edge"""
        edge_trajectory_pack = pickle.dumps(edge_trajectory)
        # print("BW lower bound:", len(edge_trajectory_pack) * 8 * self.params.ControlParams.frequency / 2**20)
        self.redis_connection.publish(channel=self.params.RedisParams.ch_edge_trajectory, message=edge_trajectory_pack)

    def ini_weights_from_cloud(self, *args):
        self.send_ready_update(True)
        weights = self.receives_weights()
        for agent in args:
            agent.actor.set_actor_weights(weights)

    def generate_action(self):
        pass

    def update_weights(self):

        while True:
            self.send_ready_update(True)

            weights = self.receives_weights()
            if self.agent_a_active:
                for i, w in enumerate(weights):
                    self.agent_b.actor.weights[i].assign(w)
                    time.sleep(0.001)
                    # asyncio.sleep(0)
            else:
                for i, w in enumerate(weights):
                    self.agent_a.actor.weights[i].assign(w)
                    time.sleep(0.001)
                    # asyncio.sleep(0)
            self.agent_a_active = not self.agent_a_active

    def receive_mode(self):
        """
        receive_mode to switch between training and testing
        """
        while True:
            mode_pack = self.training_mode_subscriber.parse_response()[2]
            mode = pickle.loads(mode_pack)
            self.training = mode[0]
            print("training:", self.training)

    def loop_sending_edge_trajectory(self):

        self.trajectory_sending_condition.acquire()
        while True:
            self.trajectory_sending_condition.wait()
            self.send_edge_trajectory(self.edge_trajectory)

    def reset_control(self):
        pass

    def initialize_plant(self):
        self.reset_control()

    def receive_reset_command(self):
        """
        receive reset command from the cloud trainer to reset the plant;
        resetting command comes when the current steps reach the max_steps of a single episode
        """
        while True:
            _ = self.plant_reset_subscriber.parse_response()[2]
            self.set_normal_mode(False)

    def run(self):
        self.t2.daemon = True
        self.t3.daemon = True
        self.t4.daemon = True
        self.t5.daemon = True

        self.t2.start()
        self.t3.start()
        self.t4.start()
        self.t5.start()

        self.generate_action()

    def set_normal_mode(self, normal_mode):
        pass

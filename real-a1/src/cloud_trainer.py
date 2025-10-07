import pickle
import dataclasses
import threading
import time
import numpy as np
from src.transition import TrajectorySegment
from src.student.utils import log_info
from src.physical_design import MATRIX_P
from src.student.ReplayMem import ReplayMemory
from job.job_config import JobConfig
from src.student.DDPG import DDPGConfig as AgentConfig   # We take DDPG as an example
from src.redis import RedisConfig

@dataclasses.dataclass
class CloudConfig:
    # self.on_target_reset_steps = 100  # num steps on target after which the episode is terminated
    sleep_after_reset: int = 2  # seconds of sleep because it makes sense
    pre_fill_steps: int = 0
    weights_update_period: int = 1
    artificial_bandwidth: int = -1  # set bandwidth between cloud and edge in MBit/s
    artificial_ping: float = 0  # set ping between cloud and edge in ms
    ethernet_bandwidth: int = 100  # bw if connected with ethernet between cloud and edge in MBit/s
    ethernet_ping: float = 0.2  # ping if connected with ethernet between cloud and edge in ms


@dataclasses.dataclass
class CloudSystemConfig:
    JobParams: JobConfig = dataclasses.field(default_factory=JobConfig)
    CloudParams: CloudConfig = dataclasses.field(default_factory=CloudConfig)
    AgentParams: AgentConfig = dataclasses.field(default_factory=AgentConfig)
    RedisParams: RedisConfig = dataclasses.field(default_factory=RedisConfig)


class CloudSystem:
    def __init__(self, params: CloudSystemConfig, agent, redis_connection, writer):
        self.params = params
        self.writer = writer

        self.redis_connection = redis_connection
        self.edge_trajectory_subscriber = self.redis_connection.subscribe(
            channel=self.params.RedisParams.ch_edge_trajectory)
        self.ep = 0

        self.agent = agent
        self.edge_status_subscriber = self.redis_connection.subscribe(
            channel=self.params.RedisParams.ch_edge_ready_update)

        self.t2 = threading.Thread(target=self.optimize)
        self.t3 = threading.Thread(target=self.waiting_edge_ready)

        self.optimize_condition = threading.Condition()
        self.replay_memory = ReplayMemory(size=self.params.AgentParams.replay_buffer_size)

        # pre-fill the replay memory using offline dataset
        # if self.params.CloudParams.pre_fill_steps > 0:
        #     self.prefill_sim(self.params.CloudParams.pre_fill_steps)

        self.training = True
        self.trainable = 0
        self.mode = "Resetting"
        self.global_step = 0
        self.send_mode_and_steps()

        if self.params.CloudParams.artificial_bandwidth != -1.0:
            # get the size of the weights packet
            packet = pickle.dumps(self.agent.actor.get_weights())
            ethernet_time = (len(packet) * 8 / 2 ** 20) / self.params.CloudParams.ethernet_bandwidth + \
                            self.params.CloudParams.ethernet_ping / 1000
            self.sending_time = (len(packet) * 8 / 2 ** 20) / self.params.CloudParams.artificial_bandwidth + \
                                self.params.CloudParams.artificial_ping / 1000 - ethernet_time
            print(f"Setting sending time for actor weights to {self.sending_time} seconds")
            print(f"Update actor weights every {self.sending_time * 30} steps")
        else:
            self.sending_time = 0

        print("Cloud training system is initialized ...")

    def run(self):
        """It's triple threads"""
        self.t2.daemon = True
        self.t3.daemon = True
        self.t2.start()
        self.t3.start()
        self.store_trajectory()

    def store_trajectory(self):

        best_acc_reward = 0.0  # best accumulated reward
        moving_average_acc_reward = 0.0

        while self.global_step < self.params.AgentParams.total_training_steps:

            self.ep += 1

            if self.ep % self.params.AgentParams.eval_period == 0 and \
                    self.global_step > self.params.AgentParams.learning_starts:
                mode_pack = pickle.dumps([False])
                self.redis_connection.publish(self.params.RedisParams.ch_edge_mode, mode_pack)
                self.training = False
                print("EVALUATING!!!!!!")
            else:
                mode_pack = pickle.dumps([True])
                self.redis_connection.publish(self.params.RedisParams.ch_edge_mode, mode_pack)
                self.training = True

            accumulated_reward = self.run_episode(self.training)
            self.agent.save_weights(self.params.JobParams.output_path + '/agent_model/')

            if not self.training:
                moving_average_acc_reward = 0.95 * moving_average_acc_reward + 0.05 * accumulated_reward
                if moving_average_acc_reward > best_acc_reward:
                    self.agent.save_weights(self.params.JobParams.output_path + '/agent_model_best/')
                    best_acc_reward = moving_average_acc_reward

    def run_episode(self, training):

        self.mode = "Training" if training else "Evaluating"
        traj_segment = self.receive_edge_trajectory()

        step_count = self.params.AgentParams.max_episode_steps
        reward_list = []
        teacher_correct_counts = 0
        for _ in range(step_count):
            last_seg = traj_segment
            traj_segment = self.receive_edge_trajectory()

            r = self.calculate_reward(last_seg.observations, traj_segment.observations,
                                      MATRIX_P, traj_segment.last_action)

            if training:
                if traj_segment.sequence_number == (last_seg.sequence_number + 1):
                    # only save the experience if the two trajectory are consecutive
                    self.replay_memory.add((last_seg.observations,traj_segment.last_action,
                                            r, traj_segment.observations, traj_segment.failed))
                    self.global_step += 1
                    self.trainable += 1

                    if self.optimize_condition.acquire(False):
                        self.optimize_condition.notify_all()
                        self.optimize_condition.release()
                else:
                    print("Package loss... terrible loss :/")

            if not traj_segment.student_activate:
                teacher_correct_counts += 1

            reward_list.append(r)
            self.send_mode_and_steps() # for progress monitoring

            if not traj_segment.normal_operation: # Termination condition is triggered
                break

        self.initiate_reset()
        self.mode = "Resetting"
        self.send_mode_and_steps()
        time.sleep(self.params.CloudParams.sleep_after_reset) # wait a bit for the edge reset
        # Clean the received trajectory
        stale_segments = self.edge_trajectory_subscriber.parse_response(block=False)
        while stale_segments is not None:
            stale_segments = self.edge_trajectory_subscriber.parse_response(block=False)

        if training:
            self.writer.add_scalar("training/episodic_return", np.sum(reward_list), self.global_step)
            self.writer.add_scalar("training/episodic_length", len(reward_list), self.global_step)
            self.writer.add_scalar("training/teacher_counts", teacher_correct_counts, self.global_step)
        else:
            self.writer.add_scalar("evaluation/episodic_return", np.sum(reward_list), self.global_step)
            self.writer.add_scalar("evaluation/episodic_length", len(reward_list), self.global_step)
            self.writer.add_scalar("evaluation/teacher_counts", teacher_correct_counts, self.global_step)

            # todo implement a convergence condition to automatically stop the training
            # if self.model_stats.converge_eval_episode >= self.params.stats_params.converge_episodes:
            #     self.agent.save_weights(self.params.stats_params.model_name + '_converge')
            #     print("Converging, training stopped")
            #     self.mode = "Training Finished"
            #     self.send_mode_and_steps()
            #     sys.exit()

        accumulated_reward = sum(reward_list)
        return accumulated_reward

    def optimize(self):
        optimize_times = 0

        while True:
            if self.trainable == 0:
                self.optimize_condition.acquire()
                self.optimize_condition.wait()

            if self.replay_memory.get_size() > self.params.AgentParams.learning_starts:
                for _ in range(self.params.AgentParams.iterations_per_step):
                    mini_batch = self.replay_memory.sample(self.params.AgentParams.batch_size)
                    training_info = self.agent.optimize(mini_batch)
                    optimize_times += 1
                    self.trainable -= 1
                    log_info(self.writer, int(optimize_times / self.params.AgentParams.iterations_per_step),
                             training_info, 'agent', period=1000)

    def waiting_edge_ready(self):
        weights = self.agent.actor.get_weights()
        self.send_weights(weights)

        while True:
            edge_status = self.receive_edge_status()
            if edge_status:
                weights = self.agent.actor.get_weights()
                if self.sending_time > 0:
                    time.sleep(self.sending_time)
                self.send_weights(weights)
                print("[{}] ===>  Training: current training finished, sending weights, mem_size: {}, backlog: {}"
                      .format(get_current_time(), self.replay_memory.get_size(), self.trainable))

    def send_mode_and_steps(self):
        mode_and_steps_pack = pickle.dumps([self.mode, self.global_step])
        self.redis_connection.publish(channel=self.params.RedisParams.ch_training_steps, message=mode_and_steps_pack)

    def receive_edge_status(self):
        edge_status_pack = self.edge_status_subscriber.parse_response()
        edge_status = pickle.loads(edge_status_pack[2])
        return edge_status

    def send_weights(self, weights):
        weights_pack = pickle.dumps(weights)
        self.redis_connection.publish(channel=self.params.RedisParams.ch_edge_weights, message=weights_pack)

    def receive_edge_trajectory(self):
        edge_trajectory_pack = self.edge_trajectory_subscriber.parse_response()[2]
        edge_seg = TrajectorySegment.pickle_load_pack(edge_trajectory_pack)
        return edge_seg

    def initiate_reset(self):
        reset_pack = pickle.dumps(True)
        self.redis_connection.publish(channel=self.params.RedisParams.ch_plant_reset, message=reset_pack)

    def calculate_reward(self, observation_cur, observation_pre, p_matrix, action=None):
        # here the lyapunov-like reward, observation is the tracking error
        observation_cur = observation_cur[:, np.newaxis]
        observation_pre = observation_pre[:, np.newaxis]
        s_pre = observation_pre[2:]
        s_cur = observation_cur[2:]
        ly_reward_pre = s_pre.T @ p_matrix @ s_pre
        ly_reward_cur = s_cur.T @ p_matrix @ s_cur
        reward = ly_reward_pre - ly_reward_cur
        return np.squeeze(reward)


    def prefill_mem(self):
        # todo implement to prefill the replay memory for the future
        pass

def get_current_time():
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
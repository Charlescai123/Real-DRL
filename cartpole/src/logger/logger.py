import shutil
import threading

import numpy as np
import distutils.util
import tensorflow as tf

from matplotlib import pyplot as plt
from omegaconf import DictConfig

from src.logger.plotter.fig_plotter import FigPlotter
from src.logger.plotter.live_plotter import LivePlotter
from src.logger.utils import check_dir, is_dir_empty


class Logger:
    def __init__(self, logger_cfg: DictConfig):

        self.params = logger_cfg
        self.mode = logger_cfg.mode
        self.log_dir = self.params.log_dir
        self.model_dir = self.params.model_save_dir

        self.check_dir()
        self.clear_cache()
        self.training_log_writer = tf.summary.create_file_writer(self.log_dir + '/training')
        self.evaluation_log_writer = tf.summary.create_file_writer(self.log_dir + '/eval')

        # Figure plotter
        self.fig_plotter = FigPlotter(
            plotter_cfg=logger_cfg.fig_plotter
        )

        # Live plotter
        self.live_plotter = LivePlotter(
            live_cfg=logger_cfg.live_plotter
        )

        # Status record
        self.state_list = []
        self.action_list = []
        self.action_mode_list = []
        self.energy_list = []

    def create_thread(self):
        self.p = threading.Thread(target=self.animation_run)
        self.p.setDaemon(True)

    def plot_phase(self, x_set, theta_set, epsilon, p_mat, idx):
        self.fig_plotter.phase_portrait(
            state_list=self.state_list,
            action_mode_list=self.action_mode_list,
            x_set=x_set,
            theta_set=theta_set,
            epsilon=epsilon,
            p_mat=p_mat,
            idx=idx
        )

    def plot_trajectory(self, x_set, theta_set, action_set, freq, idx):
        self.fig_plotter.plot_trajectory(
            state_list=self.state_list,
            action_list=self.action_list,
            action_mode_list=self.action_mode_list,
            energy_list=self.energy_list,
            x_set=x_set,
            theta_set=theta_set,
            action_set=action_set,
            freq=freq,
            fig_idx=idx
        )

    def clear_logs(self):
        self.state_list.clear()
        self.action_list.clear()
        self.action_mode_list.clear()
        self.energy_list.clear()

    def update_logs(self, state, action, action_mode, energy):
        self.state_list.append(state)
        self.action_list.append(action)
        self.action_mode_list.append(action_mode)
        self.energy_list.append(energy)

    def check_dir(self):
        print(f"checking logger directories...")
        check_dir(self.log_dir)
        check_dir(self.model_dir)

    def log(self, locs, mode='train'):
        if mode == 'train':
            with self.training_log_writer.as_default():
                n_steps = locs['global_steps'] - len(locs['reward_list'])
                step_list = range(n_steps, locs['global_steps'])

                # Separate reward
                if len(locs['ep_infos']) > 0:
                    for key in locs['ep_infos'][0]:
                        rew_list = []
                        for ep_info in locs['ep_infos']:
                            rew_list.append(ep_info[key])

                        tf.summary.scalar(f'Train_Episode/return_{key}', np.sum(rew_list), locs['ep_i'])
                        self.log_stepwise_data(f"Train_Stepwise/reward_{key}", rew_list, step_list)

                # Episodic statistics
                tf.summary.scalar('Train_Episode/return', np.sum(locs['reward_list']), locs['ep_i'])
                tf.summary.scalar('Train_Episode/distance_score', np.sum(locs['distance_score_list']), locs['ep_i'])
                tf.summary.scalar('Train_Episode/critic_loss', np.sum(locs['critic_loss_list']), locs['ep_i'])
                tf.summary.scalar('Train_Episode/energy_cost', np.sum(locs['energy_list']), locs['ep_i'])

                # Stepwise statistics
                self.log_stepwise_data("Train_Stepwise/reward", locs['reward_list'], step_list)
                self.log_stepwise_data("Train_Stepwise/distance_score", locs['distance_score_list'], step_list)
                self.log_stepwise_data("Train_Stepwise/critic_loss", locs['critic_loss_list'], step_list)
                self.log_stepwise_data("Train_Stepwise/energy_cost", locs['energy_list'], step_list)

                # Activation ratio statistics
                total_cnt = locs['student_cnt'] + locs['teacher_cnt']
                tf.summary.scalar('Activation/student_count', locs['student_cnt'], locs['ep_i'])
                tf.summary.scalar('Activation/teacher_count', locs['teacher_cnt'], locs['ep_i'])
                tf.summary.scalar('Activation/ratio', locs['teacher_cnt']/total_cnt, locs['ep_i'])

        elif mode == 'eval':
            with self.evaluation_log_writer.as_default():
                tf.summary.scalar('Eval/average_reward', locs['mean_reward'], locs['global_steps'])
                tf.summary.scalar('Eval/distance_score', locs['mean_distance_score'], locs['global_steps'])
                tf.summary.scalar('Eval/critic_loss', locs['mean_loss'], locs['global_steps'])
                tf.summary.scalar('Eval/distance_score_and_survived',
                                  locs['mean_distance_score'] * (1 - locs['failed']), locs['global_steps'])
        else:
            raise Exception(f"Unknown logging mode: {mode}")

    def log_training_data(self, average_reward, average_distance_score, critic_loss, failed, global_steps):
        with self.training_log_writer.as_default():
            tf.summary.scalar('train_eval/Average_Reward', average_reward, global_steps)
            tf.summary.scalar('train_eval/distance_score', average_distance_score, global_steps)
            tf.summary.scalar('train_eval/critic_loss', critic_loss, global_steps)
            tf.summary.scalar('train_eval/distance_score_and_survived', average_distance_score * (1 - failed),
                              global_steps)

    def log_evaluation_data(self, average_reward, average_distance_score, failed, global_steps):
        with self.evaluation_log_writer.as_default():
            tf.summary.scalar('Eval/Average_Reward', average_reward, global_steps)
            tf.summary.scalar('Eval/distance_score', average_distance_score, global_steps)
            tf.summary.scalar('Eval/distance_score_and_survived', average_distance_score * (1 - failed),
                              global_steps)

    def clear_cache(self):
        if not is_dir_empty(self.log_dir):
            if self.params.force_override:
                shutil.rmtree(self.log_dir)
            else:
                print(self.log_dir, 'already exists.')
                resp = input('Override log file? [Y/n]\n')
                if resp == '' or distutils.util.strtobool(resp):
                    print('Deleting old log dir')
                    shutil.rmtree(self.log_dir)
                else:
                    print('Okay, exit program')
                    exit(1)

    def change_mode(self, mode):
        current_mode = self.mode
        self.fig_plotter.change_dir(old_dir=str(current_mode), new_dir=str(mode))
        self.mode = mode

    def log_stepwise_data(self, name, data_list, step_list):
        """Log stepwise data (list)"""
        assert len(data_list) == len(step_list)
        with self.training_log_writer.as_default():
            for i in range(len(data_list)):
                tf.summary.scalar(name, data_list[i], step=step_list[i])


def plot_trajectory(trajectory_tensor, reference_trajectory_tensor=None):
    """
   trajectory_tensor: a numpy array [n, 4], where n is the length of the trajectory,
                      5 is the dimension of each point on the trajectory, containing [x, x_dot, theta, theta_dot]
   """
    trajectory_tensor = np.array(trajectory_tensor)
    reference_trajectory_tensor = np.array(
        reference_trajectory_tensor) if reference_trajectory_tensor is not None else None

    n, c = trajectory_tensor.shape

    y_label_list = ["safety", "x", "x_dot", "theta", "theta_dot"]

    plt.figure(figsize=(9, 6))

    for i in range(c):

        plt.subplot(c, 1, i + 1)
        plt.plot(np.arange(n), trajectory_tensor[:, i], label=y_label_list[i])

        if reference_trajectory_tensor is not None:
            plt.plot(np.arange(n), reference_trajectory_tensor[:, i], label=y_label_list[i])

        plt.legend(loc='best')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("trajectory.png", dpi=300)
    # plt.show()

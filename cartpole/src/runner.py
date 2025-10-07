import copy
import time

import imageio
import warnings
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

from src.envs.rewards import Rewards
from src.utils.physical_design import MATRIX_P, F, LEARNING_P
from src.phy_teacher.phy_teacher import PHYTeacher
from src.drl_student.agents.ddpg import DDPGAgent
from src.drl_student.storage.solo_replay_mem import SoloReplayMemory
from src.drl_student.storage.dual_replay_mem import DualReplayMemory
from src.trigger.trigger import Trigger
from src.utils.utils import ActionMode
from src.envs.utils import energy_value, get_unknown_distribution
from src.logger.utils import logger
from src.envs.utils import state2observations
from src.envs.cart_pole import Cartpole
from src.logger.logger import Logger

np.set_printoptions(suppress=True)


class EnvRunner:
    def __init__(self, config):

        self.params = config
        self.gamma = config.drl_student.phydrl.gamma  # Contribution ratio of Data/Model action

        # Random seed
        tf.random.set_seed(config.general.seed)
        np.random.seed(config.general.seed)

        # Environment (Cartpole simulates the physical plant)
        self.cartpole = Cartpole(config.cartpole, seed=config.general.seed)

        # DRL-Student
        self.student_params = config.drl_student
        self.agent_params = config.drl_student.agents
        self.shape_observations = self.cartpole.state_observations_dim
        self.shape_action = self.cartpole.action_dim

        # Dual Replay Buffer
        if config.drl_student.phydrl.use_dual_replay_buffer:
            self.replay_mem = DualReplayMemory(dual_buffer_config=config.drl_student.agents.replay_buffer.dual_buffer,
                                               seed=config.general.seed)
        else:  # Solo Replay Buffer
            self.replay_mem = SoloReplayMemory(solo_buffer_config=config.drl_student.agents.replay_buffer.solo_buffer,
                                               seed=config.general.seed)

        self.agent = DDPGAgent(agent_cfg=config.drl_student.agents,
                               taylor_cfg=config.drl_student.taylor,
                               shape_observations=self.shape_observations,
                               buffer_batch_sample_size=self.replay_mem.batch_sample_size,
                               shape_action=self.shape_action,
                               mode=config.logger.mode
                               )
        # PHY-Teacher
        self.teacher_params = config.phy_teacher
        self.phy_teacher = PHYTeacher(teacher_cfg=config.phy_teacher, cartpole_cfg=config.cartpole)

        # Trigger
        self.trigger = Trigger(config.trigger)

        # Logger and Plotter
        self.logger = Logger(config.logger)

        # Reward function
        self.reward_fcn = Rewards(self, config.cartpole.rewards)

        # Variables for caching
        self._initial_loss = self.agent_params.initial_loss
        self._action_magnitude = config.drl_student.agents.action.magnitude
        self._max_steps_per_episode = self.agent_params.max_steps_per_episode
        self._terminate_on_failure = self.params.cartpole.terminate_on_failure

        self.failed_times = 0

    def interaction_step(self, mode=None):

        current_state = copy.deepcopy(self.cartpole.state)
        observations, _ = state2observations(current_state)

        self.phy_teacher.update(state=np.asarray(current_state[:4]))  # Teacher update

        terminal_action, nominal_action = self.get_terminal_action(state=current_state, mode=mode)

        # Update logs
        self.logger.update_logs(
            state=copy.deepcopy(self.cartpole.state[:4]),
            action=self.trigger.plant_action,
            action_mode=self.trigger.action_mode,
            energy=energy_value(state=np.array(self.cartpole.state[:4]), p_mat=LEARNING_P)
        )

        # Inject Terminal Action
        next_state = self.cartpole.step(action=terminal_action)

        observations_next, failed = state2observations(next_state)

        teacher_flag = True if self.trigger.action_mode == ActionMode.TEACHER else False

        # Sum rewards
        sum_rew, rew_info = self.reward_fcn.compute_reward()
        # reward, distance_score = self.cartpole.reward_fcn(current_state, nominal_action, next_state, teacher_flag=teacher_flag)

        # Distance score
        distance_score = self.cartpole.get_distance_score()

        return observations, nominal_action, observations_next, failed, sum_rew, distance_score, teacher_flag, rew_info

    def train(self):
        episode = 0
        global_steps = 0
        best_dsas = 0.0  # Best distance score and survived
        moving_average_dsas = 0.0
        optimize_time = 0

        friction_change_round = self.params.general.friction_cart_change.episode_round
        origin_friction_cart = self.cartpole.friction_cart

        # Run for max training episodes
        for ep_i in range(int(self.agent_params.max_training_episodes)):

            # Reset all modules
            if self.params.cartpole.random_reset.train:
                self.cartpole.random_reset()
            else:
                self.cartpole.reset()

            # Change friction to test dual buffer
            if self.params.general.test_dual_buffer:
                if ep_i < friction_change_round:
                    self.cartpole.friction_cart = self.params.general.friction_cart_change.friction
                else:
                    self.cartpole.friction_cart = origin_friction_cart
                    # self.phy_teacher.teacher_enable = False

            self.phy_teacher.reset(state=np.array(self.cartpole.state[:4]))
            self.trigger.reset(state=np.array(self.cartpole.state[:4]),
                               learning_space=self.phy_teacher.learning_space)

            # Logging clear for each episode
            self.logger.clear_logs()

            print(f"Training at {ep_i} init_cond: {self.cartpole.state[:4]}")
            pbar = tqdm(total=self._max_steps_per_episode, desc="Episode %d" % ep_i)

            reward_list = []
            distance_score_list = []
            critic_loss_list = []
            ep_infos = []
            energy_list = []
            failed = False

            student_cnt = 0
            teacher_cnt = 0
            ep_steps = 0

            for step in range(int(self._max_steps_per_episode)):

                observations, action, observations_next, failed, r, distance_score, teacher_flag, rew_info = \
                    self.interaction_step(mode='train')

                if self.teacher_params.teacher_correct is False and teacher_flag is True:
                    # Test Learning efficiency for Runtime Learning Machine
                    logger.info("DRL-Student doesn't learn from PHY-Teacher, skip model optimizing...")
                else:
                    action_type = ActionMode.TEACHER if teacher_flag else ActionMode.STUDENT
                    self.replay_mem.add_transition((observations, action, r, observations_next, failed), action_type)

                    if self.replay_mem.is_prefilled:
                        minibatch = self.replay_mem.sample()
                        critic_loss = self.agent.optimize(minibatch)
                        optimize_time += 1
                    else:
                        critic_loss = self._initial_loss
                    critic_loss_list.append(critic_loss)
                    reward_list.append(r)
                    distance_score_list.append(distance_score)

                    # Update learning stepwise info
                    ep_infos.append(rew_info)
                    global_steps += 1
                    ep_steps += 1

                    energy_list.append(self.cartpole.ut * self.cartpole.ut / self.params.cartpole.frequency)

                if teacher_flag:
                    teacher_cnt += 1
                else:
                    student_cnt += 1

                pbar.update(1)

                if failed and self._terminate_on_failure:
                    self.failed_times += 1 * failed
                    print(f"Cartpole system failed, terminate for safety concern!")
                    pbar.close()
                    break

            # Plot Phase
            if self.params.logger.fig_plotter.phase.plot:
                self.logger.plot_phase(
                    x_set=self.params.cartpole.safety_set.x,
                    theta_set=self.params.cartpole.safety_set.theta,
                    epsilon=self.teacher_params.epsilon,
                    p_mat=MATRIX_P,
                    idx=ep_i
                )

            # Plot Trajectories
            if self.params.logger.fig_plotter.trajectory.plot:
                self.logger.plot_trajectory(
                    x_set=self.params.cartpole.safety_set.x,
                    theta_set=self.params.cartpole.safety_set.theta,
                    action_set=self.params.cartpole.force_bound,
                    freq=self.params.cartpole.frequency,
                    idx=ep_i
                )

            mean_reward = np.mean(reward_list)
            mean_distance_score = np.mean(distance_score_list)
            mean_critic_loss = np.mean(critic_loss_list)

            self.logger.log(locals(), mode='train')

            print(f"Average_reward: {mean_reward:.6}\n"
                  f"Distance_score: {mean_distance_score:.6}\n"
                  f"Critic_loss: {mean_critic_loss:.6}\n"
                  f"Total_steps_ep: {ep_steps} ")

            # Save weights per episode
            self.agent.save_weights(self.logger.model_dir)

            # Save per 5 episodes
            if (ep_i + 1) % 2 == 0:
                self.agent.save_weights(f"{self.logger.model_dir}_{ep_i + 1}")

            if (ep_i + 1) % self.student_params.agents.evaluation_period == 0:
                eval_mean_reward, eval_mean_distance_score, eval_failed = self.evaluation(mode='eval', idx=ep_i)
                self.logger.change_mode(mode='train')  # Change mode back
                self.logger.log_evaluation_data(eval_mean_reward, eval_mean_distance_score, eval_failed,
                                                global_steps)
                moving_average_dsas = 0.95 * moving_average_dsas + 0.05 * eval_mean_distance_score
                if moving_average_dsas > best_dsas:
                    self.agent.save_weights(self.logger.model_dir + '-best')
                    best_dsas = moving_average_dsas

            episode += 1
            # np.savetxt(f"{self.params.general.id}_ep{episode}_reward.txt", reward_list, fmt="%.8f")
            # print(f"global_steps is: {global_steps}")

            # Whether to terminate training based on training_steps
            if global_steps > self.agent_params.max_training_steps and self.agent_params.training_by_steps:
                np.savetxt(f"{self.logger.log_dir}/failed_times.txt",
                           [self.failed_times, episode, self.failed_times / episode])
                print(f"Final_optimize time: {optimize_time}")
                print("Total failed:", self.failed_times)
                exit("Reach maximum steps, exit...")

        np.savetxt(f"{self.logger.log_dir}/failed_times.txt",
                   [self.failed_times, episode, self.failed_times / episode])
        print(f"Final_optimize time: {optimize_time}")
        print("Total failed:", self.failed_times)
        exit("Reach maximum episodes, exit...")

    def evaluation(self, reset_state=None, mode=None, idx=0):

        if self.params.cartpole.random_reset.eval:
            self.cartpole.random_reset()
        else:
            self.cartpole.reset(reset_state)

        print(f"Evaluating at {idx} init_cond: {self.cartpole.state[:4]}")

        trajectory = []
        trajectory.append(self.cartpole.state[:4])

        reward_list = []
        distance_score_list = []
        failed = False
        ani_frames = []

        self.phy_teacher.reset(state=np.array(self.cartpole.state[:4]))
        self.trigger.reset(state=np.array(self.cartpole.state[:4]),
                           learning_space=self.phy_teacher.learning_space)

        self.logger.change_mode(mode=mode)  # Change mode
        self.logger.clear_logs()  # Clear logs

        # Visualization flag
        visual_flag = (self.params.logger.live_plotter.animation.show
                       or self.params.logger.live_plotter.live_trajectory.show)

        if visual_flag:
            plt.ion()

        print(f"friction: {self.cartpole.friction_cart}")
        # time.sleep(123)

        for step in range(self.agent_params.max_evaluation_steps):
            print(f"cartpole state: {self.cartpole.state}")

            observations, action, observations_next, failed, r, distance_score, _, _ = \
                self.interaction_step(mode=mode)

            trajectory.append(self.cartpole.state[:4])

            # Visualize Cart-pole animation
            if self.params.logger.live_plotter.animation.show:
                frame = self.cartpole.render(mode='rgb_array', idx=step)
                ani_frames.append(frame)

            # Visualize Live trajectory
            if self.params.logger.live_plotter.live_trajectory.show:
                self.logger.live_plotter.animation_run(
                    x_set=self.params.cartpole.safety_set.x,
                    theta_set=self.params.cartpole.safety_set.theta,
                    action_set=self.params.cartpole.force_bound,
                    state=self.logger.state_list[-1],
                    action=self.logger.action_list[-1],
                    action_mode=self.logger.action_mode_list[-1],
                    energy=self.logger.energy_list[-1],
                )
                plt.pause(0.01)
            reward_list.append(r)
            distance_score_list.append(distance_score)

            if failed and self.params.cartpole.terminate_on_failure:
                print(f"cartpole failed...")
                # break

        mean_reward = np.mean(reward_list)
        mean_distance_score = np.mean(distance_score_list)

        # Save as a GIF (Cart-pole animation)
        if self.params.logger.live_plotter.animation.save_to_gif:
            if len(ani_frames) == 0:
                warnings.warn("Failed to save animation as gif, please set animation.show to True")
            else:
                logger.debug(f"Animation frames: {ani_frames}")
                last_frame = ani_frames[-1]
                for _ in range(5):
                    ani_frames.append(last_frame)
                gif_path = self.params.logger.live_plotter.animation.gif_path
                fps = self.params.logger.live_plotter.animation.fps
                print(f"Saving animation frames to {gif_path}")
                imageio.mimsave(gif_path, ani_frames, fps=fps, loop=0)

        # Save as a GIF (Cart-pole trajectory)
        if self.params.logger.live_plotter.live_trajectory.save_to_gif:
            if len(ani_frames) == 0:
                warnings.warn("Failed to save live trajectory as gif, please set live_trajectory.show to True")
            else:
                last_frame = self.logger.live_plotter.frames[-1]
                for _ in range(5):
                    self.logger.live_plotter.frames.append(last_frame)
                gif_path = self.params.logger.live_plotter.live_trajectory.gif_path
                fps = self.params.logger.live_plotter.live_trajectory.fps
                print(f"Saving live trajectory frames to {gif_path}")
                imageio.mimsave(gif_path, self.logger.live_plotter.frames, fps=fps, loop=0)

        # Close and reset
        if visual_flag:
            self.cartpole.close()
            self.logger.live_plotter.reset()
            plt.ioff()
            plt.close()

        # Plot Phase
        if self.params.logger.fig_plotter.phase.plot:
            self.logger.plot_phase(
                x_set=self.params.cartpole.safety_set.x,
                theta_set=self.params.cartpole.safety_set.theta,
                epsilon=self.teacher_params.epsilon,
                p_mat=MATRIX_P,
                idx=idx
            )

        # Plot Trajectory
        if self.params.logger.fig_plotter.trajectory.plot:
            self.logger.plot_trajectory(
                x_set=self.params.cartpole.safety_set.x,
                theta_set=self.params.cartpole.safety_set.theta,
                action_set=self.params.cartpole.force_bound,
                freq=self.params.cartpole.frequency,
                idx=idx
            )

        # Reset live plotter
        self.logger.live_plotter.reset()
        # np.savetxt(f"trajectory2.txt", trajectory, fmt="%.8f")

        return mean_reward, mean_distance_score, failed

    def test2(self):
        total_fails = 0
        for i in range(100):
            _, _, fails = self.evaluation(mode='test', idx=i)
            total_fails += int(fails)
        print(f"Total fails: {total_fails}")

    def test(self):
        self.evaluation(mode='test', reset_state=self.params.cartpole.initial_condition)

    def get_terminal_action(self, state, mode=None):

        observations, _ = state2observations(state)
        s = np.asarray(state[:4])

        # DRL Action
        drl_raw_action = self.agent.get_action(observations, mode)

        # Add unknown unknowns
        if self.agent_params.unknown_distribution.apply:
            logger.debug(f"apply unknown unknowns to the drl agent")
            drl_raw_action += get_unknown_distribution()
            # Truncate to [-1, 1]
            drl_raw_action = np.clip(drl_raw_action, -1, 1)

        drl_action = drl_raw_action * self._action_magnitude

        # Model-based Action
        phy_action = F @ s

        # Student Action (Residual form)
        student_action = drl_action * 1 + phy_action * self.gamma

        # Teacher Action
        teacher_action, dwell_flag = self.phy_teacher.get_action()

        # Terminal Action by Trigger
        terminal_action, action_mode = self.trigger.get_terminal_action(student_action=student_action,
                                                                        teacher_action=teacher_action,
                                                                        plant_state=s,
                                                                        dwell_flag=dwell_flag,
                                                                        learning_space=self.phy_teacher.learning_space)
        # Used for debugging
        # terminal_action = teacher_action
        # action_mode = ActionMode.TEACHER

        # print(f"cartpole state -------------: {self.cartpole.state}")

        logger.debug(f"teacher_action: {teacher_action}")
        logger.debug(f"student_action: {student_action}")
        logger.debug(f"terminal_action: {terminal_action}")
        logger.debug(f"action_mode: {action_mode}")

        # Decide nominal action to store into replay buffer
        if action_mode == ActionMode.TEACHER:
            if self.phy_teacher.teacher_correct:  # Correction from PHY-Teacher
                # if teacher_action == None:
                #     teacher_action = 0.0
                nominal_action = (teacher_action - phy_action) / self._action_magnitude
            else:
                nominal_action = drl_raw_action
        elif action_mode == ActionMode.STUDENT:
            nominal_action = drl_raw_action
        else:
            raise NotImplementedError(f"Unknown action mode: {action_mode}")

        return terminal_action, nominal_action

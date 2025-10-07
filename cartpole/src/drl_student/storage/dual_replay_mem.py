import numpy as np

from src.envs.utils import observations2state
from src.drl_student.storage.replay_mem import ReplayMemory
from src.utils.physical_design import LEARNING_P
from src.utils.utils import ActionMode
from src.envs.utils import energy_value


class DualReplayMemory:
    """
    Replay memory class to store trajectories
    """

    def __init__(self, dual_buffer_config, combined_experience_replay=False, seed=42):
        """
        initializing the replay memory
        """
        self.stu_replay_buffer = ReplayMemory(dual_buffer_config.student, combined_experience_replay, seed)
        self.tea_replay_buffer = ReplayMemory(dual_buffer_config.teacher, combined_experience_replay, seed)

        self.batch_sample_size = int(dual_buffer_config.batch_size)
        self.experience_prefill_size = int(dual_buffer_config.experience_prefill_size)
        
        self.rho1 = dual_buffer_config.rho1
        self.rho2 = dual_buffer_config.rho2

        # Data type pointer
        self.last_data_type = None

    def add_transition(self, experience, action_type: ActionMode):
        self.last_data_type = action_type

        if action_type == ActionMode.STUDENT:
            self.stu_replay_buffer.add_transition(experience)

        elif action_type == ActionMode.TEACHER:
            self.tea_replay_buffer.add_transition(experience)

        else:
            raise Exception(f"Unknown action type: {action_type}")

    def sample(self, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_sample_size

        #######################   Replay Buffer Batch Sample   #######################
        # Get last row of DRL-Student Buffer for computing safety status
        if self.last_data_type == ActionMode.STUDENT:
            idx = self.stu_replay_buffer.idx - 1
            obs = self.stu_replay_buffer.memory[0][idx]
            is_fail = self.stu_replay_buffer.memory[4][idx]
            boundary_state = observations2state(obs, is_fail)

        # Get last row of HA-Teacher Buffer for computing safety status
        elif self.last_data_type == ActionMode.TEACHER:
            idx = self.tea_replay_buffer.idx - 1
            obs = self.tea_replay_buffer.memory[0][idx]
            is_fail = self.tea_replay_buffer.memory[4][idx]
            boundary_state = observations2state(obs, is_fail)
        else:
            raise RuntimeError(f"Unknown action type pointer: {self.last_data_type}")

        # Calculate the safety status indicator by sT*P*s
        Vs = energy_value(np.asarray(boundary_state[:4]), LEARNING_P)
        scaled_Vs = self.rho1 * Vs + self.rho2

        # Batch size for Teacher Buffer
        tea_batch_size = max(min(batch_size - 1, int(batch_size * scaled_Vs)), 1)

        # Batch size for Student Buffer
        stu_batch_size = batch_size - tea_batch_size

        teacher_batches = self.tea_replay_buffer.sample(batch_size=tea_batch_size)
        student_batches = self.stu_replay_buffer.sample(batch_size=stu_batch_size)
        # print(f"Vs: {Vs}")
        # print(f"teacher_batch_size: {tea_batch_size}")
        # print(f"PHY-Teacher num of transition: {self.tea_replay_buffer.transition_count}")

        # Stacked transitions from student and teacher
        stacked = [np.vstack([x, y]) for x, y in zip(teacher_batches[:5], student_batches[:5])]
        stacked.append(np.append(teacher_batches[5], student_batches[5]))

        return stacked

    @property
    def is_prefilled(self):
        tot_transitions = self.tea_replay_buffer.transition_count + self.stu_replay_buffer.transition_count
        return tot_transitions >= self.experience_prefill_size

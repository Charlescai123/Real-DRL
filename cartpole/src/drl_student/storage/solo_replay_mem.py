import pickle
import numpy as np

from src.envs.utils import gym_seed
from src.drl_student.storage.replay_mem import ReplayMemory


class SoloReplayMemory(ReplayMemory):
    """
    Replay memory class to store trajectories
    """

    def __init__(self, solo_buffer_config, combined_experience_replay=False, seed=42):
        """
        initializing the replay memory
        """
        super().__init__(solo_buffer_config, combined_experience_replay, seed)
        self.exp_prefill_size = int(solo_buffer_config.experience_prefill_size)

    @property
    def is_prefilled(self):
        """Whether the buffer is prefilled for sampling"""
        return self.transition_count >= self.exp_prefill_size

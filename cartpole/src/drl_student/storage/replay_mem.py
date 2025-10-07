import numpy as np
import pickle
from abc import ABC, abstractmethod
from src.envs.utils import gym_seed


def clip_or_wrap_func(a, a_min, a_max, clip_or_wrap):
    if clip_or_wrap == 0:
        return np.clip(a, a_min, a_max)
    return (a - a_min) % (a_max - a_min) + a_min


def type_of(exp):
    if type(exp) is bool:
        return bool
    else:
        return float


def shape(exp):
    if type(exp) is list:
        return len(exp)
    if type(exp) is np.ndarray:
        return len(exp)
    else:
        return 1


class ReplayMemory:
    """
    Replay memory class to store trajectories
    """

    def __init__(self, buffer_config, combined_experience_replay=False, seed=42):
        """
        initializing the replay memory
        """
        self.buffer_size = int(buffer_config.buffer_size)
        self.batch_sample_size = int(getattr(buffer_config, "batch_size", 0))
        self.combined_experience_replay = combined_experience_replay
        self.new_head = False
        self.idx = 0
        self.head = -1
        self.full = False
        self.np_rand = gym_seed(seed)

        self.memory = None

    def reset(self):
        self.memory = None
        self.full = False
        self.idx = 0
        self.head = -1
        self.new_head = False

    def initialize(self, experience):
        self.memory = [np.zeros(shape=(self.buffer_size, shape(exp)), dtype=type_of(exp)) for exp in experience]
        self.memory.append(np.zeros(shape=self.buffer_size, dtype=float))

    def add_transition(self, experience, action_type=None):
        if self.memory is None:
            self.initialize(experience)
            print("initialized done")

        if len(experience) + 1 != len(self.memory):
            raise Exception('Experiment not the same size as memory', len(experience), '!=', len(self.memory))

        for e, mem in zip(experience, self.memory):
            mem[self.idx] = e

        self.head = self.idx
        self.new_head = True
        self.idx += 1
        if self.idx >= self.buffer_size:
            self.idx = 0  # replace the oldest one with the latest one
            self.full = True

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_sample_size

        current_size = self.idx if not self.full else self.buffer_size

        # batch size for sampling
        batch_size_to_sample = current_size if current_size < batch_size else batch_size

        random_idx = self.np_rand.choice(current_size, size=batch_size_to_sample, replace=False)

        if self.combined_experience_replay:
            if self.new_head:
                random_idx[0] = self.head  # always add the latest one
                self.new_head = False

        return [mem[random_idx] for mem in self.memory]

    def get(self, start, length):
        return [mem[start:start + length] for mem in self.memory]

    @property
    def transition_count(self):
        if self.full:
            return self.buffer_size
        return self.idx

    @property
    def max_size(self):
        return self.buffer_size

    def shuffle(self):
        """
        to shuffle the whole memory
        """
        self.memory = self.sample(batch_size=self.buffer_size)

    def save2file(self, file_path):
        with open(file_path, 'wb') as fp:
            pickle.dump(self.memory, fp)

    def load_memory_caches(self, path):

        with open(path, 'rb') as fp:
            memory = pickle.load(fp)
            if self.memory is None:
                self.memory = memory
            else:
                self.memory = np.hstack((self.memory, memory))

        print("Load memory caches, pre-filled replay memory!")


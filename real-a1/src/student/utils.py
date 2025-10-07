import numpy as np
import tensorflow as tf
import random
import os
import logging


def clip_or_wrap_func(a, a_min, a_max, clip_or_wrap):
    if clip_or_wrap == 0:
        return np.clip(a, a_min, a_max)
    return (a - a_min) % (a_max - a_min) + a_min


class ActionNoise:
    def __init__(self, action_dim, bounds, clip_or_wrap):
        self.action_dim = action_dim
        self.bounds = bounds
        self.clip_or_wrap = clip_or_wrap

    def sample(self) -> np.ndarray:
        pass

    def clip_or_wrap_action(self, action):
        if len(action) == 1:
            return clip_or_wrap_func(action, self.bounds[0], self.bounds[1], self.clip_or_wrap)
        return np.array([clip_or_wrap_func(a, self.bounds[0][k], self.bounds[1][k], self.clip_or_wrap[k]) for k, a in
                         enumerate(action)])

    def add_noise(self, action):
        sample = self.sample()
        action = self.clip_or_wrap_action(action + sample)
        return action


class OrnsteinUhlenbeckActionNoise(ActionNoise):

    def __init__(self, action_dim, bounds=(-1, 1), clip_or_wrap=0, mu=0, theta=0.15, sigma=0.1, dt=0.04):
        super().__init__(action_dim, bounds, clip_or_wrap)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X) * self.dt
        dx = dx + self.sigma * np.random.randn(len(self.X)) * np.sqrt(self.dt)
        self.X = self.X + dx
        return self.X


def seed_everything(seed):
    """Set the seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def setup_environment(cfg):
    """Configure environment settings based on GPU availability."""
    if not cfg.JobParams.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            tf.config.optimizer.set_jit(True)  # Enable XLA
            os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce verbosity
        except Exception as e:
            exit(f"GPU allocation failed: {e}")


def log_info(writer, global_step, info_dict, prefix, period=500):
    """Log information to Tensorboard."""
    if global_step % period == 0:
        for key, value in info_dict.items():
            writer.add_scalar(f"{prefix}/{key}", value, global_step)
    else:
        return

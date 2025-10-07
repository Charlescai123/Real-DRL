from __future__ import annotations
import numpy as np

from src.drl_student.agents import Agent
from src.drl_student.env import VecEnv


class DoubleStorageAgent(Agent):
    def __init__(
            self,
            env: VecEnv,
            action_max: float = np.inf,
            action_min: float = -np.inf,
            benchmark: bool = False,
            device: str = "cpu",
            gamma: float = 0.99,
    ):
        """Creates an DRL-Agent (contains both hp-student buffer and ha-teacher buffer)

        Args:
            env (VecEnv): The environment of the agent.
            action_max (float): The maximum action value.
            action_min (float): The minimum action value.
            benchmark (bool): Whether to benchmark src.
            device (str): The device to use for computation.
            gamma (float): The environment discount factor.
        """
        super().__init__(env=env, action_max=action_max, action_min=action_min, benchmark=benchmark, device=device,
                         gamma=gamma)

        del self.storage

        self.stu_storage = None  # Replay buffer for DRL-Student
        self.tea_storage = None  # Replay buffer for PHY-Teacher

    @property
    def initialized(self) -> bool:
        """Whether the agent has been initialized."""
        return self.stu_storage.initialized and self.tea_storage.initialized

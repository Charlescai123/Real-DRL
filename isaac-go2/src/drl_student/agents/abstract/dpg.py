import torch
from typing import Any, Dict, Tuple, Union

from src.drl_student.agents.abstract.actor_critic import AbstractActorCritic
from src.drl_student.env import VecEnv
from src.drl_student.storage.dual_replay_storage import DualReplayStorage
from src.drl_student.storage.replay_storage import ReplayStorage


class AbstractDPG(AbstractActorCritic):
    def __init__(
            self, env: VecEnv, action_noise_scale: float = 0.1, storage_initial_size=0, storage_size=100000,
            use_dual_replay_buffer=False, **kwargs
    ):
        """
        Args:
            env (VecEnv): A vectorized environment.
            action_noise_scale (float): The scale of the gaussian action noise.
            storage_initial_size (int): Initial size of the replay storage.
            storage_size (int): Maximum size of the replay storage.
        """
        assert action_noise_scale > 0

        super().__init__(env, **kwargs)

        # Dual Replay Buffer
        if use_dual_replay_buffer:
            self.storage = DualReplayStorage(
                self.env.num_envs, stu_max_size=int(storage_size / 2), tea_max_size=int(storage_size / 2),
                stu_initial_size=int(storage_initial_size / 2), tea_initial_size=int(storage_initial_size / 2),
                device=self.device,
            )
        else:
            self.storage = ReplayStorage(
                self.env.num_envs, max_size=storage_size, device=self.device, initial_size=storage_initial_size
            )

        self._register_serializable("storage")

        self._action_noise_scale = action_noise_scale

        self._register_serializable("_action_noise_scale")

    def draw_actions(
            self,
            obs: torch.Tensor,
            env_info: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]:
        actor_obs, critic_obs = self._process_observations(obs, env_info)

        actions = self.actor.forward(actor_obs)
        noise = torch.normal(torch.zeros_like(actions), torch.ones_like(actions) * self._action_noise_scale)
        noisy_actions = self._process_actions(actions + noise)

        data = {"actor_observations": actor_obs.clone(), "critic_observations": critic_obs.clone()}

        return noisy_actions, data

    def register_terminations(self, terminations: torch.Tensor) -> None:
        pass

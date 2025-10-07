import os
import sys

from omegaconf import DictConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cloud_trainer import CloudSystemConfig, CloudSystem
import hydra
from tensorboardX import SummaryWriter
from src.student.DDPG import DDPGAgent, DDPGConfig
from src.redis import RedisConnection
import dataclasses


@dataclasses.dataclass
class Config(CloudSystemConfig):
    AgentParams: DDPGConfig = dataclasses.field(default_factory=DDPGConfig)  # you can choose to use different


@hydra.main(version_base=None, config_path="../config", config_name="base_config.yaml")
def main(cfg: DictConfig):
    cfg_cloud = cfg.cloud
    cfg_cloud.JobParams.output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    writer = SummaryWriter(os.path.join(cfg_cloud.JobParams.output_path, cfg_cloud.JobParams.job_name))

    shape_observations = 12
    shape_action = 6

    redis_connection = RedisConnection(cfg_cloud.RedisParams)
    agent = DDPGAgent(cfg_cloud.AgentParams, shape_observations, shape_action)

    cloud_system = CloudSystem(cfg_cloud, agent, redis_connection, writer)
    cloud_system.run()


if __name__ == "__main__":
    main()

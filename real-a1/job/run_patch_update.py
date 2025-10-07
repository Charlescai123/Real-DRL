import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.teacher.patch_update import PatchUpdater
import hydra
from omegaconf import DictConfig
from src.redis import RedisConnection


@hydra.main(version_base=None, config_path="../config", config_name="base_config.yaml")
def main(cfg: DictConfig):
    cfg_edge = cfg.edge
    redis_connection = RedisConnection(cfg_edge.RedisParams)
    patch_updater = PatchUpdater(redis_connection)
    patch_updater.run()

if __name__ == "__main__":
    main()
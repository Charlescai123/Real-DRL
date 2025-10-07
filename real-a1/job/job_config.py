import dataclasses
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclasses.dataclass
class JobConfig:
    """using gpu"""
    gpu: bool = True
    """the name of this experiment"""
    job_name: str = ""
    """the running mode of this experiment (train/test/collect_data)"""
    run_mode: str = "train"
    """seed of the experiment"""
    seed: int = 1
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    capture_video: bool = False
    """saving path for the running logs, models and etc"""
    output_path: str = ''
    experiment_name: str = 'quadruped robot'

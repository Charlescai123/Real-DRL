import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf
import argparse

from job.run_cloud_train import Config as CloudSystemConfig
from job.run_edge_control import Config as EdgeControlConfig

ALL_JOBS = {
    "CloudSystemConfig_template": CloudSystemConfig,
    "EdgeControlConfig_template": EdgeControlConfig
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', action='store_true', help='Listing all available jobs')
    parser.add_argument('--generate', default='', help='Generate a config file for a specific job')
    parser.add_argument('--all', action='store_true', help='Generate all config files')
    args = parser.parse_args()

    if args.list:
        print("Available jobs:")
        for job_name in ALL_JOBS.keys():
            print(f"\t{job_name}")
        exit(0)

    if args.generate != '':
        config_directory = "./config"
        if not os.path.exists(config_directory):
            os.makedirs(config_directory)

        if args.generate not in ALL_JOBS:
            print(f"Job {args.generate} not found.")
            exit(1)
        OmegaConf.save(ALL_JOBS[args.generate], config_directory + f'/{args.generate}.yaml')
        print(f"Config file generated at {config_directory}/{args.generate}.yaml")
        exit(0)

    if args.all:
        config_directory = "./config"
        if not os.path.exists(config_directory):
            os.makedirs(config_directory)
        for job_name in ALL_JOBS.keys():
            OmegaConf.save(ALL_JOBS[job_name], config_directory + f'/{job_name}.yaml')
            print(f"Config file generated at {config_directory}/{job_name}.yaml")
        exit(0)

    print("Please specify the job to generate configurations --list to list all available jobs.")
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hydra
from src.edge_a1 import A1EdgeControl
import dataclasses
from src.edge_control import EdgeControlConfig
from src.student.DDPG import DDPGConfig as AgentConfig
from src.student.DDPG import DDPGAgent
from src.redis import RedisConnection
from src.teacher.phy_teacher import PHYTeacher
from envs.robot.unitree_a1 import a1  # A1 robot
from envs.simulator.setup_pybullet import setup_pybullet
from envs.robot.locomotion_controller_mpc import LocomotionController
from omegaconf import DictConfig
from src.trigger.Trigger import Trigger, TriggerConfig
from src.physical_design import MATRIX_P


@dataclasses.dataclass
class Config(EdgeControlConfig):
    AgentParams: AgentConfig = dataclasses.field(default_factory=AgentConfig)  # you can choose to use another agent
    TriggerParams: TriggerConfig = dataclasses.field(default_factory=TriggerConfig)


@hydra.main(version_base=None, config_path="../config", config_name="base_config.yaml")
def main(cfg: DictConfig):
    shape_observations = 12
    shape_action = 6

    envs_cfg = cfg.envs.simulator
    robot_cfg = cfg.envs.robot
    edge_cfg = cfg.edge

    redis_connection = RedisConnection(cfg.edge.RedisParams)

    agent_a = DDPGAgent(edge_cfg.AgentParams, shape_observations, shape_action)
    agent_b = DDPGAgent(edge_cfg.AgentParams, shape_observations, shape_action)

    p = setup_pybullet(envs_cfg)

    robot = a1.A1(
        pybullet_client=p,
        cmd_params=robot_cfg.command,
        a1_params=robot_cfg.interface,
        gait_params=robot_cfg.gait_scheduler,
        swing_params=robot_cfg.swing_controller,
        stance_params=robot_cfg.stance_controller,
        motor_params=robot_cfg.motor_group,
        vel_estimator_params=robot_cfg.com_velocity_estimator
    )

    mpc_controller = LocomotionController(
        robot=robot,
        desired_speed=(robot_cfg.command.desired_vx, robot_cfg.command.desired_vy),
        desired_twisting_speed=robot_cfg.command.desired_wz,
        desired_com_height=robot_cfg.command.mpc_body_height,
        mpc_body_mass=robot_cfg.command.mpc_body_mass,
        mpc_body_inertia=robot_cfg.command.mpc_body_inertia,
        gait_config=robot_cfg.gait_scheduler,
        swing_config=robot_cfg.swing_controller,
        stance_config=robot_cfg.stance_controller,
        vel_estimator_config=robot_cfg.com_velocity_estimator,
    )

    safety_trigger = Trigger(edge_cfg.TriggerParams, redis_connection,
                             MATRIX_P, ref_state=mpc_controller.ref_point)

    teacher = PHYTeacher(initial_ref_point=mpc_controller.ref_point)

    edge_control = A1EdgeControl(edge_cfg, robot, mpc_controller,
                                 teacher, safety_trigger,
                                 agent_a, agent_b, redis_connection)

    edge_control.run()


if __name__ == "__main__":
    main()

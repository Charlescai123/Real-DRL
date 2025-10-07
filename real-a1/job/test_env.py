from envs.robot.unitree_a1 import a1
import hydra
from omegaconf import DictConfig
import pybullet
from pybullet_utils import bullet_client
from envs.robot.locomotion_controller_mpc import LocomotionController
from envs.simulator.worlds import plane_world, slope_world, stair_world, uneven_world
import copy


def setup_pybullet(envs_cfg):
    p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(envs_cfg.fixed_time_step)
    p.setAdditionalSearchPath(envs_cfg.envs_path)
    p.setPhysicsEngineParameter(numSolverIterations=envs_cfg.num_solver_iterations)
    p.setPhysicsEngineParameter(enableConeFriction=envs_cfg.enable_cone_friction)

    if envs_cfg.camera.is_on:
        p.resetDebugVisualizerCamera(
            cameraDistance=envs_cfg.camera.distance,
            cameraYaw=envs_cfg.camera.yaw,
            cameraPitch=envs_cfg.camera.pitch,
            cameraTargetPosition=envs_cfg.camera.target_position
        )

    p.addUserDebugLine([0, 0, 0], [1, 0, 0], lineColorRGB=[255, 0, 0])
    p.addUserDebugLine([0, 0, 0], [0, 1, 0], lineColorRGB=[0, 255, 0])
    p.addUserDebugLine([0, 0, 0], [0, 0, 1], lineColorRGB=[0, 0, 255])
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    WORLD_NAME_TO_CLASS_MAP = dict(plane=plane_world.PlaneWorld,
                                   slope=slope_world.SlopeWorld,
                                   stair=stair_world.StairWorld,
                                   uneven=uneven_world.UnevenWorld)

    world = WORLD_NAME_TO_CLASS_MAP['plane'](p, search_path=envs_cfg.envs_path)
    world.build_world()
    # Record a video or not
    if envs_cfg.video.record:
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, envs_cfg.video.save_path)

    return p


@hydra.main(version_base=None, config_path="../config", config_name="base_config.yaml")
def main(cfg: DictConfig):
    envs_cfg = cfg.envs.simulator
    robot_cfg = cfg.envs.robot

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
    controller = LocomotionController(
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

    for i in range(10):
        robot.reset()
        controller.reset_controllers()
        for step in range(10000):
            state_observation = copy.deepcopy(controller.robot_state)  # the state is estimated from the mpc controller
            controller.update()
            robot.step(action=controller.get_action()[0])
            print(state_observation)


if __name__ == "__main__":
    main()

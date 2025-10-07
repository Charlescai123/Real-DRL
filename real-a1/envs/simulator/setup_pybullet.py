import pybullet
from pybullet_utils import bullet_client
from envs.simulator.worlds import plane_world, slope_world, stair_world, uneven_world


def setup_pybullet(envs_cfg, enable_gui=True):
    if enable_gui:
        p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    else:
        p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

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
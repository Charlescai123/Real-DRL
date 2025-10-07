import io
import copy
import matlab
import matlab.engine
import numpy as np
import cvxpy as cp
from ml_collections import ConfigDict
from omegaconf import DictConfig, OmegaConf


class MatEngine:

    def __init__(self, cfg: DictConfig):
        self.engine = None

        # Matlab engine output
        self.out = io.StringIO() if cfg.stdout is False else None
        self.err = io.StringIO() if cfg.stderr is False else None

        self.matlab_engine_launch(working_path=cfg.working_path)

        # CVX Tool Setup
        if cfg.cvx_toolbox.setup:
            self.cvx_setup(cvx_path=cfg.cvx_toolbox.relative_path)

    def matlab_engine_launch(self, working_path="src/phy_teacher/matlab/"):
        print("Launching Matlab Engine...")
        self.engine = matlab.engine.start_matlab()
        self.engine.cd(working_path)
        print("Matlab current working directory is ---->>>", self.engine.pwd())

    def cvx_setup(self, cvx_path="./cvx"):
        current_path = self.engine.pwd()
        self.engine.cd(cvx_path)
        print("Setting up the CVX Toolbox...")
        _ = self.engine.cvx_setup
        print("CVX Toolbox setup done, switch back to original working path")
        self.engine.cd(current_path)

    def patch_lmi(self, tracking_err):
        tracking_err = matlab.double(tracking_err.tolist())
        F_kp, F_kd, t_min = self.engine.patch_lmi(tracking_err, nargout=3, stdout=self.out, stderr=self.err)
        return F_kp, F_kd, t_min <= 0


if __name__ == '__main__':
    tracking_error = np.array([0.0073, -0.2821, -0.2257, 0.0309, 0.1311, 0.2719, 0.0342, 0.4687, -0.3929, -0.6300])

    matlab_engine_config = ConfigDict()
    cvx_toolbox_config = ConfigDict()
    matlab_engine_config.stdout = True
    matlab_engine_config.stderr = True
    matlab_engine_config.working_path = "src/phy_teacher/matlab/"
    cvx_toolbox_config.setup = False
    cvx_toolbox_config.relative_path = "./cvx"
    matlab_engine_config.cvx_toolbox = cvx_toolbox_config

    matlab_engine_cfg = OmegaConf.create(matlab_engine_config.to_dict())

    mat = MatEngine(matlab_engine_cfg)
    Kp, Kd, tmin = mat.patch_lmi(tracking_error)

    import torch
    print(type(torch.tensor(Kp)))
    print(f"Kp: {np.array(Kp)}")
    print(f"Kp: {np.array(Kd)}")

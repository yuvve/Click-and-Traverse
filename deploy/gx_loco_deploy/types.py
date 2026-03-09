from dataclasses import dataclass
import numpy as np


@dataclass
class PolicyCfg:
    onnx_path: str
    # obs-action
    action_scale: float
    action_dim: int
    obs_limits: tuple = None  # obs_min, obs_max

    # record
    log_dir: str = None
    tag: str = None

    # policy
    is_debug: bool = False
    infer_dt: float = 0.02
    # jnt_range: tuple = None  # obs_min, obs_max


@dataclass
class Obs:
    command: np.ndarray
    root_gyro: np.ndarray
    root_quat: np.ndarray  # w, x, y, z
    qpos: np.ndarray
    qvel: np.ndarray
    obs_pf: np.ndarray = None
    torque: np.ndarray = 0.0
    timestamp: float = 0.0
    height: np.ndarray = 1.0


@dataclass
class Act:
    action: np.ndarray
    motor_targets: np.ndarray
    timestamp: float = 0.0


class Policy:
    cfg: PolicyCfg

    def reset(self):
        raise NotImplementedError

    def infer(self, obs: Obs):
        raise NotImplementedError

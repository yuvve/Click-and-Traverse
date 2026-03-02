import tyro
import numpy as np
from dataclasses import dataclass
from g1 import G1, G1Config
from ros_adapter import ROS2IO
from constants import ACTION_JOINT_NAMES, OBS_JOINT_NAMES, DEFAULT_QPOS, KPs, KDs

JOINT_ID_LOOKUP = {
    # Name: ID
    # ...
}


@dataclass
class Args:
    onnx_model_path: str


def play(args: Args):
    onnx_model_path = args.onnx_model_path

    action_joint_ids = []
    for joint_name in ACTION_JOINT_NAMES:
        action_joint_ids.append(JOINT_ID_LOOKUP[joint_name])

    obs_joint_ids = []
    for joint_name in OBS_JOINT_NAMES:
        obs_joint_ids.append(JOINT_ID_LOOKUP[joint_name])

    config = G1Config(
        action_joint_ids=action_joint_ids,
        obs_joint_ids=obs_joint_ids,
        default_qpos=np.array(DEFAULT_QPOS[7:]),
        joint_ranges=None,  # TODO: initialize to correct values
        kp_gains=KPs,
        kd_gains=KDs,
        sensor_name_to_id_map=None,  # TODO: initialize to correct values
    )
    io = ROS2IO()
    robot = G1(config, io)
    robot.load_model(onnx_model_path)
    while True:
        robot.step()


if __name__ == "__main__":
    args = tyro.cli(Args)
    play(args)

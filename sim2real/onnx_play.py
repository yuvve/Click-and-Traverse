import tyro
from dataclasses import dataclass
from g1 import G1, G1Config
from constants import ACTION_JOINT_NAMES

ACTION_JOINT_ID_LOOKUP = {
    # Name: ID
    # ...
}

action_joint_ids = []
for joint_name in ACTION_JOINT_NAMES:
    action_joint_ids.append(ACTION_JOINT_ID_LOOKUP[joint_name])


@dataclass
class Args:
    onnx_model_path: str


def play(args: Args):
    onnx_model_path = args.onnx_model_path
    config = G1Config(action_joint_ids, action_scale=0.5)
    robot = G1(config)
    robot.load_model(onnx_model_path)
    while True:
        robot.run()


if __name__ == "__main__":
    args = tyro.cli(Args)
    play(args)

import tyro
import numpy as np
import onnxruntime as rt
from dataclasses import dataclass
from g1 import G1


@dataclass
class Args:
    onnx_model_path: str


def play(args: Args):
    onnx_model_path = args.onnx_model_path
    output_names = ["continuous_actions"]
    policy = rt.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    robot = G1()
    while True:
        state = robot.get_observation().reshape(1, -1).astype(np.float32)
        onnx_input = {"obs": state}
        action = policy.run(output_names, onnx_input)[0]
        action = action[0]
        robot.actuate(action)


if __name__ == "__main__":
    args = tyro.cli(Args)
    play(args)

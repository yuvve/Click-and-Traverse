import numpy as np
from abc import ABC, abstractmethod


class Humanoid(ABC):
    @abstractmethod
    def step(self):
        """Ask the model for the next action and perform it"""
        pass

    @abstractmethod
    def load_model(self, onnx_model_path: str):
        """Loads an onnx model to be used for inference"""
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Creates the input to the neural network"""
        pass

    @abstractmethod
    def actuate(self, onnx_output: dict):
        """Actuates the robot based on the output from the neural network"""
        pass

    @abstractmethod
    def reset(self):
        """Resets the robot to its initial state"""
        pass


class HumanoidAsyncIO(ABC):
    @abstractmethod
    def send(self, channel, data):
        pass

    @abstractmethod
    def subscribe(self, channel):
        pass

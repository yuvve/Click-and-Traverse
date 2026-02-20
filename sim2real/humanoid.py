from abc import ABC, abstractmethod


class Humanoid(ABC):
    @abstractmethod
    def get_observation(self) -> dict:
        """Provides the input to the neural network"""
        pass

    @abstractmethod
    def actuate(self, onnx_output: dict):
        """Actuates the robot based on the output from the neural network"""
        pass

    @abstractmethod
    def get_last_action(self) -> dict:
        """Provides the last action taken by the robot"""
        pass

    @abstractmethod
    def reset(self):
        """Resets the robot to its initial state"""
        pass

    @abstractmethod
    def get_command(self) -> dict:
        """Provides the current command to the robot"""
        pass

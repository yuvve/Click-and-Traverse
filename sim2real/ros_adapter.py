from abc import abstractmethod
from robot import RobotIO


class ROS2IO(RobotIO):
    @abstractmethod
    def get_sensor_data(self, sensor_name):
        pass

    @abstractmethod
    def send_actuation(self, actuation):
        pass

    @abstractmethod
    def reset(self):
        pass

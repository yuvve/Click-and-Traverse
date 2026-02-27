from robot import RobotIO


class ROS2IO(RobotIO):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def get_sensor_data(self, sensor_name):
        raise NotImplementedError

    def send_actuation(self, actuation):
        raise NotImplementedError

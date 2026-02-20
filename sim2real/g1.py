import jax
import jax.numpy as jp
from humanoid import Humanoid


class G1(Humanoid):
    def __init__(self):
        super().__init__()
        self.state = {}
        self.actions = []
        self.commands = []
        self.reset()

    def get_observation(self) -> dict:
        # Implement the observation logic for G1
        state = jp.hstack(
            [
                self._get_gyro_pelvis(),
                self._get_gvec_pelvis(),
                self._get_joint_angles(),
                self._get_joint_vel(),
                self.get_last_action(),
                self._get_motor_targets(),
                self.get_command(),
                self._get_foot_height(),
                self._get_gait_phase(),
                self._get_pf(),
            ]
        )

    def actuate(self, onnx_output: dict):
        action = self._get_actuation_action(onnx_output)
        self._send_actuation_via_ros(action)

    def _get_actuation_action(self, onnx_output: dict) -> dict:
        pass

    def _send_actuation_via_ros(self, action: dict):
        pass

    def get_last_action(self) -> dict:
        pass

    def reset(self):
        pass

    def get_command(self) -> dict:
        pass

    def _get_gyro_pelvis(self) -> jax.Array:
        pass

    def _get_gvec_pelvis(self) -> jax.Array:
        pass

    def _get_joint_angles(self) -> jax.Array:
        pass

    def _get_joint_vel(self) -> jax.Array:
        pass

    def _get_motor_targets(self) -> jax.Array:
        pass

    def _get_foot_height(self) -> jax.Array:
        pass

    def _get_gait_phase(self) -> jax.Array:
        pass

    def _get_pf(self) -> jax.Array:
        pass

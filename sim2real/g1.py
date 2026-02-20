import jax
import jax.numpy as jp
from dataclasses import dataclass
from cat_ppo.envs.g1.constants import ACTION_JOINT_NAMES
from humanoid import Humanoid
from onnx_model import ONNXPolicy


@dataclass
class G1Config:
    action_joint_ids: list
    action_scale: float = 0.5


class G1(Humanoid):
    def __init__(self, config: G1Config):
        super().__init__()
        self.config = config
        self.state = None
        self.actions = None
        self.commands = None
        self.current_model = None
        self.reset()

    def run(self):
        state = self.get_state().reshape(1, -1).astype(np.float32)
        if self.current_model is None:
            raise ValueError("Model not loaded")
        actions = self.current_model.inference(state)
        action = actions[0]
        self.actuate(action)

    def load_model(self, onnx_model_path: str):
        self.current_model = ONNXPolicy(onnx_model_path)

    def get_state(self) -> dict:
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
        return state

    def actuate(self, onnx_output: dict):
        action = self._get_actuation_action(onnx_output)
        self._send_actuation_via_ros(action)

    def _get_actuation_action(self, onnx_output) -> dict:
        lower_motor_targets = jp.clip(
            _get_motor_targets()[self.config.action_joint_ids] + onnx_output * self.config.action_scale,
        #     self._soft_lowers[self.action_joint_ids],
        #     self._soft_uppers[self.action_joint_ids],
        # )
        # motor_targets = self._default_qpos.copy()
        # motor_targets = motor_targets.at[self.action_joint_ids].set(lower_motor_targets)
        # state.info["rng"], data = torque_step(
        #     state.info["rng"],
        #     self.mjx_model,
        #     state.data,
        #     motor_targets,
        #     kps=self._kps,
        #     kds=self._kds,
        #     kp_scale=state.info["kp_scale"],
        #     kd_scale=state.info["kd_scale"],
        #     rfi_lim_scale=state.info["rfi_lim_scale"],
        #     torque_limit=self.torque_limit,
        #     n_substeps=self.n_substeps,
        # )

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

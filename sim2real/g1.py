import numpy as np
from dataclasses import dataclass
from cat_ppo.envs.g1.env_cat import world_to_navi_pos
from cat_ppo.envs.g1.play_cat import base2navi_transform, world_to_navi_vel
from humanoid import Humanoid, HumanoidAsyncIO
from onnx_model import ONNXPolicy


@dataclass
class G1Config:
    action_joint_ids: list
    obs_joint_ids: list
    default_qpos: np.ndarray
    action_scale: float = 0.5
    foot_height: float = 0.07

    def __init__(self):
        self.default_motor_targets = self.default_qpos


class G1IO(HumanoidAsyncIO):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def send(self, channel, data):
        raise NotImplementedError

    def subscribe(self, channel):
        raise NotImplementedError


class G1(Humanoid):
    def __init__(self, config: G1Config, io: G1IO):
        super().__init__()
        self.config = config
        self.io = io
        self.reset()

    def step(self):
        state = self.get_state().reshape(1, -1).astype(np.float32)
        if self.current_model is None:
            raise ValueError("Model not loaded")
        actions = self.current_model.inference(state)
        action = actions[0]
        self.actuate(action)

    def load_model(self, onnx_model_path: str):
        self.current_model = ONNXPolicy(onnx_model_path)

    def get_state(self) -> np.ndarray:
        """
        :returns the current state of the robot:
        np.ndarray
        dtype: dtype('float64')
        shape: (162,)
        """

        navi2world_pose = self._get_navi2world_pose()

        state = np.hstack(
            [
                self._get_gyro_pelvis(),  # 3
                self._get_gvec_pelvis(),  # 3
                # joint state
                (self._get_joint_angles() - self.config.default_qpos)[self.config.obs_joint_ids],  # 23
                self._get_joint_vel()[self.config.obs_joint_ids],  # 23
                self._get_last_action(),  # 12
                self._get_motor_targets()[self.config.action_joint_ids],  # num_actions
                # commands
                [self._get_last_flags()[1]],
                self._get_command(navi2world_pose),  # 4
                self.config.foot_height,  # 1
                self._get_gait_phase(),  # 4
                self._get_gf(navi2world_pose, "head"),
                self._get_bf(navi2world_pose, "head"),
                self._get_df("head"),
                self._get_gf(navi2world_pose, "pelv"),
                self._get_bf(navi2world_pose, "pelv"),
                self._get_df("pelv"),
                self._get_gf(navi2world_pose, "tors"),
                self._get_bf(navi2world_pose, "tors"),
                self._get_df("tors"),
                self._get_gf(navi2world_pose, "feet"),
                self._get_bf(navi2world_pose, "feet"),
                self._get_df("feet"),
                self._get_gf(navi2world_pose, "hands"),
                self._get_bf(navi2world_pose, "hands"),
                self._get_df("hands"),
                self._get_gf(navi2world_pose, "knees"),
                self._get_bf(navi2world_pose, "knees"),
                self._get_df("knees"),
                self._get_gf(navi2world_pose, "shlds"),
                self._get_bf(navi2world_pose, "shlds"),
                self._get_df("shlds"),
            ]
        )
        return state

    def actuate(self, onnx_output: dict):
        action = self._get_actuation_action(onnx_output)
        self._send_actuation(action)

    def _get_gyro_pelvis(self) -> np.ndarray:
        # return self.mj_data.sensordata[sensor_adr : sensor_adr + sensor_dim]
        raise NotImplementedError

    def _get_gvec_pelvis(self) -> np.ndarray:
        # gvec_pelvis = self.mj_data.site_xmat[self._pelvis_imu_site_id].reshape(
        #     3, 3
        # ).T @ np.array([0, 0, -1])
        # return gvec_pelvis
        raise NotImplementedError

    def _get_joint_angles(self) -> np.ndarray:
        # joint_angles = self.mj_data.qpos[7:]
        # return joint_angles
        raise NotImplementedError

    def _get_joint_vel(self) -> np.ndarray:
        # joint_vel = self.mj_data.qvel[6:]
        # return joint_vel
        raise NotImplementedError

    def _get_last_action(self):
        # return info["last_act"]
        raise NotImplementedError

    def _get_motor_targets(self):
        return self.motor_targets

    def _get_last_flags(self) -> list[np.float64]:
        # return info["last_flags"]
        raise NotImplementedError

    def _get_pelvis2world_rot(self) -> np.ndarray:
        # pelvis2world_rot = self.mj_data.site_xmat[self._pelvis_imu_site_id].reshape(
        #     3, 3
        # )
        # return pelvis2world_rot
        raise NotImplementedError

    def _get_pelvis_imu(self) -> np.ndarray:
        # return self.mj_data.site_xpos[self._pelvis_imu_site_id]
        raise NotImplementedError

    def _get_gait_phase(self) -> np.ndarray:
        # gait_phase = np.hstack([np.cos(info["phase"]), np.sin(info["phase"])])
        # return gait_phase
        raise NotImplementedError

    def _get_navi2world_pose(self) -> np.ndarray:
        pelvis2world_rot = self._get_pelvis2world_rot()
        navi2world_rot = base2navi_transform(pelvis2world_rot)
        navi2world_pose = np.eye(4)
        navi2world_pose[:3, :3] = navi2world_rot
        navi2world_pose[:2, 3] = self._get_pelvis_imu()[:2]
        navi2world_pose[2, 3] = 0.75
        return navi2world_pose

    def _get_field(self, field_name) -> np.ndarray:
        # return info[field_name].copy()
        raise NotImplementedError

    def _get_gf(self, navi2world_pose, field_name) -> np.ndarray:
        field = self._get_field(field_name + "gf")
        field = world_to_navi_pos(navi2world_pose, field.reshape(-1, 3))
        return field.reshape(-1)

    def _get_bf(self, navi2world_pose, field_name) -> np.ndarray:
        field = self._get_field(field_name + "bf")
        field = world_to_navi_pos(navi2world_pose, field.reshape(-1, 3))
        field = field * (field < 0.5)
        return field.reshape(-1)

    def _get_df(self, field_name) -> np.ndarray:
        field = self._get_field(field_name + "df")
        field = np.clip(field, -1.0, 0.5)
        return field.reshape(-1)

    def _get_command(self, navi2world_pose) -> np.ndarray:
        """
        np.ndarray
        dtype: dtype('float64')
        shape: (3,)
        size: 3
        """
        command = self.command.copy()
        command = world_to_navi_vel(navi2world_pose, command.reshape(-1, 3)).reshape(3)
        command[-1] = 0
        return command

    def _get_actuation_action(self, onnx_output) -> dict:
        # lower_motor_targets = jp.clip(
        #     _get_motor_targets()[self.config.action_joint_ids] + onnx_output * self.config.action_scale,
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
        raise NotImplementedError

    def _send_actuation(self, action: dict):
        pass

    def reset(self):
        self.current_model = None
        self.motor_targets = self.config.default_motor_targets
        self.actions = None
        self.command = None  # TODO: see below
        # command = self.compute_cmd_from_rtf(
        #     pelvgf.reshape(-1), np.concat([headgf, feetgf, handsgf]), np.concat([headbf, feetbf, handsbf])
        # )

import time
import numpy as np
from dataclasses import dataclass

from cat_ppo.envs.g1.env_cat import world_to_navi_pos
from cat_ppo.envs.g1.play_cat import base2navi_transform, world_to_navi_vel

from sim2real.robot import Robot, RobotIO, RobotConfig
from sim2real.onnx_model import ONNXPolicy
from sim2real.ros_adapter import ROS2IO
from sim2real.constants import Kp, Kd, G1JointIndex, G1_NUM_MOTOR, Mode

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient


class G1RO2SIO(ROS2IO):
    def __init__(self):
        super().__init__()
        self.time_ = 0.0
        self.control_dt_ = 0.002  # [2ms]
        self.duration_ = 3.0  # [3 s]
        self.counter_ = 0
        self.mode_pr_ = Mode.PR
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.update_mode_machine_ = False
        self.crc = CRC()

    def Init(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result["name"]:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        # create publisher #
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        # create subscriber #
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def get_sensor_data(self, sensor_name):
        raise NotImplementedError

    def send_actuation(self, actuation):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True

        self.counter_ += 1
        if self.counter_ % 500 == 0:
            self.counter_ = 0
            print(self.low_state.imu_state.rpy)

    def LowCmdWrite(self):
        self.time_ += self.control_dt_

        if self.time_ < self.duration_:
            # [Stage 1]: set robot to zero posture
            for i in range(G1_NUM_MOTOR):
                ratio = np.clip(self.time_ / self.duration_, 0.0, 1.0)
                self.low_cmd.mode_pr = Mode.PR
                self.low_cmd.mode_machine = self.mode_machine_
                self.low_cmd.motor_cmd[i].mode = 1  # 1:Enable, 0:Disable
                self.low_cmd.motor_cmd[i].tau = 0.0
                self.low_cmd.motor_cmd[i].q = (1.0 - ratio) * self.low_state.motor_state[i].q
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = Kp[i]
                self.low_cmd.motor_cmd[i].kd = Kd[i]

        elif self.time_ < self.duration_ * 2:
            # [Stage 2]: swing ankle using PR mode
            max_P = np.pi * 30.0 / 180.0
            max_R = np.pi * 10.0 / 180.0
            t = self.time_ - self.duration_
            L_P_des = max_P * np.sin(2.0 * np.pi * t)
            L_R_des = max_R * np.sin(2.0 * np.pi * t)
            R_P_des = max_P * np.sin(2.0 * np.pi * t)
            R_R_des = -max_R * np.sin(2.0 * np.pi * t)

            self.low_cmd.mode_pr = Mode.PR
            self.low_cmd.mode_machine = self.mode_machine_
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnklePitch].q = L_P_des
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnkleRoll].q = L_R_des
            self.low_cmd.motor_cmd[G1JointIndex.RightAnklePitch].q = R_P_des
            self.low_cmd.motor_cmd[G1JointIndex.RightAnkleRoll].q = R_R_des

        else:
            # [Stage 3]: swing ankle using AB mode
            max_A = np.pi * 30.0 / 180.0
            max_B = np.pi * 10.0 / 180.0
            t = self.time_ - self.duration_ * 2
            L_A_des = max_A * np.sin(2.0 * np.pi * t)
            L_B_des = max_B * np.sin(2.0 * np.pi * t + np.pi)
            R_A_des = -max_A * np.sin(2.0 * np.pi * t)
            R_B_des = -max_B * np.sin(2.0 * np.pi * t + np.pi)

            self.low_cmd.mode_pr = Mode.AB
            self.low_cmd.mode_machine = self.mode_machine_
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnkleA].q = L_A_des
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnkleB].q = L_B_des
            self.low_cmd.motor_cmd[G1JointIndex.RightAnkleA].q = R_A_des
            self.low_cmd.motor_cmd[G1JointIndex.RightAnkleB].q = R_B_des

            max_WristYaw = np.pi * 30.0 / 180.0
            L_WristYaw_des = max_WristYaw * np.sin(2.0 * np.pi * t)
            R_WristYaw_des = max_WristYaw * np.sin(2.0 * np.pi * t)
            self.low_cmd.motor_cmd[G1JointIndex.LeftWristRoll].q = L_WristYaw_des
            self.low_cmd.motor_cmd[G1JointIndex.RightWristRoll].q = R_WristYaw_des

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)


@dataclass
class G1Config(RobotConfig):
    action_joint_ids: list
    obs_joint_ids: list
    default_qpos: np.ndarray
    joint_ranges: np.ndarray
    kp_gains: np.ndarray
    kd_gains: np.ndarray
    sensor_name_to_id_map: dict
    dt: float = 0.02  # TODO:is this simulation-specific?
    init_phase: np.ndarray = np.array([0.0, np.pi])  # TODO:is this simulation-specific?
    action_scale: float = 0.5
    foot_height: float = 0.07
    gait_freq: float = 1.5  # TODO:is this simulation-specific?
    soft_joint_pos_limit_factor: float = 0.95

    def __init__(self):
        self.default_motor_targets = self.default_qpos
        self._calc_joint_ranges()
        self._calc_phase()

    def _calc_joint_ranges(self):
        self.lowers, self.uppers = self.joint_ranges[1:].T
        c = (self.lowers + self.uppers) / 2
        r = self.uppers - self.lowers
        self.soft_lowers = c - 0.5 * r * self.soft_joint_pos_limit_factor
        self.soft_uppers = c + 0.5 * r * self.soft_joint_pos_limit_factor

    def _calc_phase(self):
        self.phase_dt = 2 * np.pi * self.dt * self.gait_freq


class Fields:
    def __init__(self) -> None:
        # TODO: initialize to the correct values ("None" just a placeholder)
        self.gf = None
        self.bf = None
        self.df = None

    def get_gf(self):
        return self.gf

    def get_bf(self):
        return self.bf

    def get_df(self):
        return self.df


class G1(Robot):
    def __init__(self, config: G1Config, io: RobotIO):
        super().__init__()
        self.config = config
        self.io = io
        self.fields = {
            "head": Fields(),
            "pelv": Fields(),
            "tors": Fields(),
            "feet": Fields(),
            "hands": Fields(),
            "knees": Fields(),
            "shlds": Fields(),
        }

        self.reset()

    def step(self):
        state = self.get_state().reshape(1, -1).astype(np.float32)
        if self.current_model is None:
            raise ValueError("Model not loaded")
        model_outputs = self.current_model.inference(state)
        model_output = model_outputs[0]
        self.actuate(model_output)
        self.model_output = model_output

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
        self._update_command(navi2world_pose)

        state = np.hstack(
            [
                self._get_gyro_pelvis(),  # 3
                self._get_gvec_pelvis(),  # 3
                # joint state
                (self._get_joint_angles() - self.config.default_qpos)[self.config.obs_joint_ids],  # 23
                self._get_joint_vel()[self.config.obs_joint_ids],  # 23
                self._get_last_model_output(),  # 12
                self._get_motor_targets()[self.config.action_joint_ids],  # num_actions
                # commands
                [self._get_last_flags()[1]],
                self._get_command(),  # 4
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
        torques = self._get_actuation_action(onnx_output)
        self._send_torques(torques)

    def _get_gyro_pelvis(self) -> np.ndarray:
        return self.io.get_sensor_data(self.config.sensor_name_to_id_map["pelvis_gyro"])

    def _get_gvec_pelvis(self) -> np.ndarray:
        return self.io.get_sensor_data(self.config.sensor_name_to_id_map["pelvis_gvec"]).reshape(3, 3).T @ np.array(
            [0, 0, -1]
        )

    def _get_joint_angles(self) -> np.ndarray:
        return self.io.get_sensor_data(self.config.sensor_name_to_id_map["joint_angles"])[7:]

    def _get_joint_vel(self) -> np.ndarray:
        return self.io.get_sensor_data(self.config.sensor_name_to_id_map["joint_vel"])[6:]

    def _get_last_model_output(self):
        return self._get_last_model_output()

    def _get_motor_targets(self) -> np.ndarray:
        return self.motor_targets

    def _set_motor_targets(self, new_motor_targets):
        self.motor_targets = new_motor_targets

    def _get_last_flags(self) -> list[np.float64]:
        return self.last_flags

    def _get_pelvis_imu(self) -> np.ndarray:
        return self.io.get_sensor_data(self.config.sensor_name_to_id_map["pelvis_imu"])

    def _get_gait_phase(self) -> np.ndarray:
        return np.hstack([np.cos(self.phase), np.sin(self.phase)])

    def _get_navi2world_pose(self) -> np.ndarray:
        pelvis2world_rot = self._get_pelvis_imu().reshape(3, 3)
        navi2world_rot = base2navi_transform(pelvis2world_rot)
        navi2world_pose = np.eye(4)
        navi2world_pose[:3, :3] = navi2world_rot
        navi2world_pose[:2, 3] = self._get_pelvis_imu()[:2]
        navi2world_pose[2, 3] = 0.75
        return navi2world_pose

    def _get_gf(self, navi2world_pose, part_name) -> np.ndarray:
        field = self.fields[part_name].get_gf()
        field = world_to_navi_pos(navi2world_pose, field.reshape(-1, 3))
        return field.reshape(-1)

    def _get_bf(self, navi2world_pose, part_name) -> np.ndarray:
        field = self.fields[part_name].get_bf()
        field = world_to_navi_pos(navi2world_pose, field.reshape(-1, 3))
        field = field * (field < 0.5)
        return field.reshape(-1)

    def _get_df(self, part_name) -> np.ndarray:
        field = self.fields[part_name].get_df()
        field = np.clip(field, -1.0, 0.5)
        return field.reshape(-1)

    def _update_command(self, navi2world_pose):
        """
        np.ndarray
        dtype: dtype('float64')
        shape: (3,)
        size: 3
        """
        self.last_command = self.command.copy()
        command = self.command.copy()
        command = world_to_navi_vel(navi2world_pose, command.reshape(-1, 3)).reshape(3)
        command[-1] = 0
        self.command = command

    def _get_command(self) -> np.ndarray:
        return self.command

    def _get_actuation_action(self, onnx_output) -> np.ndarray:
        lower_motor_targets = np.clip(
            self._get_motor_targets()[self.config.action_joint_ids] + onnx_output * self.config.action_scale,
            self.config.soft_lowers[self.config.action_joint_ids],
            self.config.soft_uppers[self.config.action_joint_ids],
        )

        motor_targets = self.config.default_motor_targets.copy()
        motor_targets[self.config.action_joint_ids] = lower_motor_targets
        self._set_motor_targets(motor_targets.copy())

        q = self._get_joint_angles()
        qd = self._get_joint_vel()

        torques = self.config.kp_gains * (self.motor_targets - q) + self.config.kd_gains * (-qd)

        return torques

    def _send_torques(self, torques: np.ndarray):
        self.io.send_actuation(torques)

    def reset(self):
        self.current_model = None
        self.model_output = np.zeros(12)
        self.command = None  # TODO: this is probably incorrect
        self.last_command = None  # TODO: this is probably incorrect
        self.flags = np.zeros(2)
        self.last_flags = np.zeros(2)
        self.motor_targets = self.config.default_motor_targets.copy()
        self.phase = self.config.init_phase
        # TODO: reset fields object

    # def reset(self):
    #     self.mj_data.qpos[:7] = consts.DEFAULT_QPOS[:7]
    #     self.mj_data.qpos[7:] = self._default_qpos
    #
    #     mujoco.mj_forward(self.mj_model, self.mj_data)
    #     if not self.headless:
    #         self.viewer.sync()
    #     phase_dt = 2 * np.pi * self.dt * self.gait_freq
    #
    #     head_pos = self.mj_data.site_xpos[self._head_site_id]
    #     head_vel = np.zeros_like(head_pos)
    #     feet_pos = self.mj_data.site_xpos[self._feet_site_id]
    #     feet_vel = np.zeros_like(feet_pos)
    #     hands_pos = self.mj_data.site_xpos[self._hands_site_id]
    #     hands_vel = np.zeros_like(hands_pos)
    #     knees_pos = self.mj_data.site_xpos[self._knees_site_id]
    #     shlds_pos = self.mj_data.site_xpos[self._shlds_site_id]
    #     pelv_pos = self.mj_data.site_xpos[self._pelvis_imu_site_id].reshape(1, -1)
    #     tors_pos = self.mj_data.site_xpos[self._torso_imu_site_id].reshape(1, -1)
    #     all_poses = np.concatenate(
    #         [
    #             head_pos.reshape(1, -1),
    #             pelv_pos.reshape(1, -1),
    #             tors_pos.reshape(1, -1),
    #             feet_pos,
    #             hands_pos,
    #             knees_pos,
    #             shlds_pos,
    #         ],
    #         axis=0,
    #     )
    #     all_gf = self.sample_field(self.gf, all_poses)
    #     all_bf = self.sample_field(self.bf, all_poses)
    #     all_df = self.sample_field(self.sdf, all_poses)
    #     headgf, pelvgf, torsgf, feetgf, handsgf, kneesgf, shldsgf = np.split(all_gf, [1, 2, 3, 5, 7, 9], axis=0)
    #     headbf, pelvbf, torsbf, feetbf, handsbf, kneesbf, shldsbf = np.split(all_bf, [1, 2, 3, 5, 7, 9], axis=0)
    #     headdf, pelvdf, torsdf, feetdf, handsdf, kneesdf, shldsdf = np.split(all_df, [1, 2, 3, 5, 7, 9], axis=0)
    #
    #     command = self.compute_cmd_from_rtf(
    #         pelvgf.reshape(-1), np.concat([headgf, feetgf, handsgf]), np.concat([headbf, feetbf, handsbf])
    #     )
    #
    #     info = {
    #         "step": 0,
    #         "command": command.copy(),
    #         "last_command": command.copy(),
    #         "flags": np.zeros(2),
    #         "last_flags": np.zeros(2),
    #         "last_act": np.zeros(12),
    #         "phase_dt": phase_dt,
    #         "phase": self._init_phase.copy(),
    #         "foot_height": self.foot_height,
    #         "motor_targets": self._default_qpos.copy(),  # NOTE
    #         "timestamp_move2stop": 100,
    #         "gait_mask": np.zeros(2),
    #         "odom_delay": self.mj_data.qpos[:7],
    #         "headgf": headgf.copy(),
    #         "headbf": headbf.copy(),
    #         "headdf": headdf.copy(),
    #         "head_pos": head_pos.copy(),
    #         "head_vel": head_vel.copy(),
    #         "feetgf": feetgf.copy(),
    #         "feetbf": feetbf.copy(),
    #         "feetdf": feetdf.copy(),
    #         "feet_pos": feet_pos.copy(),
    #         "feet_vel": feet_vel.copy(),
    #         "handsgf": handsgf.copy(),
    #         "handsbf": handsbf.copy(),
    #         "handsdf": handsdf.copy(),
    #         "hands_pos": hands_pos.copy(),
    #         "hands_vel": hands_vel.copy(),
    #         "kneesgf": kneesgf.copy(),
    #         "kneesbf": kneesbf.copy(),
    #         "kneesdf": kneesdf.copy(),
    #         "knees_pos": knees_pos.copy(),
    #         "shldsgf": shldsgf.copy(),
    #         "shldsbf": shldsbf.copy(),
    #         "shldsdf": shldsdf.copy(),
    #         "shlds_pos": shlds_pos.copy(),
    #         "pelvgf": pelvgf.copy(),
    #         "pelvbf": pelvbf.copy(),
    #         "pelvdf": pelvdf.copy(),
    #         "pelv_pos": pelv_pos.copy(),
    #         "torsgf": torsgf.copy(),
    #         "torsbf": torsbf.copy(),
    #         "torsdf": torsdf.copy(),
    #         "tors_pos": tors_pos.copy(),
    #     }
    #     # breakpoint()
    #     obs = self.get_obs(info)
    #     return State(info, obs)

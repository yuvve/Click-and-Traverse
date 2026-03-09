import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
import onnxruntime as rt

from gx_loco_deploy.types import PolicyCfg, Policy, Obs, Act
from gx_loco_deploy.utils.recorder import Recoder
from gx_loco_deploy.policies.g1_cat import constants as consts


def quat2gevc(quat, g_axis=np.float32([0.0, 0.0, -1.0])):
    # quat: w, x, y, z
    rot = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix()  # Quaternion to rotation matrix
    gvec = rot.T @ g_axis  # Rotate the gravity axis to the body frame

    # Calculate roll and pitch
    roll = np.arctan2(gvec[1], gvec[2])  # Roll (around x-axis)
    pitch = np.arctan2(-gvec[0], np.sqrt(gvec[1]**2 + gvec[2]**2))  # Pitch (around y-axis)

    return gvec, roll, pitch


def torque_clip(qpos_des, qpos, qvel, kps, kds, torque_limit):
    joint_pos_low = (kds * qvel - torque_limit) / kps + qpos
    joint_pos_high = (kds * qvel + torque_limit) / kps + qpos
    clipped_qpos_des = np.clip(qpos_des, joint_pos_low, joint_pos_high)
    return clipped_qpos_des


def get_move_flag(cmd_vel):
    return np.linalg.norm(cmd_vel) > 0.2


class OnnxPolicy(Policy):

    @property
    def phase_dt(self):
        return 2 * np.pi * self.infer_dt * self.gait_freq
    
    def __init__(self, cfg: PolicyCfg):
        self.cfg = cfg
        self.infer_dt = cfg.infer_dt
        self.action_scale = cfg.action_scale
        self.obs_limits = cfg.obs_limits

        # modules
        self._output_names = ["continuous_actions"]
        # self.infer_fn = rt.InferenceSession(cfg.onnx_path, providers=["CUDAExecutionProvider"]) # CUDAExecutionProvider / CPUExecutionProvider
        self.infer_fn = rt.InferenceSession(cfg.onnx_path, providers=["CPUExecutionProvider"]) # CUDAExecutionProvider / CPUExecutionProvider
        if cfg.log_dir is not None:
            datatime = datetime.now().strftime("%m%d%H%M")
            tag = f"_{cfg.tag}" if cfg.tag else ""
            self.log_dir = f"{cfg.log_dir}/{datatime}{tag}"
            self._recorder = Recoder()
        else:
            self._recorder = None

        self._init_phase = np.array([0, np.pi])
        self._stance_phase = np.array([0.0, 0.0])
        self.actuators_ids_active = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        self.obs_joint_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                       17, 18, 22, 23, 24, 25])
        # fmt: off
        self._default_qpos = consts.DEFAULT_QPOS.copy()
        # fmt: on
        self._num_joints = len(self._default_qpos)

        self.gait_freq = 1.5
        # self.gait_freq = 1.1
        # self.phase_dt = 2 * np.pi * self.infer_dt * self.gait_freq
        self.height = 1.0
        # buffers
        self._phase = np.array([0, np.pi])
        self._foot_height = 0.07
        # self._foot_height = 0.3
        self._last_move_flag = 0
        self.skill_mask = np.zeros(4)

        # history_len = 15
        # num_obs = 85
        self._last_action = np.zeros(cfg.action_dim, dtype=np.float32)
        self._info = {
            # "state_sensor_mem": np.zeros((15, 64))
        }
        self.motor_targets = self._default_qpos.copy()
        # self._obs_history = np.zeros((history_len, num_obs), dtype=np.float32)

        # self._lowers, self._uppers = self.cfg.jnt_range
        # c = (self._lowers + self._uppers) / 2
        # r = self._uppers - self._lowers
        self._soft_lowers = consts.SOFT_LOWERS.copy()
        self._soft_uppers = consts.SOFT_UPPERS.copy()
        
        self.step = 0
        self.stop_timestep = 50
        self.move_flag = 0
        self.obs_history = []


    def infer(self, obs: Obs) -> Act:
        nn_obs = self._get_nn_obs(obs)
        nn_action = self.infer_fn.run(self._output_names, {"obs": nn_obs})[0][0]

        self._update_phase(obs.command)

        # record
        if self._recorder is not None:
            self._recorder.add(obs, nn_action)
        self.step += 1

        return nn_action
    
    def update_motor_targets(self, nn_action):
        lower_motor_targets = np.clip(
            self.motor_targets[self.actuators_ids_active]
            + nn_action * self.action_scale,
            self._soft_lowers[self.actuators_ids_active],
            self._soft_uppers[self.actuators_ids_active],
        )
        self.motor_targets = self._default_qpos.copy()
        self.motor_targets[self.actuators_ids_active] = lower_motor_targets

        
        # clipped_motor_targets = torque_clip(
        #     motor_targets, obs.qpos, obs.qvel, consts.KPs, consts.KDs, consts.TORQUE_LIMIT)

        action = Act(
            nn_action.copy(),
            self.motor_targets.copy(),
            timestamp=time.perf_counter(),
        )

        # update
        self._last_action = nn_action.copy()
        return action

    def _get_nn_obs(self, obs: Obs):
        assert len(obs.command) == 3
        gvec_pelvis, r, p = quat2gevc(np.array(obs.root_quat))
        gait_phase = np.hstack([np.cos(self._phase), np.sin(self._phase)])

        gait_cycle = np.cos(self._phase)
        gait_mask = np.where(gait_cycle > 0.6, 1, 0)
        gait_mask = np.where(gait_cycle < -0.6, -1, gait_mask)
        gait_mask = np.float32(gait_mask)

        cmd_vel = obs.command
        obs_cmd = np.hstack([[self.move_flag * self.gait_freq], cmd_vel])
        # print(obs_cmd)
        
        
        state_sensor = np.hstack(
            [
                obs.root_gyro, 
                gvec_pelvis,  
                (obs.qpos - self._default_qpos)[self.obs_joint_ids],  
                obs.qvel[self.obs_joint_ids],
                self._last_action,
                self.motor_targets[self.actuators_ids_active],
                self._foot_height,
                gait_phase,
                obs.obs_pf,
                # r,p,
            ]
        )
        # print(state_sensor)
        state_command = np.hstack([
            # self.skill_mask,
            gait_mask,
            obs_cmd,
        ])
        # print(state_command)
        

        state = np.hstack([state_sensor, state_command])

        self.obs_history.append(state)
        self.obs_history = self.obs_history[-32:] # NOTE
        state = np.array(self.obs_history)[None]
        

        # _state_sensor_mem = np.roll(self._info["state_sensor_mem"], 1, axis=0)
        # _state_sensor_mem[0] = state_sensor
        # self._info["state_sensor_mem"] = _state_sensor_mem
        # state_mem = np.hstack([_state_sensor_mem.reshape(-1), state_command])

        return np.float32(state)
    
    def _update_phase_cmd(self, obs):
        command_height = obs.height
        self._update_phase(obs.command)
        self.height = self.height + np.clip(
            command_height,
            -0.005,
            0.005,
        )
        self.height = np.clip(
            self.height,
            0.6,
            1.0,
        )

    def _update_phase(self, command):
        has_vel = get_move_flag(command)
        had_vel = self._last_move_flag
        move2stop = (had_vel == 1.0) & (has_vel == 0.0)
        stop2move = (had_vel == 0.0) & (has_vel == 1.0)

        self.stop_timestep = np.where(move2stop, self.step + 50, self.stop_timestep)
        after_delay = self.step > self.stop_timestep
        self.move_flag = np.where((has_vel == 0.0) & after_delay, 0.0, 1.0)
        phase = self._phase + self.phase_dt
        phase = np.fmod(phase + np.pi, 2 * np.pi) - np.pi
        phase = np.where(self.move_flag == 1.0, phase, self._stance_phase)
        phase = np.where(stop2move == 1.0, self._init_phase, phase)
        self._phase = phase
        # print(self.phase_dt)

        # update
        self._last_move_flag = has_vel

    def end(self):
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        record_filename = f"{self.log_dir}/obs_act_record.pkl"
        if self._recorder is not None:
            self._recorder.save(record_filename)
        else:
            print("No recorder, skip saving.")
        print(f"Record saved to {record_filename}")

    @property
    def default_qpos(self):
        return self._default_qpos.copy()



class OnnxPolicy_pf_cmd(OnnxPolicy):
    @property
    def phase_dt(self):
        return 2 * np.pi * self.infer_dt * self.gait_freq
    
    def __init__(self, cfg: PolicyCfg):
        self.cfg = cfg
        self.infer_dt = cfg.infer_dt
        self.action_scale = cfg.action_scale
        self.obs_limits = cfg.obs_limits

        # modules
        self._output_names = ["continuous_actions"]
        # self.infer_fn = rt.InferenceSession(cfg.onnx_path, providers=["CUDAExecutionProvider"]) # CUDAExecutionProvider / CPUExecutionProvider
        self.infer_fn = rt.InferenceSession(cfg.onnx_path, providers=["CPUExecutionProvider"]) # CUDAExecutionProvider / CPUExecutionProvider
        if cfg.log_dir is not None:
            datatime = datetime.now().strftime("%m%d%H%M")
            tag = f"_{cfg.tag}" if cfg.tag else ""
            self.log_dir = f"{cfg.log_dir}/{datatime}{tag}"
            self._recorder = Recoder()
        else:
            self._recorder = None

        self._init_phase = np.array([0, np.pi])
        self._stance_phase = np.array([0.0, 0.0])
        self.actuators_ids_active = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        self.obs_joint_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                       17, 18, 22, 23, 24, 25])
        # fmt: off
        self._default_qpos = consts.DEFAULT_QPOS.copy()
        # fmt: on
        self._num_joints = len(self._default_qpos)

        self.gait_freq = 1.5
        # self.gait_freq = 1.1
        # buffers
        self._phase = np.array([0, np.pi])
        self._foot_height = 0.07
        # self._foot_height = 0.3
        self._last_move_flag = 0

        self._last_action = np.zeros(cfg.action_dim, dtype=np.float32)
        self._info = {
            # "state_sensor_mem": np.zeros((15, 64))
        }
        self.motor_targets = self._default_qpos.copy()
        self._soft_lowers = consts.SOFT_LOWERS.copy()
        self._soft_uppers = consts.SOFT_UPPERS.copy()
        self.step = 0
        self.stop_timestep = 0
        self.move_flag = np.zeros([])
        self.obs_history = []

    def _get_nn_obs(self, obs: Obs):
        assert len(obs.command) == 3
        gvec_pelvis, r, p = quat2gevc(np.array(obs.root_quat))
        gait_phase = np.hstack([np.cos(self._phase), np.sin(self._phase)])

        # gait_cycle = np.cos(self._phase)
        # gait_mask = np.where(gait_cycle > 0.6, 1, 0)
        # gait_mask = np.where(gait_cycle < -0.6, -1, gait_mask)
        # gait_mask = np.float32(gait_mask)

        cmd_vel = obs.command
        obs_cmd = np.hstack([[self.move_flag], cmd_vel])
        # print(obs_cmd)
        
        
        state = np.hstack(
            [
                obs.root_gyro, 
                gvec_pelvis,  
                (obs.qpos - self._default_qpos)[self.obs_joint_ids],  
                obs.qvel[self.obs_joint_ids],
                self._last_action,
                self.motor_targets[self.actuators_ids_active],
                obs_cmd,
                self._foot_height,
                gait_phase,
                obs.obs_pf,
            ]
        )

        return np.float32(state)[None]


class OnnxPolicy_velocity_cmd(OnnxPolicy):
    @property
    def phase_dt(self):
        return 2 * np.pi * self.infer_dt * self.gait_freq
    
    def __init__(self, cfg: PolicyCfg):
        self.cfg = cfg
        self.infer_dt = cfg.infer_dt
        self.action_scale = cfg.action_scale
        self.obs_limits = cfg.obs_limits

        # modules
        self._output_names = ["continuous_actions"]
        # self.infer_fn = rt.InferenceSession(cfg.onnx_path, providers=["CUDAExecutionProvider"]) # CUDAExecutionProvider / CPUExecutionProvider
        self.infer_fn = rt.InferenceSession(cfg.onnx_path, providers=["CPUExecutionProvider"]) # CUDAExecutionProvider / CPUExecutionProvider
        if cfg.log_dir is not None:
            datatime = datetime.now().strftime("%m%d%H%M")
            tag = f"_{cfg.tag}" if cfg.tag else ""
            self.log_dir = f"{cfg.log_dir}/{datatime}{tag}"
            self._recorder = Recoder()
        else:
            self._recorder = None

        self._init_phase = np.array([0, np.pi])
        self._stance_phase = np.array([0.0, 0.0])
        self.actuators_ids_active = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        self.obs_joint_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                       17, 18, 22, 23, 24, 25])
        # fmt: off
        self._default_qpos = consts.DEFAULT_QPOS.copy()
        # fmt: on
        self._num_joints = len(self._default_qpos)

        self.gait_freq = 1.5
        # self.gait_freq = 1.1
        # buffers
        self._phase = np.array([0, np.pi])
        self._foot_height = 0.07
        # self._foot_height = 0.3
        self._last_move_flag = 0

        self._last_action = np.zeros(cfg.action_dim, dtype=np.float32)
        self._info = {
            # "state_sensor_mem": np.zeros((15, 64))
        }
        self.motor_targets = self._default_qpos.copy()
        self._soft_lowers = consts.SOFT_LOWERS.copy()
        self._soft_uppers = consts.SOFT_UPPERS.copy()
        self.step = 0
        self.stop_timestep = 0
        self.move_flag = np.zeros([])
        self.obs_history = []

    def _get_nn_obs(self, obs: Obs):
        assert len(obs.command) == 3
        gvec_pelvis, r, p = quat2gevc(np.array(obs.root_quat))
        gait_phase = np.hstack([np.cos(self._phase), np.sin(self._phase)])

        # gait_cycle = np.cos(self._phase)
        # gait_mask = np.where(gait_cycle > 0.6, 1, 0)
        # gait_mask = np.where(gait_cycle < -0.6, -1, gait_mask)
        # gait_mask = np.float32(gait_mask)

        cmd_vel = obs.command
        obs_cmd = np.hstack([[self.move_flag], cmd_vel])
        # print(obs_cmd)
        
        
        state = np.hstack(
            [
                obs.root_gyro, 
                gvec_pelvis,  
                (obs.qpos - self._default_qpos)[self.obs_joint_ids],  
                obs.qvel[self.obs_joint_ids],
                self._last_action,
                self.motor_targets[self.actuators_ids_active],
                obs_cmd,
                self._foot_height,
                gait_phase,
                # obs.obs_pf,
            ]
        )

        return np.float32(state)[None]





from typing import Union
import numpy as np
import time
import mujoco
from collections import deque
from rclpy.node import Node
from gx_loco_deploy.policies.g1_cat.octomap_idl import VoxelMap
from octomap_msgs.msg import Octomap
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_ as Odometry
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_ as PoseStamped
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from gx_loco_deploy.common.command_helper import (
    create_damping_cmd,
    create_zero_cmd,
    init_cmd_hg,
    MotorMode,
)
from gx_loco_deploy.types import Policy, Obs
from gx_loco_deploy.common.remote_controller import RemoteController, KeyMap
from gx_loco_deploy.policies.g1_cat import constants as consts
from gx_loco_deploy.policies.g1_cat.pf import PotentialField

from multiprocessing import shared_memory

#### share
SHM_NAME = "pf_shm"
# SHAPE = (140, 140, 40, 3+3+1)  # gf (3), bf (3), sdf (1)
SHAPE = (128, 128, 35, 3+3+1)  # gf (3), bf (3), sdf (1)
DTYPE = np.float32
nbytes_pf = int(np.prod(SHAPE) * np.dtype(DTYPE).itemsize)
nbytes = int((np.prod(SHAPE)+2) * np.dtype(DTYPE).itemsize)
try:
    existing_shm = shared_memory.SharedMemory(name=SHM_NAME)
    print(f"[create_shared_memory] 找到已有共享内存: {existing_shm.name}")
    existing_shm.close()
    existing_shm.unlink()  # 解除链接
    print(f"[create_shared_memory] 解除链接并删除现有共享内存: {SHM_NAME}")
except FileNotFoundError:
    print(f"[create_shared_memory] 没有找到共享内存: {SHM_NAME}, 继续创建新的共享内存")
SHM = shared_memory.SharedMemory(name=SHM_NAME, create=False, size=nbytes)

class LowLevelController:
    def __init__(
            self,
            policy_vel: Policy,
            policy_pf: Policy,
            default_qpos: np.ndarray,
            policy_init_qpos: np.ndarray,
            kps: np.ndarray,
            kds: np.ndarray,
            ros_node,
            infer_dt=0.02,
            is_debug=True,
            pf_path=None,
            xml_path=None,
    ) -> None:
        self.is_debug = is_debug
        self.policy_vel = policy_vel
        self.policy_pf = policy_pf
        self.infer_dt = infer_dt
        self.num_motor = len(default_qpos)
        self.default_qpos = default_qpos
        self.motor_kps = kps
        self.motor_kds = kds
        self.joint2motor_ids = consts.JOINT2MOTOR_IDX_LEG + consts.JOINT2MOTOR_IDX_WAIST_ARM

        self.pose_list = []

        self._eps = 0.3                # 带宽 epsilon，可调 0.01~0.05m
        self._ransac_dist = 0.02
        self._ransac_n = 5
        self._ransac_iters = 1000
        self._init_frames = 10          # 前 N 帧用于估计 floor 高度
        self._z_samples = []
        self._floor_ready = False
        self._z_floor = 0.0             # floor_init 的 z=0 对应 gravity_init 的高度
        self.filtered_pub = None        # 如果你要发布，初始化 Publisher（Unitree Channel/ROS2二选一）

        # self.ros_node = ros_node 
        # self.tf_broadcaster = TransformBroadcaster(self.ros_node)
        self.policy_init_qpos = policy_init_qpos

        # modules
        self.remote_controller = RemoteController()

        # buffer
        self.delay_buf = deque(maxlen=100)
        self._last_verbose_time = time.time()
        self._last_command = np.zeros(3)
        self.move_flag = np.zeros(1)

        self.qpos = np.zeros(self.num_motor, dtype=np.float32)
        self.qvel = np.zeros(self.num_motor, dtype=np.float32)
        self.torque = np.zeros(self.num_motor, dtype=np.float32)
        self.action = np.zeros(self.num_motor, dtype=np.float32)
        self.target_dof_pos = default_qpos.copy()
        self._last_start = 0
        self.counter = 0
        self.sdf_save = False
        # if consts.MSG_TYPE == "hg":
        # g1 and h1_2 use the hg msg type
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0
        
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.pf_path = pf_path
        self.pf = PotentialField(self.mj_model, self.pf_path)
        
        self.lowcmd_publisher_ = ChannelPublisher(consts.TOPIC_LOWCMD, LowCmdHG)
        self.lowcmd_publisher_.Init()
        
        self.lowstate_subscriber = ChannelSubscriber(consts.TOPIC_LOWSTATE, LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        
        self.sdf_subscriber = ChannelSubscriber(consts.TOPIC_VOXEL, VoxelMap)
        self.sdf_subscriber.Init(self.VoxelMapHandler, 10)
        self.odometry_subscriber = ChannelSubscriber(consts.TOPIC_NEWODOM, Odometry)
        self.odometry_subscriber.Init(self.OdometryHandler, 10)

        # /goal_pose            
        # Type: geometry_msgs/msg/PoseStamped
        # Publisher count: 1
        # Subscription count: 0


        # wait for the subscriber to receive data
        self.wait_for_low_state()
        # Initialize the command msg
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

        self._soft_lowers = consts.SOFT_LOWERS.copy()
        self._soft_uppers = consts.SOFT_UPPERS.copy()

        # record slam odo
        self.recording_pose_ori = False      # 记录开关
        self.odo_pose_list = []                  # 已存 pose 列表
        self.odo_ori_list = []           # 已存 ori 四元数列表
        self.qpos_list = []           # 已存 qpos 列表
        from datetime import datetime
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_file_path = "lidar_log/pose_ori_record"
        self.pose_ori_file = f"{base_file_path}_{current_time}.npz"
        #### share
        # SHM = shared_memory.SharedMemory(name=SHM_NAME, create=False)
        self.pf_arr = np.ndarray(SHAPE, dtype=DTYPE, buffer=SHM.buf)
        self.Sxy = np.ndarray(
            (2,),                  # 形状是 (2,)
            dtype=DTYPE,
            buffer=SHM.buf,
            offset=nbytes_pf    # 从数组结束的位置开始
        )
        self.finish_signal = np.ones(1, dtype=np.float32)
        # SHM.close()
 
    def Start(self):
        self.run_step_ptr = RecurrentThread(
            interval=self.infer_dt,
            target=self.run_step,
            name="run_step",
        )
        self.run_step_ptr.Start()
        self.recording_pose_ori = True


    def VoxelMapHandler(self, msg: VoxelMap):
        if self.pf_path is not None:
            return

        pf_arr=self.pf_arr.copy()
        self.pf.gf = pf_arr[:,:,:,0:3]
        self.pf.bf = pf_arr[:,:,:,3:6]
        self.pf.sdf = pf_arr[:,:,:,6:7]
        Sxy = self.Sxy.copy()
        self.pf.cfg.Sx = int(Sxy[0])
        self.pf.cfg.Sy = int(Sxy[1])

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)
        
        # _start = self.remote_controller.button[KeyMap.start]
        # _start_up = (self._last_start == 0) and (_start == 1)
        # _start_down = (self._last_start == 1) and (_start == 0)
        # self._last_start = _start

    # def GoalHandler(self, msg: PoseStamped):
    #     pass
    #     self.goal = msg.pose


    def OdometryHandler(self, msg: Odometry):
        pose = msg.pose
        self.pose = pose
        # print(f'Odometry: {self.pose.pose}')
        self.pose_list.append(self.pose)
        self.mj_data.qpos[0:3] = np.array([self.pose.pose.position.x, self.pose.pose.position.y, self.pose.pose.position.z])
        ori = self.pose.pose.orientation
        self.mj_data.qpos[3:7] = np.array([ori.w, ori.x, ori.y, ori.z])
        # self.mj_data.qpos[3:7] = np.array(self.low_state.imu_state.quaternion)
        self.gyroscope = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
        self.mj_data.qvel[3:6] = self.gyroscope
        self.mj_data.qpos[7:] = self.qpos # dof
        
        mujoco.mj_forward(self.mj_model, self.mj_data)
        if self.recording_pose_ori:
            self.odo_pose_list.append(np.array([self.pose.pose.position.x, self.pose.pose.position.y, self.pose.pose.position.z]))
            self.odo_ori_list.append(np.array([ori.w, ori.x, ori.y, ori.z]))
            self.qpos_list.append(self.qpos.copy())

    def save_pose_ori(self):
        if self.odo_pose_list == []:
            print("No pose and orientation data to save.")
            return
        pose_array = np.vstack(self.odo_pose_list)
        orientation_array = np.vstack(self.odo_ori_list)
        qpos_array =  np.vstack(self.qpos_list)

        # 保存为 npz
        np.savez(self.pose_ori_file,
                 position=pose_array,
                 orientation=orientation_array,
                 qpos=qpos_array)
        print(f"Saved pose and orientation to {self.pose_ori_file}")

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.infer_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.infer_dt)

    def damping_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_damping_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            # print(self.remote_controller.button[KeyMap.start])
            time.sleep(self.infer_dt)
        
    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 5
        # num_step = int(total_time / self.infer_dt)
        dof_size = len(self.joint2motor_ids)

        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            motor_id = self.joint2motor_ids[i]
            init_dof_pos[i] = self.low_state.motor_state[motor_id].q

        self.move_to(init_dof_pos, self.policy_init_qpos, duration=total_time)

        # for i in range(num_step):
        #     alpha = i / num_step
        #     tar_qpos = init_dof_pos * (1 - alpha) + self.default_qpos * alpha
        #     self.apply_motor_target(tar_qpos, self.motor_kps, self.motor_kds)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            self.apply_motor_target(self.default_qpos, self.motor_kps, self.motor_kds)
            time.sleep(self.infer_dt)

    def run_step(self):
        self.counter += 1
        # Get the current joint position and velocity
        for i in range(len(self.joint2motor_ids)):
            self.qpos[i] = self.low_state.motor_state[self.joint2motor_ids[i]].q
            self.qvel[i] = self.low_state.motor_state[self.joint2motor_ids[i]].dq
            self.torque[i] = self.low_state.motor_state[self.joint2motor_ids[i]].tau_est
        
        # imu_state quaternion: w, x, y, z
        quat_pelvis = self.low_state.imu_state.quaternion
        
        gyro_pelvis = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)

        # joint state
        qpos = self.qpos.copy()
        qvel = self.qvel.copy()

        obs_pf, command = self.pf.get_potential_field_1204(self.mj_data, move_flag=self.move_flag*self.finish_signal)

        if not self.remote_controller.button[KeyMap.up]:
            command = np.float32([self.remote_controller.ly * 0.4,
                                self.remote_controller.lx * -1 * 0.4,
                                self.remote_controller.rx * -1 * 0.5])
            command = self._last_command + np.clip(command - self._last_command, self.infer_dt * -1, self.infer_dt * 1)
            self.finish_signal[:] = 1.0
        else:
            self.finish_signal[:] *= self.move_flag

        self._last_command = command.copy()

        obs = Obs(
            command=command.copy(),
            root_gyro=gyro_pelvis.copy(),
            root_quat=quat_pelvis.copy(),
            qpos=qpos.copy(),
            qvel=qvel.copy(),
            obs_pf=obs_pf.copy(),
            torque=self.torque.copy(),
            timestamp=time.perf_counter(),
        )

        act_vel = self.policy_vel.infer(obs)
        act_pf = self.policy_pf.infer(obs)
        if self.remote_controller.button[KeyMap.up]:
            nn_action = act_pf
            self.move_flag = self.policy_pf.move_flag
        else:
            nn_action = act_vel
            self.move_flag = self.policy_vel.move_flag
        act = self.policy_vel.update_motor_targets(nn_action)
        self.policy_pf.update_motor_targets(nn_action)

        
        motor_targets = act.motor_targets
        self.apply_motor_target(motor_targets, self.motor_kps, self.motor_kds)
        # self.apply_motor_target(self.default_qpos, self.motor_kps, self.motor_kds)
        
        if self.is_debug: 
            self.delay_buf.append(time.perf_counter())
            if time.time() - self._last_verbose_time > 1.0:
                if len(self.delay_buf) > 1:
                    avg_delay = 1 / np.mean(np.diff(self.delay_buf))
                    print(f"avg control freq: {avg_delay:.1f} Hz")
                    self._last_verbose_time = time.time()

    def apply_motor_target(self, tar_qpos, kps, kds):
        for i in range(len(self.joint2motor_ids)):
            motor_idx = self.joint2motor_ids[i]
            self.low_cmd.motor_cmd[motor_idx].q = tar_qpos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        if self.is_debug:
            create_damping_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)

    def move_to(self, curr_qpos, tar_qpos, duration, kp=None, kd=None):
        if kp is None:
            kp = self.motor_kps
        if kd is None:
            kd = self.motor_kds

        num_steps = int(duration / self.infer_dt)
        for i in range(1, num_steps + 1):
            alpha = i / num_steps
            qpos = curr_qpos * (1 - alpha) + tar_qpos * alpha
            self.apply_motor_target(qpos, kp, kd)
            time.sleep(self.infer_dt)


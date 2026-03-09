from typing import Union
import numpy as np
import time
from collections import deque

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
import open3d as o3d
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo

from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_ as OdomModeState
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_ as Odometry
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_ as PoseStamped

from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_ as PointCloudState

from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread


from gx_loco_deploy.types import Policy, Obs
from gx_loco_deploy.common.remote_controller import RemoteController, KeyMap
from gx_loco_deploy.types import PolicyCfg
from gx_loco_deploy.policies.g1_cat import constants as consts
from gx_loco_deploy.policies.g1_cat.g1_onnx_policy import OnnxPolicy_pf_cmd, OnnxPolicy_velocity_cmd

import rclpy

from gx_loco_deploy.policies.g1_cat.low_level_controller import LowLevelController

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
SHM = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=nbytes)

def wait_for_shm(timeout: float = 10.0, interval: float = 0.05):
    start = time.time()
    last_err = None
    while time.time() - start < timeout:
        try:
            shm = shared_memory.SharedMemory(name=SHM_NAME)
            return
        except FileNotFoundError as e:
            last_err = e
            time.sleep(interval)

    raise FileNotFoundError(f"🛑 🛑 🛑 等待共享内存 {SHM_NAME} 超时，最后错误: {last_err}")

import subprocess
import time
import signal
import os

TMUX_SESSION = "ros_multi"

def terminal_sh_tmux():
    """
    启动 5 个 ROS / Python 节点到一个 tmux 会话中。
    """
    subprocess.run(f"tmux kill-session -t {TMUX_SESSION}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run(f"tmux new-session -d -s {TMUX_SESSION} -n livox", shell=True)

    cmds = [
        # 窗口 1：Livox 驱动
        """
        conda deactivate;
        source /opt/ros/foxy/setup.bash;
        source ~/unitree_ros2/setup.sh;
        source ~/workspace/Click-and-Traverse/deploy/Click-and-Traverse-SLAM/lidar_ws/install/setup.bash;
        ros2 launch livox_ros_driver2 msg_MID360_launch.py;
        exec bash
        """,

        # 窗口 2：Octomap Bridge
        """
        conda activate deploy_cat;
        source /opt/ros/foxy/setup.bash;
        source ~/unitree_ros2/setup.sh;
        source ~/workspace/Click-and-Traverse/deploy/Click-and-Traverse-SLAM/lidar_ws/install/setup.bash;
        python -m scripts.exp_dis_pf.octomap_bridge;
        exec bash
        """,

        # 窗口 3：FastLIO
        """
        source /opt/ros/foxy/setup.bash;
        source ~/unitree_ros2/setup.sh;
        source ~/workspace/Click-and-Traverse/deploy/Click-and-Traverse-SLAM/lidar_ws/install/setup.bash;
        ros2 launch fast_lio mapping.launch.py config_file:=mid360.yaml rviz:=true;
        exec bash
        """,

        # 窗口 4：Filter Ground
        """
        conda activate deploy_cat;
        cd /home/ubuntu/workspace/Click-and-Traverse/deploy
        python -m scripts.exp_dis_pf.filter_ground;
        exec bash
        """,

        # 窗口 5：Octomap Server
        """
        conda activate deploy_cat;
        source /opt/ros/foxy/setup.bash;
        source ~/unitree_ros2/setup.sh;
        source ~/workspace/Click-and-Traverse/deploy/Click-and-Traverse-SLAM/lidar_ws/install/setup.bash;
        ros2 launch octomap_server octomap_mapping.launch.xml;
        exec bash
        """
    ]

    for i, cmd in enumerate(cmds):
        if i == 0:
            subprocess.run(f"tmux send-keys -t {TMUX_SESSION}:0 '{cmd}' C-m", shell=True)
        else:
            subprocess.run(f"tmux new-window -t {TMUX_SESSION} -n node{i+1} '{cmd}'", shell=True)
        if i==2:
            time.sleep(0.5)

    print(f"✅ 所有任务已在 tmux 会话 [{TMUX_SESSION}] 中启动。")
    print(f"👉 查看运行状态：tmux attach -t {TMUX_SESSION}")
    print(f"🛑 关闭所有终端：tmux kill-session -t {TMUX_SESSION}")



def close_all_tmux():
    """关闭整个 tmux 会话"""
    subprocess.run(f"tmux kill-session -t {TMUX_SESSION}", shell=True)
    print(f"🛑 已关闭 tmux 会话 [{TMUX_SESSION}]")


def main():
    xml_path = "/home/ubuntu/workspace/Click-and-Traverse/data/assets/unitree_g1/scene_mjx_feetonly_mesh.xml"
    pf_path = None # update real-time pf from octomap_bridge.py
    
    onnx_path = "/home/ubuntu/workspace/Click-and-Traverse/data/logs/G1_mj_axis/11221955_G1LocoPFRz_v6highsmooth_vel/checkpoints/000403046400/policy.onnx"
    onnx_path_pf = "/home/ubuntu/workspace/Click-and-Traverse/data/logs/G1_mj_axis/11221955_G1LocoPFR10_v6highsmooth_pf/checkpoints/000403046400/policy.onnx"


    def extract_g1_token(onnx_path):
        tokens = onnx_path.split("/")
        for token in tokens:
            if 'G1' in token:
                return token
        return ''
    policy_vel_cfg = PolicyCfg(
        onnx_path=onnx_path,
        infer_dt=0.02,
        action_scale=0.5,
        action_dim=12,
        obs_limits=(-100., 100.),
        log_dir="action_log",
        tag=f"real-{extract_g1_token(onnx_path)}",
    )

    policy_pf_cfg = PolicyCfg(
        onnx_path=onnx_path_pf,
        infer_dt=0.02,
        action_scale=0.5,
        action_dim=12,
        obs_limits=(-100., 100.),
        log_dir="action_log",
        tag=f"real-{extract_g1_token(onnx_path_pf)}",
    )

    policy_vel = OnnxPolicy_velocity_cmd(policy_vel_cfg) # cmd (vx, vy, wyaw) manual
    policy_pf = OnnxPolicy_pf_cmd(policy_pf_cfg) # cmd (vx, vy, wyaw) from pf, no manual
    
    # Initialize DDS communication
    ChannelFactoryInitialize(0)

    rclpy.init()
    controller = LowLevelController(
        policy_vel,
        policy_pf,
        default_qpos=policy_vel.default_qpos,
        policy_init_qpos=policy_vel.default_qpos,
        kps=consts.KPs,
        kds=consts.KDs,
        ros_node=None,
        is_debug=True,
        pf_path=pf_path,
        xml_path=xml_path,)
    wait_for_shm(timeout=10.0, interval=0.05)

    # Enter the zero torque state
    controller.damping_state()

    terminal_sh_tmux()

    # Press the start key to move to the default position
    controller.move_to_default_pos()
    controller.default_pos_state()

    # Press the A key to start the policy control loop
    controller.Start()
    while True:
        if controller.remote_controller.button[KeyMap.X]: # NOTE refresh map
            controller.odo_pose_list=[]
            controller.odo_ori_list=[]
            controller.qpos_list=[]
            terminal_sh_tmux()
            time.sleep(0.5)
        try:
            time.sleep(0.02)
        except KeyboardInterrupt:
            break
        if controller.remote_controller.button[KeyMap.select] == 1:
            break
    print('Manual interrupt received, exit.')
    controller.save_pose_ori()
    policy_vel.end() # save record
    policy_pf.end()

    # Enter the damping(debug) state for safety kill
    controller.is_debug=True
    close_all_tmux()
    SHM.close()
    SHM.unlink()
    time.sleep(1)
    print("Exit")
    

if __name__ == "__main__":
    
    main()


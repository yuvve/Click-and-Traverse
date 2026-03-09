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

from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Pose_, Point_, Quaternion_


from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_ as Odometry

from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_ as PointCloudState

from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from gx_loco_deploy.policies.g1_cat import constants as consts

# from scripts.exp_dis_pf.g1_onnx_policy import OnnxPolicy
import math
import mujoco
# from scripts.exp_dis_pf.pf import PotentialField

import rclpy.time
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as R

def pointcloud2_to_xyz(msg):
    """将 Unitree SDK 的 PointCloud2_ 转为 (N,3) numpy 数组"""
    num_points = int(msg.width) * int(msg.height)
    if num_points == 0 or msg.data is None or len(msg.data) == 0:
        return np.empty((0, 3), dtype=np.float32)

    # ✅ 有的 SDK 版本 msg.data 是 list[int]，要转成 bytes
    if isinstance(msg.data, list):
        msg_data = bytes(msg.data)
    else:
        msg_data = msg.data

    step_f32 = (msg.point_step // 4) if getattr(msg, "point_step", 12) else 3
    points = np.frombuffer(msg_data, dtype=np.float32).reshape((num_points, step_f32))
    xyz = points[:, :3].astype(np.float32)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    return xyz


def xyz_to_pointcloud2(points_xyz, frame_id="", stamp=None):
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_ as PointCloudState
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointField_ as PointField
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_

    points_xyz = np.asarray(points_xyz, dtype=np.float32)

    # ✅ 构造 Header（必须带两个参数）
    if stamp is None:
        # 构造一个空的时间戳对象（Unitree SDK 通常是结构体类型）
        from unitree_sdk2py.idl.builtin_interfaces.msg.dds_ import Time_
        stamp = Time_(sec=0, nanosec=0)

    header = Header_(stamp, frame_id)

    # ✅ 定义点字段 (x, y, z)
    fields = [
        PointField(name="x", offset=0, datatype=7, count=1),
        PointField(name="y", offset=4, datatype=7, count=1),
        PointField(name="z", offset=8, datatype=7, count=1),
    ]

    # ✅ 点云基本参数
    height = 1
    width = points_xyz.shape[0]
    is_bigendian = False
    point_step = 12  # 每个点 3×float32
    row_step = point_step * width
    is_dense = True

    # ✅ data 必须是 list[int]
    data_bytes = points_xyz.tobytes()
    data = list(data_bytes)

    # ✅ 构造消息（必须显式传所有参数）
    msg = PointCloudState(
        header,
        height,
        width,
        fields,
        is_bigendian,
        point_step,
        row_step,
        data,
        is_dense,
    )

    return msg

def quat_wxyz_to_R(q):
    """
    使用 SciPy 将四元数 [w, x, y, z] 转换为旋转矩阵 (3x3)
    """
    # SciPy 默认要求四元数顺序为 [x, y, z, w]
    q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
    rot = R.from_quat(q_xyzw)
    return rot.as_matrix() 


def R_to_rpy(R_mat):
    """
    使用 SciPy 将旋转矩阵转为欧拉角 (roll, pitch, yaw)
    采用 ZYX 顺序（即 yaw→pitch→roll）
    返回值单位：弧度
    """
    rot = R.from_matrix(R_mat)
    roll, pitch, yaw = rot.as_euler('ZYX', degrees=False)[::-1]  # reverse to roll, pitch, yaw
    return roll, pitch, yaw

def R_from_rpy(roll, pitch, yaw):
    """
    使用 SciPy 从欧拉角 (roll, pitch, yaw) 生成旋转矩阵。
    顺序为 ZYX (即 yaw→pitch→roll)，与 ROS2/REP103 一致。
    返回值：3×3 numpy 数组
    """
    # 注意 SciPy 的 as_euler('ZYX') → 返回 [yaw, pitch, roll]
    rot = R.from_euler('ZYX', [yaw, pitch, roll], degrees=False)
    return rot.as_matrix()

def transform_points_with_t0(t0: TransformStamped.transform, pts_ci):
    """使用 t0 将点云从 camera_init 转到 floor_init"""
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    # 1️⃣ 提取 translation 和 rotation
    t = np.array([
        t0.translation.x,
        t0.translation.y,
        t0.translation.z,
    ], dtype=float)

    q = np.array([
        t0.rotation.x,
        t0.rotation.y,
        t0.rotation.z,
        t0.rotation.w,
    ], dtype=float)

    # 2️⃣ 得到旋转矩阵 (camera_init → floor_init)
    R_cf = R.from_quat(q).as_matrix()

    # 3️⃣ 求逆：floor_init ← camera_init
    R_fc = R_cf.T
    t_fc = -R_fc @ t

    # 4️⃣ 应用到点云
    pts_fi = (R_fc @ pts_ci.T).T + t_fc
    return pts_fi

def transform_points_with_t0_to_camera(t0: TransformStamped.transform, pts_ci):
    """使用 t0 将点云从 camera_init 转到 floor_init"""
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    # 1️⃣ 提取 translation 和 rotation
    t = np.array([
        t0.translation.x,
        t0.translation.y,
        t0.translation.z,
    ], dtype=float)

    q = np.array([
        t0.rotation.x,
        t0.rotation.y,
        t0.rotation.z,
        t0.rotation.w,
    ], dtype=float)

    # 2️⃣ 得到旋转矩阵 (camera_init → floor_init)
    R_cf = R.from_quat(q).as_matrix()

    # 4️⃣ 应用到点云
    pts_fi = (R_cf @ pts_ci.T).T + t
    return pts_fi

def coordinate_from_plane(a, b, c, d):
    """
    根据平面方程 ax + by + cz + d = 0 构造一个局部坐标系。
    
    输出：4x4 齐次矩阵 T_plane
    其定义为：
        - 原点：平面上的一点（距原点最近点）
        - z 轴：平面法向量方向（单位化）
        - x/y 轴：在平面内任意正交基（右手系）
    """
    # 法向量
    n = np.array([a, b, c], dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        raise ValueError("法向量长度过小，可能是退化平面")
    n /= n_norm

    # 平面上最近原点的一点
    p0 = -d * n  # 满足 a*x+b*y+c*z+d=0

    # 构造平面内的 x/y 轴
    # 随机选一个与法向量不共线的参考向量
    ref = np.array([0.0, 1.0, 0.0])
    # if abs(np.dot(ref, n)) > 0.9:
    #     ref = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(ref, n)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(n, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # 构造旋转矩阵（列为坐标轴）
    R = np.stack([x_axis, y_axis, n], axis=1)

    # 构造齐次变换矩阵
    # T = np.eye(4)
    # T[:3, :3] = 
    # T[:3, 3] = p0

    return R, p0
def transform_pose_with_t0(t0: TransformStamped.transform, pose_ci,R_rp):
    """使用 t0 将位姿从 camera_init 转到 floor_init。
    假设 t0 表示 camera_init -> floor_init。

    Args:
        pose_ci: geometry_msgs.msg.Pose 或 unitree_sdk2py.idl.geometry_msgs.msg.dds_.Pose_
            在 camera_init 下的位姿。
    Returns:
        pose_fi: geometry_msgs.msg.Pose 或等价结构
            在 floor_init 下的位姿。
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    # 1️⃣ 提取 translation 和 rotation（来自 t0）
    t = np.array([
        t0.translation.x,
        t0.translation.y,
        t0.translation.z,
    ], dtype=float)

    q = np.array([
        t0.rotation.x,
        t0.rotation.y,
        t0.rotation.z,
        t0.rotation.w,
    ], dtype=float)

    # 2️⃣ 得到旋转矩阵 (camera_init → floor_init)
    R_cf = R.from_quat(q).as_matrix()

    # 3️⃣ 求逆：floor_init ← camera_init
    R_fc = R_cf.T 
    R_c2f = R_fc
    t_fc = -R_fc @ t
    # 4️⃣ 提取输入 pose 的位置与姿态
    pos_ci = np.array([
        pose_ci.pose.position.x,
        pose_ci.pose.position.y,
        pose_ci.pose.position.z, # 0.47618
    ], dtype=float)

    quat_ci = np.array([
        pose_ci.pose.orientation.x,
        pose_ci.pose.orientation.y,
        pose_ci.pose.orientation.z,
        pose_ci.pose.orientation.w,
    ], dtype=float)

    # 5️⃣ 应用变换：camera_init → floor_init
    # print(R_from_rpy(math.pi, math.pi/180*2, 0.0).T @ np.array([0.0, 0.0, 0.47618]))
    # pos_fp = R_fc @ (pos_ci- R_from_rpy(math.pi, math.pi/180*2, 0.0).T @ np.array([0.0, 0.0, 0.49])) + t_fc
    R_c2l = R_rp @ R_from_rpy(math.pi, 0.0, 0.0) @ R_from_rpy(0.0, -math.pi/180*2, 0.0)
    # pos_fp = R_c2f @ (pos_ci- R_from_rpy(math.pi, math.pi/180*2, 0.0).T @ np.array([0.0, 0.0, 0.49])) + t_fc
    # R_f2c = R_from_rpy(0.0, -math.pi/180*2, 0.0).T @ R_from_rpy(math.pi, 0.0, 0.0).T @ R_rp.T @ R_t2f

    # R_t2f, p0 = coordinate_from_plane(self._axis_mean[0], self._axis_mean[1], self._axis_mean[2], self._axis_mean[3])
    # R_c2f = R_t2f.T @ R_rp @ R_from_rpy(math.pi, 0.0, 0.0) @ R_from_rpy(0.0, -math.pi/180*2, 0.0)
    # R_c2l = R_rp @ R_from_rpy(math.pi, 0.0, 0.0) @ R_from_rpy(0.0, -math.pi/180*2, 0.0)

    # print('----------')
    # print(t_fc)
    # print(R_fc @ pos_ci)
    # print(pos_fi)
    R_ci = R.from_quat(quat_ci).as_matrix()
    # print('----------')
    # print(quat_ci)
    # print(R.from_matrix(R_fc).as_quat())
    R_fp = R_c2f @ R_ci @ R_from_rpy(math.pi, math.pi/180*2, 0.0).T
    quat_fp = R.from_matrix(R_fp).as_quat()  # [x, y, z, w]


    pos_fp = R_c2f @ pos_ci + t_fc + R_c2f @ R_ci @ np.array([0.0, 0.0, 0.49])
    # print(quat_fp)
    # 6️⃣ 构造新的 Pose_ 消息
    pose_fp = Pose_(position=Point_(x=pos_fp[0], y=pos_fp[1], z=pos_fp[2]), orientation=Quaternion_(
        x=quat_fp[0], y=quat_fp[1], z=quat_fp[2], w=quat_fp[3]))

    return pose_fp

from rclpy.node import Node
class TFNode(Node):
    def __init__(self):
        super().__init__('floor_tf_node')

class NewOdomPublisher:
    def __init__(
            self,
            ros_node: TFNode,
            infer_dt=0.02,
            is_debug=True,
            xml_path=None,
    ) -> None:
        self.is_debug = is_debug
        self.infer_dt = infer_dt
        # self.joint2motor_ids = consts.JOINT2MOTOR_IDX_LEG + consts.JOINT2MOTOR_IDX_WAIST_ARM
        # print(f"self.joint2motor_ids: {self.joint2motor_ids}")

        self._eps = 0.2                # **带宽 epsilon，可调 0.01~0.05m
        self._ransac_dist = 0.04       # ***
        self._ransac_n = 3             # ***
        self._ransac_iters = 10000      # ***
        self._init_frames = 10          # 前 N 帧用于估计 floor 高度
        self._z_samples = []
        self._axis_samples = np.ndarray((self._init_frames, 4), dtype=float)  # plane coefficients samples
        self._floor_ready = False
        self._z_floor = 0.0             # floor_init 的 z=0 对应 gravity_init 的高度
        self.filtered_pub = None        # 如果你要发布，初始化 Publisher（Unitree Channel/ROS2二选一）

        self.ros_node = ros_node
        self.tf_broadcaster = StaticTransformBroadcaster(self.ros_node)
        self.tf_broadcaster_local_floor = TransformBroadcaster(self.ros_node)

        # modules
        # self.remote_controller = RemoteController()

        # buffer
        self.delay_buf = deque(maxlen=100)
        self._last_verbose_time = time.time()
        self._last_command = np.zeros(3)

        self._last_start = 0
        self.counter = 0

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_machine_ = 0
        
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        self.lowstate_subscriber = ChannelSubscriber(consts.TOPIC_LOWSTATE, LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        
        self.pointcloud_subscriber = ChannelSubscriber(consts.TOPIC_POINTCLOUD, PointCloudState)
        self.pointcloud_subscriber.Init(self.PointCloudHandler, 10)
        
        self.odometry_subscriber = ChannelSubscriber(consts.TOPIC_ODOM, Odometry)
        self.odometry_subscriber.Init(self.OdometryHandler, 10)

        self.newodometry_publisher = ChannelPublisher(consts.TOPIC_NEWODOM, Odometry)
        self.newodometry_publisher.Init()

        self.filtered_pub = ChannelPublisher(consts.TOPIC_PUBPOINTCLOUD, PointCloudState)
        self.filtered_pub.Init()
        self.pose_pos = None
        # wait for the subscriber to receive data
        self.wait_for_low_state()
        # Initialize the command msg
        
        # NOTE mj handler, pf handler
        # odo -> mj prop -> fk -> link pos -> scene -> sdf
        # octo/premap+fastlio -> scene -> sdf


    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
    def publish_camera_to_floor_tf(self):
        if self.tf_broadcaster is None:
            return  # 没有 Node 就不发布


        q_wxyz = np.array(self.low_state.imu_state.quaternion, dtype=np.float64)
        R_imu = quat_wxyz_to_R(q_wxyz)
        roll, pitch, yaw = R_to_rpy(R_imu)
        self.first_yaw = yaw
        R_rp = R_from_rpy(roll, pitch, 0)
         
        # R_cf = R_rp @ R_from_rpy(math.pi, 0.0, 0.0)
        # R_cf = R_rp @ R_from_rpy(math.pi, math.pi/180*3, 0.0)
        R_t2f, p0 = coordinate_from_plane(self._axis_mean[0], self._axis_mean[1], self._axis_mean[2], self._axis_mean[3])
        R_c2f = R_t2f.T @ R_rp @ R_from_rpy(math.pi, 0.0, 0.0) @ R_from_rpy(0.0, -math.pi/180*2, 0.0)
        R_f2c = R_from_rpy(0.0, -math.pi/180*2, 0.0).T @ R_from_rpy(math.pi, 0.0, 0.0).T @ R_rp.T @ R_t2f

        # R_cf = R_from_rpy(math.pi, math.pi/180*3, 0.0) @ R_rp
        # R_cf = R_from_rpy(math.pi/4, 0.0, 0.0) @ R_from_rpy( 0.0, math.pi/500, 0.0)
        # R_cf = R_from_rpy(0.0, -math.pi/6,  0.0) @ R_from_rpy(math.pi, 0.0, 0.0)
        rot = R.from_matrix(R_f2c)

        # SciPy 默认输出顺序 [x, y, z, w]
        qx, qy, qz, qw = rot.as_quat()

        t = TransformStamped()
        t.header.stamp = self.ros_node.get_clock().now().to_msg()
        t.header.frame_id = "camera_init"
        t.child_frame_id = "floor_init"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = R_f2c @ np.array([0.0, 0.0, self._z_floor])
        # print(t.transform.translation)
        
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.t0 = t.transform
        self.tf_broadcaster.sendTransform(t)

    def publish_local_floor_tf(self):
        if self.tf_broadcaster_local_floor is None:
            return  # 没有 Node 就不发布
        if self.pose_pos is None:
            return
        rot = R.from_matrix(R_from_rpy(0.0, 0, 0.0))

        # SciPy 默认输出顺序 [x, y, z, w]
        qx, qy, qz, qw = rot.as_quat()

        t = TransformStamped()
        t.header.stamp = self.ros_node.get_clock().now().to_msg()
        t.header.frame_id = "floor_init"
        t.child_frame_id = "floor_local"
        t.transform.translation.x = self.pose.position.x * 1.0
        t.transform.translation.y = self.pose.position.y * 1.0
        t.transform.translation.z = 0.0
        # t.transform.translation.x =  self.pose_pos[0]
        # t.transform.translation.y =  self.pose_pos[1]
        # t.transform.translation.z = 0.0
        # print(t.transform.translation.z)
        # print(t.transform.translation)
        # t.transform.translation.x +=1
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster_local_floor.sendTransform(t)



    def OdometryHandler(self, msg: Odometry):
        pose = msg.pose
        # print(msg)
        if not self._floor_ready:
            return
        

        q_wxyz = np.array(self.low_state.imu_state.quaternion, dtype=np.float64)
        R_imu = quat_wxyz_to_R(q_wxyz)
        roll, pitch, yaw = R_to_rpy(R_imu)
        R_rp = R_from_rpy(roll, pitch, yaw - self.first_yaw)
        rot = R.from_matrix(R_rp)
        qx, qy, qz, qw = rot.as_quat()
        self.pose = transform_pose_with_t0(self.t0, pose, R_rp)

        self.pose_pos = np.array([
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z, # 0.47618
        ], dtype=float)

        self.pose_R = R.from_quat(np.array([
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ], dtype=float)).as_matrix()
        # print('!!!!!!!!!!!')
        # print(pose.pose.position.y)
        # print(self.pose_pos)
        # print(self.pose_R)
        # q_wxyz2 = np.array([self.pose.orientation.w,self.pose.orientation.x,self.pose.orientation.y,self.pose.orientation.z], dtype=np.float64)
        
        # print(f"imu:{roll, pitch, yaw- self.first_yaw}")
        # print(f"fastlio:{R_to_rpy(quat_wxyz_to_R(q_wxyz2))}")
        # self.pose.orientation = Quaternion_(x=qx, y=qy, z=qz, w=qw)
        # print(self.pose)
        odom_fi = msg
        odom_fi.header = msg.header
        odom_fi.header.frame_id = "floor_init"
        odom_fi.child_frame_id = "body"

        odom_fi.pose.pose = self.pose
        odom_fi.pose.covariance = msg.pose.covariance  # 保留协方差
        odom_fi.twist = msg.twist  # 保留速度信息（不改变参考系方向）
        self.newodometry_publisher.Write(odom_fi)


    def PointCloudHandler(self, msg: PointCloudState):
        # 0) 点云 → numpy
        # print("receive pointcloud")
        pts_ci = pointcloud2_to_xyz(msg)  # camera_init 下的点云（fastlio 输出）
        if pts_ci.size == 0:
            return

        if not self._floor_ready:
            # 1) IMU 姿态：从四元数取 roll/pitch，去掉 yaw，构造只含 roll/pitch 的旋转
            #   假设 imu_state.quaternion 是 [w,x,y,z]，且其姿态是相对 camera_init（或已对齐）
            q_wxyz = np.array(self.low_state.imu_state.quaternion, dtype=np.float64)
            R_imu = quat_wxyz_to_R(q_wxyz)          # camera_init -> body(imu) 的旋转（或等价）
            roll, pitch, yaw = R_to_rpy(R_imu)
            R_rp = R_from_rpy(roll, pitch, 0.0)     # 去掉 yaw，仅 roll/pitch
            # 将点从 camera_init 旋转到 gravity_init：去除水平旋转，使 z 与重力对齐
            # 这里采用 R_rp 作为“把点转到只有俯仰/横滚的重力系”的旋转，使用其转置作为主动旋转
            # R_cf = R_from_rpy(math.pi, 0.0, 0.0) @  R_rp
            # R_cf = R_rp @ R_from_rpy(math.pi, math.pi/180*3, 0.0)
            
            R_f2c = R_from_rpy(0.0, -math.pi/180*2, 0.0).T @ R_from_rpy(math.pi, 0.0, 0.0).T @ R_rp.T
            pts_gi = (R_f2c.T @ pts_ci.T).T          # (N,3) pc->pf
            z = pts_gi[:, 2]
            # pts_fi = self.transform_points_with_t0(pts_ci)
        else:
            # pts_fi = (R_cf.T @ pts_ci.T).T          # (N,3)
            pts_fi = transform_points_with_t0(self.t0, pts_ci)
            # 2) 取最低 10% 的点，计算平均 z0
            z = pts_fi[:, 2]
        
        if z.size < 5:
            return
        z_th = np.percentile(z, 15.0)   # **
        low_mask = z <= z_th
        if not np.any(low_mask):
            return
        z0 = z[low_mask].mean()

        # 3) 带宽筛选 [z0-eps, z0+eps]，获取候选地面点
        band_mask = ((z0 - self._eps) <= z) & (z <= (z0 + self._eps))
        if not self._floor_ready:
            band_pts = pts_gi[band_mask]
        else:
            band_pts = pts_fi[band_mask]
        if band_pts.shape[0] < 10:  # 太少不做 RANSAC
            return

        # 4) Open3D RANSAC 拟合平面，得到地面高度 z（注意法向朝上）
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(band_pts)
        try:
            plane_model, inliers = cloud.segment_plane(
                distance_threshold=self._ransac_dist,
                ransac_n=self._ransac_n,
                num_iterations=self._ransac_iters
            )
        except RuntimeError:
            return

        if len(inliers) == 0:
            return

        a, b, c, d = plane_model
        # 让法向朝上（c >= 0），便于取高度
        if c < 0:
            a, b, c, d = -a, -b, -c, -d
        if abs(c) < 1e-6:
            return  # 退化

        # 平面高度（在 gravity_init 下，z = -d/c）
        z_plane = -d / c
        # print(z_plane)

        # 5) 前 N 帧平均，得到 floor_init 的 z̄
        if not self._floor_ready:
            self._z_samples.append(z_plane)
            self._axis_samples[len(self._z_samples)-1, :] = np.array([a, b, c, d], dtype=float)
            if len(self._z_samples) >= self._init_frames:
                self._axis_mean = np.mean(self._axis_samples, axis=0)
                self._z_floor = float(np.mean(self._z_samples))
                self._floor_ready = True
                print("publish floor tf!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                self.publish_camera_to_floor_tf()
                # 这里你可以发布 TF：gravity_init -> floor_init（平移 (0,0,-z̄)）
                # 或在 Unitree 的坐标系统中缓存这一步变换

            # 初始化阶段不发布点云，直接返回
            return
        t = TransformStamped()
        t.header.stamp = self.ros_node.get_clock().now().to_msg()
        # print(t.header.stamp)
        t.header.frame_id = "camera_init"
        t.child_frame_id = "floor_init"
        t.transform = self.t0
        self.tf_broadcaster.sendTransform(t)

        band_inlier_idx = np.zeros(band_pts.shape[0], dtype=bool)
        band_inlier_idx[np.asarray(inliers, dtype=int)] = True
        ground_inliers_gi = band_pts[band_inlier_idx]

        # ② 从 RANSAC 平面系数中提取法向量和平面方程

        normal = np.array([a, b, c])
        norm = np.linalg.norm(normal)
        normal /= norm
        d /= norm
        if c < 0:
            normal = -normal
            d = -d
        # ③ 对所有点（包括带宽外）计算到该平面的距离
        #    dist = |a x + b y + c z + d| / sqrt(a^2 + b^2 + c^2)
        all_dists = pts_fi @ normal + d

        # ④ 根据距离阈值重新分类为地面点/非地面点
        #    扩展平面范围，允许带宽外的点也被吸纳为地面
        dist_threshold = 0.10  # ≈ 5 cm，可根据 IMU 浮动调整
        ground_mask = all_dists < dist_threshold
        # pts_fi[:,2] < dist_threshold
        obstacles_fi = pts_fi[ground_mask]

        # ⑤ 非地面点为剩余部分
        obstacles_gi = pts_fi[~ground_mask]

        # ⑥ gravity_init -> floor_init：仅 z 平移，使地面高度为 0
        obstacles_gi = obstacles_gi.copy()

        obstacles_gis = transform_points_with_t0_to_camera(self.t0, obstacles_gi)
        obstacles_fi = transform_points_with_t0_to_camera(self.t0, obstacles_fi)
        band_pts = transform_points_with_t0_to_camera(self.t0, band_pts)
        pts_fi = transform_points_with_t0_to_camera(self.t0, pts_ci)

        # pts_fi = (R_cf @ pts_ci.T).T + t

        # R_fc = R_cf.T
        # t_fc = -R_fc @ t
        obstacles_gi_local = (self.pose_R.T @ obstacles_gis.T).T - self.pose_R.T @ self.pose_pos
        pts_local = (self.pose_R @ pts_ci.T).T + self.pose_pos
        # 8) 发布（或缓存）在 floor_init 下的“非地面点云”
        out_msg = xyz_to_pointcloud2(
            # obstacles_gis,
            # band_pts,
            # obstacles_fi,
            # pts_ci,
            obstacles_gi_local,
            # pts_local,
            frame_id="body",
            # frame_id="camera_init",
            stamp=msg.header.stamp
        )
        # 你可以用 Unitree Channel 发布；若未初始化，则缓存
        if self.filtered_pub is not None:
            self.filtered_pub.Write(out_msg)
        self.publish_local_floor_tf()
        self.filtered_pointcloud = out_msg

    # def LowStateGoHandler(self, msg: LowStateGo):
    #     self.low_state = msg
    #     self.remote_controller.set(self.low_state.wireless_remote)


    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.infer_dt)
        print("Successfully connected to the robot.")



def main():
    from pathlib import Path
    xml_path = "/home/ubuntu/workspace/Click-and-Traverse/data/assets/unitree_g1/scene_mjx_feetonly_mesh.xml"


    # fmt: off

    # fmt: on


    # Initialize DDS communication
    ChannelFactoryInitialize(0)

    rclpy.init()
    tf_node = TFNode()
    controller = NewOdomPublisher(
        ros_node=tf_node,
        is_debug=True,
        xml_path=xml_path,
    )

    while True:
        time.sleep(0.02)
        # try:
        #     time.sleep(0.02)
        # except KeyboardInterrupt:
        #     break
        # if controller.remote_controller.button[KeyMap.select] == 1:
        #     break

if __name__ == "__main__":
    main()

"""
1. pub floor axis wrt camera_init
imu+pointcloud+RANSAC

2. get odom wrt floor


"""
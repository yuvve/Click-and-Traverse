
import time
import zlib
import skfmm
import base64
import trimesh
import octomap_py
import numpy as np
from skimage import measure
from rclpy.node import Node
from gx_loco_deploy.policies.g1_cat.pf import PFConfig, make_pf_for_octomap
from gx_loco_deploy.policies.g1_cat.octomap_idl import VoxelMap
from octomap_msgs.msg import Octomap
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_ as Odometry
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_ as PoseStamped
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize

from gx_loco_deploy.common.remote_controller import RemoteController, KeyMap
from gx_loco_deploy.policies.g1_cat import constants as consts
from multiprocessing import shared_memory

SHM_NAME = "pf_shm"
# SHAPE = (75, 50, 38, 3+3+1)  # gf (3), bf (3), sdf (1)
SHAPE = (128, 128, 35, 3+3+1)  # gf (3), bf (3), sdf (1)
DTYPE = np.float32
nbytes_pf = int(np.prod(SHAPE) * np.dtype(DTYPE).itemsize)
nbytes = int((np.prod(SHAPE)+2) * np.dtype(DTYPE).itemsize)
SHM = shared_memory.SharedMemory(name=SHM_NAME, create=False, size=nbytes)

def marching_cubes_mesh(occ, spacing):
    if occ.sum() == 0:
        # raise ValueError("occupancy 全为 0，无法抽取表面。")
        return trimesh.Trimesh()
    verts, faces, normals, _ = measure.marching_cubes(occ.astype(np.uint8), level=0.5, spacing=spacing)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    return mesh
from scipy.ndimage import binary_closing, binary_opening, binary_erosion
from scipy.ndimage import convolve
def closing_opening_padded(occ, iters=1, kernel=3):
    assert kernel in (3,5,7)
    # se = np.ones((3, 3, 3), dtype=bool)
    # occ_closed_opened = binary_opening(occ, structure=se, iterations=1)

    se = np.ones((5, 5, 5), dtype=bool)
    occ = binary_closing(occ, structure=se, iterations=2)

    kernel = np.ones((3, 3, 3))
    neighbor_count = convolve(occ.astype(np.int32), kernel, mode='constant', cval=0)
    occ = occ & (neighbor_count > 1)

    # se = np.ones((3, 3, 3), dtype=bool)
    # occ_closed_opened = binary_opening(occ_closed, structure=se, iterations=1)

    return occ

from sklearn.linear_model import RANSACRegressor

def remove_ground_ransac(voxel_np, voxel_size=0.04,
                         ground_center=64, ground_band=1,
                         threshold=0.06, max_iters=2000):
    """
    基于 z≈ground_center 的先验先切片，再用 RANSAC 去除地面。

    voxel_np: occupancy [128,128,128], 1=occupied
    voxel_size: voxel resolution (m)
    ground_center: 地面大约在这个 z index
    ground_band: 在 [ground_center ± ground_band] 内切片做 RANSAC
    threshold: RANSAC 内点距离阈值
    """

    # ------------------------------------------------------------
    # 1) 提取所有 occupancy 的 voxel 坐标
    # ------------------------------------------------------------
    xyz_idx = np.column_stack(np.where(voxel_np == 1))  # (N,3)
    if xyz_idx.shape[0] == 0:
        return voxel_np.copy()

    # ------------------------------------------------------------
    # 2) 根据地面先验切片（极大加速）
    # ------------------------------------------------------------
    z_low = ground_center - ground_band
    z_high = ground_center + ground_band

    z_mask = (xyz_idx[:, 2] >= z_low) & (xyz_idx[:, 2] <= z_high)
    ground_candidates_idx = xyz_idx[z_mask]

    # 如果候选点太少，不做RANSAC
    if ground_candidates_idx.shape[0] < 50:
        return voxel_np.copy()

    # 转为 metric 坐标
    ground_candidates = ground_candidates_idx * voxel_size

    # ------------------------------------------------------------
    # 3) RANSAC 拟合平面： z = ax + by + c
    # ------------------------------------------------------------
    X = ground_candidates[:, :2]
    y = ground_candidates[:, 2]

    ransac = RANSACRegressor(
        residual_threshold=threshold,
        max_trials=max_iters
    )
    ransac.fit(X, y)

    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_

    # 内点 mask (只用于 debug，不必使用)
    # inliers = ransac.inlier_mask_

    # ------------------------------------------------------------
    # 4) 全体点判断是否属于地面
    # ------------------------------------------------------------
    all_points_metric = xyz_idx * voxel_size

    # 平面模型:  z_pred = a x + b y + c
    z_pred = all_points_metric[:, 0] * a + all_points_metric[:, 1] * b + c

    # 距离
    dist = np.abs(all_points_metric[:, 2] - z_pred)

    # 过滤：保留距离 > 阈值 的点（非地面）
    non_ground_mask = dist > 2* threshold
    non_ground_idx = xyz_idx[non_ground_mask]

    # ------------------------------------------------------------
    # 5) 重新构建 voxel（无 for 循环）
    # ------------------------------------------------------------
    filtered = np.zeros_like(voxel_np, dtype=voxel_np.dtype)

    # 去掉越界（几乎不会发生）
    # valid = np.all((non_ground_idx >= 0) &
    #                (non_ground_idx < voxel_np.shape), axis=1)
    # non_ground_idx = non_ground_idx[valid]

    # vectorized 赋值
    filtered[
        non_ground_idx[:, 0],
        non_ground_idx[:, 1],
        non_ground_idx[:, 2]
    ] = 1

    return filtered


class ROS2BridgeNode(Node):
    def __init__(self):
        super().__init__('octomap_bridge_node')
        self.get_logger().info("Octomap Bridge Node has been started.")
        print(VoxelMap.__idl_typename__)
        self.pub1 = ChannelPublisher("rt/Voxelmap", VoxelMap)
        self.pub1.Init()

        self.goal_subscriber = ChannelSubscriber(consts.TOPIC_GOAL, PoseStamped)
        self.goal_subscriber.Init(self.GoalHandler, 10)
        self.odometry_subscriber = ChannelSubscriber(consts.TOPIC_NEWODOM, Odometry)
        self.odometry_subscriber.Init(self.OdometryHandler, 10)
        self.pf_cfg = PFConfig()
        # SHM = shared_memory.SharedMemory(name=SHM_NAME, create=False)
        self.pf_arr = np.ndarray(SHAPE, dtype=DTYPE, buffer=SHM.buf)
        self.Sxy = np.ndarray(
            (2,),                  # 形状是 (2,)
            dtype=DTYPE,
            buffer=SHM.buf,
            offset=nbytes_pf    # 从数组结束的位置开始
        )
        # SHM.close()
        self.resolution = 0.04
        # self.pos = np.array([0, 0, 0],dtype=np.int32)
        self.O = 256
        self.remote_controller = RemoteController()
        self.lowstate_subscriber = ChannelSubscriber(consts.TOPIC_LOWSTATE, LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
    
    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def GoalHandler(self, msg: PoseStamped):
        goal = msg.pose
        odo_B_x = self.pf_cfg.Sx * self.resolution
        odo_B_y = self.pf_cfg.Sy * self.resolution
        goal_x = goal.position.x - odo_B_x
        self.pf_cfg.goal_w[1] = goal.position.y - odo_B_y
        if goal.position.x < (odo_B_x - 2.5) or goal.position.x > (odo_B_x + 2.5):
            return
        if goal.position.y < (odo_B_y - 2.5) or goal.position.y > (odo_B_y + 2.5):
            return
        self.pf_cfg.goal_w[0] = goal.position.x - odo_B_x
        self.pf_cfg.goal_w[1] = goal.position.y - odo_B_y

    def OdometryHandler(self, msg: Odometry):
        self.pose = msg.pose
        if self.remote_controller.button[KeyMap.B]:
            self.pf_cfg.Sx = int((self.pose.pose.position.x)/self.resolution)
            self.pf_cfg.Sy = int((self.pose.pose.position.y)/self.resolution)
            # print(self.pf_cfg.Sx, self.pf_cfg.Sy, self.pf_cfg.Sz)
    
    def OctomapHandler(self, msg: Octomap):
        print("relative goal", self.pf_cfg.goal_w)
        print("Sx Sy", self.pf_cfg.Sx * self.resolution, self.pf_cfg.Sy * self.resolution)
        t0 = time.time()        
        bt_data = bytes([(x + 256) % 256 for x in msg.data])
        header = (
            "# Octomap OcTree binary file\n"
            f"id {msg.id}\n"
            f"size 1\n"
            f"res {msg.resolution}\n"
            "data\n"
        )
        full_bt = header.encode("utf-8") + bt_data
        
        self.obs_pf_save_path = "/home/ubuntu/workspace/Click-and-Traverse/data/assets/R2SObs/expname/"
        self.obs_pf_save = False

        voxel_np = octomap_py.bt_buffer_to_voxel(full_bt, msg.resolution, 512)
        assert self.resolution == msg.resolution

        voxel_np = voxel_np[
            self.O+self.pf_cfg.Sx-self.pf_cfg.Nx0:self.O+self.pf_cfg.Sx+self.pf_cfg.Nx1,
            self.O+self.pf_cfg.Sy-self.pf_cfg.Ny0:self.O+self.pf_cfg.Sy+self.pf_cfg.Ny1,
            self.O+self.pf_cfg.Sz-self.pf_cfg.Nz0:self.O+self.pf_cfg.Sz+self.pf_cfg.Nz1]
        assert voxel_np.shape[0] == (self.pf_cfg.Nx0 + self.pf_cfg.Nx1)
        assert voxel_np.shape[1] == (self.pf_cfg.Ny0 + self.pf_cfg.Ny1)
        assert voxel_np.shape[2] == (self.pf_cfg.Nz0 + self.pf_cfg.Nz1)

        # t0 = time.time()
        # voxel_np = remove_ground_ransac(voxel_np)
        t1 = time.time()
        voxel_np = closing_opening_padded(voxel_np, kernel=3)
        # t2 = time.time()
        z_ceil_thresh = self.pf_cfg.Nz0+20
        mask_ceil = voxel_np[:, :, z_ceil_thresh:] == 1
        filled = np.cumsum(mask_ceil[:, :, :], axis=2)[:, :, ::1] > 0
        voxel_np[:, :, z_ceil_thresh:] = filled.astype(np.uint8)
        
        # t3 = time.time()
        z_bar_thresh = self.pf_cfg.Nz0+5
        mask_bar = voxel_np[:, :, :z_bar_thresh] == 1
        filled = np.cumsum(mask_bar[:, :, ::-1], axis=2)[:, :, ::-1] > 0
        voxel_np[:, :, :z_bar_thresh] = filled.astype(np.uint8)
        t2 = time.time()        
        # voxel_np[:, :, :z_check_thresh] = 0
        # voxel_np_clip = voxel_np[
        #     self.pf_cfg.Sx:self.pf_cfg.Sx+self.pf_cfg.Nx, 
        #     self.pf_cfg.Sy:self.pf_cfg.Sy+self.pf_cfg.Ny, 
        #     self.pf_cfg.Sz:self.pf_cfg.Sz+self.pf_cfg.Nz]
        # self.pf_cfg.update()
        obs_mask = voxel_np > 0.5
        assert voxel_np.shape[0] == (self.pf_cfg.Nx0 + self.pf_cfg.Nx1), voxel_np.shape
        assert voxel_np.shape[1] == (self.pf_cfg.Ny0 + self.pf_cfg.Ny1), voxel_np.shape
        assert voxel_np.shape[2] == (self.pf_cfg.Nz0 + self.pf_cfg.Nz1), voxel_np.shape
        if not obs_mask.any():
            obs_mask[0,0,-1]=1
            obs_mask[0,-1,-1]=1
        sdf, bf, gf = make_pf_for_octomap(self.pf_cfg, obs_mask)
        t3 = time.time()
        
        self.pf_arr[:, :, :, 0:3] = gf
        self.pf_arr[:, :, :, 3:6] = bf
        self.pf_arr[:, :, :, 6:7] = sdf[..., np.newaxis]
        self.Sxy[0] = self.pf_cfg.Sx
        self.Sxy[1] = self.pf_cfg.Sy
        # print('111111111111111')
        # data_1 = base64.b64encode(zlib.compress(sdf.tobytes())).decode("ascii")
        # data_2 = base64.b64encode(zlib.compress(bf.tobytes())).decode("ascii")
        # msg = VoxelMap(length=sdf.shape[0], width=sdf.shape[1], height=sdf.shape[2], data_sdf=(data_1), data_bf=(data_2) , resolution=0.04)
        # data_1 = sdf.tobytes()
        # data_2 = bf.tobytes()
        # msg = VoxelMap(length=sdf.shape[0], width=sdf.shape[1], height=sdf.shape[2], data_sdf=str(data_1), data_bf=str(data_2) , resolution=0.04)
        # if self.pub1.Write(msg, 0.1):
        #     print("pub1:publish sdf success.")
        # else:
        #     print("pub1:Waiting for sub.")
        
        if self.obs_pf_save == True and time.time() - getattr(self, 'save_time', 0) > 3:
            self.save_time = time.time()
            # print('save')
        #     np.save(self.obs_pf_save_path + f"voxel_{time.time()}.npy", voxel_np)
            # np.save(self.obs_pf_save_path + f"sdf_{time.time()}.npy", sdf)
            # np.save(self.obs_pf_save_path + f"bf_{time.time()}.npy", bf)
            # np.save(self.obs_pf_save_path + f"gf_{time.time()}.npy", gf)
            np.save(self.obs_pf_save_path + f"voxel_{time.time()}.npy", voxel_np)
            # np.save(self.obs_pf_save_path + f"sdf.npy", sdf)
            # np.save(self.obs_pf_save_path + f"bf.npy", bf)
            # np.save(self.obs_pf_save_path + f"gf.npy", gf)
            mesh = marching_cubes_mesh(voxel_np, spacing=(0.04, 0.04, 0.04))
            mesh.export(self.obs_pf_save_path + f"voxel_mesh_{time.time()}.obj")
        #     # mesh.export(self.obs_pf_save_path + f"voxel_mesh.obj")
        self.pub1.Write(VoxelMap(), 0.1)
        # print("sdf calculation is ok.")
        t4 = time.time()        
        # print(f"gf calculation delay:{t4 - self.all_time}" )
        # print("t1-t0load:", t1 - t0)
        # print("t2-t1process:", t2 - t1)
        # print("t3-t2:pfcal", t3 - t2)
        # print("t4-t3:save", t4 - t3)
        # print("total:", t4 - t0)

        # except Exception as e:
        #     print(f"[LowLevelController] ⚠️ Octomap parsing failed: {e}")


if __name__ == '__main__':
    import rclpy
    ChannelFactoryInitialize(0)
    rclpy.init()
    node = ROS2BridgeNode()
    node.create_subscription(Octomap, "/octomap_binary", node.OctomapHandler, 10)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
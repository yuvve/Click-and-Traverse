import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
import skfmm
import numpy as np
from scipy.spatial.transform import Rotation as R
FEET_SITES = [
    "left_foot",
    "right_foot",
]

HAND_SITES = [
    "left_palm",
    "right_palm",
]

KNEE_SITES = [
    "left_knee",
    "right_knee",
]

SHOULDER_SITES = [
    "left_shoulder",
    "right_shoulder",
]

TSDF = 0.5

class PFConfig:
    voxel: float = 0.04         # voxel resolution

    Nx0 = 64                      
    Nx1 = 64
    Ny0 = 64
    Ny1 = 64
    Nz0 = 0
    Nz1 = 35
    Sx = 0                       # start point, updated by real-time localization
    Sy = 0
    Sz = 0

    Nx = Nx0 + Nx1
    Ny = Ny0 + Ny1
    Nz = Nz0 + Nz1
    Lx: float = Nx * voxel  
    Ly: float = Ny * voxel  
    Lz: float = Nz * voxel  
    v_max: float = 0.6          
    k_decay: float = 0.6       
    goal_seed_r: float = 0.12   

    start_w:  np.ndarray = np.array([0.0,  0.0, 0.75], dtype=np.float32)
    goal_w:   np.ndarray = np.array([2.0,  0.0, 0.75], dtype=np.float32)
    origin_w = np.array([- Nx0 * voxel, - Ny0 * voxel, - Nz0 * voxel], dtype=np.float32)

def make_axes(cfg: PFConfig):
    Nx = int(round(cfg.Lx / cfg.voxel))
    Ny = int(round(cfg.Ly / cfg.voxel))
    Nz = int(round(cfg.Lz / cfg.voxel))
    assert Nx == cfg.Nx and Ny == cfg.Ny and Nz == cfg.Nz, "网格尺寸与配置不符，请检查！"
    ox, oy, oz = cfg.origin_w
    xv = ox + (np.arange(Nx) + 0.5) * cfg.voxel
    yv = oy + (np.arange(Ny) + 0.5) * cfg.voxel
    zv = oz + (np.arange(Nz) + 0.5) * cfg.voxel
    return xv, yv, zv

def make_grid(cfg: PFConfig):
    xv, yv, zv = make_axes(cfg)
    X, Y, Z = np.meshgrid(xv, yv, zv, indexing='ij')
    return (xv, yv, zv), (X, Y, Z)


def make_sdf(obs_mask: np.ndarray, voxel: float) -> np.ndarray:
    phi_obs = np.ones(obs_mask.shape, dtype=float)
    phi_obs[obs_mask] = -1.0
    sdf = skfmm.distance(phi_obs, dx=voxel).astype(np.float32)  # 有符号距离(米)
    return sdf

def grad3(scalar_field: np.ndarray, voxel: float):
    dfx, dfy, dfz = np.gradient(scalar_field, voxel, voxel, voxel, edge_order=2)
    return np.stack([dfx, dfy, dfz], axis=-1).astype(np.float32)

def make_guidance_field_progressive(cfg, grids, obs_mask, goal_local, bf, sdf, r_proj=None):
    """
    cfg 需要: voxel, v_max, k_decay
    T        : np.ndarray, 距离势
    obs_mask : bool ndarray, True=障碍物内部
    bf       : (...,3) 法向场（建议为SDF外法向）
    sdf      : np.ndarray, 与障碍物的有符号距离(外正内负). 若None则用obs_mask构造
    r_proj   : 法向消去的影响半径（米）。若None默认 2*voxel ~ 3*voxel
    返回:
      T, gf  : 指导速度场（已归一化并做速度衰减）
    """
    # t0 = time.time()
    voxel = cfg.voxel
    eps = 1e-9
    if r_proj is None:
        r_proj = 5.0 * voxel  # 可按机器人尺寸调大/调小

    X, Y, Z = grids
    # 目标负区（小球）
    phi = np.ones(obs_mask.shape, dtype=float)
    goal_seed = ((X - goal_local[0])**2 + (Y - goal_local[1])**2 + (Z - goal_local[2])**2) <= cfg.goal_seed_r**2
    assert goal_seed.any(), goal_local
    phi[goal_seed] = -1.0
    phi = np.ma.MaskedArray(phi, mask=obs_mask)

    # Fast Marching: T（到φ=0的最短距离，障碍绕行）
    T_ma = skfmm.distance(phi, dx=cfg.voxel)
    # t1 = time.time()
    T_free_max = np.max(T_ma[~T_ma.mask]) if np.any(~T_ma.mask) else 0.0
    T = T_ma.filled(T_free_max).astype(np.float32)
    # 2) -∇T（不做归一化、不做速度缩放）
    dTx, dTy, dTz = np.gradient(T, voxel, voxel, voxel, edge_order=2)
    g = np.stack([-dTx, -dTy, -dTz], axis=-1).astype(np.float32)     # (...,3)
    # t2 = time.time()

    # 3) 单位化法向 b̂；零范数处跳过
    bnorm = np.linalg.norm(bf, axis=-1, keepdims=True)
    # bunit = np.zeros_like(bf, dtype=np.float32)
    # valid_b = bnorm[..., 0] > eps
    # bunit[valid_b] = bf[valid_b] / bnorm[valid_b]
    bunit = bf / (bnorm + eps)

    # 4) 去法向分量: g_perp = g - (g·b̂) b̂
    proj = np.sum(g * bunit, axis=-1, keepdims=True)
    # proj = np.clip(proj, -1.0, 0.0)
    g_perp = g - proj * bunit

    # 5) 距离权重 w(d): d>=0 取外侧距离；w=1(贴边) -> w=0(远离)
    # d_out = np.maximum(sdf, 0.0)
    # 使用smoothstep: s = clamp((d/r)^2 * (3-2*d/r), 0,1)
    t = np.clip(sdf / (r_proj + eps), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    # w = 2 * (1.0 - t)[..., None]  # (...,1) 0-2 0.5-1 1-0
    w = (1.0 - smooth)[..., None]  # (...,1)
    # w = 0

    # 6) 组合: 贴边更“切向”，远处还原原始导航
    g_mix = (1.0 - w) * g + w * g_perp
    # g_mix[g_mix[...,0]<0] *= -1

    # 7) 障碍物内部：用法向（通常用外法向把场推离障碍；若想指向内部可取 -bunit）
    # from scipy.ndimage import binary_dilation
    # obs_mask = binary_dilation(obs_mask, iterations=1)
    g_mix[obs_mask] = bunit[obs_mask]

    # 8) 统一归一化（最后一步）+ 速度标量
    mag = np.linalg.norm(g_mix, axis=-1, keepdims=True)
    dir_unit = np.zeros_like(g_mix, dtype=np.float32)
    nz = (mag[..., 0] > eps)
    dir_unit[nz] = g_mix[nz] / mag[nz]

    # 速度：远处 vmax，近 goal 衰减到 0
    Tpos = np.maximum(T, 0.0)
    # speed = cfg.v_max * (Tpos / (Tpos + cfg.k_decay))
    # speed = np.clip(speed, 0.0, cfg.v_max).astype(np.float32)[..., None]
    T_thresh = 0.3
    p = 3.0  # >1 越大下降越陡
    goal_dist = (((X - goal_local[0])**2 + (Y - goal_local[1])**2))**0.5

    speed = np.where(
        goal_dist > T_thresh,
        cfg.v_max,
        cfg.v_max * (goal_dist / T_thresh) ** p
    )
    speed = speed.astype(np.float32)[..., None]


    gf = (dir_unit * speed).astype(np.float32)

    # t3 = time.time()
    # print(f"[Timing] SDF: {t1 - t0:.3f} s, Grad: {t2 - t1:.3f} s, GF: {t3 - t2:.3f} s")
    return T, gf

# =========================
# 保存
# =========================
def save_all(cfg: PFConfig, sdf, bf, gf, obs_mask, meta_extra=None):
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "sdf.npy", sdf)
    np.save(outdir / "bf.npy",  bf)
    np.save(outdir / "gf.npy",  gf)
    np.save(outdir / "obs.npy", obs_mask.astype(np.uint8))
    meta = {
        "voxel": cfg.voxel,
        "origin": cfg.origin_w,
        "shape_xyz": np.array(sdf.shape, dtype=np.int32),
        "start_w": cfg.start_w,
        "goal_w": cfg.goal_w,
        "scene": cfg.scene
    }
    if meta_extra:
        meta.update(meta_extra)
    np.save(outdir / "meta.npy", meta)
    print(f"[OK] Saved to {outdir}")

# =========================
# 可视化（三视图 + 导航矢量）
# =========================
def visualize_all(xv, yv, zv, sdf, T, gf, obs_mask, start_l, goal_l, title_prefix=""):
    # z≈start_z 的顶视图 (xy)
    kz = int(np.argmin(np.abs(zv - start_l[2])))
    plt.figure(figsize=(7,5))
    im = plt.imshow(sdf[:, :, kz].T, origin='lower',
                    extent=[xv[0], xv[-1], yv[0], yv[-1]],
                    aspect='equal', cmap='coolwarm')
    plt.colorbar(im, label="SDF (m)")
    obs_xy = obs_mask[:, :, kz].T
    plt.contour(obs_xy, levels=[0.5], colors='k',
                extent=[xv[0], xv[-1], yv[0], yv[-1]])
    step = 3
    X2, Y2 = np.meshgrid(xv[::step], yv[::step], indexing='ij')
    U = gf[::step, ::step, kz, 0]; V = gf[::step, ::step, kz, 1]
    plt.quiver(X2, Y2, U, V, pivot='mid', scale=30, color='w')
    plt.scatter([start_l[0]],[start_l[1]], c='w', s=50, edgecolors='k', label='start')
    plt.scatter([goal_l[0]],[goal_l[1]], c='r', s=60, edgecolors='k', marker='*', label='goal')
    plt.title(f"{title_prefix} Top view (z≈{zv[kz]:.2f} m)")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.legend(); plt.tight_layout(); #plt.show()
    plt.savefig(f"{title_prefix}_top.png", dpi=300)
    plt.close()

    # y≈start_y 的侧视图 (xz)
    ky = int(np.argmin(np.abs(yv - start_l[1])))
    plt.figure(figsize=(7,5))
    im = plt.imshow(sdf[:, ky, :].T, origin='lower',
                    extent=[xv[0], xv[-1], zv[0], zv[-1]],
                    aspect='equal', cmap='coolwarm')
    plt.colorbar(im, label="SDF (m)")
    obs_xz = obs_mask[:, ky, :].T
    plt.contour(obs_xz, levels=[0.5], colors='k',
                extent=[xv[0], xv[-1], zv[0], zv[-1]])
    X2, Z2 = np.meshgrid(xv[::step], zv[::step], indexing='ij')
    U = gf[::step, ky, ::step, 0]; W = gf[::step, ky, ::step, 2]
    plt.quiver(X2, Z2, U, W, pivot='mid', scale=30, color='w')
    plt.scatter([start_l[0]],[start_l[2]], c='w', s=50, edgecolors='k')
    plt.scatter([goal_l[0]],[goal_l[2]], c='r', s=60, edgecolors='k', marker='*')
    plt.title(f"{title_prefix} Side view (y≈{yv[ky]:.2f} m)")
    plt.xlabel("x (m)"); plt.ylabel("z (m)")
    plt.tight_layout(); #plt.show()
    plt.savefig(f"{title_prefix}_side.png", dpi=300)
    plt.close()

    # x≈中线 的正视图 (yz)
    kx = int(np.argmin(np.abs(xv - 1.)))  # 取中线（可按需改）
    plt.figure(figsize=(7,5))
    im = plt.imshow(sdf[kx, :, :].T, origin='lower',
                    extent=[yv[0], yv[-1], zv[0], zv[-1]],
                    aspect='equal', cmap='coolwarm')
    plt.colorbar(im, label="SDF (m)")
    obs_yz = obs_mask[kx, :, :].T
    plt.contour(obs_yz, levels=[0.5], colors='k',
                extent=[yv[0], yv[-1], zv[0], zv[-1]])
    Y2, Z2 = np.meshgrid(yv[::step], zv[::step], indexing='ij')
    V = gf[kx, ::step, ::step, 1]; W = gf[kx, ::step, ::step, 2]
    plt.quiver(Y2, Z2, V, W, pivot='mid', scale=30, color='w')
    plt.scatter([start_l[1]],[start_l[2]], c='w', s=50, edgecolors='k')
    plt.scatter([goal_l[1]],[goal_l[2]], c='r', s=60, edgecolors='k', marker='*')
    plt.title(f"{title_prefix} Front view (x≈{xv[kx]:.2f} m)")
    plt.xlabel("y (m)"); plt.ylabel("z (m)")
    plt.tight_layout(); #plt.show()
    plt.savefig(f"{title_prefix}_front.png", dpi=300)
    plt.close()

import os
import numpy as np
import trimesh
from trimesh.creation import box

def make_pf_for_octomap(cfg: PFConfig, obs_mask: np.ndarray):
    # SDF 与导数
    sdf = make_sdf(obs_mask, cfg.voxel)
    bf  = grad3(sdf, cfg.voxel)

    # 网格
    (xv, yv, zv), (X, Y, Z) = make_grid(cfg)
    # Eikonal 导航场
    T, gf = make_guidance_field_progressive(cfg, (X, Y, Z), obs_mask, cfg.goal_w, bf, sdf)
    return sdf, bf, gf

def world_to_navi_vel(navi2world_pose: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """
    将速度从 world 坐标系转换到 navi 坐标系

    Args:
        navi2world_pose: (4,4) navi->world 齐次矩阵
        vel: (N,3) 速度向量

    Returns:
        (N,3) navi 坐标系下的速度
    """
    world2navi = np.linalg.inv(navi2world_pose)
    R = world2navi[:3, :3]
    return (R @ vel.T).T

def base2navi_transform(base2world: np.ndarray) -> np.ndarray:
    x_proj = base2world[:, 0]
    x_proj /= (np.linalg.norm(x_proj)+1e-10)
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_proj)
    y_axis /= (np.linalg.norm(y_axis)+1e-10)
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack((x_axis, y_axis, z_axis))

class PotentialField:
    def __init__(self, mj_model, pf_path=None):
        # pf_path = config.pf_config.path
        self.dx = 0.04
        self.cfg = PFConfig()
        # self.sdf = np.load(f"{pf_path}/sdf.npy")[...,None]    # (Nx,Ny,Nz)
        # self.bf  = np.load(f"{pf_path}/bf.npy")    # (Nx,Ny,Nz,3)
        if pf_path is not None:
            self.sdf = np.load(f"{pf_path}/sdf.npy")[...,None] #+1# (Nx,Ny,Nz) NOTE
            self.bf  = np.load(f"{pf_path}/bf.npy")    # (Nx,Ny,Nz,3)
            self.gf  = np.load(f"{pf_path}/gf.npy")    # (Nx,Ny,Nz,3)
            self.Nx, self.Ny, self.Nz = self.sdf.shape[:3]
        else:
            self.Nx, self.Ny, self.Nz = self.cfg.Nx, self.cfg.Ny, self.cfg.Nz
            self.sdf = None
            self.bf = None
            self.gf = None
        print(self.Nx, self.Ny, self.Nz)
        # print(self.gf)
        self._head_site_id = mj_model.site("head").id
        self._feet_site_id = np.array(
            [mj_model.site(name).id for name in FEET_SITES]
        )
        self._hands_site_id = np.array(
            [mj_model.site(name).id for name in HAND_SITES]
        )
        self._knees_site_id = np.array(
            [mj_model.site(name).id for name in KNEE_SITES]
        )
        self._shlds_site_id = np.array(
            [mj_model.site(name).id for name in SHOULDER_SITES]
        )
        self._torso_imu_site_id = mj_model.site("imu_in_torso").id
        self._pelvis_imu_site_id = mj_model.site("imu_in_pelvis").id
    
    def get_default_field_1204(self):
        headgf = np.zeros((1,3))
        headbf = np.zeros((1,3))
        headdf = np.array([[TSDF]])
        feetgf = np.zeros((2,3))
        feetbf = np.zeros((2,3))
        feetdf = np.array([[TSDF],[TSDF]])
        handsgf = np.zeros((2,3))
        handsbf = np.zeros((2,3))
        handsdf = np.array([[TSDF],[TSDF]])
        kneesgf = np.zeros((2,3))
        kneesbf = np.zeros((2,3))
        kneesdf = np.array([[TSDF],[TSDF]])
        shldgf = np.zeros((2,3))
        shldsbf = np.zeros((2,3))
        shldsdf = np.array([[TSDF],[TSDF]])
        pelvgf = np.zeros((1,3))
        pelvbf = np.zeros((1,3))
        pelvdf = np.array([[TSDF]])
        torsgf = np.zeros((1,3))
        torsbf = np.zeros((1,3))
        torsdf = np.array([[TSDF]])
        return np.hstack([
            headgf.reshape(-1), headbf.reshape(-1), headdf.reshape(-1),
            pelvgf.reshape(-1), pelvbf.reshape(-1), pelvdf.reshape(-1),
            torsgf.reshape(-1), torsbf.reshape(-1), torsdf.reshape(-1),
            feetgf.reshape(-1), feetbf.reshape(-1), feetdf.reshape(-1),
            handsgf.reshape(-1), handsbf.reshape(-1), handsdf.reshape(-1),
            kneesgf.reshape(-1), kneesbf.reshape(-1), kneesdf.reshape(-1),
            shldgf.reshape(-1), shldsbf.reshape(-1), shldsdf.reshape(-1)]), np.zeros((3,))
        

    def get_potential_field_1204(self, mj_data, move_flag=np.zeros(1)):
        if self.sdf is None or self.bf is None or self.gf is None:
            return self.get_default_field_1204()
        self.pf_origin = self.cfg.origin_w + np.array([self.cfg.Sx, self.cfg.Sy, self.cfg.Sz], dtype=np.float32 + 0.5) * self.cfg.voxel
        # print('pf_origin:', self.pf_origin, self.cfg.Sx, self.cfg.Sy)
        # print('root pos:', mj_data.site_xpos[self._pelvis_imu_site_id][:2])
        pelvis2world_rot = mj_data.site_xmat[self._pelvis_imu_site_id].reshape(
            3, 3
        )
        navi2world_rot = base2navi_transform(pelvis2world_rot)
        navi2world_pose = np.eye(4)
        navi2world_pose[:3, :3]=navi2world_rot
        navi2world_pose[:2, 3] = mj_data.site_xpos[self._pelvis_imu_site_id][:2]
        # print(navi2world_pose)
        navi2world_pose[2, 3] = 0.75
        head_pos = mj_data.site_xpos[self._head_site_id]
        pelv_pos = mj_data.site_xpos[self._pelvis_imu_site_id]
        tors_pos = mj_data.site_xpos[self._torso_imu_site_id]
        feet_pos = mj_data.site_xpos[self._feet_site_id]
        hands_pos = mj_data.site_xpos[self._hands_site_id]
        knees_pos = mj_data.site_xpos[self._knees_site_id]
        shlds_pos = mj_data.site_xpos[self._shlds_site_id]
        all_poses = np.concatenate([
            head_pos.reshape(1,-1),
            pelv_pos.reshape(1,-1),
            tors_pos.reshape(1,-1),
            feet_pos,
            hands_pos,
            knees_pos,
            shlds_pos
        ], axis=0)
        all_gf = self.sample_field(self.gf.copy(), all_poses)
        all_bf = self.sample_field(self.bf.copy(), all_poses)
        all_df = self.sample_field(self.sdf.copy(), all_poses)
        headgf, pelvgf, torsgf, feetgf, handsgf, kneesgf, shldsgf = np.split(all_gf, [1,2,3,5,7,9], axis=0)
        headbf, pelvbf, torsbf, feetbf, handsbf, kneesbf, shldsbf = np.split(all_bf, [1,2,3,5,7,9], axis=0)
        rtf = pelvgf.reshape(-1)
        command = self.compute_cmd_from_rtf(rtf, np.concatenate([headgf,feetgf,handsgf], axis=0), np.concatenate([headbf,feetbf,handsbf], axis=0))

        # print('command:', command)
        # print('all_gf:', all_gf)
        all_gf = world_to_navi_vel(navi2world_pose, all_gf)
        all_bf = world_to_navi_vel(navi2world_pose, all_bf)
        command = world_to_navi_vel(navi2world_pose, command.reshape(-1,3))
        all_gf = all_gf * (move_flag[None] > 0.5) / (np.linalg.norm(all_gf, axis=-1, keepdims=True) + 1e-9)
        all_bf = all_bf / (np.linalg.norm(all_bf, axis=-1, keepdims=True) + 1e-9)

        all_bf = all_bf * (all_df < TSDF)
        all_df = np.clip(all_df, -1.0, TSDF)
        headgf, pelvgf, torsgf, feetgf, handsgf, kneesgf, shldsgf = np.split(all_gf, [1,2,3,5,7,9], axis=0)
        headbf, pelvbf, torsbf, feetbf, handsbf, kneesbf, shldsbf = np.split(all_bf, [1,2,3,5,7,9], axis=0)
        headdf, pelvdf, torsdf, feetdf, handsdf, kneesdf, shldsdf = np.split(all_df, [1,2,3,5,7,9], axis=0)
        # print(self.Nx, self.Ny, self.Nz)
        # print(f'headpos: {head_pos}, feetpos: {feet_pos}, handpos: {hands_pos}, kneespos: {knees_pos}, shldspos: {shlds_pos}')

        # print(f'headgf: {headgf}, feetgf: {feetgf}, handsgf: {handsgf}, command: {command}')
        # print(f"headbf:{headbf}, feetbf:{feetbf}, handsbf:{handsbf}, kneesbf:{kneesbf}, shldsbf:{shldsbf}")
        # print(f"headdf:{headdf}, feetdf:{feetdf}, handsdf:{handsdf}, kneesdf:{kneesdf}, shldsdf:{shldsdf}")

        return np.hstack([
            headgf.reshape(-1), headbf.reshape(-1), headdf.reshape(-1),
            pelvgf.reshape(-1), pelvbf.reshape(-1), pelvdf.reshape(-1),
            torsgf.reshape(-1), torsbf.reshape(-1), torsdf.reshape(-1),
            feetgf.reshape(-1), feetbf.reshape(-1), feetdf.reshape(-1),
            handsgf.reshape(-1), handsbf.reshape(-1), handsdf.reshape(-1),
            kneesgf.reshape(-1), kneesbf.reshape(-1), kneesdf.reshape(-1),
            shldsgf.reshape(-1), shldsbf.reshape(-1), shldsdf.reshape(-1)]), command.reshape(-1)

    def world_to_grid(self, pos):
        """ 世界坐标 -> voxel index (浮点) """
        rel = pos - self.pf_origin
        idx = rel / self.dx
        return idx

    def sample_field(self, field, pos):
        """
        三线性插值 (完全向量化，无 if/for)
        field: (Nx, Ny, Nz, C)
        pos:   (N, 3)  世界坐标
        return: (N, C)
        """
        # ---- 连续索引 (N,3) ----
        idx = self.world_to_grid(pos)                         # (N,3)
        # print(idx)
        x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]            # (N,)
        # x[4] = 68.4604
        # y[4] = 58.6561
        # z[4] = 0.31921
        # ---- 边界裁剪到 [0, N-2]，保证八邻域可访问 ----
        x = np.clip(x, 0, self.Nx - 2)
        y = np.clip(y, 0, self.Ny - 2)
        z = np.clip(z, 0, self.Nz - 2)
        # print(x[4], y[4], z[4])

        # ---- 下取整与小数部分 ----
        xi = np.floor(x).astype(np.int32)                    # (N,)
        yi = np.floor(y).astype(np.int32)
        zi = np.floor(z).astype(np.int32)
        xd = x - xi                                          # (N,)
        yd = y - yi
        zd = z - zi

        # ---- 8 个 corner 的整数索引：广播构造 (N,8,3) ----
        offsets = np.array([
            [0,0,0],[1,0,0],[0,1,0],[1,1,0],
            [0,0,1],[1,0,1],[0,1,1],[1,1,1]
        ], dtype=np.int32)                                   # (8,3)

        base = np.stack([xi, yi, zi], axis=1)                # (N,3)
        corners = base[:, None, :] + offsets[None, :, :]     # (N,8,3)
        # print(corners[4])

        # ---- 按 8 个 corner 一次性 gather：得到 (N,8,C) ----
        vals = field[corners[..., 0], corners[..., 1], corners[..., 2], :]  # (N,8,C)
        # print(vals)
        # ---- 8 个权重：外积后 reshape 为 (N,8) ----
        wx = np.stack([1.0 - xd, xd], axis=1)                # (N,2)
        wy = np.stack([1.0 - yd, yd], axis=1)                # (N,2)
        wz = np.stack([1.0 - zd, zd], axis=1)                # (N,2)
        # print(field[0,0], vals[4])

        w = (wx[:, :, None, None] *
            wy[:, None, :, None] *
            wz[:, None, None, :]).reshape(-1, 8)            # (N,8)

        # ---- 加权求和：沿 8 个角点聚合到 (N,C) ----
        out = np.einsum('ne,nec->nc', w, vals)               # (N,C)
        # print(out[4])
        return out
    

    def compute_cmd_from_rtf(self, rtf, cgf, cbf):
        # rtf: (3,) 原始速度 (x,y,z)
        # cgf: (M,3) head query guidance field
        # cbf: (M,3) head query barrier directions

        v = rtf[:2]* 0.7  # 只取 xy 分量

        # 单位化 bf，避免除0
        bnorm = np.linalg.norm(cbf[:, :2], axis=-1, keepdims=True) + 1e-9
        b_hat = cbf[:, :2] / bnorm  # (M,2)

        # 下界 L = b^T cgf
        Ls = np.sum(b_hat * cgf[:, :2], axis=-1)  # (M,)

        # 当前 b^T v
        bv = np.sum(b_hat * v, axis=-1)           # (M,)

        # 投影修正量 Δv_i = ((L - b^T v)/||b||^2) b
        diff = (Ls - bv)[:, None] / (np.sum(b_hat * b_hat, axis=-1, keepdims=True) + 1e-9)
        delta = diff * b_hat  # (M,2)

        # 只在 L > b^T v 时生效
        mask = (Ls > bv)[:, None]  # (M,1)
        delta = np.where(mask, delta, 0.0)

        # 一次 sweep: 把所有修正量加起来
        v_new = v + np.mean(delta, axis=0)

        # 拼接成 command
        command = np.hstack([v_new[0], v_new[1], 0.0]) * 0.75

        # 小速度停止
        small_cond = np.linalg.norm(command) < 0.2
        command = np.where(small_cond, 0.0, command)
        return command
import time

import mujoco.viewer
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from gx_loco_deploy.types import PolicyCfg

from gx_loco_deploy.policies.g1_cat import constants as consts
from gx_loco_deploy.policies.g1_cat.g1_onnx_policy import OnnxPolicy_pf_cmd, Obs

import os

os.environ["MUJOCO_GL"] = "egl"
np.set_printoptions(precision=3)
from gx_loco_deploy.policies.g1_cat.pf import PotentialField

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_sensor_data(mj_model, mj_data, sensor_name: str) -> np.ndarray:
    """Gets sensor data given sensor name."""
    sensor_id = mj_model.sensor(sensor_name).id
    sensor_adr = mj_model.sensor_adr[sensor_id]
    sensor_dim = mj_model.sensor_dim[sensor_id]
    return mj_data.sensordata[sensor_adr: sensor_adr + sensor_dim]


if __name__ == "__main__":
    xml_path = "/home/ubuntu/workspace/Click-and-Traverse/data/assets/unitree_g1/scene_mjx_feetonly_mesh.xml"
    pf_path="/home/ubuntu/workspace/Click-and-Traverse/data/assets/R2SObs/door"
    
    onnx_path = "/home/ubuntu/workspace/Click-and-Traverse/data/logs/G1_mj_axis/11221955_G1LocoPFRz_v6highsmooth_vel/checkpoints/000403046400/policy.onnx"
    onnx_path = "/home/ubuntu/workspace/Click-and-Traverse/data/logs/G1_mj_axis/11221955_G1LocoPFR10_v6highsmooth_pf/checkpoints/000403046400/policy.onnx"

    counter = 0
    infer_dt = 0.02
    sim_dt = 0.005
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.timestep = sim_dt
    mj_data = mujoco.MjData(mj_model)
    pf = PotentialField(mj_model, pf_path)
    decimation = int(infer_dt / sim_dt)
    # lower_limit, upper_limit = mj_model.jnt_range[1:].T
    # fmt: off
    def extract_g1_token(onnx_path):
        tokens = onnx_path.split("/")
        for token in tokens:
            if 'G1' in token:
                return token
        return ''
    policy_cfg = PolicyCfg(
        onnx_path=onnx_path,
        infer_dt=infer_dt,
        action_scale=0.5,
        action_dim=12,
        obs_limits=(-100., 100.),
        log_dir="action_log",
        tag=f"sim-{extract_g1_token(onnx_path)}",
        is_debug=False,
    )
    # fmt: on
    policy = OnnxPolicy_pf_cmd(policy_cfg)
    target_dof_pos = policy._default_qpos.copy()

    mj_data.qpos[7:] = policy._default_qpos.copy()

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time():
            try:
                step_start = time.time()
                
                
                if counter % decimation == 0:
                    qpos = mj_data.qpos[7:]
                    qvel = mj_data.qvel[6:]
                    quat_pelvis = mj_data.qpos[3:7]
                    gyro_pelvis = mj_data.qvel[3:6]

                    obs_pf, command = pf.get_potential_field_1204(mj_data)

                    obs = Obs(
                        command=command.copy(),
                        root_gyro=gyro_pelvis.copy(),
                        root_quat=quat_pelvis.copy(),
                        qpos=qpos.copy(),
                        qvel=qvel.copy(),
                        obs_pf=obs_pf.copy(),
                        # torque=tau.copy(),
                        timestamp=time.perf_counter(),
                        # height=height,
                    )
                    act = policy.infer(obs)
                    target_dof_pos = act.motor_targets
                # print('qpos', mj_data.qpos[7:])
                tau = pd_control(
                    target_dof_pos,
                    mj_data.qpos[7:],
                    consts.KPs,
                    np.zeros_like(consts.KDs),
                    mj_data.qvel[6:],
                    consts.KDs,
                )
                # print('tau', tau)
                mj_data.ctrl[:] = tau
                mujoco.mj_step(mj_model, mj_data)
                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()
                counter += 1

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
            except KeyboardInterrupt:
                policy.end()
                break

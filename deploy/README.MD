# Deployment of Click-and-Traverse

## Unitree G1 environment configuration

Please refer to Unitree official website (e.g. [unitree_ros2](https://support.unitree.com/home/zh/G1_developer/ros2_communication_routine) and [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)) for more details.

### Conda Environment

```bash
conda create -n deploy_cat python=3.8
conda activate deploy_cat
```

```bash
# Clone the SDK repository
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
# Enter the repository directory
cd unitree_sdk2_python
# Install the SDK
pip install -e .
conda deactivate
```
**Note**: Deactivate conda env before installing ros2-related environment.

### ROS 2 Environment
Installation steps refer to [Unitree G1 ROS2 Communication](https://support.unitree.com/home/en/G1_developer/ros2_communication_routine)

Below is an example of a ROS 2 environment installation for Ubuntu 20.04 (CMake 3.23.3 and GCC 9.4.0), using ROS 2 Foxy as an example.

For ROS2 Foxy installation, please refer to:  
[https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html](https://docs.ros.org/en/fo)

**Note**: After installation, it is recommended to comment out `source /opt/ros/foxy/setup.bash` in `.bashrc` to prevent conflicts.


- Install and Compile unitree_ros2:
```bash
# Clone the repository
git clone https://github.com/unitreerobotics/unitree_ros2
# Install dependencies
pip install empy==3.3.4 catkin_pkg==1.0.0 lark==1.2.2
sudo apt install ros-foxy-rmw-cyclonedds-cpp
sudo apt install ros-foxy-rosidl-generator-dds-idl # Comment out "source /opt/ros/foxy/setup.bash" in ~/.bashrc
```

- Compile cyclonedds (ensure ROS 2 environment is not sourced)
Since the G1 uses CycloneDDS version 0.10.2, it is necessary to change the default DDS implementation in ROS2. For more details, refer to:  
[https://docs.ros.org/en/foxy/Concepts/About-Different-Middleware-Vendors.html](https://docs.ros.org/en/foxy/Concepts/About-Different-Middleware-Vendors.html)

Before compiling CycloneDDS, make sure no ROS2 environment variables are sourced when launching the terminal. Otherwise, CycloneDDS may fail to compile.  
You can check your `~/.bashrc` file for any lines such as `source /opt/ros/foxy/setup.bash` and comment them out or remove them if present.

- Compile:
```bash
cd ~/unitree_ros2/cyclonedds_ws/src
git clone https://github.com/ros2/rmw_cyclonedds -b foxy
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd ..
colcon build --packages-select cyclonedds
```
- Enable the ROS 2 environment and build the `unitree_ros2` project
```bash
source /opt/ros/foxy/setup.bash
colcon build
```

### Network Configuration and Verification

1. Connect the robot, and check your network interface name and IP using `ifconfig`, e.g., name `eno1`, IP `192.168.123.161`.

2. Configure your host network to be in the same subnet as the robot, e.g.:
   - IP: `192.168.123.222`
   - Netmask: `255.255.255.0`
   - Gateway: `192.168.123.1`

3. Modify the network interface configuration: In `~/unitree_ros2/setup.sh`, change `NetworkInterface name="eno1"` to match your actual interface name.

4. Switch to the ROS 2 environment:
```bash
source /opt/ros/foxy/setup.sh; source ~/unitree_ros2/setup.sh
```
5. Run `ros2 topic list` to verify the connection. If robot-related topics appear, the configuration is successful.


## SLAM-CAT
### install
```bash
cd Click-and-Traverse/deploy
source /opt/ros/foxy/setup.bash
cd Click-and-Traverse-SLAM/lidar_ws
colcon build --symlink-install
```

```bash
conda activate deploy_cat
cd ../octomap_py_pkg
pip install -e .
cd ..
```
### usage
Please refer to `scripts/exp_dis_pf/deploy_real_gf.py:84-131` for more details.

We modify the default config file as follows for the mid360 deployment (suitable for the G1 robot):

- `Click-and-Traverse-SLAM/lidar/src/FAST_LIO/config/mid360.yaml`

- `Click-and-Traverse-SLAM/lidar/src/octomap_mapping/octomap_server/launch/octomap_mapping.launch.xml`

The result is as follows, user can use **2D Goal Pose** to click the target position:
<p align="center">
  <img src="../assets/slam_vis.png" width="100%">
</p>

## Real world deploy
Start G1 robot and set to debug mod, then
```bash
cd Click-and-Traverse/deploy
source /opt/ros/foxy/setup.bash
source Click-and-Traverse-SLAM/lidar/install/setup.bash
conda activate deploy_cat
pip install scikit-fmm
python -m scripts.exp_dis_pf.deploy_real_gf
```

You can replay the whole body trajectory in mujoco simulator with:
```bash
TODO
```

### Important Notes
1. `is_debug` in `deploy_real_gf.py:202` is used to control whether to send torque commands to the robot. If it is set to `True`, the robot will behave dampingly (kp = 0 for position control).

2. Tmux is used to monitor the robot's status. Please refer to `scripts/exp_dis_pf/deploy_real_gf.py:84-131` for more details. You can manually check the tmux sessions for debugging.

3. Shared memory SHM is used to share potential field data between the robot and the SlAM. Please refer to `scripts/exp_dis_pf/deploy_real_gf.py:40-54` `gx_loco_policies/g1_cat/low_level_controller.py` and `scripts/exp_dis_pf/octomap_bridge.py` for more details.

4. `obs_pf_save` in `octomap_bridge.py:279` is used to control whether to save observation data. It's very useful for debugging and data analysis. It's also used to save point cloud data for real-to-sim construction.

## Real-to-sim construction (for sim-to-sim test and finetuning)
You can construct the real world environment via point cloud with `obs_pf_save` in `octomap_bridge.py:279`.

Then, you can construct the real-to-sim environment with
```bash
cd Click-and-Traverse/deploy
python scripts/exp_dis_pf/R2S_set_point.py
```
`R2S_set_point.py` will let you set start and goal positions respectively via clicking.

Edit `procedural_obstacle_generation/main.py` and call:

```python
generate_pf(scene_type, pc_path)
```
You can generate pf and mesh for sim-to-sim test and finetuning.

### sim-to-sim test

```bash
python -m scripts.exp_dis_pf.deploy_mj
```

or replace obs_path when mj_onnx_play
```bash
python -m cat_ppo.eval.mj_onnx_play --task G1Cat --exp_name exp_name --obs_path obs_path
```

### finetuning
Replace obs_path when train ppo:
```bash
python -m train_ppo --task {task} --restore_name {restore_name} --exp_name {exp_name}  --ground {ground} --lateral {lateral} --overhead {overhead} --term_collision_threshold {term_collision_threshold} --obs_path {obs_path}
```

You can set `restore_name` to `none` if you want to train from scratch.

### Important Notes
Scene occupancy grids from procedural_obstacle_generation and real2sim construction are different in size. Check `Lx` `Ly` `Lz` and `origin_w` in `procedural_obstacle_generation/pf_modular.py` and `gx_loco_deploy/policies/g1_cat/pf.py`. 
    
- For training, you need to adjust `pf_config.origin` in `cat_ppo/envs/g1/env_cat.py` accordingly.
    
- For inference, you need to manually set the scene in the xml file.
    ```xml
        <asset>
            <mesh name="scene_mesh" file="path-to/obs.obj"/>
        </asset>

        <worldbody>
            <body name="scene" pos="pf_config.origin">
            <geom type="mesh" mesh="scene_mesh" rgba="0.7 0.7 0.7 1" contype="1" conaffinity="1" material="mat_brick_cube"/>
            </body>
        </worldbody>
    ```
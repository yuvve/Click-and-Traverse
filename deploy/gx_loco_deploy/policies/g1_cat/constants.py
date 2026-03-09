import numpy as np

MSG_TYPE = "hg"  # "hg" or "go"
IMU_TYPE = "pelvis"  # "torso" or "pelvis"

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_ODOMMODESTATE = "rt/odommodestate"
TOPIC_POINTCLOUD = "rt/cloud_registered"
TOPIC_PUBPOINTCLOUD = "rt/filtered_point_cloud"
TOPIC_ODOM = "rt/Odometry"
TOPIC_NEWODOM = "rt/NewOdometry"
TOPIC_VOXEL = "rt/Voxelmap"
TOPIC_GOAL = "rt/goal_pose"

# fmt: off
JOINT2MOTOR_IDX_LEG = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
JOINT2MOTOR_IDX_WAIST_ARM = [
    12, 13, 14,
    15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28
]
JOINT2MOTOR_IDX_RIGHT_ARM = [
    22, 23, 24, 25, 26, 27, 28
]

TORQUE_LIMIT = np.array([
    88.,  139.,  88., 139.,  50.,  50.,
    88.,  139.,  88., 139.,  50.,  50.,
    88.,  50.,  50.,
    25.,  25.,  25.,  25.,  25.,   5.,   5.,
    25.,  25.,  25.,  25.,  25.,   5.,   5.,
])

DEFAULT_QPOS = np.float32([
    -0.1, 0, 0, 0.3, -0.2, 0,
    -0.1, 0, 0, 0.3, -0.2, 0,
    0, 0, 0,
    0.2, 0.3, 0, 1.28, 0, 0, 0,
    0.2, -0.3, 0, 1.28, 0, 0, 0,
])

# v1
KPs = np.float32([
    100, 100, 100, 200, 80, 20,
    100, 100, 100, 200, 80, 20,
    300, 300, 300,
    90, 60, 20, 60, 20, 20, 20,
    90, 60, 20, 60, 20, 20, 20,
])

KDs = np.float32([
    2, 2, 2, 4, 2, 1,
    2, 2, 2, 4, 2, 1,
    10, 10, 10,
    2, 2, 1, 1, 1, 1, 1,
    2, 2, 1, 1, 1, 1, 1,
])


ACTION_JOINT_NAMES = [
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist
    # "waist_yaw_joint",
    # "waist_roll_joint",
    # "waist_pitch_joint",
]

OBS_JOINT_NAMES = [
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

SOFT_LOWERS = np.float32([
    -2.3954375 , -0.4363325 , -2.61972   , -0.01309033, -0.83776325,
    -0.24871   , -2.3954375 , -0.4363325 , -2.61972   , -0.01309033,
    -0.83776325, -0.24871   , -2.4871    , -0.494     , -0.494     ,
    -2.94521   , -1.4922075 , -2.4871    , -0.96866   , -1.873609  ,
    -1.5337085 , -1.5337085 , -2.94521   , -2.1555075 , -2.4871    ,
    -0.96866   , -1.873609  , -1.5337085 , -1.5337085 
])

SOFT_UPPERS = np.float32([
    2.7445375 , 2.8798325 , 2.61972   , 2.80562333, 0.48869325,
    0.24871   , 2.7445375 , 2.8798325 , 2.61972   , 2.80562333,
    0.48869325, 0.24871   , 2.4871    , 0.494     , 0.494     ,
    2.52641   , 2.1555075 , 2.4871    , 2.01586   , 1.873609  ,
    1.5337085 , 1.5337085 , 2.52641   , 1.4922075 , 2.4871    ,
    2.01586   , 1.873609  , 1.5337085 , 1.5337085
])

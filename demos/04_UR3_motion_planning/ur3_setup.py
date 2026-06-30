"""
UR3 Robot Arm Setup
-------------------
- Instantiate the UR3 robot arm from URDF description.
- Define a robot arm motion trajectory, computed in joint space.
"""
import numpy as np

from python_robot.manipulator import URDFManipulator
from python_robot.base import Frame
from python_robot.motion import JointTrajectory, MultiPointMotionProfileType, IKMask, IKTarget


# Instantiate the UR3 robot arm model. Set global plot and animation options.
ur3 = URDFManipulator(
    file_path="ur_description/urdf/ur3_joint_limited_robot.urdf.xacro",
    plot_options={
        "extent": 0.6,
        "spacing": 0.1,
        "world_frame_scale": 0.4,
        "frame_scale": 0.075,
    },
    anim_options={
        "extent": 0.6,
        "spacing": 0.1,
        "world_frame_scale": 0.4,
        "frame_scale": 0.075,
    }
)


# Override the joint limits of the robot arm links. Allow full rotation of the
# revolute joints in both the positive and negative turning direction.
joint_limits = (-2 * np.pi, 2 * np.pi)
for link in ur3.links:
    link.q_lim = joint_limits


# Define an IKMask that allows arbitrary orientation of target frames along
# a trajectory. The IKMask is used by the IK-solver when computing the values of
# the joint coordinates that correspond with a given target frame in Cartesian
# space.
free_orient = IKMask(alpha=False, beta=False, gamma=False)


# Define the target frames that represent the motion trajectory points in 3D
# space. IKMask free_orient indicates to the IK-solver that the orientation of
# the frames doesn't matter.
targets = [
    IKTarget(
        Frame(origin=(0.2, 0.2, 0.0), rpy_angles=(0.0, 0.0, 0.0)),
        free_orient
    ),
    IKTarget(
        Frame(origin=(0.1, 0.1, 0.1), rpy_angles=(0.0, 0.0, 0.0)),
        free_orient
    ),
    IKTarget(
        Frame(origin=(0.2, 0.2, 0.2), rpy_angles=(0.0, 0.0, 0.0)),
        free_orient
    ),
    IKTarget(
        Frame(origin=(0.1, 0.1, 0.3), rpy_angles=(0.0, 0.0, 0.0)),
        free_orient
    ),
    IKTarget(
        Frame(origin=(0.2, 0.2, 0.4), rpy_angles=(0.0, 0.0, 0.0)),
        free_orient
    ),
    IKTarget(
        Frame(origin=(0.1, 0.1, 0.5), rpy_angles=(0.0, 0.0, 0.0)),
        free_orient
    ),
    IKTarget(
        Frame(origin=(0.2, 0.2, 0.6), rpy_angles=(0.0, 0.0, 0.0))
        # No IKMask specified: the default IKMask will be used. The default
        # IKMask requires that both position and orientation of the target frame
        # are reached.
    )
]

# Compute the motion of the robot arm joints in joint space. Use a linear path
# with parabolic blends between path points as indicated by
# ``mp_type=MultiPointMotionProfileType.LINEAR```. All parabolic blends are
# calculated with the same blend acceleration.
joint_trajectory = JointTrajectory.create(
    targets=targets,
    dt_segments=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    manipulator=ur3,
    num_t_samples=600,
    mp_type=MultiPointMotionProfileType.LINEAR,
    blend_accels=20.0
)

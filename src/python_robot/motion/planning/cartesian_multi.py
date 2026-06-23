"""
Multi-segment Cartesian straight-line motion planning.

This module converts target frames to six-dimensional pose vectors, constructs
a vector-valued linear path with parabolic blends, and samples the resulting
Cartesian trajectory.
"""

from typing import Sequence

from dataclasses import dataclass

import numpy as np

from ...base.types import NumpyArray
from ...base import Frame

from ..profiles_6D import MultiLinearVectorPath


__all__ = ["CartesianMultiStraightLineMotion"]


@dataclass
class CartesianMultiStraightLineMotion:
    """
    Class for finding a smooth Cartesian straight-line path through multiple
    3D poses of the manipulator's end-effector frame.

    The motion is planned in a six-dimensional Cartesian pose space. The first
    three components describe the position of the frame origin. The last three
    components describe the frame orientation as an angle-axis vector.

    The six-dimensional pose vector has the form

        [x, y, z, rx, ry, rz]

    where [rx, ry, rz] is an angle-axis vector. Its magnitude is the rotation
    angle and its direction is the rotation axis.

    The six pose components are interpolated with a vector-valued linear path
    with parabolic blends. All components share the same blend times. This is
    important for Cartesian straight-line motion, because the position
    components must remain synchronized during the linear pieces.

    Attributes
    ----------
    target_frames: Sequence[Frame]
        Sequence of target poses (frames) through which the Cartesian path is
        defined.
    dt_segments: Sequence[float]
        Sequence with the travel durations of each segment between two
        successive target poses.
    dt_blends: float | Sequence[float]
        Single blend time or sequence of blend times at the target poses.
    num_t_samples: int, default = 100
        Number of time samples used to generate the trajectory.
    """
    target_frames: Sequence[Frame]
    dt_segments: Sequence[float]
    dt_blends: float | Sequence[float]
    num_t_samples: int = 100

    def __post_init__(self) -> None:
        if len(self.target_frames) < 2:
            raise ValueError("At least two target frames are required.")

        if len(self.dt_segments) != len(self.target_frames) - 1:
            raise ValueError(
                f"Number of segment durations ({len(self.dt_segments)}) does "
                f"not match the number of segments "
                f"({len(self.target_frames) - 1})."
            )

        # Transform the target frames to "pose vectors" [x, y, z, rx, ry, rz],
        # where:
        # - [x, y, z] are the Cartesian components of the origin of a target
        #   frame, and
        # - [rx, ry, rz] are the Cartesian components of the angle-axis vector
        #   representation of a target frame's orientation.
        # When going from one target frame to the next, the change of frame
        # orientation must happen with the smallest rotation possible.
        self.pose_vectors = self._frames_to_pose_vectors(self.target_frames)

        # Once we have the target "pose vectors", we can use them to construct
        # the motion time-functions (position, velocity and acceleration) of the
        # "pose variables" x, y, z, rx, ry, and rz. This can be done in a
        # similar fashion as with joint-space motion. Each of the position
        # paths x(t), y(t), z(t), rx(t), ry(t), and rz(t) is composed of linear
        # sections and parabolic blends.
        # To ensure that the resultant motion between path points is a straight
        # line in 3D space, the blend time of the parabolic blends must be the
        # same for each of the "pose variables" x, y, z, rx, ry, and rz.
        self._motion_profile = MultiLinearVectorPath(
            pose_vectors=self.pose_vectors,
            dt_segments=self.dt_segments,
            dt_blends=self.dt_blends,
        )

        # Time-sampling of the motion profile
        res = self._time_sampling(self._motion_profile, self.num_t_samples)
        self._t_arr, self._p_arr, self._V_arr, self._A_arr = res

    @staticmethod
    def _choose_equivalent_rotvec(
        rotvec: NumpyArray,
        previous_rotvec: NumpyArray,
    ) -> NumpyArray:
        """
        Chooses the equivalent angle-axis vector closest to the previous one.

        The angle-axis representation is not unique. This method selects an
        equivalent representation that minimizes the Euclidean distance to the
        previous angle-axis vector. This avoids unnecessary large orientation
        changes between successive target frames.
        """
        theta = float(np.linalg.norm(rotvec))

        if np.isclose(theta, 0.0):
            return rotvec

        axis = rotvec / theta
        candidates = []

        for n in range(-2, 3):
            candidates.append((theta + 2.0 * np.pi * n) * axis)
            candidates.append((-theta + 2.0 * np.pi * n) * axis)

        distances = [
            np.linalg.norm(candidate - previous_rotvec)
            for candidate in candidates
        ]
        i_min = int(np.argmin(distances))

        return candidates[i_min]

    @classmethod
    def _frames_to_pose_vectors(
        cls,
        frames: Sequence[Frame],
    ) -> NumpyArray:
        """
        Converts a sequence of frames to six-dimensional Cartesian pose vectors.

        The angle-axis part of each pose vector is chosen so that the
        orientation change with respect to the previous target frame remains
        as small as possible.
        """
        pose_vectors = []

        for i, frame in enumerate(frames):
            pose_vector = frame.to_pose_vector()

            if i > 0:
                pose_vector[3:] = cls._choose_equivalent_rotvec(
                    pose_vector[3:],
                    pose_vectors[-1][3:]
                )

            pose_vectors.append(pose_vector)

        return np.array(pose_vectors)

    # @staticmethod
    # def _time_sampling(
    #     motion_profile: MultiLinearVectorPath,
    #     n_samples: int
    # ) -> tuple[NumpyArray, ...]:
    #     """
    #     Takes uniform distributed time samples of the Cartesian motion profile.
    #
    #     Returns
    #     -------
    #     t_arr: NumpyArray
    #         The time moments at which the values of the pose vectors are
    #         calculated.
    #     p_arr: NumpyArray
    #         The values of the pose vectors in the course of time. The
    #         number of rows of the array is equal to the number of time samples.
    #         The number of columns is equal to 6 (x, y, z, rx, ry, rz).
    #     pd_arr: NumpyArray
    #         The values of the pose vector velocities in the course of time. The
    #         number of rows of the array is equal to the number of time samples.
    #         The number of columns is equal to 6 (x_dot, y_dot, z_dot, rx_dot,
    #         ry_dot, rz_dot).
    #     pdd_arr: NumpyArray
    #         The values of the pose vector accelerations in the course of time.
    #         The number of rows of the array is equal to the number of time
    #         samples. The number of columns is equal to 6 (x_ddot, y_ddot, z_ddot,
    #         rx_ddot, ry_ddot, rz_ddot).
    #     """
    #     t_arr, p_arr = motion_profile.position_profile(n_samples)
    #     _, pd_arr = motion_profile.velocity_profile(n_samples)
    #     _, pdd_arr = motion_profile.acceleration_profile(n_samples)
    #     return t_arr, p_arr, pd_arr, pdd_arr

    @staticmethod
    def _time_sampling(
        motion_profile: MultiLinearVectorPath,
        n_samples: int,
    ) -> tuple[NumpyArray, ...]:
        t_arr, p_arr = motion_profile.position_profile(n_samples)
        _, V_arr = motion_profile.spatial_velocity_profile(n_samples)
        _, A_arr = motion_profile.acceleration_profile(n_samples)
        return t_arr, p_arr, V_arr, A_arr

    def trajectory(self) -> tuple[NumpyArray, list[Frame]]:
        """
        Returns the trajectory of the end-effector frame in Cartesian space.

        Returns
        -------
        t_arr: NumpyArray
            Numpy array with time moments.
        frames: list[Frame]
            List of frame poses along the trajectory at the time instants in
            `t_arr`.
        """
        t_arr, path = self._motion_profile.position_profile(self.num_t_samples)
        frames = [
            Frame.from_pose_vector(pose_vector)
            for pose_vector in path
        ]
        return t_arr, frames

    @property
    def motion_profile(self) -> MultiLinearVectorPath:
        """
        Returns the underlying motion profile of the end-effector frame in
        Cartesian space.
        """
        return self._motion_profile

    # @property
    # def motion_samples(self) -> tuple[NumpyArray, ...]:
    #     """
    #     Returns time samples of the end-effector frame motion in Cartesian
    #     space.
    #
    #     Returns
    #     -------
    #     t_arr: NumpyArray
    #         The time moments at which the values of the pose vectors are
    #         calculated.
    #     p_arr: NumpyArray
    #         The values of the pose vectors in the course of time. The
    #         number of rows of the array is equal to the number of time samples.
    #         The number of columns is equal to 6 (x, y, z, rx, ry, rz).
    #     pd_arr: NumpyArray
    #         The values of the pose vector velocities in the course of time. The
    #         number of rows of the array is equal to the number of time samples.
    #         The number of columns is equal to 6 (x_dot, y_dot, z_dot, rx_dot,
    #         ry_dot, rz_dot).
    #     pdd_arr: NumpyArray
    #         The values of the pose vector accelerations in the course of time.
    #         The number of rows of the array is equal to the number of time
    #         samples. The number of columns is equal to 6 (x_ddot, y_ddot, z_ddot,
    #         rx_ddot, ry_ddot, rz_ddot).
    #     """
    #     return self._t_arr, self._p_arr, self._pd_arr, self._pdd_arr

    @property
    def motion_samples(self) -> tuple[NumpyArray, ...]:
        """
        Return sampled Cartesian motion data.

        Returns
        -------
        tuple[NumpyArray, ...]
            Time samples, pose-vector samples, spatial velocity samples, and
            spatial acceleration samples.
        """
        return self._t_arr, self._p_arr, self._V_arr, self._A_arr

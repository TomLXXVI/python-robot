from typing import Sequence

from dataclasses import dataclass

import numpy as np
from spatialmath import SO3, SE3

from ...base.types import NumpyArray
from ...base import Frame

from ..profiles_6D import MultiLinearSegmentVectorPath


__all__ = ["CartesianMultiStraightLineMotion"]


@dataclass
class CartesianMultiStraightLineMotion:
    """
    Class for finding a smooth Cartesian straight-line path through multiple
    3D poses of a frame.

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
        Sequence of target frames through which the Cartesian path is defined.
    dt_segments: Sequence[float]
        Sequence with the travel durations of each segment between two
        successive target frames.
    dt_blends: float | Sequence[float]
        Single blend time or sequence of blend times at the target frames.
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

        # Transform target frames to target "pose vectors" [x, y, z, rx, ry, rz].
        # When going from one target frame to the next, the change of
        # frame orientation must happen with the smallest rotation possible.
        self.pose_vectors = self._frames_to_pose_vectors(self.target_frames)

        # Once we have the target "pose vectors" for each path point, we can use
        # a similar method for generating the paths of the "pose variables"
        # [x, y, z, rx, ry, rz]. Each path is composed of linear sections and
        # parabolic blends. To ensure that the resultant motion from path point
        # to path point in 3D space will be a straight line, the blend time used
        # for the parabolic blend must be the same for each of the "pose
        # variables".
        self._mvp = MultiLinearSegmentVectorPath(
            path_points=self.pose_vectors,
            dt_segments=self.dt_segments,
            dt_blends=self.dt_blends,
        )

    @staticmethod
    def _rotation_matrix_to_rotvec(R: NumpyArray) -> NumpyArray:
        """
        Converts a rotation matrix to an angle-axis vector.

        The magnitude of the returned vector is the rotation angle in radians.
        Its direction is the rotation axis.
        """
        trace = np.trace(R)
        cos_theta = 0.5 * (trace - 1.0)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        theta = np.arccos(cos_theta)

        if np.isclose(theta, 0.0):
            return np.zeros(3)

        if np.isclose(theta, np.pi):
            axis = np.empty(3)

            axis[0] = np.sqrt(max((R[0, 0] + 1.0) / 2.0, 0.0))
            axis[1] = np.sqrt(max((R[1, 1] + 1.0) / 2.0, 0.0))
            axis[2] = np.sqrt(max((R[2, 2] + 1.0) / 2.0, 0.0))

            axis[1] = np.copysign(axis[1], R[0, 1] + R[1, 0])
            axis[2] = np.copysign(axis[2], R[0, 2] + R[2, 0])

            norm = np.linalg.norm(axis)
            if np.isclose(norm, 0.0):
                return np.zeros(3)

            return theta * axis / norm  # type: ignore

        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) / (2.0 * np.sin(theta))

        return theta * axis

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
    def _frame_to_pose_vector(cls, frame: Frame) -> NumpyArray:
        """
        Converts a Frame object to a six-dimensional Cartesian pose vector.
        """
        position = np.asarray(frame.origin, dtype=float)
        # R = np.asarray(frame.orient_mat.A, dtype=float)
        # rotvec = cls._rotation_matrix_to_rotvec(R)
        R = frame.orient_mat
        theta, axis = R.angvec()
        rotvec = theta * axis
        return np.concatenate((position, rotvec))

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
            pose_vector = cls._frame_to_pose_vector(frame)

            if i > 0:
                pose_vector[3:] = cls._choose_equivalent_rotvec(
                    pose_vector[3:],
                    pose_vectors[-1][3:]
                )

            pose_vectors.append(pose_vector)

        return np.array(pose_vectors)

    @staticmethod
    def _pose_vector_to_frame(pose_vector: NumpyArray) -> Frame:
        """
        Converts a six-dimensional Cartesian pose vector to a Frame object.
        """
        position = pose_vector[:3]
        rotvec = pose_vector[3:]
        theta = np.linalg.norm(rotvec)

        if np.isclose(theta, 0.0):
            R = SO3()
        else:
            axis = rotvec / theta
            R = SO3.AngleAxis(float(theta), axis, unit="rad")

        T = SE3.Rt(R, position)
        return Frame.from_matrix(T)

    def pose_vector(self, t: float) -> NumpyArray:
        """
        Returns the six-dimensional Cartesian pose vector at time t.
        """
        return self._mvp.position(t)

    def pose_vector_velocity(self, t: float) -> NumpyArray:
        """
        Returns the time derivative of the Cartesian pose vector at time t.
        """
        return self._mvp.velocity(t)

    def pose_vector_acceleration(self, t: float) -> NumpyArray:
        """
        Returns the second time derivative of the Cartesian pose vector at time t.
        """
        return self._mvp.acceleration(t)

    def frame(self, t: float) -> Frame:
        """
        Returns the frame pose at time t.
        """
        pose_vector = self.pose_vector(t)
        return self._pose_vector_to_frame(pose_vector)

    def trajectory(self) -> tuple[NumpyArray, list[Frame]]:
        """
        Returns the Cartesian trajectory.

        Returns
        -------
        t_arr: NumpyArray
            Numpy array with time moments.
        frames: list[Frame]
            List of frame poses along the trajectory at the time instants in
            `t_arr`.
        """
        t_arr, pose_vector_arr = self._mvp.position_profile(self.num_t_samples)
        frames = [
            self._pose_vector_to_frame(pose_vector)
            for pose_vector in pose_vector_arr
        ]
        return t_arr, frames

    def pose_vector_profile(self) -> tuple[NumpyArray, NumpyArray]:
        """
        Returns the sampled six-dimensional Cartesian pose-vector profile.
        """
        return self._mvp.position_profile(self.num_t_samples)

    def pose_vector_velocity_profile(self) -> tuple[NumpyArray, NumpyArray]:
        """
        Returns the sampled velocity profile of the Cartesian pose vector.
        """
        return self._mvp.velocity_profile(self.num_t_samples)

    def pose_vector_acceleration_profile(self) -> tuple[NumpyArray, NumpyArray]:
        """
        Returns the sampled acceleration profile of the Cartesian pose vector.
        """
        return self._mvp.acceleration_profile(self.num_t_samples)

    @property
    def motion_profile(self) -> MultiLinearSegmentVectorPath:
        """
        Returns the underlying multipoint vector motion profile.
        """
        return self._mvp

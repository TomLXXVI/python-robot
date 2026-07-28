"""
Multi-segment Cartesian straight-line motion planning.

This module converts target frames to six-dimensional pose vectors, constructs
a vector-valued linear path with parabolic blends, and samples the resulting
Cartesian trajectory.
"""

from typing import Sequence

from dataclasses import dataclass

import numpy as np

from ....base.types import NumpyArray
from ....base import Frame, SpatialAcceleration, SpatialVelocity

from .profile import BlendedVectorProfile


__all__ = ["BlendedCartesianMotion", "CartesianMultiLineMotion"]


@dataclass
class BlendedCartesianMotion:
    """
    Represent continuous blended Cartesian motion through target frames.

    Target frames are converted to continuous six-dimensional pose vectors.
    A :class:`BlendedVectorProfile` defines their evolution in time. This class
    does not choose sample times or construct a sampled trajectory.

    Parameters
    ----------
    target_frames : Sequence[Frame]
        Frames through which the Cartesian path is defined.
    segment_durations : Sequence[float]
        Travel duration between each pair of successive target frames.
    blend_durations : float | Sequence[float]
        Blend duration at every target frame, or one shared duration.
    """
    target_frames: Sequence[Frame]
    segment_durations: Sequence[float]
    blend_durations: float | Sequence[float]

    def __post_init__(self) -> None:
        if len(self.target_frames) < 2:
            raise ValueError("At least two target frames are required.")

        if len(self.segment_durations) != len(self.target_frames) - 1:
            raise ValueError(
                f"Number of segment durations "
                f"({len(self.segment_durations)}) does "
                f"not match the number of segments "
                f"({len(self.target_frames) - 1})."
            )

        self.pose_vectors = self._frames_to_pose_vectors(self.target_frames)

        self._profile = BlendedVectorProfile(
            pose_vectors=self.pose_vectors,
            dt_segments=self.segment_durations,
            dt_blends=self.blend_durations,
        )

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

    @property
    def duration(self) -> float:
        """
        Return the total motion duration.

        Returns
        -------
        float
            Total duration in seconds.
        """
        return self._profile.dt_tot

    @property
    def target_times(self) -> NumpyArray:
        """
        Return the nominal times of the target frames.

        Returns
        -------
        NumpyArray
            One-dimensional array containing one time per target frame.
        """
        return self._profile.knot_times.copy()

    @property
    def profile(self) -> BlendedVectorProfile:
        """
        Return the underlying continuous vector profile.

        Returns
        -------
        BlendedVectorProfile
            Profile used to evaluate the Cartesian motion.
        """
        return self._profile

    def pose_vector_at(self, time: float) -> NumpyArray:
        """
        Evaluate the Cartesian pose vector at a time.

        Parameters
        ----------
        time : float
            Evaluation time in seconds.

        Returns
        -------
        NumpyArray
            Pose vector ``(x, y, z, rx, ry, rz)``.
        """
        return self._profile.pose(time)

    def frame_at(self, time: float) -> Frame:
        """
        Evaluate the end-effector frame at a time.

        Parameters
        ----------
        time : float
            Evaluation time in seconds.

        Returns
        -------
        Frame
            Cartesian frame at ``time``.
        """
        return Frame.from_pose_vector(self.pose_vector_at(time))

    def spatial_velocity_at(self, time: float) -> NumpyArray:
        """
        Evaluate the spatial velocity at a time.

        Parameters
        ----------
        time : float
            Evaluation time in seconds.

        Returns
        -------
        NumpyArray
            Spatial velocity ``(vx, vy, vz, wx, wy, wz)``.
        """
        pose = self._profile.pose(time)
        pose_velocity = self._profile.velocity(time)
        velocity = SpatialVelocity.from_pose(pose, pose_velocity)
        return np.asarray(velocity, dtype=float)

    def spatial_acceleration_at(self, time: float) -> NumpyArray:
        """
        Evaluate the spatial acceleration at a time.

        Parameters
        ----------
        time : float
            Evaluation time in seconds.

        Returns
        -------
        NumpyArray
            Spatial acceleration ``(ax, ay, az, alphax, alphay, alphaz)``.
        """
        pose = self._profile.pose(time)
        pose_velocity = self._profile.velocity(time)
        pose_acceleration = self._profile.acceleration(time)
        acceleration = SpatialAcceleration.from_pose(
            pose,
            pose_velocity,
            pose_acceleration,
        )
        return np.asarray(acceleration, dtype=float)


class CartesianMultiLineMotion(BlendedCartesianMotion):
    """
    Provide compatibility with the former sampled multi-line motion API.

    New code should use :class:`BlendedCartesianMotion` and construct a
    :class:`CartesianTrajectory` through its ``from_motion`` factory.

    Parameters
    ----------
    target_frames : Sequence[Frame]
        Cartesian target frames.
    dt_segments : Sequence[float]
        Travel durations between successive target frames.
    dt_blends : float | Sequence[float]
        Blend durations at the target frames.
    num_t_samples : int, default=100
        Number of samples generated for the compatibility API.
    """

    def __init__(
        self,
        target_frames: Sequence[Frame],
        dt_segments: Sequence[float],
        dt_blends: float | Sequence[float],
        num_t_samples: int = 100,
    ) -> None:
        """
        Initialize the compatibility motion and generate its samples.
        """
        super().__init__(
            target_frames=target_frames,
            segment_durations=dt_segments,
            blend_durations=dt_blends,
        )
        self.dt_segments = dt_segments
        self.dt_blends = dt_blends
        self.num_t_samples = num_t_samples
        self._sample_legacy_motion()

    def _sample_legacy_motion(self) -> None:
        """
        Populate arrays required by the former sampled motion API.
        """
        self._t_arr = np.linspace(0.0, self.duration, self.num_t_samples)
        self._p_arr = np.array([
            self.pose_vector_at(time)
            for time in self._t_arr
        ])
        self._V_arr = np.array([
            self.spatial_velocity_at(time)
            for time in self._t_arr
        ])
        self._A_arr = np.array([
            self.spatial_acceleration_at(time)
            for time in self._t_arr
        ])

    def trajectory(self) -> tuple[NumpyArray, list[Frame]]:
        """
        Return the legacy sampled frame trajectory.

        Returns
        -------
        tuple[NumpyArray, list[Frame]]
            Sample times and corresponding Cartesian frames.
        """
        frames = [
            Frame.from_pose_vector(pose_vector)
            for pose_vector in self._p_arr
        ]
        return self._t_arr, frames

    @property
    def motion_profile(self) -> BlendedVectorProfile:
        """
        Return the underlying profile under its former property name.
        """
        return self.profile

    @property
    def motion_samples(self) -> tuple[NumpyArray, ...]:
        """
        Return the samples required by the former motion API.

        Returns
        -------
        tuple[NumpyArray, ...]
            Times, poses, spatial velocities and spatial accelerations.
        """
        return self._t_arr, self._p_arr, self._V_arr, self._A_arr

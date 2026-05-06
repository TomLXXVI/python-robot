from typing import Sequence

from enum import StrEnum

import numpy as np

from ...base.types import NumpyArray
from ...base import Frame
from ...manipulator import SerialLinkManipulator
from ..profiles_1D.multi_point import MultiPointCubicPath, MultiLinearPath

__all__ = [
    "MultiPointMotionProfile",
    "MultiPointMotionProfileType",
    "JointSpaceMotion"
]

MultiPointMotionProfile = MultiPointCubicPath | MultiLinearPath


class MultiPointMotionProfileType(StrEnum):
    CUBIC = "cubic"
    LINEAR = "linear"


class JointSpaceMotion:

    def __init__(
        self,
        target_frames: Sequence[Frame],
        dt_segments: Sequence[float],
        manipulator: SerialLinkManipulator,
        mp_type: MultiPointMotionProfileType = MultiPointMotionProfileType.CUBIC,
        blend_accels: float | Sequence[float] | None = None,
        num_t_samples: int = 100,
        ini_guess: Sequence[float] | None = None,
    ) -> None:
        """
        Creates a JointSpaceMotion object.

        The class performs three steps:
        -   In the first step the target frames in Cartesian space are mapped
            to corresponding joint configurations in joint space.
        -   In the second step the target joint configurations and the travel times
            between them are used to construct a motion profile for each of the
            joints.
        -   In the third step time samples are taken from the motion profile of
            each joint.

        Parameters
        ----------
        target_frames: Sequence[Frame]
            List of end-effector frames (positions and orientations) in
            Cartesian space.
        dt_segments: Sequence[float]
            List with the desired/required travel times between two successive
            target frames.
        manipulator: SerialLinkManipulator
            The manipulator that needs to execute the motion.
        mp_type: MotionProfileType, default = MotionProfileType.CUBIC
            Indicates the type of multipoint motion profile to use for
            calculating the motion paths of the joints in the manipulator.
        blend_accels: float | Sequence[float] | None, optional
            Acceleration values at the joint-space path points for calculating
            the parabolic blends. If a single value is given, it is used for all
            path points.
            Only relevant when MultiPointMotionProfileType is set to LINEAR. If
            mp_type is CUBIC, leave it to None.
        num_t_samples: int, default = 100
            Number of time samples that must be taken when time-sampling the
            motion paths.
        ini_guess: Sequence[float] | None, optional
            Initial joint coordinates for the inverse kinematics of the first
            target frame. If None, the current joint coordinates of the
            manipulator are used. Each following target frame uses the previous
            inverse-kinematics solution as its initial guess.
        """
        self.manipulator = manipulator
        self.target_frames = target_frames
        self.mp_type = mp_type
        self.blend_accels = blend_accels
        self.ini_guess = ini_guess

        self.n_segments = len(self.target_frames) - 1
        self.n_joints = len(self.manipulator)

        if self.n_segments != len(dt_segments):
            raise ValueError(
                f"The number of segment travel times ({len(dt_segments)}) is "
                f"not equal to the number of segments in the trajectory "
                f"({self.n_segments})."
            )
        self.dt_segments = dt_segments

        self._q_sets = self._map_to_joints(self.target_frames, self.ini_guess)
        self._motion_profiles = self._motion_profiling(self._q_sets)
        res = self._time_sampling(self._motion_profiles, num_t_samples)
        self._t_arr, self._q_arr, self._qd_arr, self._qdd_arr = res

    def _map_to_joints(
        self,
        frames: Sequence[Frame],
        ini_guess: Sequence[float] | None = None,
    ) -> NumpyArray:
        """
        Maps the end-effector frames from "Cartesian space" to "joint space",
        i.e., the poses of the end-effector frames (w.r.t. the fixed base frame
        of the manipulator) are translated to sets of corresponding joint
        coordinates through application of the inverse kinematics of the
        manipulator.

        Returns
        -------
        q_sets: NumpyArray
            2D-array: The number of rows of the array is equal to the number of
            frames. The number of columns equals the number of joints of the
            manipulator.
        """
        q_sets = []
        q_guess = ini_guess if ini_guess is not None else self.manipulator.joint_coords

        for frame in frames:
            q = self.manipulator.inv_kin(frame, ini_guess=q_guess)  # TODO: Ensure that q_guess and q have radians
            q_sets.append(q)
            q_guess = q

        return np.array(q_sets)

    def _motion_profiling(self, q_sets: NumpyArray) -> list[MultiPointMotionProfile]:
        """
        From the joint coordinate sets determined by inverse kinematics,
        calculates the motion paths for each joint in the manipulator.

        Returns
        -------
        list[MultiPointMotionProfile]
            Either a list of MultiPointCubicPath objects or a list of
            MultiLinearSegmentPath objects, depending on the type of motion
            profile selected.
        """
        if self.mp_type == MultiPointMotionProfileType.CUBIC:
            motion_profiles = [
                MultiPointCubicPath(
                    path_points=q_sets[:, i],
                    dt_segments=self.dt_segments,
                    v_start=0.0,
                    v_end=0.0,
                )
                for i in range(self.n_joints)
            ]
            return motion_profiles
        elif self.mp_type == MultiPointMotionProfileType.LINEAR and self.blend_accels is not None:
            motion_profiles = [
                MultiLinearPath(
                    path_points=q_sets[:, i],
                    dt_segments=self.dt_segments,
                    blend_accels=self.blend_accels
                )
                for i in range(self.n_joints)
            ]
            return motion_profiles
        elif self.mp_type == MultiPointMotionProfileType.LINEAR and self.blend_accels is None:
            # noinspection GrazieInspectionRunner
            raise ValueError(
                "If motion profile type is set to LINEAR, parameter "
                "'blend_accels' cannot be None."
            )
        else:
            raise NotImplementedError(
                f"No implementation available for motion profile type"
                f" '{self.mp_type}'. "
            )

    def _time_sampling(
        self,
        motion_profiles: list[MultiPointMotionProfile],
        n_samples: int
    ) -> tuple[NumpyArray, ...]:
        """
        Takes evenly distributed time samples of the motion profiles of the
        joints.

        Returns
        -------
        t_arr: NumpyArray
            The time moments at which the values of the joint coordinates are
            calculated.
        q_arr: NumpyArray
            The values of the joint coordinates in the course of time. The
            number of rows of the array is equal to the number of time samples.
            The number of columns is equal to the number of joints of the
            manipulator.
        qd_arr: NumpyArray
            The values of the joint velocities in the course of time. The
            number of rows of the array is equal to the number of time samples.
            The number of columns is equal to the number of joints of the
            manipulator.
        qdd_arr: NumpyArray
            The values of the joint accelerations in the course of time. The
            number of rows of the array is equal to the number of time samples.
            The number of columns is equal to the number of joints of the
            manipulator.
        """
        t_arr = np.linspace(0.0, sum(self.dt_segments), n_samples)

        q_arr = np.column_stack([
            np.array([mp.position(t) for t in t_arr], dtype=float)
            for mp in motion_profiles
        ])

        qd_arr = np.column_stack([
            np.array([mp.velocity(t) for t in t_arr], dtype=float)
            for mp in motion_profiles
        ])

        qdd_arr = np.column_stack([
            np.array([mp.acceleration(t) for t in t_arr], dtype=float)
            for mp in motion_profiles
        ])

        return t_arr, q_arr, qd_arr, qdd_arr

    @property
    def target_coordinates(self) -> NumpyArray:
        """
        Returns the joint coordinates (joint positions) of the manipulator that
        correspond with the target end-effector frames in Cartesian space.

        Returns
        -------
        NumpyArray
            2D-array: The number of rows of the array is equal to the number of
            frames. The number of columns equals the number of joints in the
            manipulator.
        """
        return self._q_sets

    @property
    def motion_profiles(self) -> list[MultiPointMotionProfile]:
        """
        Returns the motion profiles of the joints of the manipulator.

        Returns
        -------
        list[MultiPointMotionProfile]
            Either a list of MultiPointCubicPath objects or a list of
            MultiLinearSegmentPath objects, depending on the type of motion
            profile selected. The order corresponds with the order of the
            joints in the manipulator, from base to tool-end.
        """
        return self._motion_profiles

    @property
    def motion_samples(self) -> tuple[NumpyArray, ...]:
        """
        Returns motion samples of the joints of the manipulator.

        Returns
        -------
        t_arr: NumpyArray
            The time moments at which the values of the joint coordinates are
            taken.
        q_arr: NumpyArray
            The values of the joint coordinates in the course of time. The
            number of rows of the array is equal to the number of time samples.
            The number of columns is equal to the number of joints of the
            manipulator.
        qd_arr: NumpyArray
            The values of the joint velocities in the course of time. The
            number of rows of the array is equal to the number of time samples.
            The number of columns is equal to the number of joints of the
            manipulator.
        qdd_arr: NumpyArray
            The values of the joint accelerations in the course of time. The
            number of rows of the array is equal to the number of time samples.
            The number of columns is equal to the number of joints of the
            manipulator.
        """
        return self._t_arr, self._q_arr, self._qd_arr, self._qdd_arr

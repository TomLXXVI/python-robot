"""
Multi-point joint-space motion planning from Cartesian targets.

The module maps Cartesian target frames to joint coordinates with inverse
kinematics, creates one motion profile per joint, and samples joint position,
velocity, and acceleration trajectories.
"""

from typing import Sequence

from enum import StrEnum
from dataclasses import dataclass

import numpy as np

from automation_motion.base.types import NumpyArray
from automation_motion.profiles_1D.multi_point import MultiPointCubicPath, MultiLinearPath

from ...base import Frame
from ...manipulator import SerialLinkManipulator

__all__ = [
    "MultiPointMotionProfile",
    "MultiPointMotionProfileType",
    "JointSpaceMotion",
    "IKMask",
    "IKTarget",
]


MultiPointMotionProfile = MultiPointCubicPath | MultiLinearPath


class MultiPointMotionProfileType(StrEnum):
    """
    Available one-dimensional multi-point profile families for joint motion.
    """

    CUBIC = "cubic"
    LINEAR = "linear"


@dataclass(frozen=True)
class IKMask:
    """
    Degrees of freedom enabled during inverse-kinematics target solving.

    Parameters
    ----------
    x, y, z : bool, default = True
        Enable translation constraints along the corresponding Cartesian axes.
    alpha, beta, gamma : bool, default = True
        Enable orientation constraints around the corresponding axes.
    """

    x: bool = True
    y: bool = True
    z: bool = True
    alpha: bool = True
    beta: bool = True
    gamma: bool = True

    def to_array(self) -> NumpyArray:
        """
        Return the inverse-kinematics mask as a Boolean NumPy array.

        Returns
        -------
        NumpyArray
            Boolean array ordered as ``x, y, z, alpha, beta, gamma``.
        """
        return np.array(
            [self.x, self.y, self.z,
             self.alpha, self.beta, self.gamma],
            dtype=bool
        )


@dataclass
class IKTarget:
    """
    Cartesian target frame with an optional inverse-kinematics mask.

    Parameters
    ----------
    frame : Frame
        Desired end-effector frame.
    ik_mask : IKMask, optional
        Degrees of freedom enabled for inverse kinematics. If omitted, all
        translational and rotational constraints are enabled.
    """

    frame: Frame
    ik_mask: IKMask | None = None

    def __post_init__(self):
        if self.ik_mask is None:
            self.ik_mask = IKMask()


class JointSpaceMotion:
    """
    Convert Cartesian targets into sampled joint-space motion.

    The class solves inverse kinematics for each target frame, builds a
    multipoint profile for each joint, and samples joint positions, velocities,
    and accelerations over the complete movement.
    """

    def __init__(
        self,
        targets: Sequence[IKTarget] | Sequence[Frame],
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
            to corresponding target joint coordinates in joint space.
        -   In the second step the target joint coordinates and the travel
            times between them are used to define a motion profile for each
            of the joints.
        -   In the third step time samples are taken from the motion profile of
            each joint.

        Parameters
        ----------
        targets: Sequence[IKTarget] | Sequence[Frame]
            Sequence of Cartesian end-effector target frames optionally combined
            with an IK-mask indicating the degrees of freedom the IK-solver has
            available to determine a corresponding set of joint coordinates.
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
        self.targets = targets
        self.mp_type = mp_type
        self.blend_accels = blend_accels
        self.ini_guess = ini_guess

        self.n_segments = len(self.targets) - 1
        self.n_joints = len(self.manipulator)

        if self.n_segments != len(dt_segments):
            raise ValueError(
                f"The number of segment travel times ({len(dt_segments)}) is "
                f"not equal to the number of segments in the trajectory "
                f"({self.n_segments})."
            )
        self.dt_segments = dt_segments

        self._q_sets = self._map_to_joints(self.targets, self.ini_guess)
        self._motion_profiles = self._motion_profiling(self._q_sets)
        res = self._time_sampling(self._motion_profiles, num_t_samples)
        self._t_arr, self._q_arr, self._qd_arr, self._qdd_arr = res

    def _map_to_joints(
        self,
        targets: Sequence[IKTarget] | Sequence[Frame],
        ini_guess: Sequence[float] | None = None,
    ) -> NumpyArray:
        """
        Maps the end-effector target frames from "Cartesian space" to
        "joint space", i.e., the pose of each target frame (w.r.t. the fixed
        base frame of the manipulator) is translated to a corresponding set of
        joint coordinates using the inverse kinematics of the manipulator.

        Returns
        -------
        q_sets: NumpyArray
            2D-array: The number of rows of the array is equal to the number of
            frames. The number of columns equals the number of joints of the
            manipulator.
        """
        def choose_closest_revolute_equivalent(q, previous_q, joints, q_lims=None):
            q = np.asarray(q, dtype=float)
            previous_q = np.asarray(previous_q, dtype=float)

            q_equiv = q.copy()

            for i, joint in enumerate(joints):
                if not joint.is_revolute:
                    continue

                qi = q[i]
                qi_prev = previous_q[i]

                k0 = int(np.round((qi_prev - qi) / (2.0 * np.pi)))
                candidates = qi + 2.0 * np.pi * np.arange(k0 - 2, k0 + 3)

                if q_lims is not None and q_lims[i] is not None:
                    lo, hi = q_lims[i]
                    candidates = candidates[(candidates >= lo) & (candidates <= hi)]

                if len(candidates) > 0:
                    q_equiv[i] = candidates[np.argmin(np.abs(candidates - qi_prev))]

            return q_equiv

        joints = self.manipulator.links
        q_lims = [link.q_lim for link in self.manipulator.links]
        q_sets = []
        q_guess = ini_guess if ini_guess is not None else self.manipulator.joint_coords

        if isinstance(targets[0], Frame):
            targets = [IKTarget(frame) for frame in targets]

        for target in targets:
            q = self.manipulator.inv_kin(
                target.frame,
                ini_guess=q_guess,
                mask=target.ik_mask.to_array()  # type: ignore
            )
            # For revolute joints, a branch continuity step is performed after
            # every IK solution: optionally add ±2π to revolute joints so that
            # they lie as close as possible to the previous joint configuration,
            # as long as they remain within q_lim.
            if len(q_sets) >= 1:
                previous_q = q_sets[-1]
                q = choose_closest_revolute_equivalent(q, previous_q, joints, q_lims)
            q_sets.append(q)
            q_guess = q

        return np.array(q_sets)

    def _motion_profiling(self, q_sets: NumpyArray) -> list[MultiPointMotionProfile]:
        """
        From the sets of target joint coordinates that were determined by
        the inverse kinematics, calculates a motion profile for each joint of
        the manipulator.

        Returns
        -------
        list[MultiPointMotionProfile]
            Either a list of MultiPointCubicPath objects or a list of
            MultiLinearSegmentPath objects, depending on the type of motion
            profile that was selected.
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
        Returns uniformly distributed time samples of the joint-motion profiles.

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

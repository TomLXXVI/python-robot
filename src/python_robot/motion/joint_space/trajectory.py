"""
Joint-space trajectory planning and sampled trajectory containers.

This module keeps the joint-space planning helpers close to the trajectory
object they support. Cartesian targets are mapped to joint coordinates with
inverse kinematics, profiled per joint, sampled, and exposed through the
``JointTrajectory`` public interface.
"""

from __future__ import annotations
from typing import Sequence

from enum import StrEnum
from dataclasses import dataclass

import numpy as np

from automation_motion.base.types import NumpyArray, AngleUnit
from automation_motion.charts import LineChart, BarChart
from automation_motion.profiles_1D.multi_point import MultiPointCubicPath, MultiLinearPath

from ...base import Frame
from ...manipulator import SerialLinkManipulator, ConfigurationError
from ...utils import array_to_table

__all__ = [
    "MultiPointMotionProfile",
    "MultiPointMotionProfileType",
    "IKMask",
    "IKTarget",
    "JointTrajectoryBuilder",
    "JointTrajectory",
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


class JointTrajectoryBuilder:
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
        Creates a JointTrajectoryBuilder object.

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


class JointTrajectory:
    """
    Given a sequence of end-effector frames (positions and orientations) in
    Cartesian space, determines appropriate motion paths for the joints of the
    manipulator.
    """
    def __init__(
        self,
        t_arr: NumpyArray,
        q_arr: NumpyArray,
        qd_arr: NumpyArray,
        qdd_arr: NumpyArray,
        *,
        target_frames: Sequence[Frame],
        dt_segments: Sequence[float],
        manipulator: SerialLinkManipulator,
        q_sets: NumpyArray,
        motion_profiles: Sequence[MultiPointMotionProfile],
        angle_unit: AngleUnit = "deg",
    ) -> None:
        """
        Create a sampled joint-space trajectory.

        Parameters
        ----------
        t_arr : NumpyArray
            Sampled time moments.
        q_arr : NumpyArray
            Sampled joint positions ordered from base to tool.
        qd_arr : NumpyArray
            Sampled joint velocities ordered from base to tool.
        qdd_arr : NumpyArray
            Sampled joint accelerations ordered from base to tool.
        target_frames : Sequence[Frame]
            Cartesian end-effector frames used as motion targets.
        dt_segments : Sequence[float]
            Travel durations between successive target frames.
        manipulator : SerialLinkManipulator
            Manipulator for which the trajectory was generated.
        q_sets : NumpyArray
            Target joint coordinates corresponding to ``target_frames``.
        motion_profiles : Sequence[MultiPointMotionProfile]
            Analytic per-joint profiles, if available.
        angle_unit : AngleUnit, default = "deg"
            Angle unit used when rendering table output.
        """
        self._t_arr = t_arr
        self._q_arr = q_arr
        self._qd_arr = qd_arr
        self._qdd_arr = qdd_arr

        self.target_frames = target_frames
        self.dt_segments = dt_segments
        self.manipulator = manipulator
        self._q_sets = q_sets
        self._motion_profiles = motion_profiles

        self.n_joints = len(self.manipulator)
        self.n_segments = len(self.dt_segments)
        
        self._tables = _JointSpaceTables(self, angle_unit)

    @classmethod
    def create(
        cls,
        targets: Sequence[IKTarget] | Sequence[Frame],
        dt_segments: Sequence[float],
        manipulator: SerialLinkManipulator,
        *,
        mp_type: MultiPointMotionProfileType = MultiPointMotionProfileType.CUBIC,
        blend_accels: float | Sequence[float] | None = None,
        num_t_samples: int = 100,
        ini_guess: Sequence[float] | None = None,
        angle_unit: AngleUnit = "deg",
    ) -> JointTrajectory:
        """
        Creates a JointTrajectory object.

        Parameters
        ----------
        targets: Sequence[IKTarget] | Sequence[Frame]
            Sequence of Cartesian end-effector target frames optionally combined
            with an IK-mask indicating the degrees of freedom the IK-solver has
            available to determine a corresponding set of joint coordinates.
        dt_segments: Sequence[float]
            List with the required travel times of each segment between two
            successive frames.
        manipulator: SerialLinkManipulator
            Manipulator for which a joint-space trajectory is to be created.
        mp_type: MultiPointMotionProfileType, default = MultiPointMotionProfileType.CUBIC
            Indicates the type of multipoint motion profile to use for
            calculating the motion paths of each joint in the manipulator.
        blend_accels: float | Sequence[float] | None, optional
            Acceleration values at the joint-space path points for creating
            the parabolic blends in case MotionProfileType is set to LINEAR.
            If a single value is given, it is used for all path points. If
            mp_type is set to CUBIC, leave it to None.
        num_t_samples: int, default = 100
            Number of time samples to be taken from the position profile of
            the motion paths of the joints in the manipulator.
        ini_guess: Sequence[float] | None, optional
            Initial joint coordinates for the inverse kinematics of the first
            target frame. If None, the current joint coordinates of the
            manipulator are used. Each following target frame uses the previous
            inverse-kinematics solution as its initial guess.
        angle_unit: AngleUnit, default = "deg"
            Angle unit to be used in the tables.
        """
        jm = JointTrajectoryBuilder(
            targets, dt_segments, manipulator,
            mp_type, blend_accels, num_t_samples,
            ini_guess
        )
        t_arr, q_arr, qd_arr, qdd_arr = jm.motion_samples
        q_sets = jm.target_coordinates
        motion_profiles = jm.motion_profiles

        if isinstance(targets[0], IKTarget):
            target_frames = [target.frame for target in targets]
        else:
            target_frames = targets

        return cls(
            t_arr, q_arr, qd_arr, qdd_arr,
            target_frames=target_frames,  # type: ignore
            dt_segments=dt_segments,
            manipulator=manipulator,
            q_sets=q_sets,
            motion_profiles=motion_profiles,
            angle_unit=angle_unit,
        )

    def to_cartesian_space(self) -> CartesianTrajectory:
        """
        Convert this joint-space trajectory to Cartesian space.

        Returns
        -------
        CartesianTrajectory
            Cartesian trajectory obtained by applying forward kinematics at each
            sampled joint configuration.
        """
        from python_robot.motion.cartesian_space import CartesianTrajectory
        return CartesianTrajectory.from_joint_space(self)

    @classmethod
    def from_cartesian_space(
        cls,
        css: CartesianTrajectory,
        manipulator: SerialLinkManipulator,
        ini_guess: Sequence[float] | None = None,
    ) -> JointTrajectory:
        """
        Convert a Cartesian-space trajectory to joint space.

        Parameters
        ----------
        css : CartesianTrajectory
            Cartesian trajectory to convert.
        manipulator : SerialLinkManipulator
            Manipulator used for inverse kinematics and Jacobian mapping.
        ini_guess : Sequence[float], optional
            Initial inverse-kinematics guess for the first trajectory sample.

        Returns
        -------
        JointTrajectory
            Joint-space trajectory sampled from the Cartesian trajectory.
        """
        return css.to_joint_space(manipulator, ini_guess)

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
    def motion_profiles(self) -> Sequence[MultiPointMotionProfile]:
        """
        Returns the motion paths/motion profiles of the joints of the
        manipulator.

        Returns
        -------
        Sequence[MultiPointMotionProfile]
            Either a list of MultiPointCubicPath objects or a list of
            MultiLinearSegmentPath objects, depending on the type of motion
            profile selected. The order corresponds with the order of the
            joints in the manipulator, from base to tool-end.
        """
        return self._motion_profiles

    @property
    def has_motion_profiles(self) -> bool:
        """
        Returns True if this trajectory has analytic joint motion profiles.

        Trajectories created directly in joint space have analytic per-joint motion
        profiles. Trajectories converted from Cartesian space only have sampled
        joint positions, velocities, and accelerations.
        """
        return len(self._motion_profiles) > 0

    @property
    def scheme(self) -> NumpyArray:
        """
        Returns a "timetable" of the joint coordinates (joint positions) that
        move the manipulator from the initial to the final end-effector pose in
        Cartesian space.

        Returns
        -------
        NumpyArray
            2D-array: The first column of the array contains the sampled time
            moments. The following columns contain the coordinates/positions of
            each joint, starting at the joint closest to the base and proceeding
            to the joint closest to the tool-end in the manipulator.
        """
        return np.column_stack((self._t_arr, self._q_arr, self._qd_arr, self._qdd_arr))

    @property
    def tables(self) -> _JointSpaceTables:
        """
        Return formatted table views for the joint-space trajectory.
        """
        return self._tables

    @property
    def time_samples(self) -> NumpyArray:
        """
        Return the sampled time moments of the joint-space trajectory.
        """
        return self._t_arr

    @property
    def positions(self) -> NumpyArray:
        """
        Returns the array of sampled joint positions q.

        The number of rows is equal to the number time samples. The number of
        columns is equal to the number of joints in the manipulator.
        """
        return self._q_arr

    @property
    def velocities(self) -> NumpyArray:
        """
        Returns the array of sampled joint velocities (qd).

        The number of rows is equal to the number time samples. The number of
        columns is equal to the number of joints in the manipulator.
        """
        return self._qd_arr

    @property
    def accelerations(self) -> NumpyArray:
        """
        Returns the array of sampled joint accelerations (qdd).

        The number of rows is equal to the number time samples. The number of
        columns is equal to the number of joints in the manipulator.
        """
        return self._qdd_arr

    def dynamics(self) -> NumpyArray:
        """
        Returns the torques/forces acting on the joints of the manipulator.

        Note that the effect of any external wrench applied to the end-effector
        is not included (included are inertia, centripetal, Coriolis, friction
        and gravity effects).

        Returns
        -------
        NumpyArray
            2D array: The first column of the array contains the sampled time
            moments. The following columns contain the torques/forces acting on
            the joints.
        """
        if self.manipulator.has_dynamics():
            tau_arr = np.array([
                self.manipulator.inv_dyn(q, qd, qdd)
                for q, qd, qdd in zip(self._q_arr, self._qd_arr, self._qdd_arr)
            ])
            return np.column_stack((self._t_arr, tau_arr))
        raise ConfigurationError("The manipulator's dynamics is not defined.")

    def dynamics_distribution(
        self,
        bins: int | Sequence[float] = 10,
        *,
        absolute: bool = True,
        normalize: bool = False,
    ) -> NumpyArray:
        """
        Returns a time-weighted distribution of the joint torques/forces.

        This is useful for judging how heavily a servo motor is loaded during
        the defined motion. While ``dynamics()`` returns the torque/force time
        history, this method summarizes how much movement time is spent in each
        torque/force interval.

        Parameters
        ----------
        bins : int | Sequence[float], default = 10
            Number of equally spaced bins, or explicit bin edges.
        absolute : bool, default = True
            If True, use absolute torque/force values. If False, keep signed
            values.
        normalize : bool, default = False
            If True, return fractions of the total movement time instead of
            seconds.

        Returns
        -------
        NumpyArray
            2D array: The first two columns contain the lower and upper bin
            edge. The following columns contain, for each joint, the time in
            seconds or fraction of the movement time spent inside that bin.
        """
        tau_arr = self.dynamics()[:, 1:]
        tau_values = np.abs(tau_arr) if absolute else tau_arr
        sample_weights = self._time_sample_weights()

        if isinstance(bins, int):
            if bins < 1:
                raise ValueError("bins must be at least 1.")

            tau_min = 0.0 if absolute else float(np.min(tau_values))
            tau_max = float(np.max(tau_values))
            if tau_min == tau_max:
                if absolute:
                    tau_max = 1.0 if tau_max == 0.0 else tau_max * 1.05
                else:
                    span = 1.0 if tau_max == 0.0 else abs(tau_max) * 0.05
                    tau_min -= span
                    tau_max += span
            bin_edges = np.linspace(tau_min, tau_max, bins + 1)
        else:
            bin_edges = np.asarray(bins, dtype=float)
            if (
                bin_edges.ndim != 1
                or len(bin_edges) < 2
                or np.any(np.diff(bin_edges) <= 0.0)
            ):
                raise ValueError("bins must contain strictly increasing bin edges.")

        distributions = []
        total_time = float(np.sum(sample_weights))
        for i in range(self.n_joints):
            hist, _ = np.histogram(
                tau_values[:, i],
                bins=bin_edges,
                weights=sample_weights,
            )
            if normalize and total_time > 0.0:
                hist = hist / total_time
            distributions.append(hist)

        return np.column_stack((
            bin_edges[:-1],
            bin_edges[1:],
            np.array(distributions).T,
        ))

    def _time_sample_weights(self) -> NumpyArray:
        """
        Returns the representative time span around each sampled time moment.
        """
        if len(self._t_arr) == 1:
            return np.ones(1)

        dt_arr = np.diff(self._t_arr)
        if np.any(dt_arr < 0.0):
            raise ValueError("time samples must be sorted in ascending order.")

        weights = np.empty_like(self._t_arr, dtype=float)
        weights[0] = dt_arr[0] / 2.0
        weights[-1] = dt_arr[-1] / 2.0
        if len(self._t_arr) > 2:
            weights[1:-1] = (dt_arr[:-1] + dt_arr[1:]) / 2.0
        return weights

    def plot_positions(self, show_targets: bool = False) -> LineChart:
        """
        Plots the positions q(t) of the joints and returns the LineChart
        object. Call show() on this object to see the plot.

        Returns
        -------
        LineChart
        """
        def _to_degrees(i: int, q: NumpyArray) -> NumpyArray:
            # convert joint angles to degrees if required
            if self.manipulator.links[i].is_revolute and self._tables.angle_unit == "deg":
                return np.rad2deg(q)
            return q

        target_times = np.concatenate(([0.0], np.cumsum(self.dt_segments)))

        chart = LineChart()

        for i in range(self.n_joints):
            chart.add_xy_data(
                label=f"q{i+1}",
                x1_values=self._t_arr,
                y1_values=_to_degrees(i, self._q_arr[:, i]),
            )
            if show_targets:
                chart.add_xy_data(
                    label=f"q{i+1}, targets",
                    x1_values=target_times,
                    y1_values=_to_degrees(i, self._q_sets[:, i]),
                    style_props={
                        "marker": "o",
                        "linestyle": "none",
                    },
                )

        chart.x1.add_title("time, s")
        chart.y1.add_title("joint coordinate")
        columns = 1
        if self.n_joints == 1:
            columns = 1
        elif self.n_joints == 2:
            columns = 2
        elif self.n_joints >= 3:
            columns = 3
        chart.add_legend(columns=columns)
        return chart

    def plot_velocities(self, show_targets: bool = False) -> LineChart:
        """
        Plots the velocities qd(t) of the joints and returns the LineChart
        object. Call show() on this object to see the plot.

        Returns
        -------
        LineChart
        """
        if self.has_motion_profiles:
            target_times = np.concatenate(([0.0], np.cumsum(self.dt_segments)))
            target_qd_arr = np.array([
                [mp.velocity(t) for mp in self._motion_profiles]
                for t in target_times
            ])

        chart = LineChart()
        for i in range(self.n_joints):
            chart.add_xy_data(
                label=f"qd{i+1}",
                x1_values=self._t_arr,
                y1_values=self._qd_arr[:, i],
            )
            if self.has_motion_profiles and show_targets:
                # noinspection PyUnboundLocalVariable
                chart.add_xy_data(
                    label=f"qd{i+1}, targets",
                    x1_values=target_times,
                    y1_values=target_qd_arr[:, i],
                    style_props={
                        "marker": "o",
                        "linestyle": "none",
                    },
                )

        chart.x1.add_title("time, s")
        chart.y1.add_title("joint velocity")
        columns = 1
        if self.n_joints == 1:
            columns = 1
        elif self.n_joints == 2:
            columns = 2
        elif self.n_joints >= 3:
            columns = 3
        chart.add_legend(columns=columns)
        return chart

    def plot_accelerations(self, show_targets: bool = False) -> LineChart:
        """
        Plots the accelerations qdd(t) of the joints and returns the
        LineChart object. Call show() on this object to see the plot.

        Returns
        -------
        LineChart
        """
        if self.has_motion_profiles:
            target_times = np.concatenate(([0.0], np.cumsum(self.dt_segments)))
            target_qdd_arr = np.array([
                [mp.acceleration(t) for mp in self._motion_profiles]
                for t in target_times
            ])

        chart = LineChart()
        for i in range(self.n_joints):
            chart.add_xy_data(
                label=f"qdd{i + 1}",
                x1_values=self._t_arr,
                y1_values=self._qdd_arr[:, i],
            )
            if self.has_motion_profiles and show_targets:
                # noinspection PyUnboundLocalVariable
                chart.add_xy_data(
                    label=f"qdd{i + 1}, targets",
                    x1_values=target_times,
                    y1_values=target_qdd_arr[:, i],
                    style_props={
                        "marker": "o",
                        "linestyle": "none",
                    },
                )

        chart.x1.add_title("time, s")
        chart.y1.add_title("joint acceleration")
        columns = 1
        if self.n_joints == 1:
            columns = 1
        elif self.n_joints == 2:
            columns = 2
        elif self.n_joints >= 3:
            columns = 3
        chart.add_legend(columns=columns)
        return chart

    def plot_dynamics(self) -> LineChart:
        """
        Plot sampled joint torques or forces over time.

        Returns
        -------
        LineChart
            Chart containing one torque/force trace per manipulator joint.

        Raises
        ------
        ConfigurationError
            If the manipulator has no dynamic parameters.
        """
        if self.manipulator.has_dynamics():
            tau_arr = self.dynamics()
            t_arr = tau_arr[:, 0]

            chart = LineChart()

            for i in range(1, self.n_joints + 1):
                chart.add_xy_data(
                    label=f"tau{i}",
                    x1_values=t_arr,
                    y1_values=tau_arr[:, i]
                )

            chart.x1.add_title("time, s")
            chart.y1.add_title("joint torque/force")
            columns = 1
            if self.n_joints == 1:
                columns = 1
            elif self.n_joints == 2:
                columns = 2
            elif self.n_joints >= 3:
                columns = 3
            chart.add_legend(columns=columns)
            return chart
        raise ConfigurationError("The manipulator's dynamics is not defined.")

    def plot_dynamics_distribution(
        self,
        bins: int | Sequence[float] = 10,
        *,
        absolute: bool = True,
        normalize: bool = False,
    ) -> BarChart:
        """
        Plots the time-weighted distribution of the joint torques/forces.

        Parameters
        ----------
        bins : int | Sequence[float], default = 10
            Number of equally spaced bins, or explicit bin edges.
        absolute : bool, default = True
            If True, use absolute torque/force values. If False, keep signed
            values.
        normalize : bool, default = False
            If True, plot fractions of the total movement time instead of
            seconds.

        Returns
        -------
        BarChart
        """
        distribution = self.dynamics_distribution(
            bins=bins,
            absolute=absolute,
            normalize=normalize,
        )
        lower_edges = distribution[:, 0]
        upper_edges = distribution[:, 1]
        bin_widths = upper_edges - lower_edges
        bin_centers = (lower_edges + upper_edges) / 2.0

        chart = BarChart()
        bar_widths = 0.85 * bin_widths / self.n_joints

        for i in range(self.n_joints):
            offset = (i - (self.n_joints - 1) / 2.0) * bar_widths
            chart.add_xy_data(
                label=f"tau{i+1}",
                x1_values=bin_centers + offset,  # type: ignore
                y1_values=distribution[:, i + 2],
                style_props={
                    "width": bar_widths,
                    "align": "center",
                },
            )

        chart.x1.add_title(
            "absolute joint torque/force" if absolute else "joint torque/force"
        )
        chart.y1.add_title(
            "movement time fraction" if normalize else "movement time, s"
        )
        columns = 1
        if self.n_joints == 2:
            columns = 2
        elif self.n_joints >= 3:
            columns = 3
        chart.add_legend(columns=columns)
        return chart


class _JointSpaceTables:
    """
    Helper class of JointTrajectory.
    """
    def __init__(
        self,
        jss: JointTrajectory,
        angle_unit: str = "deg"
    ) -> None:
        """
        Create formatted table views for a joint-space trajectory.

        Parameters
        ----------
        jss : JointTrajectory
            Joint-space trajectory to render as tables.
        angle_unit : str, default = "deg"
            Unit used for revolute joint angles in textual tables.
        """
        self._jss = jss
        self._angle_unit = angle_unit

    @property
    def angle_unit(self) -> str:
        """
        Return the angle unit used for revolute-joint table values.
        """
        return self._angle_unit

    @angle_unit.setter
    def angle_unit(self, val: AngleUnit) -> None:
        """
        Set the angle unit used for revolute-joint table values.
        """
        self._angle_unit = val

    @property
    def target_coordinates(self) -> str:
        """
        Return target joint coordinates as a formatted table string.
        """
        _coordinates = self._jss.target_coordinates.copy()
        n_cols = _coordinates.shape[1]
        links = self._jss.manipulator.links

        headers = []
        for i in range(n_cols):
            if links[i].is_revolute and self._angle_unit == "deg":
                col = np.rad2deg(_coordinates[:, i])
                _coordinates[:, i] = col
            headers.append(f"q{i+1}")

        table = array_to_table(
            _coordinates,
            headers=headers,
            index=True,
            index_header="frame"
        )
        return table

    @property
    def scheme(self) -> str:
        """
        Return the sampled joint-space trajectory as a formatted table string.
        """
        _scheme = self._jss.scheme.copy()
        links = self._jss.manipulator.links

        headers = ["time"]

        for i in range(1, self._jss.n_joints + 1):
            if links[i-1].is_revolute and self._angle_unit == "deg":
                col = np.rad2deg(_scheme[:, i])
                _scheme[:, i] = col
            headers.append(f"q{i}")

        x = 1
        for i in range(self._jss.n_joints + 1, 2 * self._jss.n_joints + 1):
            headers.append(f"qd{x}")
            x += 1

        x = 1
        for i in range(2 * self._jss.n_joints + 1, 3 * self._jss.n_joints + 1):
            headers.append(f"qdd{x}")
            x += 1

        table = array_to_table(
            _scheme,
            headers=headers,
        )
        return table

    @property
    def dynamics(self) -> str:
        """
        Return sampled joint dynamics as a formatted table string.
        """
        if self._jss.manipulator.has_dynamics():
            tau_arr = self._jss.dynamics()
            headers = ["time"]
            headers.extend([f"tau{i+1}" for i in range(self._jss.n_joints)])
            table = array_to_table(tau_arr, headers=headers)
            return table
        raise ConfigurationError("The manipulator's dynamics is not defined.")

    @property
    def dynamics_distribution(self) -> str:
        """
        Return the joint torque/force distribution as a formatted table string.
        """
        if self._jss.manipulator.has_dynamics():
            distribution = self._jss.dynamics_distribution()
            headers = ["tau_min", "tau_max"]
            headers.extend([f"tau{i+1}, s" for i in range(self._jss.n_joints)])
            return array_to_table(distribution, headers=headers)
        raise ConfigurationError("The manipulator's dynamics is not defined.")

"""
Motion scheme containers for joint-space and Cartesian-space trajectories.

This module wraps sampled motion data in higher-level objects that can convert
between joint space and Cartesian space, expose table views, calculate dynamics,
and plot motion profiles or end-effector trajectories.
"""

from __future__ import annotations
from typing import Sequence, Literal

import numpy as np

from automation_motion.base.types import NumpyArray, AngleUnit
from automation_motion.charts import LineChart, CompositeLineChart, BarChart

from ...base import Frame, SpatialVelocity
from ...manipulator import SerialLinkManipulator, ConfigurationError
from ...visualisation.core import WorldScene
from ...utils import array_to_table
from ...utils.introspection import get_valid_keyword_parameters
from .joint_multi import JointSpaceMotion, MultiPointMotionProfile, MultiPointMotionProfileType, IKTarget
from .cartesian_multi import CartesianMultiStraightLineMotion

__all__ = [
    "JointSpaceScheme",
    "CartesianSpaceScheme"
]


class JointSpaceScheme:
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
        Create a sampled joint-space motion scheme.

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
            Manipulator for which the scheme was generated.
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
    ) -> JointSpaceScheme:
        """
        Creates a JointSpaceScheme object.

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
            Manipulator for which a joint-space scheme is to be created.
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
        jm = JointSpaceMotion(
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

    def to_cartesian_space(self) -> CartesianSpaceScheme:
        """
        Convert this joint-space scheme to Cartesian space.

        Returns
        -------
        CartesianSpaceScheme
            Cartesian scheme obtained by applying forward kinematics at each
            sampled joint configuration.
        """
        return CartesianSpaceScheme.from_joint_space(self)

    @classmethod
    def from_cartesian_space(
        cls,
        css: CartesianSpaceScheme,
        manipulator: SerialLinkManipulator,
        ini_guess: Sequence[float] | None = None,
    ) -> JointSpaceScheme:
        """
        Convert a Cartesian-space scheme to joint space.

        Parameters
        ----------
        css : CartesianSpaceScheme
            Cartesian scheme to convert.
        manipulator : SerialLinkManipulator
            Manipulator used for inverse kinematics and Jacobian mapping.
        ini_guess : Sequence[float], optional
            Initial inverse-kinematics guess for the first trajectory sample.

        Returns
        -------
        JointSpaceScheme
            Joint-space scheme sampled from the Cartesian trajectory.
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
        Returns True if this scheme has analytic joint motion profiles.

        Schemes created directly in joint space have analytic per-joint motion
        profiles. Schemes converted from Cartesian space only have sampled
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
        Return formatted table views for the joint-space scheme.
        """
        return self._tables

    @property
    def time_samples(self) -> NumpyArray:
        """
        Return the sampled time moments of the joint-space scheme.
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


class CartesianSpaceScheme:
    """
    Given a sequence of end-effector poses (frames) in Cartesian space, creates
    a smooth straight-line path between them.
    """
    def __init__(
        self,
        t_arr: NumpyArray,
        traj_frames: list[Frame],
        target_frames: Sequence[Frame],
        dt_segments: Sequence[float],
        p_arr: NumpyArray | None = None,
        V_arr: NumpyArray | None = None,
        A_arr: NumpyArray | None = None,
        target_V_arr: NumpyArray | None = None,
        target_A_arr: NumpyArray | None = None,
    ) -> None:
        """
        Create a sampled Cartesian-space motion scheme.

        Parameters
        ----------
        t_arr : NumpyArray
            Sampled time moments.
        traj_frames : list[Frame]
            Sampled end-effector frames along the trajectory.
        target_frames : Sequence[Frame]
            Original Cartesian target frames.
        dt_segments : Sequence[float]
            Travel durations between successive target frames.
        p_arr : NumpyArray, optional
            Sampled pose vectors ``(x, y, z, rx, ry, rz)``.
        V_arr : NumpyArray, optional
            Sampled spatial velocities.
        A_arr : NumpyArray, optional
            Sampled spatial accelerations.
        target_V_arr : NumpyArray, optional
            Spatial velocities evaluated at the target times.
        target_A_arr : NumpyArray, optional
            Spatial accelerations evaluated at the target times.
        """
        self._t_arr = t_arr
        self._traj_frames = traj_frames

        self._target_frames = list(target_frames) if target_frames is not None else []
        self.dt_segments = list(dt_segments) if dt_segments is not None else []

        self._p_arr = p_arr
        self._V_arr = V_arr
        self._A_arr = A_arr
        self._target_V_arr = target_V_arr
        self._target_A_arr = target_A_arr

        self._traj_viewer = _CartesianTrajectoryPlotter(self)
        self._tables = _CartesianSpaceTables(self)

    @classmethod
    def create(
        cls,
        target_frames: Sequence[Frame],
        dt_segments: Sequence[float],
        dt_blends: float | Sequence[float] = 0.1,
        num_t_samples: int = 100
    ) -> CartesianSpaceScheme:
        """
        Creates a smooth Cartesian straight-line path through multiple 3D poses 
        of a frame.

        Parameters
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

        Returns
        -------
        CartesianSpaceScheme
        """
        cm = CartesianMultiStraightLineMotion(
            target_frames, dt_segments,
            dt_blends, num_t_samples
        )
        t_arr, traj_frames = cm.trajectory()
        _, p_arr, V_arr, A_arr = cm.motion_samples

        target_times = np.concatenate(([0.0], np.cumsum(dt_segments)))
        target_V_arr = np.array([
            cm.motion_profile.spatial_velocity(t)
            for t in target_times
        ])
        target_A_arr = np.array([
            cm.motion_profile.acceleration(t)
            for t in target_times
        ])

        return cls(
            t_arr=t_arr,
            traj_frames=traj_frames,
            target_frames=target_frames,
            dt_segments=dt_segments,
            p_arr=p_arr,
            V_arr=V_arr,
            A_arr=A_arr,
            target_V_arr=target_V_arr,
            target_A_arr=target_A_arr,
        )

    def to_joint_space(
        self, 
        manipulator: SerialLinkManipulator,
        ini_guess: Sequence[float] | None = None,
    ) -> JointSpaceScheme:
        """
        Converts the Cartesian-space scheme to a joint-space scheme.

        Parameters
        ----------
        manipulator: SerialLinkManipulator
            Manipulator for which a joint-space scheme is to be created.
        ini_guess: Sequence[float] | None, default = None
            Initial joint coordinates for the inverse kinematics of the first
            target frame. If None, the current joint coordinates of the
            manipulator are used. Each following target frame uses the previous
            inverse-kinematics solution as its initial guess.
        """
        if self._V_arr is not None and self._A_arr is not None:
            converter = _CartesianToJointSpaceConverter(
                css=self,
                manipulator=manipulator,
                ini_guess=ini_guess,
            )
            return converter.convert()
        raise ValueError(
            "Joint-space scheme cannot be created. "
            "Spatial velocity and acceleration are not available."
        )

    @classmethod
    def from_joint_space(cls, jss: JointSpaceScheme) -> CartesianSpaceScheme:
        """
        Converts a joint-space scheme to Cartesian space through application
        of the forward kinematics of the manipulator.
        """
        frames = [jss.manipulator.fwd_kin(row) for row in jss.positions]
        p_arr = np.array([frame.to_pose_vector() for frame in frames])
        jacobians = [jss.manipulator.jacobian(row) for row in jss.positions]
        V_arr = np.array([J @ qd for J, qd in zip(jacobians, jss.velocities)])

        if len(V_arr[0]) == 3:  # planar motion [v_x, v_y, w_z]
            V_arr = np.array([[V[0], V[1], 0.0, 0.0, 0.0, V[2]] for V in V_arr])

        return cls(
            t_arr=jss.time_samples,
            traj_frames=frames,
            target_frames=jss.target_frames,
            dt_segments=jss.dt_segments,
            p_arr=p_arr,
            V_arr=V_arr,
        )

    @property
    def scheme(self) -> NumpyArray:
        """
        Returns a "timetable" of the trajectory in Cartesian space.

        Returns
        -------
        NumpyArray
            2D-array: The first column of the array contains the sampled time
            moments. The following three columns contain the coordinates of the
            origin of each trajectory frame. The last three columns contain the
            orientation angles about the x-axis, y-axis, and z-axis of the
            trajectory frames.
        """
        xyz_coords = np.array([
            np.asarray(fr.origin, dtype=float)
            for fr in self._traj_frames]
        )
        xyz_angles = np.array([
            np.asarray(fr.rpy_angles, dtype=float)
            for fr in self._traj_frames]
        )
        scheme = np.column_stack([self._t_arr, xyz_coords, xyz_angles])
        return scheme

    @property
    def tables(self) -> _CartesianSpaceTables:
        """
        Return formatted table views for the Cartesian-space scheme.
        """
        return self._tables

    @property
    def trajectory_frames(self) -> list[Frame]:
        """
        Return the sampled end-effector frames along the Cartesian trajectory.

        Returns
        -------
        list[Frame]
            Frames sampled at ``time_samples``.
        """
        return self._traj_frames

    @property
    def trajectory_points(self) -> NumpyArray:
        """
        Returns the sampled end-effector origins along the Cartesian trajectory.

        Returns
        -------
        NumpyArray
            2D array with shape (n_samples, 3).
        """
        return np.array([
            np.asarray(frame.origin, dtype=float)
            for frame in self._traj_frames
        ])

    @property
    def target_frames(self) -> list[Frame]:
        """
        Returns the original target frames used to create the joint motion
        scheme.
        """
        return self._target_frames

    @property
    def target_points(self) -> NumpyArray:
        """
        Returns the origins of the original target frames.

        Returns
        -------
        NumpyArray
            2D array with shape (n_target_frames, 3).
        """
        return np.array([
            np.asarray(frame.origin, dtype=float)
            for frame in self._target_frames
        ])

    @property
    def time_samples(self) -> NumpyArray:
        """
        Return the sampled time moments of the Cartesian-space scheme.
        """
        return self._t_arr

    @property
    def poses(self) -> NumpyArray:
        """
        Return the sampled Cartesian pose vectors.

        Returns
        -------
        NumpyArray
            Array with columns ``x, y, z, rx, ry, rz``.
        """
        if self._p_arr is not None:
            return self._p_arr
        raise ValueError("Time paths are not available.")

    def spatial_velocities(
        self,
        ref_frame: Literal["world", "end-effector"] = "world"
    ) -> NumpyArray:
        """
        Returns the array of sampled spatial velocities V.

        Parameters
        ----------
        ref_frame : Literal["world", "end-effector"], default = "world"
            Indicates the reference frame in which the spatial velocities are
            observed. By default, this reference frame is het world frame. If
            ref_frame is "end-effector", spatial velocities are transformed to
            the frame of the end-effector.
        """
        def transform(frame_0E: Frame, VE_0: NumpyArray) -> NumpyArray:
            frame_E0 = ~frame_0E
            VE_E = frame_E0.transform(SpatialVelocity(VE_0), is_frame=False)
            return np.asarray(VE_E, dtype=float)

        if self._V_arr is not None:
            if ref_frame == "world":
                return self._V_arr
            if ref_frame == "end-effector":
                V_arr = np.array([
                    transform(frame, V)
                    for frame, V
                    in zip(self.trajectory_frames, self._V_arr)
                ])
                return V_arr
        raise ValueError(
            f"Spatial velocities are not available in "
            f"reference frame {ref_frame}."
        )

    @property
    def spatial_accelerations(self) -> NumpyArray:
        """
        Return the sampled spatial accelerations.

        Returns
        -------
        NumpyArray
            Array with columns ``a_x, a_y, a_z, alpha_x, alpha_y, alpha_z``.
        """
        if self._A_arr is not None:
            return self._A_arr
        raise ValueError("Spatial accelerations are not available.")

    @staticmethod
    def _pose_vectors_from_frames(frames: Sequence[Frame]) -> NumpyArray:
        return CartesianMultiStraightLineMotion._frames_to_pose_vectors(frames)

    def plot_poses(self, show_targets: bool = False) -> CompositeLineChart:
        """
        Plots the Cartesian pose paths p(t) of the end-effector frame and
        returns the LineChart object. Call show() on this object to see the
        plot.
        """
        labels = ("x", "y", "z", "rx", "ry", "rz")
        p_arr = self.poses

        target_times = np.concatenate(([0.0], np.cumsum(self.dt_segments)))
        target_p_arr = self._pose_vectors_from_frames(self._target_frames)
        has_targets = len(target_times) == len(target_p_arr)

        parent = self
        class PlotTimePaths(CompositeLineChart):
            def add_data(self) -> None:
                for i, label in enumerate(labels):
                    if i < 3:
                        chart = self.top_chart
                    else:
                        chart = self.bottom_chart

                    chart.add_xy_data(
                        label=label,
                        x1_values=parent._t_arr,
                        y1_values=p_arr[:, i],
                    )
                    if has_targets and show_targets:
                        chart.add_xy_data(
                            label=f"{label}, targets",
                            x1_values=target_times,
                            y1_values=target_p_arr[:, i],
                            style_props={
                                "marker": "o",
                                "linestyle": "none",
                            },
                        )
                self.bottom_chart.x1.add_title("time, s")
                self.top_chart.y1.add_title("linear position")
                self.bottom_chart.y1.add_title("angular position")
                self.top_chart.add_legend(anchor="lower center", position=(0.5, 1.05), columns=3)
                self.bottom_chart.add_legend(anchor="upper center", position=(0.5, -0.25), columns=3)

        return PlotTimePaths()

    def plot_spatial_velocities(
        self,
        ref_frame: Literal["world", "end-effector"] = "world",
        show_targets: bool = False
    ) -> CompositeLineChart:
        """
        Plots the spatial velocity paths V(t) of the end-effector frame and
        returns the LineChart object. Call show() on this object to see the
        plot.

        Parameters
        ----------
        ref_frame : Literal["world", "end-effector"], default = "world"
            Indicates the reference frame in which the spatial velocities are
            observed. By default, this reference frame is het world frame. If
            ref_frame is "end-effector", spatial velocities are transformed to
            the frame of the end-effector.
        show_targets : bool, default = False
            If True, the spatial velocities of the initial target frames used
            to construct the trajectory are indicated by dot markers on the
            line charts.
        """
        labels = ("v_x", "v_y", "v_z", "w_x", "w_y", "w_z")
        V_arr = self.spatial_velocities(ref_frame)
        target_times = np.concatenate(([0.0], np.cumsum(self.dt_segments)))
        has_targets = (
            self._target_V_arr is not None
            and len(target_times) == len(self._target_V_arr)
        )

        parent = self
        class PlotSpatialVelocities(CompositeLineChart):
            def add_data(self) -> None:
                for i, label in enumerate(labels):
                    if i < 3:
                        chart = self.top_chart
                    else:
                        chart = self.bottom_chart

                    chart.add_xy_data(
                        label=label,
                        x1_values=parent._t_arr,
                        y1_values=V_arr[:, i],
                    )
                    if has_targets and show_targets:
                        # noinspection PyUnresolvedReferences
                        chart.add_xy_data(
                            label=f"{label}, targets",
                            x1_values=target_times,
                            y1_values=parent._target_V_arr[:, i],
                            style_props={
                                "marker": "o",
                                "linestyle": "none",
                            },
                        )
                self.bottom_chart.x1.add_title("time, s")
                self.top_chart.y1.add_title("linear velocity")
                self.bottom_chart.y1.add_title("angular velocity")
                self.top_chart.add_legend(anchor="lower center", position=(0.5, 1.05), columns=3)
                self.bottom_chart.add_legend(anchor="upper center", position=(0.5, -0.25), columns=3)

        return PlotSpatialVelocities()

    def plot_spatial_accelerations(self, show_targets: bool = False) -> CompositeLineChart:
        """
        Plots the spatial acceleration paths A(t) of the end-effector frame and
        returns the LineChart object. Call show() on this object to see the
        plot.
        """
        labels = ("a_x", "a_y", "a_z", "alpha_x", "alpha_y", "alpha_z")
        A_arr = self.spatial_accelerations
        target_times = np.concatenate(([0.0], np.cumsum(self.dt_segments)))
        has_targets = (
            self._target_A_arr is not None
            and len(target_times) == len(self._target_A_arr)
        )

        parent = self
        class PlotSpatialAccelerations(CompositeLineChart):
            def add_data(self) -> None:
                for i, label in enumerate(labels):
                    if i < 3:
                        chart = self.top_chart
                    else:
                        chart = self.bottom_chart

                    chart.add_xy_data(
                        label=label,
                        x1_values=parent._t_arr,
                        y1_values=A_arr[:, i],
                    )
                    if has_targets and show_targets:
                        # noinspection PyUnresolvedReferences
                        chart.add_xy_data(
                            label=f"{label}, targets",
                            x1_values=target_times,
                            y1_values=parent._target_A_arr[:, i],
                            style_props={
                                "marker": "o",
                                "linestyle": "none",
                            },
                        )

                self.bottom_chart.x1.add_title("time, s")
                self.top_chart.y1.add_title("linear acceleration")
                self.bottom_chart.y1.add_title("angular acceleration")
                self.top_chart.add_legend(anchor="lower center", position=(0.5, 1.05), columns=3)
                self.bottom_chart.add_legend(anchor="upper center", position=(0.5, -0.25), columns=3)

        return PlotSpatialAccelerations()

    def plot_trajectory(
        self,
        show_points: bool = False,
        show_path: bool = True,
        show_frames: bool = False,
        show_target_path: bool = True,
        show_target_points: bool = False,
        show_target_frames: bool = False,
        point_step: int = 1,
        frame_step: int = 10,
        point_color: str = "black",
        path_color: str = "orange",
        target_point_color: str = "blue",
        target_path_color: str = "blue",
        point_size: float = 8.0,
        target_point_size: float = 10.0,
        path_width: float = 2.0,
        target_path_width: float = 2.0,
        frame_scale: float = 0.2,
        target_frame_scale: float = 0.2,
        **kwargs
    ) -> None:
        """
        Plot the Cartesian end-effector trajectory in 3D space.

        Parameters
        ----------
        show_points : bool, default=True
            If True, draw sampled end-effector positions as point markers.
        show_path : bool, default=True
            If True, draw line segments between successive trajectory points.
        show_frames : bool, default=False
            If True, draw some sampled end-effector frames along the trajectory.
        show_target_path : bool, default=True
            If True, draw line segments between successive target points.
        show_target_points : bool, default=True
            If True, draw end-effector target positions as point markers.
        show_target_frames : bool, default=True
            If True, draw target end-effector frames along the trajectory.
        point_step : int, default=1
            Draw every `point_step`-th point.
        frame_step : int, default=10
            Draw every `frame_step`-th frame if `show_frames` is True.
        point_color : str, default="black"
            Color of the trajectory points.
        path_color : str, default="orange"
            Color of the trajectory path.
        target_point_color : str, default="blue"
            Color of the target points.
        target_path_color : str, default="blue"
            Color of the target path.
        point_size : float, default=8.0
            Size of the point markers.
        target_point_size : float, default=14.0
            Size of the target point markers.
        path_width : float, default=3.0
            Width of the trajectory path.
        target_path_width : float, default=2.0
            Width of the target path.
        frame_scale : float, default=0.2
            Scale of the optional end-effector frames.
        target_frame_scale : float, default=0.25
            Scale of the optional end-effector target frames.
        **kwargs
            Additional keyword arguments passed to WorldScene.

        Returns
        -------
        None
        """
        self._traj_viewer.plot_trajectory(
            show_points, show_path, show_frames, show_target_path,
            show_target_points, show_target_frames, point_step,
            frame_step, point_color, path_color, target_point_color,
            target_path_color, point_size, target_point_size,
            path_width, target_path_width, frame_scale,
            target_frame_scale, **kwargs
        )

    async def plot_trajectory_async(
        self,
        show_points: bool = True,
        show_path: bool = True,
        show_frames: bool = False,
        show_target_path: bool = True,
        show_target_points: bool = True,
        show_target_frames: bool = True,
        point_step: int = 1,
        frame_step: int = 10,
        point_color: str = "black",
        path_color: str = "orange",
        target_point_color: str = "blue",
        target_path_color: str = "blue",
        point_size: float = 8.0,
        target_point_size: float = 14.0,
        path_width: float = 3.0,
        target_path_width: float = 2.0,
        frame_scale: float = 0.2,
        target_frame_scale: float = 0.25,
        **kwargs
    ) -> None:
        """
        Plot the Cartesian end-effector trajectory asynchronously.

        This is mainly useful in Jupyter notebooks.

        Parameters
        ----------
        show_points : bool, default=True
            If True, draw sampled end-effector positions as point markers.
        show_path : bool, default=True
            If True, draw line segments between successive trajectory points.
        show_frames : bool, default=False
            If True, draw some sampled end-effector frames along the trajectory.
        show_target_path : bool, default=True
            If True, draw line segments between successive target points.
        show_target_points : bool, default=True
            If True, draw end-effector target positions as point markers.
        show_target_frames : bool, default=True
            If True, draw target end-effector frames along the trajectory.
        point_step : int, default=1
            Draw every `point_step`-th point.
        frame_step : int, default=10
            Draw every `frame_step`-th frame if `show_frames` is True.
        point_color : str, default="black"
            Color of the trajectory points.
        path_color : str, default="orange"
            Color of the trajectory path.
        target_point_color : str, default="blue"
            Color of the target points.
        target_path_color : str, default="blue"
            Color of the target path.
        point_size : float, default=8.0
            Size of the point markers.
        target_point_size : float, default=14.0
            Size of the target point markers.
        path_width : float, default=3.0
            Width of the trajectory path.
        target_path_width : float, default=2.0
            Width of the target path.
        frame_scale : float, default=0.2
            Scale of the optional end-effector frames.
        target_frame_scale : float, default=0.25
            Scale of the optional end-effector target frames.
        **kwargs
            Additional keyword arguments passed to WorldScene.
        """
        await self._traj_viewer.plot_trajectory_async(
            show_points, show_path, show_frames, show_target_path,
            show_target_points, show_target_frames, point_step,
            frame_step, point_color, path_color, target_point_color,
            target_path_color, point_size, target_point_size,
            path_width, target_path_width, frame_scale,
            target_frame_scale, **kwargs
        )


class _CartesianToJointSpaceConverter:
    """
    Helper class of CartesianSpaceScheme to convert Cartesian trajectory
    samples to joint-space trajectory samples.
    """
    def __init__(
        self,
        css: CartesianSpaceScheme,
        manipulator: SerialLinkManipulator,
        ini_guess: Sequence[float] | None = None,
    ) -> None:
        """
        Create a converter from Cartesian trajectory samples to joint samples.

        Parameters
        ----------
        css : CartesianSpaceScheme
            Cartesian scheme to convert.
        manipulator : SerialLinkManipulator
            Manipulator used for inverse kinematics and Jacobian mapping.
        ini_guess : Sequence[float], optional
            Initial inverse-kinematics guess for the first trajectory frame.
        """
        self._css = css
        self._manipulator = manipulator
        self._ini_guess = ini_guess

    def _trajectory_joint_coords(self) -> NumpyArray:
        q_list = []
        q_guess = (
            self._ini_guess
            if self._ini_guess is not None
            else self._manipulator.joint_coords
        )
        for frame in self._css._traj_frames:
            q = self._manipulator.inv_kin(frame, ini_guess=q_guess)
            q_list.append(np.asarray(q, dtype=float))
            q_guess = q

        return np.array(q_list, dtype=float)

    def _target_joint_coords(self) -> NumpyArray:
        q_sets = []
        q_guess = (
            self._ini_guess
            if self._ini_guess is not None
            else self._manipulator.joint_coords
        )
        for frame in self._css._target_frames:
            q = self._manipulator.inv_kin(frame, ini_guess=q_guess)
            q_sets.append(np.asarray(q, dtype=float))
            q_guess = q

        if not q_sets:
            return np.empty((0, len(self._manipulator)))
        return np.array(q_sets, dtype=float)

    @staticmethod
    def _reduce_to_jacobian_space(jacobian: NumpyArray, vector: NumpyArray) -> NumpyArray:
        if jacobian.shape[0] == vector.shape[0]:
            return vector
        if jacobian.shape[0] == 3 and vector.shape[0] == 6:
            return vector[[0, 1, 5]]
        raise ConfigurationError(
            f"Cannot map a {vector.shape[0]}D Cartesian vector to a "
            f"{jacobian.shape[0]}D manipulator Jacobian."
        )

    def _joint_velocity(self, q: NumpyArray, V: NumpyArray) -> NumpyArray:
        J = self._manipulator.jacobian(q)
        V_reduced = self._reduce_to_jacobian_space(J, V)
        return self._manipulator.jacobian_pinv(q) @ V_reduced

    def _joint_acceleration(
        self,
        q: NumpyArray,
        qd: NumpyArray,
        A: NumpyArray,
    ) -> NumpyArray:
        J = self._manipulator.jacobian(q)
        A_reduced = self._reduce_to_jacobian_space(J, A)
        J_dot = self._manipulator.jacobian_dot(q, qd)
        return self._manipulator.jacobian_pinv(q) @ (A_reduced - J_dot @ qd)

    def convert(self) -> JointSpaceScheme:
        """
        Converts the Cartesian motion of the end-effector frame to the
        corresponding motion of the manipulator's joints.

        Returns
        -------
        JointSpaceScheme
        """
        # Get the joint coordinates that correspond with the end-effector frames
        # along the Cartesian trajectory (= the trajectory frames).
        q_arr = self._trajectory_joint_coords()

        # Get the spatial velocity and spatial acceleration of the trajectory
        # frames.
        if self._css._V_arr is not None and self._css._A_arr is not None:
            V_arr = self._css._V_arr
            A_arr = self._css._A_arr
        else:
            raise ValueError(
                "Joint-space scheme cannot be created. "
                "Spatial velocity and acceleration are not available."
            )

        # Given the spatial velocities of the trajectory frames, calculate the
        # corresponding velocities of the manipulator's joints.
        qd_arr = np.array([
            self._joint_velocity(q, V)
            for q, V in zip(q_arr, V_arr)
        ])

        # Given the spatial accelerations of the trajectory frames, calculate
        # the corresponding accelerations of the manipulator's joints.
        qdd_arr = np.array([
            self._joint_acceleration(q, qd, A)
            for q, qd, A in zip(q_arr, qd_arr, A_arr)
        ])

        return JointSpaceScheme(
            self._css._t_arr, q_arr, qd_arr, qdd_arr,
            target_frames=self._css._target_frames,
            dt_segments=self._css.dt_segments,
            manipulator=self._manipulator,
            q_sets=self._target_joint_coords(),
            motion_profiles=[],
        )


class _CartesianTrajectoryPlotter:
    """
    Helper class of CartesianMotionScheme to plot a Cartesian trajectory.
    """
    def __init__(self, cms: CartesianSpaceScheme) -> None:
        """
        Create a trajectory plotter for a Cartesian-space scheme.

        Parameters
        ----------
        cms : CartesianSpaceScheme
            Cartesian scheme whose trajectory should be plotted.
        """
        self._cms = cms

    @staticmethod
    def _create_scene(**kwargs) -> WorldScene:
        world_frame_scale = kwargs.pop("world_frame_scale", 1.0)

        scene_params = get_valid_keyword_parameters(WorldScene.__init__, exclude={"self"})
        scene_kwargs = {k: v for k, v in kwargs.items() if k in scene_params}

        scene = WorldScene(**scene_kwargs)
        scene.camera.enable_view_shortcuts()
        scene.add_plane_grid()
        scene.add_world_frame(frame_scale=world_frame_scale)

        return scene

    @staticmethod
    def _add_path_to_scene(
        scene: WorldScene,
        points: NumpyArray,
        color: str,
        line_width: float,
    ) -> None:
        """
        Add a path to the scene as a sequence of line segments.
        """
        if len(points) < 2:
            return

        for p1, p2 in zip(points[:-1], points[1:]):
            scene.add_link(
                p1=p1,
                p2=p2,
                color=color,
                line_width=line_width,
            )

    def _plot_trajectory(
        self,
        show_points: bool = True,
        show_path: bool = True,
        show_frames: bool = False,
        show_target_path: bool = True,
        show_target_points: bool = True,
        show_target_frames: bool = True,
        point_step: int = 1,
        frame_step: int = 10,
        point_color: str = "black",
        path_color: str = "orange",
        target_point_color: str = "blue",
        target_path_color: str = "blue",
        point_size: float = 8.0,
        target_point_size: float = 14.0,
        path_width: float = 3.0,
        target_path_width: float = 2.0,
        frame_scale: float = 0.2,
        target_frame_scale: float = 0.25,
        **kwargs
    ) -> WorldScene:
        """
        Creates a 3D scene containing the Cartesian trajectory.

        The actual trajectory is the Cartesian path obtained by applying forward
        kinematics to the sampled joint-space scheme. The target path is the
        polyline through the original target frame origins.

        Parameters
        ----------
        show_points : bool, default=True
            If True, draw sampled end-effector positions as point markers.
        show_path : bool, default=True
            If True, draw line segments between successive trajectory points.
        show_frames : bool, default=False
            If True, draw some sampled end-effector frames along the trajectory.
        show_target_path : bool, default=True
            If True, draw line segments between successive target points.
        show_target_points : bool, default=True
            If True, draw end-effector target positions as point markers.
        show_target_frames : bool, default=True
            If True, draw target end-effector frames along the trajectory.
        point_step : int, default=1
            Draw every `point_step`-th point.
        frame_step : int, default=10
            Draw every `frame_step`-th frame if `show_frames` is True.
        point_color : str, default="black"
            Color of the trajectory points.
        path_color : str, default="orange"
            Color of the trajectory path.
        target_point_color : str, default="blue"
            Color of the target points.
        target_path_color : str, default="blue"
            Color of the target path.
        point_size : float, default=8.0
            Size of the point markers.
        target_point_size : float, default=14.0
            Size of the target point markers.
        path_width : float, default=3.0
            Width of the trajectory path.
        target_path_width : float, default=2.0
            Width of the target path.
        frame_scale : float, default=0.2
            Scale of the optional end-effector frames.
        target_frame_scale : float, default=0.25
            Scale of the optional end-effector target frames.
        **kwargs
            Additional keyword arguments passed to WorldScene.

        Returns
        -------
        WorldScene
            Scene containing the plotted trajectory.
        """
        if point_step < 1:
            raise ValueError("point_step must be at least 1.")

        if frame_step < 1:
            raise ValueError("frame_step must be at least 1.")

        scene = self._create_scene(**kwargs)

        points = self._cms.trajectory_points
        target_points = self._cms.target_points if self._cms._target_frames else np.empty((0, 3))

        # Original target path
        if show_target_path and len(target_points) >= 2:
            self._add_path_to_scene(
                scene=scene,
                points=target_points,
                color=target_path_color,
                line_width=target_path_width,
            )

        # Original target points
        if show_target_points:
            for i, point in enumerate(target_points):
                scene.add_point(
                    point=point,
                    color=target_point_color,
                    size=target_point_size,
                    name=f"T{i}",
                )

        # Original target frames
        if show_target_frames:
            for i, frame in enumerate(self._cms._target_frames):
                frame.name = f"T{i}"
                scene.add_frame(
                    frame=frame,
                    frame_scale=target_frame_scale,
                    line_width=2.0,
                    show_label=True,
                )

        # Actual Cartesian trajectory
        if show_path and len(points) >= 2:
            self._add_path_to_scene(
                scene=scene,
                points=points,
                color=path_color,
                line_width=path_width,
            )

        # Sampled actual trajectory points
        if show_points:
            for point in points[::point_step]:
                scene.add_point(
                    point=point,
                    color=point_color,
                    size=point_size,
                    name=None,
                )

        # Optional sampled actual end-effector frames
        if show_frames:
            for frame in self._cms._traj_frames[::frame_step]:
                scene.add_frame(
                    frame=frame,
                    frame_scale=frame_scale,
                    show_label=False,
                )

        return scene

    def plot_trajectory(
        self,
        show_points: bool = False,
        show_path: bool = True,
        show_frames: bool = False,
        show_target_path: bool = True,
        show_target_points: bool = False,
        show_target_frames: bool = False,
        point_step: int = 1,
        frame_step: int = 10,
        point_color: str = "black",
        path_color: str = "orange",
        target_point_color: str = "blue",
        target_path_color: str = "blue",
        point_size: float = 8.0,
        target_point_size: float = 10.0,
        path_width: float = 2.0,
        target_path_width: float = 2.0,
        frame_scale: float = 0.2,
        target_frame_scale: float = 0.2,
        **kwargs
    ) -> None:
        """
        Plot the Cartesian end-effector trajectory in 3D space.

        Parameters
        ----------
        show_points : bool, default=True
            If True, draw sampled end-effector positions as point markers.
        show_path : bool, default=True
            If True, draw line segments between successive trajectory points.
        show_frames : bool, default=False
            If True, draw some sampled end-effector frames along the trajectory.
        show_target_path : bool, default=True
            If True, draw line segments between successive target points.
        show_target_points : bool, default=True
            If True, draw end-effector target positions as point markers.
        show_target_frames : bool, default=True
            If True, draw target end-effector frames along the trajectory.
        point_step : int, default=1
            Draw every `point_step`-th point.
        frame_step : int, default=10
            Draw every `frame_step`-th frame if `show_frames` is True.
        point_color : str, default="black"
            Color of the trajectory points.
        path_color : str, default="orange"
            Color of the trajectory path.
        target_point_color : str, default="blue"
            Color of the target points.
        target_path_color : str, default="blue"
            Color of the target path.
        point_size : float, default=8.0
            Size of the point markers.
        target_point_size : float, default=14.0
            Size of the target point markers.
        path_width : float, default=3.0
            Width of the trajectory path.
        target_path_width : float, default=2.0
            Width of the target path.
        frame_scale : float, default=0.2
            Scale of the optional end-effector frames.
        target_frame_scale : float, default=0.25
            Scale of the optional end-effector target frames.
        **kwargs
            Additional keyword arguments passed to WorldScene.

        Returns
        -------
        WorldScene
            Scene containing the plotted trajectory.
        """
        scene = self._plot_trajectory(
            show_points=show_points,
            show_path=show_path,
            show_frames=show_frames,
            show_target_path=show_target_path,
            show_target_points=show_target_points,
            show_target_frames=show_target_frames,
            point_step=point_step,
            frame_step=frame_step,
            point_color=point_color,
            path_color=path_color,
            target_point_color=target_point_color,
            target_path_color=target_path_color,
            point_size=point_size,
            target_point_size=target_point_size,
            path_width=path_width,
            target_path_width=target_path_width,
            frame_scale=frame_scale,
            target_frame_scale=target_frame_scale,
            **kwargs,
        )
        scene.show()

    async def plot_trajectory_async(
        self,
        show_points: bool = True,
        show_path: bool = True,
        show_frames: bool = False,
        show_target_path: bool = True,
        show_target_points: bool = True,
        show_target_frames: bool = True,
        point_step: int = 1,
        frame_step: int = 10,
        point_color: str = "black",
        path_color: str = "orange",
        target_point_color: str = "blue",
        target_path_color: str = "blue",
        point_size: float = 8.0,
        target_point_size: float = 14.0,
        path_width: float = 3.0,
        target_path_width: float = 2.0,
        frame_scale: float = 0.2,
        target_frame_scale: float = 0.25,
        jupyter_backend: Literal["server", "client", "trame"] = "client",
        **kwargs
    ) -> None:
        """
        Plot the Cartesian end-effector trajectory asynchronously.

        This is mainly useful in Jupyter notebooks.
        """
        scene = self._plot_trajectory(
            show_points=show_points,
            show_path=show_path,
            show_frames=show_frames,
            show_target_path=show_target_path,
            show_target_points=show_target_points,
            show_target_frames=show_target_frames,
            point_step=point_step,
            frame_step=frame_step,
            point_color=point_color,
            path_color=path_color,
            target_point_color=target_point_color,
            target_path_color=target_path_color,
            point_size=point_size,
            target_point_size=target_point_size,
            path_width=path_width,
            target_path_width=target_path_width,
            frame_scale=frame_scale,
            target_frame_scale=target_frame_scale,
            **kwargs,
        )
        await scene.show_async(jupyter_backend)


class _JointSpaceTables:
    """
    Helper class of JointSpaceScheme.
    """
    def __init__(
        self,
        jss: JointSpaceScheme,
        angle_unit: str = "deg"
    ) -> None:
        """
        Create formatted table views for a joint-space scheme.

        Parameters
        ----------
        jss : JointSpaceScheme
            Joint-space scheme to render as tables.
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
        Return the sampled joint-space scheme as a formatted table string.
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


class _CartesianSpaceTables:

    def __init__(
        self,
        css: CartesianSpaceScheme,
    ) -> None:
        """
        Create formatted table views for a Cartesian-space scheme.

        Parameters
        ----------
        css : CartesianSpaceScheme
            Cartesian-space scheme to render as tables.
        """
        self._css = css

    @property
    def scheme(self) -> str:
        """
        Return the sampled Cartesian-space scheme as a formatted table string.
        """
        _scheme = self._css.scheme.copy()
        headers = ["time", "x", "y", "z", "alpha", "beta", "gamma"]
        table = array_to_table(_scheme, headers=headers)
        return table

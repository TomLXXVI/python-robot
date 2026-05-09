from __future__ import annotations
from typing import Sequence

import numpy as np

from ...base.types import NumpyArray, AngleUnit
from ...base import Frame
from ...manipulator import SerialLinkManipulator, ConfigurationError
from ...charts import LineChart, CompositeLineChart
from ...visualisation import WorldScene
from ...utils import array_to_table
from ...utils.introspection import get_valid_keyword_parameters
from .joint_multi import JointSpaceMotion, MultiPointMotionProfile, MultiPointMotionProfileType
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
        motion_paths: Sequence[MultiPointMotionProfile],
        angle_unit: AngleUnit = "deg",
    ) -> None:
        self._t_arr = t_arr
        self._q_arr = q_arr
        self._qd_arr = qd_arr
        self._qdd_arr = qdd_arr

        self.target_frames = target_frames
        self.dt_segments = dt_segments
        self.manipulator = manipulator
        self._q_sets = q_sets
        self._motion_paths = motion_paths

        self.n_joints = len(self.manipulator)
        self.n_segments = len(self.dt_segments)
        
        self._tables = _JointMotionTables(self, angle_unit)

    @classmethod
    def create(
        cls,
        target_frames: Sequence[Frame],
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
        target_frames: Sequence[Frame]
            List of end-effector frames (positions and orientations) in
            Cartesian space.
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
            target_frames, dt_segments, manipulator,
            mp_type, blend_accels, num_t_samples,
            ini_guess
        )
        t_arr, q_arr, qd_arr, qdd_arr = jm.motion_samples
        q_sets = jm.target_coordinates
        motion_paths = jm.motion_profiles
        return cls(
            t_arr, q_arr, qd_arr, qdd_arr,
            target_frames=target_frames,
            dt_segments=dt_segments,
            manipulator=manipulator,
            q_sets=q_sets,
            motion_paths=motion_paths,
            angle_unit=angle_unit,
        )

    def to_cartesian_space(self) -> CartesianSpaceScheme:
        """
        Converts the joint-space scheme to Cartesian space.
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
        Converts the Cartesian-space scheme to joint space.
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
        return self._motion_paths

    @property
    def has_motion_profiles(self) -> bool:
        """
        Returns True if this scheme has analytic joint motion profiles.

        Schemes created directly in joint space have analytic per-joint motion
        profiles. Schemes converted from Cartesian space only have sampled
        joint positions, velocities, and accelerations.
        """
        return len(self._motion_paths) > 0

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
    def tables(self) -> _JointMotionTables:
        return self._tables

    @property
    def time_samples(self) -> NumpyArray:
        """Returns the array of sampled time moments."""
        return self._t_arr

    @property
    def time_paths(self) -> NumpyArray:
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
        Returns the joint torques/forces that cause the manipulator's motion.

        Returns
        -------
        NumpyArray
            2D array: The first column of the array contains the sampled time
            moments. The following columns contain the torques/forces to be
            applied to the joints.
        """
        if self.manipulator.has_dynamics():
            tau_arr = np.array([
                self.manipulator.inv_dyn(q, qd, qdd)
                for q, qd, qdd in zip(self._q_arr, self._qd_arr, self._qdd_arr)
            ])
            return np.column_stack((self._t_arr, tau_arr))
        raise ConfigurationError("The manipulator's dynamics is not defined.")

    def plot_time_paths(self) -> LineChart:
        """
        Plots the position paths q(t) of the joints and returns the LineChart
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

    def plot_velocities(self) -> LineChart:
        """
        Plots the velocity paths qd(t) of the joints and returns the LineChart
        object. Call show() on this object to see the plot.

        Returns
        -------
        LineChart
        """
        if self.has_motion_profiles:
            target_times = np.concatenate(([0.0], np.cumsum(self.dt_segments)))
            target_qd_arr = np.array([
                [mp.velocity(t) for mp in self._motion_paths]
                for t in target_times
            ])

        chart = LineChart()
        for i in range(self.n_joints):
            chart.add_xy_data(
                label=f"qd{i+1}",
                x1_values=self._t_arr,
                y1_values=self._qd_arr[:, i],
            )
            if self.has_motion_profiles:
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

    def plot_accelerations(self) -> LineChart:
        """
        Plots the acceleration paths qdd(t) of the joints and returns the
        LineChart object. Call show() on this object to see the plot.

        Returns
        -------
        LineChart
        """
        if self.has_motion_profiles:
            target_times = np.concatenate(([0.0], np.cumsum(self.dt_segments)))
            target_qdd_arr = np.array([
                [mp.acceleration(t) for mp in self._motion_paths]
                for t in target_times
            ])

        chart = LineChart()
        for i in range(self.n_joints):
            chart.add_xy_data(
                label=f"qdd{i + 1}",
                x1_values=self._t_arr,
                y1_values=self._qdd_arr[:, i],
            )
            if self.has_motion_profiles:
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
        self._tables = _CartesianMotionTables(self)

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
        frames = [jss.manipulator.fwd_kin(row) for row in jss.time_paths]
        return cls(
            t_arr=jss.time_samples,
            traj_frames=frames,
            target_frames=jss.target_frames,
            dt_segments=jss.dt_segments,
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
            np.asarray(fr.orient_angles, dtype=float)
            for fr in self._traj_frames]
        )
        scheme = np.column_stack([self._t_arr, xyz_coords, xyz_angles])
        return scheme

    @property
    def tables(self) -> _CartesianMotionTables:
        return self._tables

    @property
    def trajectory_frames(self) -> list[Frame]:
        """
        Returns the sampled end-effector frames along the Cartesian trajectory.
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
        """Returns the array of sampled time moments."""
        return self._t_arr

    @property
    def time_paths(self) -> NumpyArray:
        """
        Returns the array of sampled pose vectors.
        """
        if self._p_arr is not None:
            return self._p_arr
        raise ValueError("Time paths are not available.")

    @property
    def spatial_velocities(self) -> NumpyArray:
        """
        Returns the array of sampled spatial velocities V.
        """
        if self._V_arr is not None:
            return self._V_arr
        raise ValueError("Spatial velocities are not available.")

    @property
    def spatial_accelerations(self) -> NumpyArray:
        """
        Returns the array of sampled spatial accelerations.
        """
        if self._A_arr is not None:
            return self._A_arr
        raise ValueError("Spatial accelerations are not available.")

    @staticmethod
    def _pose_vectors_from_frames(frames: Sequence[Frame]) -> NumpyArray:
        return CartesianMultiStraightLineMotion._frames_to_pose_vectors(frames)

    def plot_time_paths(self):
        """
        Plots the Cartesian pose paths p(t) of the end-effector frame and
        returns the LineChart object. Call show() on this object to see the
        plot.
        """
        labels = ("x", "y", "z", "rx", "ry", "rz")
        p_arr = self.time_paths

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
                    if has_targets:
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

    def plot_spatial_velocities(self):
        """
        Plots the spatial velocity paths V(t) of the end-effector frame and
        returns the LineChart object. Call show() on this object to see the
        plot.
        """
        labels = ("v_x", "v_y", "v_z", "w_x", "w_y", "w_z")
        V_arr = self.spatial_velocities
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
                    if has_targets:
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

    def plot_spatial_accelerations(self):
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
                    if has_targets:
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
            motion_paths=[],
        )


class _CartesianTrajectoryPlotter:
    """
    Helper class of CartesianMotionScheme to plot a Cartesian trajectory.
    """
    def __init__(self, cms: CartesianSpaceScheme) -> None:
        self._cms = cms

    @staticmethod
    def _create_scene(**kwargs) -> WorldScene:
        scene_params = get_valid_keyword_parameters(WorldScene.__init__, exclude={"self"})
        scene_kwargs = {k: v for k, v in kwargs.items() if k in scene_params}
        scene = WorldScene(**scene_kwargs)
        scene.camera.enable_view_shortcuts()
        scene.add_plane_grid()
        scene.add_world_frame()
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
                scene.add_frame(
                    frame=frame,
                    scale=target_frame_scale,
                    line_width=2.0,
                    name=f"T{i}",
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
                    scale=frame_scale,
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
        await scene.show_async()


class _JointMotionTables:
    """
    Helper class of JointMotionScheme.
    """
    def __init__(
        self,
        jms: JointSpaceScheme,
        angle_unit: str = "deg"
    ) -> None:
        self._jms = jms
        self._angle_unit = angle_unit

    @property
    def angle_unit(self) -> str:
        return self._angle_unit

    @angle_unit.setter
    def angle_unit(self, val: AngleUnit) -> None:
        self._angle_unit = val

    @property
    def target_coordinates(self) -> str:
        _coordinates = self._jms.target_coordinates.copy()
        n_cols = _coordinates.shape[1]
        links = self._jms.manipulator.links

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
        _scheme = self._jms.scheme.copy()
        links = self._jms.manipulator.links

        headers = ["time"]

        for i in range(1, self._jms.n_joints + 1):
            if links[i-1].is_revolute and self._angle_unit == "deg":
                col = np.rad2deg(_scheme[:, i])
                _scheme[:, i] = col
            headers.append(f"q{i}")

        x = 1
        for i in range(self._jms.n_joints + 1, 2 * self._jms.n_joints + 1):
            headers.append(f"qd{x}")
            x += 1

        x = 1
        for i in range(2 * self._jms.n_joints + 1, 3 * self._jms.n_joints + 1):
            headers.append(f"qdd{x}")
            x += 1

        table = array_to_table(
            _scheme,
            headers=headers,
        )
        return table

    @property
    def dynamics(self) -> str:
        if self._jms.manipulator.has_dynamics():
            tau_arr = self._jms.dynamics()
            headers = ["time"]
            headers.extend([f"tau{i+1}" for i in range(self._jms.n_joints)])
            table = array_to_table(tau_arr, headers=headers)
            return table
        raise ConfigurationError("The manipulator's dynamics is not defined.")


class _CartesianMotionTables:

    def __init__(
        self,
        cms: CartesianSpaceScheme,
    ) -> None:
        self._cms = cms

    @property
    def scheme(self) -> str:
        _scheme = self._cms.scheme.copy()
        headers = ["time", "x", "y", "z", "alpha", "beta", "gamma"]
        table = array_to_table(_scheme, headers=headers)
        return table

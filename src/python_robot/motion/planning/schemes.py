from __future__ import annotations
from typing import Sequence

from enum import StrEnum

import numpy as np

from ...base.types import NumpyArray
from ...base import Frame
from ...manipulator import SerialLinkManipulator
from ...charts import LineChart
from ...visualisation import WorldScene
from ..profiles.multi_point import MultiPointCubicPath, MultiLinearSegmentPath


__all__ = [
    "MultiMotionProfileType",
    "JointMotionScheme",
    "CartesianMotionScheme"
]


MultiMotionProfile = MultiPointCubicPath | MultiLinearSegmentPath


class MultiMotionProfileType(StrEnum):
    CUBIC = "cubic"
    LINEAR = "linear"


class JointMotionScheme:
    """
    Given a sequence of end-effector frames (positions and orientations) in
    Cartesian space, determine the motion paths of the joints of the
    manipulator.
    """
    def __init__(
        self,
        target_frames: Sequence[Frame],
        manipulator: SerialLinkManipulator,
        dt_segments: Sequence[float],
        mp_type: MultiMotionProfileType = MultiMotionProfileType.CUBIC,
        blend_accels: float | Sequence[float] | None = None,
        num_t_samples: int = 100
    ) -> None:
        # noinspection GrazieInspectionRunner
        """
        Creates a JointSpaceScheme object.

        Parameters
        ----------
        target_frames: Sequence[Frame]
            List of end-effector frames (positions and orientations) in
            Cartesian space.
        manipulator: SerialLinkManipulator
            Manipulator for which a joint-space scheme is to be created.
        dt_segments: Sequence[float]
            List with the required travel times of each segment between two
            successive frames.
        mp_type: MotionProfileType, default = MotionProfileType.CUBIC
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
        """
        self.manipulator = manipulator
        self.target_frames = target_frames
        self.mp_type = mp_type
        self.blend_accels = blend_accels

        self.n_segments = len(self.target_frames) - 1
        self.n_joints = len(self.manipulator)

        if self.n_segments != len(dt_segments):
            raise ValueError(
                f"The number of segment travel times ({len(dt_segments)}) is "
                f"not equal to the number of segments in the trajectory "
                f"({self.n_segments})."
            )
        self.dt_segments = dt_segments

        self._q_sets = self._map_to_joints(self.target_frames)
        self._q_paths = self._time_scaling(self._q_sets)
        self._t_arr, self._q_arr = self._time_sampling(self._q_paths, num_t_samples)

    @property
    def coordinates(self) -> NumpyArray:
        """
        Returns the joint coordinates (joint positions) of the manipulator that
        correspond with the sequence of end-effector frames in Cartesian space.

        Returns
        -------
        NumpyArray
            2D-array: The number of rows of the array is equal to the number of
            frames. The number of columns equals the number of joints in the
            manipulator.
        """
        return self._q_sets

    @property
    def paths(self) -> list[MultiMotionProfile]:
        """
        Returns the motion paths/motion profiles of the joints of the
        manipulator.

        Returns
        -------
        list[MultiMotionProfile]
            Either a list of MultiPointCubicPath objects or a list of
            MultiLinearSegmentPath objects, depending on the type of motion
            profile selected. The order corresponds with the order of the
            joints in the manipulator, from base to tool-end.
        """
        return self._q_paths

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
        return np.column_stack((self._t_arr, self._q_arr))

    def to_cartesian_space(self) -> CartesianMotionScheme:
        return CartesianMotionScheme.from_joint_motion(self)

    def plot_scheme(self) -> LineChart:
        chart = LineChart()
        for i in range(self.n_joints):
            chart.add_xy_data(
                label=f"q{i+1}",
                x1_values=self._t_arr,
                y1_values=self._q_arr[:, i],
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

    def _map_to_joints(self, frames: Sequence[Frame]) -> NumpyArray:
        """
        Maps the end-effector frames from Cartesian space to "joint space",
        i.e., the poses of the end-effector frames (w.r.t. the fixed base frame
        of the manipulator) are translated to sets of corresponding joint angles
        by application of the inverse kinematics of the manipulator.

        Returns
        -------
        q_sets: NumpyArray
            2D-array: The number of rows of the array is equal to the number of
            frames. The number of columns equals the number of joints of the
            manipulator.
        """
        q_sets = np.array([self.manipulator.inv_kin(frame) for frame in frames])
        return q_sets

    def _time_scaling(self, q_sets: NumpyArray) -> list[MultiMotionProfile]:
        """
        From the joint coordinate sets determined by inverse kinematics, creates
        motion paths for each joint in the manipulator

        Returns
        -------
        list[MotionProfile]
            Either a list of MultiPointCubicPath objects or a list of
            MultiLinearSegmentPath objects, depending on the type of motion
            profile selected.
        """
        if self.mp_type == MultiMotionProfileType.CUBIC:
            q_paths = [
                MultiPointCubicPath(
                    path_points=q_sets[:, i],
                    dt_segments=self.dt_segments,
                    v_start=0.0,
                    v_end=0.0,
                )
                for i in range(self.n_joints)
            ]
            return q_paths
        elif self.mp_type == MultiMotionProfileType.LINEAR and self.blend_accels is not None:
            q_paths = [
                MultiLinearSegmentPath(
                    path_points=q_sets[:, i],
                    dt_segments=self.dt_segments,
                    blend_accels=self.blend_accels
                )
                for i in range(self.n_joints)
            ]
            return q_paths
        elif self.mp_type == MultiMotionProfileType.LINEAR and self.blend_accels is None:
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
        q_paths: list[MultiMotionProfile],
        n_samples: int
    ) -> tuple[NumpyArray, NumpyArray]:
        """
        Takes time samples of the positions of the joints from their motion
        paths.

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
        """
        t_arr = np.linspace(0.0, sum(self.dt_segments), n_samples)

        q_arr = np.column_stack([
            np.array([q_path.position(t) for t in t_arr], dtype=float)
            for q_path in q_paths
        ])
        return t_arr, q_arr


class CartesianMotionScheme:

    def __init__(
        self,
        t_arr: NumpyArray,
        trajectory_frames: list[Frame],
        target_frames: Sequence[Frame] | None = None,
    ) -> None:
        self._t_arr = t_arr
        self._traj_frames = trajectory_frames
        self._target_frames = list(target_frames) if target_frames is not None else []

    @classmethod
    def from_joint_motion(cls, jms: JointMotionScheme) -> CartesianMotionScheme:
        frames = [jms.manipulator.fwd_kin(r[1:]) for r in jms.scheme]
        t_arr = jms.scheme[:, 0]
        return cls(t_arr, frames, jms.target_frames)

    @property
    def scheme(self) -> NumpyArray:
        xyz_coords = np.array([
            np.asarray(fr.origin, dtype=float)
            for fr in self._traj_frames]
        )
        scheme = np.column_stack([self._t_arr, xyz_coords])
        return scheme

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

    @staticmethod
    def _create_scene(**kwargs) -> WorldScene:
        scene = WorldScene(**kwargs)
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

        points = self.trajectory_points
        target_points = self.target_points if self._target_frames else np.empty((0, 3))

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
            for i, frame in enumerate(self._target_frames):
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
            for frame in self._traj_frames[::frame_step]:
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

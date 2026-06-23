"""
Animation helpers for frames and kinematic chains.

The module updates PyVista artists over sampled frame or joint-coordinate
sequences and can optionally write the result to GIF or MP4 output.
"""

from __future__ import annotations
from typing import Callable, Literal, Sequence, TYPE_CHECKING

import asyncio
from pathlib import Path
import time

import numpy as np
import pyvista as pv
from spatialmath import SE3, SO3
from spatialmath.base.types import ArrayLike3

from python_robot.base.frame import Frame
from .scene import WorldScene
from .artists import FrameArtist, LinkArtist

if TYPE_CHECKING:
    from python_robot.manipulator.kinematic_chain import KinematicChain


ToolVisual = Literal["auto", "none", "point", "frame", "both"]
ResolvedToolVisual = Literal["none", "point", "frame", "both"]
_TOOL_VISUAL_OPTIONS = {"auto", "none", "point", "frame", "both"}


class FrameAnimator:
    """
    Animate frame motion in an existing world scene.

    This class handles the time loop and output writing, while the scene object
    remains responsible for the visual representation of frames, links, and
    points.
    """
    def __init__(self, scene: WorldScene) -> None:
        """
        Create the animator.

        Parameters
        ----------
        scene : WorldScene
            Scene used for rendering the animation.
        """
        self.scene = scene

    def _prepare_animation_output(
        self,
        gif_path: str | Path | None,
        mp4_path: str | Path | None,
        fps: int,
    ) -> tuple[Path | None, Path | None]:
        """
        Open the selected animation writer if needed.
        """
        if gif_path is not None and mp4_path is not None:
            raise ValueError("Specify either gif_path or mp4_path, not both.")

        gif_file = Path(gif_path) if gif_path is not None else None
        mp4_file = Path(mp4_path) if mp4_path is not None else None

        if gif_file is not None:
            self.scene.plotter.open_gif(str(gif_file))
        elif mp4_file is not None:
            self.scene.plotter.open_movie(str(mp4_file), framerate=fps)

        return gif_file, mp4_file

    def _finalize_animation_output(
        self,
        gif_path: Path | None,
        mp4_path: Path | None,
        close_plotter: bool,
        keep_plotter_open: bool = False,
    ) -> None:
        """
        Finalize animation writing.
        """
        if gif_path is not None or mp4_path is not None:
            self._close_animation_writer()
            if not keep_plotter_open:
                self.scene.plotter.close()
        elif close_plotter:
            self.scene.plotter.close()

    def _close_animation_writer(self) -> None:
        """
        Close the current animation writer without closing the render window.
        """
        writer = getattr(self.scene.plotter, "mwriter", None)
        if writer is not None:
            writer.close()  # type: ignore
            self.scene.plotter.mwriter = None

    def _write_animation_frame(self) -> None:
        """
        Write a frame without asking the interactor to process GUI events.
        """
        plotter = self.scene.plotter
        if getattr(plotter, "_first_time", False):
            plotter._on_first_render_request()  # type: ignore[attr-defined]
            plotter.render()

        writer = plotter._get_mwriter_not_none()  # type: ignore[attr-defined]
        writer.append_data(plotter.image)

    @staticmethod
    async def _pause_for_live_view(
        show: bool,
        dt: float,
    ) -> None:
        """
        Yield to the notebook event loop while preserving animation timing.
        """
        if show:
            await asyncio.sleep(dt)
        else:
            await asyncio.sleep(0)

    def _ensure_interactor_style(self) -> None:
        """
        Make sure Trame mouse events have a valid VTK interactor style.
        """
        iren = getattr(self.scene.plotter, "iren", None)
        if iren is not None:
            iren.update_style()  # type: ignore

    async def _show_plotter_async(
        self,
        *,
        show: bool,
        interactive_update: bool,
        jupyter_backend: Literal["client", "server", "trame"] | None,
    ) -> None:
        """
        Show the plotter without blocking the active notebook event loop.
        """
        if not show:
            return

        self._ensure_interactor_style()
        await self.scene.show_async(
            jupyter_backend=jupyter_backend,
            auto_close=False,
            interactive_update=interactive_update,
        )
        await asyncio.sleep(0)

    def animate_frame_sequence(
        self,
        frames: list[Frame],
        scale: float = 1.0,
        line_width: float = 3.0,
        name: str | None = "B",
        show_label: bool = True,
        label_offset: float = 0.1,
        fps: int = 20,
        step: int = 1,
        gif_path: str | Path | None = None,
        mp4_path: str | Path | None = None,
        show: bool = True,
        interactive_update: bool = True,
        close_plotter: bool = False,
        jupyter_backend: Literal["client", "server", "trame"] | None = None,
    ) -> None:
        """
        Animate a sequence of frames.

        This is the most general animation method. Each frame can have a different
        origin and a different orientation, so both translational and rotational
        motion are supported.

        Parameters
        ----------
        frames : list[Frame]
            Sequence of frame poses, expressed in the scene/world frame.
        scale : float, default=1.0
            Length of the frame axes.
        line_width : float, default=3.0
            Width of the axis lines.
        name : str | None, default="B"
            Optional frame label.
        show_label : bool, default=True
            If True and ``name`` is not None, a label is shown.
        label_offset : float, default=0.1
            Vertical offset of the label relative to the end of the z-axis.
        fps : int, default=20
            Playback rate.
        step : int, default=1
            Use every ``step``-th sample from the sequence.
        gif_path : str | Path, optional
            If given, write the animation to a GIF file.
        mp4_path : str | Path, optional
            If given, write the animation to an MP4 file.
        show : bool, default=True
            Whether to show the PyVista window.
        interactive_update : bool, default=True
            Passed to ``plotter.show(...)``.
        close_plotter : bool, default=False
            If True, close the plotter when the animation ends and no output
            file is being written.
        jupyter_backend : Literal["client", "server", "trame"], optional
            Backend used for visualisation in Jupyter notebook.
        """
        sampled_frames = frames[::step]
        if not sampled_frames:
            raise ValueError("The frame sequence is empty.")

        gif_file, mp4_file = self._prepare_animation_output(gif_path, mp4_path, fps)

        artist = self.scene.create_frame_artist(
            frame=sampled_frames[0],
            scale=scale,
            line_width=line_width,
            name=name,
            show_label=show_label,
            label_offset=label_offset,
        )

        if show:
            self._ensure_interactor_style()
            self.scene.plotter.show(
                auto_close=False,
                interactive_update=interactive_update,
                jupyter_backend=jupyter_backend,
            )

        if gif_file is not None or mp4_file is not None:
            self._write_animation_frame()

        dt = 1.0 / fps

        for frame in sampled_frames[1:]:
            self.scene.update_frame_artist(artist, frame=frame, scale=scale, render=False)

            if gif_file is not None or mp4_file is not None:
                self.scene.plotter.render()
                self._write_animation_frame()
            else:
                self.scene.plotter.render()
                if show and interactive_update:
                    self.scene.plotter.update()
                time.sleep(dt)

        self._finalize_animation_output(
            gif_file,
            mp4_file,
            close_plotter,
            keep_plotter_open=show,
        )

    async def animate_frame_sequence_async(
        self,
        frames: list[Frame],
        scale: float = 1.0,
        line_width: float = 3.0,
        name: str | None = "B",
        show_label: bool = True,
        label_offset: float = 0.1,
        fps: int = 20,
        step: int = 1,
        gif_path: str | Path | None = None,
        mp4_path: str | Path | None = None,
        show: bool = True,
        interactive_update: bool = True,
        close_plotter: bool = False,
        jupyter_backend: Literal["client", "server", "trame"] | None = None,
    ) -> None:
        """
        Animate a sequence of frames in an active async environment.
        """
        sampled_frames = frames[::step]
        if not sampled_frames:
            raise ValueError("The frame sequence is empty.")

        gif_file, mp4_file = self._prepare_animation_output(gif_path, mp4_path, fps)

        artist = self.scene.create_frame_artist(
            frame=sampled_frames[0],
            scale=scale,
            line_width=line_width,
            name=name,
            show_label=show_label,
            label_offset=label_offset,
        )

        await self._show_plotter_async(
            show=show,
            interactive_update=interactive_update,
            jupyter_backend=jupyter_backend,
        )

        if gif_file is not None or mp4_file is not None:
            self._write_animation_frame()

        dt = 1.0 / fps

        for frame in sampled_frames[1:]:
            self.scene.update_frame_artist(artist, frame=frame, scale=scale, render=False)
            self.scene.plotter.render()

            if gif_file is not None or mp4_file is not None:
                self._write_animation_frame()
                await self._pause_for_live_view(show=show, dt=dt)
            else:
                if show and interactive_update:
                    self.scene.plotter.update()
                await asyncio.sleep(dt)

        self._finalize_animation_output(
            gif_file,
            mp4_file,
            close_plotter,
            keep_plotter_open=show,
        )

    def animate_matrix_sequence(
        self,
        matrices: list[SE3],
        scale: float = 1.0,
        line_width: float = 3.0,
        name: str | None = "B",
        show_label: bool = True,
        label_offset: float = 0.1,
        fps: int = 20,
        step: int = 1,
        gif_path: str | Path | None = None,
        mp4_path: str | Path | None = None,
        show: bool = True,
        interactive_update: bool = True,
        close_plotter: bool = False,
        jupyter_backend: Literal["client", "server", "trame"] | None = None,
    ) -> None:
        """
        Animate a sequence of SE3 poses.

        Parameters are the same as for `animate_frame_sequence()`, except that
        the pose sequence is supplied as `spatialmath.SE3` matrices instead of
        `Frame` objects.
        """
        frames = [Frame.from_matrix(matrix) for matrix in matrices]

        self.animate_frame_sequence(
            frames=frames,
            scale=scale,
            line_width=line_width,
            name=name,
            show_label=show_label,
            label_offset=label_offset,
            fps=fps,
            step=step,
            gif_path=gif_path,
            mp4_path=mp4_path,
            show=show,
            interactive_update=interactive_update,
            close_plotter=close_plotter,
            jupyter_backend=jupyter_backend,
        )

    async def animate_matrix_sequence_async(
        self,
        matrices: list[SE3],
        scale: float = 1.0,
        line_width: float = 3.0,
        name: str | None = "B",
        show_label: bool = True,
        label_offset: float = 0.1,
        fps: int = 20,
        step: int = 1,
        gif_path: str | Path | None = None,
        mp4_path: str | Path | None = None,
        show: bool = True,
        interactive_update: bool = True,
        close_plotter: bool = False,
        jupyter_backend: Literal["client", "server", "trame"] | None = None,
    ) -> None:
        """
        Animate a sequence of SE3 poses in an active async environment.
        """
        frames = [Frame.from_matrix(matrix) for matrix in matrices]

        await self.animate_frame_sequence_async(
            frames=frames,
            scale=scale,
            line_width=line_width,
            name=name,
            show_label=show_label,
            label_offset=label_offset,
            fps=fps,
            step=step,
            gif_path=gif_path,
            mp4_path=mp4_path,
            show=show,
            interactive_update=interactive_update,
            close_plotter=close_plotter,
            jupyter_backend=jupyter_backend,
        )

    def animate_orientation_sequence(
        self,
        orientations: list[SO3],
        origin: ArrayLike3 = (0.0, 0.0, 0.0),
        scale: float = 1.0,
        line_width: float = 3.0,
        name: str | None = "B",
        show_label: bool = True,
        label_offset: float = 0.1,
        fps: int = 20,
        step: int = 1,
        gif_path: str | Path | None = None,
        mp4_path: str | Path | None = None,
        show: bool = True,
        interactive_update: bool = True,
        close_plotter: bool = False,
        jupyter_backend: Literal["client", "server", "trame"] | None = None,
    ) -> None:
        """
        Animate a sequence of SO3 orientations for a frame with fixed origin.

        Parameters are the same as for `animate_frame_sequence()`, except that
        only the orientation changes. The frame origin is set by `origin`.
        """
        origin_arr = np.asarray(origin, dtype=float).reshape(3)
        matrices = [SE3.Rt(R=R, t=origin_arr) for R in orientations]

        self.animate_matrix_sequence(
            matrices=matrices,
            scale=scale,
            line_width=line_width,
            name=name,
            show_label=show_label,
            label_offset=label_offset,
            fps=fps,
            step=step,
            gif_path=gif_path,
            mp4_path=mp4_path,
            show=show,
            interactive_update=interactive_update,
            close_plotter=close_plotter,
            jupyter_backend=jupyter_backend,
        )

    async def animate_orientation_sequence_async(
        self,
        orientations: list[SO3],
        origin: ArrayLike3 = (0.0, 0.0, 0.0),
        scale: float = 1.0,
        line_width: float = 3.0,
        name: str | None = "B",
        show_label: bool = True,
        label_offset: float = 0.1,
        fps: int = 20,
        step: int = 1,
        gif_path: str | Path | None = None,
        mp4_path: str | Path | None = None,
        show: bool = True,
        interactive_update: bool = True,
        close_plotter: bool = False,
        jupyter_backend: Literal["client", "server", "trame"] | None = None,
    ) -> None:
        """
        Animate a sequence of SO3 orientations in an active async environment.
        """
        origin_arr = np.asarray(origin, dtype=float).reshape(3)
        matrices = [SE3.Rt(R=R, t=origin_arr) for R in orientations]

        await self.animate_matrix_sequence_async(
            matrices=matrices,
            scale=scale,
            line_width=line_width,
            name=name,
            show_label=show_label,
            label_offset=label_offset,
            fps=fps,
            step=step,
            gif_path=gif_path,
            mp4_path=mp4_path,
            show=show,
            interactive_update=interactive_update,
            close_plotter=close_plotter,
            jupyter_backend=jupyter_backend,
        )


class KinematicChainAnimator(FrameAnimator):
    """
    Animate a kinematic chain in an existing world scene.

    The animator updates link frames, link segments, and optionally the TCP/tool
    visualization while preserving the original joint configuration of the
    chain after the animation finishes.
    """
    @staticmethod
    def _get_link_frames(chain: KinematicChain) -> list[Frame]:
        """
        Return the poses of all links frames in the chain.
        """
        return [chain.pose(i) for i in chain.iter_indices()]

    @classmethod
    def _get_link_endpoints(
        cls,
        chain: KinematicChain,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Return the line endpoints for all links in the chain.
        """
        frames = cls._get_link_frames(chain)
        base_origin = np.asarray(chain.base_frame.origin, dtype=float).reshape(3)
        origins = [base_origin] + [np.asarray(frame.origin, dtype=float) for frame in frames]
        return list(zip(origins[:-1], origins[1:]))

    @classmethod
    def _get_end_effector_position(cls, chain: KinematicChain) -> np.ndarray:
        """
        Return the current end-effector position of the chain.
        """
        return np.asarray(cls._get_link_frames(chain)[-1].origin, dtype=float).reshape(3)

    @staticmethod
    def _tool_frame_is_identity(chain: KinematicChain, atol: float = 1e-12) -> bool:
        return np.allclose(chain.tool_frame.matrix.A, np.eye(4), atol=atol)

    @classmethod
    def _resolve_tool_visual(
        cls,
        chain: KinematicChain,
        tool_visual: ToolVisual,
    ) -> ResolvedToolVisual:
        if tool_visual not in _TOOL_VISUAL_OPTIONS:
            raise ValueError(
                "tool_visual must be one of 'auto', 'none', 'point', 'frame', or 'both'."
            )
        if tool_visual == "auto":
            return "none" if cls._tool_frame_is_identity(chain) else "frame"
        return tool_visual  # type: ignore[return-value]

    @staticmethod
    def _get_tool_frame(chain: KinematicChain) -> Frame:
        return chain.fwd_kin()

    @staticmethod
    def _get_last_link_origin(chain: KinematicChain) -> np.ndarray:
        return np.asarray(chain.pose(-1).origin, dtype=float).reshape(3)

    @staticmethod
    def _make_path_mesh(points: list[np.ndarray]) -> pv.PolyData:
        """
        Create a polyline mesh through the given path points.
        """
        path = np.asarray(points, dtype=float).reshape((-1, 3))
        mesh = pv.PolyData(path)

        if len(path) > 1:
            mesh.lines = np.concatenate(([len(path)], np.arange(len(path))))

        return mesh

    @staticmethod
    def _update_path_mesh(mesh: pv.PolyData, points: list[np.ndarray]) -> None:
        """
        Update a polyline mesh in place.
        """
        path = np.asarray(points, dtype=float).reshape((-1, 3))
        mesh.points = path

        if len(path) > 1:
            mesh.lines = np.concatenate(([len(path)], np.arange(len(path))))

    def animate_chain_sequence(
        self,
        chain: KinematicChain,
        joint_coord_sets: Sequence[Sequence[float]],
        frame_scale: float = 1.0,
        frame_line_width: float = 2.0,
        link_line_width: float = 5.0,
        show_frames: bool = True,
        frame_names: Sequence[str] | None = None,
        fps: int = 20,
        step: int = 1,
        gif_path: str | Path | None = None,
        mp4_path: str | Path | None = None,
        show: bool = True,
        interactive_update: bool = True,
        close_plotter: bool = False,
        show_ee_path: bool = False,
        ee_path_color: str = "orange",
        ee_path_line_width: float = 3.0,
        tool_visual: ToolVisual = "auto",
        tool_frame_scale: float | None = None,
        tool_frame_line_width: float = 2.0,
        tool_point_color: str = "darkorange",
        tool_point_size: float = 12.0,
        tool_link_color: str = "darkorange",
        tool_link_line_width: float = 3.0,
        tool_name: str | None = "TCP",
        camera_setup: Callable[[WorldScene], None] | None = None,
        jupyter_backend: Literal["client", "server", "trame"] | None = None,
    ) -> None:
        """
        Animate a sequence of joint configurations for a kinematic chain.

        Parameters
        ----------
        chain : KinematicChain
            The chain to animate. Its state will be restored afterward.
        joint_coord_sets : Sequence[Sequence[float]]
            Sequence of joint-coordinate vectors. Each item is one full
            configuration of the chain, with joints ordered from the base
            toward the tool end.
        frame_scale : float, default=1.0
            Axis length of links frames.
        frame_line_width : float, default=2.0
            Width of frame axes.
        link_line_width : float, default=5.0
            Width of links segments.
        show_frames : bool, default=True
            If True, draw the local links frames.
        frame_names : Sequence[str] | None, default=None
            Optional names for the links frames.
        fps : int, default=20
            Playback rate.
        step : int, default=1
            Use every ``step``-th configuration.
        gif_path, mp4_path : str | Path | None
            Optional output file.
        show : bool, default=True
            Whether to show the render window.
        interactive_update : bool, default=True
            Passed to ``plotter.show(...)``.
        close_plotter : bool, default=False
            Close plotter at end if no output file is written.
        show_ee_path : bool, default=False
            If True, draw the path traced by the last link frame.
        ee_path_color : str, default="orange"
            Color of the end-effector path.
        ee_path_line_width : float, default=3.0
            Line width of the end-effector path.
        tool_visual : {'auto', 'none', 'point', 'frame', 'both'}, default='auto'
            How to visualize the tool/TCP frame. With 'auto', no tool is drawn
            when the tool frame is the identity transform; otherwise a smaller
            TCP frame is drawn.
        tool_frame_scale : float | None, optional
            Axis length for the TCP frame. If None, a scale relative to
            `frame_scale` is used.
        tool_frame_line_width : float, default=2.0
            Line width of the TCP frame axes.
        tool_point_color : str, default='darkorange'
            Color used for the TCP point when `tool_visual` is 'point' or
            'both'.
        tool_point_size : float, default=12.0
            Marker size of the TCP point.
        tool_link_color : str, default='darkorange'
            Color of the segment from the last link frame to the TCP.
        tool_link_line_width : float, default=3.0
            Line width of the segment from the last link frame to the TCP.
        tool_name : str | None, default='TCP'
            Optional label for the TCP point or frame.
        camera_setup : Callable[[WorldScene], None] | None, default=None
            Optional callback used to configure the scene camera after the
            initial robot geometry has been created and before the animation is
            shown or written to file.
        jupyter_backend: Literal["client", "server", "trame"], optional
            Backend used by PyVista in a Jupyter notebook.
        """
        sampled_joint_sets = list(joint_coord_sets[::step])
        if not sampled_joint_sets:
            raise ValueError("The joint-coordinate sequence is empty.")

        original_joint_coords = list(chain.joint_coords)

        gif_file, mp4_file = self._prepare_animation_output(gif_path, mp4_path, fps)

        animation_failed = False
        try:
            chain.joint_coords = sampled_joint_sets[0]

            frame_artists: list[FrameArtist] = []
            if show_frames:
                frames = self._get_link_frames(chain)

                if frame_names is not None:
                    iter_ = zip(frames, frame_names)
                else:
                    iter_ = frames

                for item in iter_:
                    if isinstance(item, tuple):
                        frame, name = item
                    else:
                        frame, name = item, None
                    artist = self.scene.create_frame_artist(
                        frame=frame,  # type: ignore
                        scale=frame_scale,
                        line_width=frame_line_width,
                        name=name,
                        show_label=True,
                    )
                    frame_artists.append(artist)

            link_artists: list[LinkArtist] = []
            for p1, p2 in self._get_link_endpoints(chain):
                artist = self.scene.create_link_artist(
                    p1=p1,
                    p2=p2,
                    line_width=link_line_width,
                )
                link_artists.append(artist)

            resolved_tool_visual = self._resolve_tool_visual(chain, tool_visual)  # type: ignore
            tool_frame_artist: FrameArtist | None = None
            tool_point_artist = None
            tool_link_artist: LinkArtist | None = None
            tool_scale = tool_frame_scale if tool_frame_scale is not None else 0.7 * frame_scale

            if resolved_tool_visual != "none":
                tool_frame = self._get_tool_frame(chain)
                tcp_origin = np.asarray(tool_frame.origin, dtype=float).reshape(3)
                link_origin = self._get_last_link_origin(chain)

                if (
                    resolved_tool_visual in ("frame", "both")
                    and not np.allclose(link_origin, tcp_origin)
                ):
                    tool_link_artist = self.scene.create_link_artist(
                        p1=link_origin,
                        p2=tcp_origin,
                        color=tool_link_color,
                        line_width=tool_link_line_width,
                    )

                if resolved_tool_visual in ("point", "both"):
                    tool_point_artist = self.scene.create_point_artist(
                        point=tcp_origin,
                        color=tool_point_color,
                        size=tool_point_size,
                        name=tool_name,
                    )

                if resolved_tool_visual in ("frame", "both"):
                    tool_frame_artist = self.scene.create_frame_artist(
                        frame=tool_frame,
                        scale=tool_scale,
                        line_width=tool_frame_line_width,
                        name=tool_name,
                        show_label=tool_name is not None,
                    )

            ee_path_points: list[np.ndarray] = []
            ee_path_mesh: pv.PolyData | None = None
            if show_ee_path:
                ee_path_points.append(self._get_end_effector_position(chain))
                ee_path_mesh = self._make_path_mesh(ee_path_points)
                self.scene.plotter.add_mesh(
                    ee_path_mesh,
                    color=ee_path_color,
                    line_width=ee_path_line_width,
                )

            if camera_setup is not None:
                camera_setup(self.scene)

            if show:
                self._ensure_interactor_style()
                self.scene.plotter.show(
                    auto_close=False,
                    interactive_update=interactive_update,
                    jupyter_backend=jupyter_backend,
                )

            if gif_file is not None or mp4_file is not None:
                self._write_animation_frame()

            dt = 1.0 / fps

            for joint_coords in sampled_joint_sets[1:]:
                chain.joint_coords = joint_coords

                frames = self._get_link_frames(chain)
                links = self._get_link_endpoints(chain)

                if show_frames:
                    for artist, frame in zip(frame_artists, frames):
                        self.scene.update_frame_artist(
                            artist,
                            frame=frame,
                            scale=frame_scale,
                            render=False,
                        )

                for artist, (p1, p2) in zip(link_artists, links):
                    self.scene.update_link_artist(
                        artist,
                        p1=p1,
                        p2=p2,
                        render=False,
                    )

                if resolved_tool_visual != "none":
                    tool_frame = self._get_tool_frame(chain)
                    tcp_origin = np.asarray(tool_frame.origin, dtype=float).reshape(3)

                    if tool_link_artist is not None:
                        self.scene.update_link_artist(
                            tool_link_artist,
                            p1=self._get_last_link_origin(chain),
                            p2=tcp_origin,
                            render=False,
                        )

                    if tool_point_artist is not None:
                        self.scene.update_point_artist(
                            tool_point_artist,
                            point=tcp_origin,
                            render=False,
                        )

                    if tool_frame_artist is not None:
                        self.scene.update_frame_artist(
                            tool_frame_artist,
                            frame=tool_frame,
                            scale=tool_scale,
                            render=False,
                        )

                if ee_path_mesh is not None:
                    ee_path_points.append(self._get_end_effector_position(chain))
                    self._update_path_mesh(ee_path_mesh, ee_path_points)

                self.scene.plotter.render()

                if gif_file is not None or mp4_file is not None:
                    self._write_animation_frame()
                else:
                    if show and interactive_update:
                        self.scene.plotter.update()
                    time.sleep(dt)

        except BaseException:
            animation_failed = True
            raise
        finally:
            chain.joint_coords = original_joint_coords
            self._finalize_animation_output(
                gif_file,
                mp4_file,
                close_plotter or animation_failed,
                keep_plotter_open=show and not animation_failed,
            )

    async def animate_chain_sequence_async(
        self,
        chain: KinematicChain,
        joint_coord_sets: Sequence[Sequence[float]],
        frame_scale: float = 1.0,
        frame_line_width: float = 2.0,
        link_line_width: float = 5.0,
        show_frames: bool = True,
        frame_names: Sequence[str] | None = None,
        fps: int = 20,
        step: int = 1,
        gif_path: str | Path | None = None,
        mp4_path: str | Path | None = None,
        show: bool = True,
        interactive_update: bool = True,
        close_plotter: bool = False,
        show_ee_path: bool = False,
        ee_path_color: str = "orange",
        ee_path_line_width: float = 3.0,
        tool_visual: ToolVisual = "auto",
        tool_frame_scale: float | None = None,
        tool_frame_line_width: float = 2.0,
        tool_point_color: str = "darkorange",
        tool_point_size: float = 12.0,
        tool_link_color: str = "darkorange",
        tool_link_line_width: float = 3.0,
        tool_name: str | None = "TCP",
        camera_setup: Callable[[WorldScene], None] | None = None,
        jupyter_backend: Literal["client", "server", "trame"] | None = None,
    ) -> None:
        """
        Animate a sequence of joint configurations in an async environment.

        Use this in Jupyter notebooks (with await).
        """
        sampled_joint_sets = list(joint_coord_sets[::step])
        if not sampled_joint_sets:
            raise ValueError("The joint-coordinate sequence is empty.")

        original_joint_coords = list(chain.joint_coords)

        gif_file, mp4_file = self._prepare_animation_output(gif_path, mp4_path, fps)

        animation_failed = False
        try:
            chain.joint_coords = sampled_joint_sets[0]

            frame_artists: list[FrameArtist] = []
            if show_frames:
                frames = self._get_link_frames(chain)

                if frame_names is not None:
                    iter_ = zip(frames, frame_names)
                else:
                    iter_ = frames

                for item in iter_:
                    if isinstance(item, tuple):
                        frame, name = item
                    else:
                        frame, name = item, None
                    artist = self.scene.create_frame_artist(
                        frame=frame,  # type: ignore
                        scale=frame_scale,
                        line_width=frame_line_width,
                        name=name,
                        show_label=True,
                    )
                    frame_artists.append(artist)

            link_artists: list[LinkArtist] = []
            for p1, p2 in self._get_link_endpoints(chain):
                artist = self.scene.create_link_artist(
                    p1=p1,
                    p2=p2,
                    line_width=link_line_width,
                )
                link_artists.append(artist)

            resolved_tool_visual = self._resolve_tool_visual(chain, tool_visual)  # type: ignore
            tool_frame_artist: FrameArtist | None = None
            tool_point_artist = None
            tool_link_artist: LinkArtist | None = None
            tool_scale = tool_frame_scale if tool_frame_scale is not None else 0.7 * frame_scale

            if resolved_tool_visual != "none":
                tool_frame = self._get_tool_frame(chain)
                tcp_origin = np.asarray(tool_frame.origin, dtype=float).reshape(3)
                link_origin = self._get_last_link_origin(chain)

                if (
                    resolved_tool_visual in ("frame", "both")
                    and not np.allclose(link_origin, tcp_origin)
                ):
                    tool_link_artist = self.scene.create_link_artist(
                        p1=link_origin,
                        p2=tcp_origin,
                        color=tool_link_color,
                        line_width=tool_link_line_width,
                    )

                if resolved_tool_visual in ("point", "both"):
                    tool_point_artist = self.scene.create_point_artist(
                        point=tcp_origin,
                        color=tool_point_color,
                        size=tool_point_size,
                        name=tool_name,
                    )

                if resolved_tool_visual in ("frame", "both"):
                    tool_frame_artist = self.scene.create_frame_artist(
                        frame=tool_frame,
                        scale=tool_scale,
                        line_width=tool_frame_line_width,
                        name=tool_name,
                        show_label=tool_name is not None,
                    )

            ee_path_points: list[np.ndarray] = []
            ee_path_mesh: pv.PolyData | None = None
            if show_ee_path:
                ee_path_points.append(self._get_end_effector_position(chain))
                ee_path_mesh = self._make_path_mesh(ee_path_points)
                self.scene.plotter.add_mesh(
                    ee_path_mesh,
                    color=ee_path_color,
                    line_width=ee_path_line_width,
                )

            if camera_setup is not None:
                camera_setup(self.scene)

            await self._show_plotter_async(
                show=show,
                interactive_update=interactive_update,
                jupyter_backend=jupyter_backend,
            )

            if gif_file is not None or mp4_file is not None:
                self._write_animation_frame()

            dt = 1.0 / fps

            for joint_coords in sampled_joint_sets[1:]:
                chain.joint_coords = joint_coords

                frames = self._get_link_frames(chain)
                links = self._get_link_endpoints(chain)

                if show_frames:
                    for artist, frame in zip(frame_artists, frames):
                        self.scene.update_frame_artist(
                            artist,
                            frame=frame,
                            scale=frame_scale,
                            render=False,
                        )

                for artist, (p1, p2) in zip(link_artists, links):
                    self.scene.update_link_artist(
                        artist,
                        p1=p1,
                        p2=p2,
                        render=False,
                    )

                if resolved_tool_visual != "none":
                    tool_frame = self._get_tool_frame(chain)
                    tcp_origin = np.asarray(tool_frame.origin, dtype=float).reshape(3)

                    if tool_link_artist is not None:
                        self.scene.update_link_artist(
                            tool_link_artist,
                            p1=self._get_last_link_origin(chain),
                            p2=tcp_origin,
                            render=False,
                        )

                    if tool_point_artist is not None:
                        self.scene.update_point_artist(
                            tool_point_artist,
                            point=tcp_origin,
                            render=False,
                        )

                    if tool_frame_artist is not None:
                        self.scene.update_frame_artist(
                            tool_frame_artist,
                            frame=tool_frame,
                            scale=tool_scale,
                            render=False,
                        )

                if ee_path_mesh is not None:
                    ee_path_points.append(self._get_end_effector_position(chain))
                    self._update_path_mesh(ee_path_mesh, ee_path_points)

                self.scene.plotter.render()

                if gif_file is not None or mp4_file is not None:
                    self._write_animation_frame()
                    await self._pause_for_live_view(show=show, dt=dt)
                else:
                    if show and interactive_update:
                        self.scene.plotter.update()
                    await asyncio.sleep(dt)

        except BaseException:
            animation_failed = True
            raise
        finally:
            chain.joint_coords = original_joint_coords
            self._finalize_animation_output(
                gif_file,
                mp4_file,
                close_plotter or animation_failed,
                keep_plotter_open=show and not animation_failed,
            )

from typing import Literal, Sequence

import numpy as np

from python_robot.base.types import ArrayLike3
from python_robot.base import Frame
from python_robot.visualisation import WorldScene, KinematicChainAnimator
from python_robot.utils.introspection import get_valid_keyword_parameters

from .kinematic_chain import KinematicChain

__all__ = ["KinematicChainViewer"]


ToolVisual = Literal["auto", "none", "point", "frame", "both"]
_TOOL_VISUAL_OPTIONS = {"auto", "none", "point", "frame", "both"}


class KinematicChainViewer:
    """
    Plot and animate a kinematic chain in 3D space.

    The viewer draws the world frame, the manipulator link frames, link
    segments, and optionally the tool/TCP frame. Link segments start at the
    manipulator base frame, so a displaced base frame is reflected in the
    rendered robot geometry.
    """

    def __init__(self, kinematic_chain: KinematicChain) -> None:
        self.kinematic_chain = kinematic_chain
        # self.min_link_length = min([link.link_length for link in kinematic_chain if link.link_length])

    def _get_link_frames(self) -> list[Frame]:
        return [
            self.kinematic_chain.pose(i)
            for i in self.kinematic_chain.iter_indices()
        ]

    def _get_link_endpoints(self) -> list[tuple[ArrayLike3, ArrayLike3]]:
        base_origin = np.asarray(self.kinematic_chain.base_frame.origin, dtype=float)
        origins = [base_origin] + [frame.origin for frame in self._get_link_frames()]
        return list(zip(origins[:-1], origins[1:]))

    def _tool_frame_is_identity(self, atol: float = 1e-12) -> bool:
        return np.allclose(
            self.kinematic_chain.tool_frame.matrix.A,
            np.eye(4),
            atol=atol,
        )

    def _resolve_tool_visual(self, tool_visual: ToolVisual) -> Literal["none", "point", "frame", "both"]:
        if tool_visual not in _TOOL_VISUAL_OPTIONS:
            raise ValueError(
                "tool_visual must be one of 'auto', 'none', 'point', 'frame', or 'both'."
            )
        if tool_visual == "auto":
            return "none" if self._tool_frame_is_identity() else "frame"
        return tool_visual  # type: ignore[return-value]

    def _add_tool_visual(
        self,
        ws: WorldScene,
        tool_visual: ToolVisual,
        frame_scale: float,
        tool_frame_scale: float | None,
        tool_frame_line_width: float,
        tool_point_color: str,
        tool_point_size: float,
        tool_link_color: str,
        tool_link_line_width: float,
        tool_name: str | None,
    ) -> None:
        resolved = self._resolve_tool_visual(tool_visual)
        if resolved == "none":
            return

        tool_frame = self.kinematic_chain.fwd_kin()
        tool_frame.name = tool_name if tool_name is not None else ""
        last_link_frame = self.kinematic_chain.pose(-1)
        tcp_origin = np.asarray(tool_frame.origin, dtype=float)
        link_origin = np.asarray(last_link_frame.origin, dtype=float)

        if resolved in ("frame", "both") and not np.allclose(link_origin, tcp_origin):
            ws.add_link(
                p1=link_origin,
                p2=tcp_origin,
                color=tool_link_color,
                line_width=tool_link_line_width,
            )

        if resolved in ("point", "both"):
            ws.add_point(
                point=tcp_origin,
                color=tool_point_color,
                size=tool_point_size,
                name=tool_frame.name,
            )

        if resolved in ("frame", "both"):
            ws.add_frame(
                tool_frame,
                scale=tool_frame_scale if tool_frame_scale is not None else 0.7 * frame_scale,
                line_width=tool_frame_line_width,
            )

    @staticmethod
    def _create_scene(**kwargs) -> WorldScene:
        ws = WorldScene(**kwargs)
        ws.camera.enable_view_shortcuts()
        ws.add_plane_grid()
        ws.add_world_frame()
        return ws

    def _plot(self, **kwargs) -> WorldScene:
        """
        Create a scene for the current kinematic-chain configuration.

        This method builds the scene but does not show it. It is mainly used by
        `plot()` and `plot_async()`, but can also be useful when the caller wants
        direct access to the underlying `WorldScene`.

        Parameters
        ----------
        tool_visual : {'auto', 'none', 'point', 'frame', 'both'}, default='auto'
            How to visualize the tool/TCP frame. With 'auto', no tool is drawn
            when the tool frame is the identity transform; otherwise a smaller
            TCP frame is drawn.
        tool_frame_scale : float | None, optional
            Axis length for the TCP frame. If None, a scale relative to the link
            frame scale is used.
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
        **kwargs
            Additional keyword arguments for `WorldScene.__init__()` and
            `WorldScene.add_frame()`.

        Returns
        -------
        WorldScene
            Scene containing the current manipulator visualization.
        """
        tool_visual: ToolVisual = kwargs.pop("tool_visual", "auto")
        tool_frame_scale = kwargs.pop("tool_frame_scale", None)
        tool_frame_line_width = kwargs.pop("tool_frame_line_width", 2.0)
        tool_point_color = kwargs.pop("tool_point_color", "darkorange")
        tool_point_size = kwargs.pop("tool_point_size", 12.0)
        tool_link_color = kwargs.pop("tool_link_color", "darkorange")
        tool_link_line_width = kwargs.pop("tool_link_line_width", 3.0)
        tool_name = kwargs.pop("tool_name", "TCP")

        scene_params = get_valid_keyword_parameters(
            WorldScene.__init__,
            exclude={"self"}
        )
        frame_params = get_valid_keyword_parameters(
            WorldScene.add_frame,
            exclude={"self", "frame"}
        )
        scene_kwargs = {
            key: value for key, value in kwargs.items()
            if key in scene_params
        }
        frame_kwargs = {
            key: value for key, value in kwargs.items()
            if key in frame_params
        }

        unknown_kwargs = set(kwargs) - scene_params - frame_params
        if unknown_kwargs:
            raise TypeError(
                f"Unknown plot_frames keyword argument(s): "
                f"{', '.join(sorted(unknown_kwargs))}"
            )

        ws = self._create_scene(**scene_kwargs)
        frame_scale = frame_kwargs.get("scale", 0.5)

        for frame in self._get_link_frames():
            ws.add_frame(frame, scale=frame_scale)

        for p1, p2 in self._get_link_endpoints():
            ws.add_link(p1=p1, p2=p2)

        self._add_tool_visual(
            ws=ws,
            tool_visual=tool_visual,
            frame_scale=frame_scale,
            tool_frame_scale=tool_frame_scale,
            tool_frame_line_width=tool_frame_line_width,
            tool_point_color=tool_point_color,
            tool_point_size=tool_point_size,
            tool_link_color=tool_link_color,
            tool_link_line_width=tool_link_line_width,
            tool_name=tool_name,
        )

        return ws

    def plot(self, **kwargs) -> None:
        """
        Plot the current joint-and-link configuration in 3D space.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to `_plot()`. This includes scene
            options such as `extent`, `spacing`, `background_color`, frame
            options such as `scale`, and tool/TCP options such as
            `tool_visual`.

        Returns
        -------
        None
        """
        ws = self._plot(**kwargs)
        ws.show()

    async def plot_async(self, **kwargs) -> None:
        """
        Plot the current joint-and-link configuration asynchronously.

        This is the asynchronous counterpart of `plot()` and is intended for
        Jupyter notebooks and other async contexts. Call it with `await`.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to `_plot()`.

        Returns
        -------
        None
        """
        jupyter_backend = kwargs.pop("jupyter_backend", None)
        ws = self._plot(**kwargs)
        await ws.show_async(jupyter_backend)

    def animate(
        self,
        joint_coord_sets: Sequence[Sequence[float]],
        fps: int = 20,
        step: int = 1,
        gif_path: str | None = None,
        mp4_path: str | None = None,
        show: bool = True,
        show_ee_path: bool = False,
        ee_path_color: str = "orange",
        ee_path_line_width: float = 3.0,
        **kwargs
    ) -> None:
        """
        Animate a sequence of manipulator joint configurations.

        Parameters
        ----------
        joint_coord_sets : Sequence[Sequence[float]]
            Sequence of joint-coordinate vectors. Each item is one full
            manipulator configuration, with joints ordered from the base toward
            the tool end of the chain.
        fps : int, default=20
            Playback rate.
        step : int, default=1
            Use every ``step``-th configuration.
        gif_path, mp4_path : str | Path | None
            Optional output file.
        show : bool, default=True
            Whether to show the render window.
        show_ee_path : bool, default=False
            If True, draw the path traced by the end-effector during the
            animation.
        ee_path_color : str, default="orange"
            Color of the end-effector path.
        ee_path_line_width : float, default=3.0
            Line width of the end-effector path.
        **kwargs
            Additional keyword arguments for `WorldScene.__init__()` and
            `KinematicChainAnimator.animate_chain_sequence()`. This includes
            tool/TCP options such as `tool_visual`, `tool_frame_scale`,
            `tool_point_color`, and `tool_name`.
        """
        scene_params = get_valid_keyword_parameters(
            WorldScene.__init__,
            exclude={"self"}
        )
        animator_params = get_valid_keyword_parameters(
            KinematicChainAnimator.animate_chain_sequence,
            exclude={"self"}
        )
        scene_kwargs = {
            key: value for key, value in kwargs.items()
            if key in scene_params
        }
        animator_kwargs = {
            key: value for key, value in kwargs.items()
            if key in animator_params
        }
        unknown_kwargs = set(kwargs) - scene_params - animator_params
        if unknown_kwargs:
            raise TypeError(
                f"Unknown keyword argument(s): "
                f"{', '.join(sorted(unknown_kwargs))}"
            )

        ws = self._create_scene(**scene_kwargs)

        animator = KinematicChainAnimator(ws)
        animator.animate_chain_sequence(
            chain=self.kinematic_chain,
            joint_coord_sets=joint_coord_sets,
            frame_scale=animator_kwargs.get("frame_scale", 0.5),
            link_line_width=animator_kwargs.get("link_line_width", 5.0),
            show_frames=animator_kwargs.get("show_frames", True),
            frame_names=None,
            fps=fps,
            step=step,
            gif_path=gif_path,
            mp4_path=mp4_path,
            show=show,
            show_ee_path=show_ee_path,
            ee_path_color=ee_path_color,
            ee_path_line_width=ee_path_line_width,
            tool_visual=animator_kwargs.get("tool_visual", "auto"),
            tool_frame_scale=animator_kwargs.get("tool_frame_scale", None),
            tool_frame_line_width=animator_kwargs.get("tool_frame_line_width", 2.0),
            tool_point_color=animator_kwargs.get("tool_point_color", "darkorange"),
            tool_point_size=animator_kwargs.get("tool_point_size", 12.0),
            tool_link_color=animator_kwargs.get("tool_link_color", "darkorange"),
            tool_link_line_width=animator_kwargs.get("tool_link_line_width", 3.0),
            tool_name=animator_kwargs.get("tool_name", "TCP"),
        )

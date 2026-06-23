"""
High-level plotting and animation wrapper for kinematic chains.
"""

from __future__ import annotations
from typing import Literal, Sequence, Any, TYPE_CHECKING

import numpy as np

from python_robot.base.types import ArrayLike3
from python_robot.base import Frame
from python_robot.visualisation import WorldScene, KinematicChainAnimator
from python_robot.utils.introspection import get_valid_keyword_parameters

if TYPE_CHECKING:
    from python_robot.manipulator.kinematic_chain import KinematicChain


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
        """
        Create a viewer for a kinematic chain.

        Parameters
        ----------
        kinematic_chain : KinematicChain
            Chain whose current configuration can be plotted or animated.
        """
        self.kinematic_chain = kinematic_chain

        self._plot_scene_kwargs: dict[str, Any] = {}
        self._plot_frame_kwargs: dict[str, Any] = {}
        self._plot_tool_visual_kwargs: dict[str, Any] = {}
        self._anim_scene_kwargs: dict[str, Any] = {}
        self._anim_anim_kwargs: dict[str, Any] = {}

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
        tool_visual: ToolVisual = "auto",
        tool_frame_scale: float = 1.0,
        tool_frame_line_width: float = 2.0,
        tool_point_color: str = "darkorange",
        tool_point_size: float = 12.0,
        tool_link_color: str = "darkorange",
        tool_link_line_width: float = 3.0,
        tool_name: str | None = "TCP",
    ) -> None:
        """
        Adds a tool frame visual to the kinematic chain plot.

        Parameters
        ----------
        ws: WorldScene
            PyVista scene object used to plot the kinematic chain.
        tool_visual : {'auto', 'none', 'point', 'frame', 'both'}, default='auto'
            How to visualize the tool/TCP frame. With 'auto', no tool is drawn
            when the tool frame is the identity transform; otherwise a smaller
            TCP frame is drawn.
        tool_frame_scale : float, default=1.0
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

        Returns
        -------
        None
        """
        resolved = self._resolve_tool_visual(tool_visual)  # type: ignore
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
                frame_scale=tool_frame_scale,
                line_width=tool_frame_line_width,
            )

    def _plot_kwargs_dispatcher(self, **kwargs) -> tuple[dict[str, Any], ...]:
        world_frame_scale = kwargs.pop(
            "world_frame_scale",
            self._plot_scene_kwargs.get("world_frame_scale", 1.0)
        )

        scene_params = get_valid_keyword_parameters(
            WorldScene.__init__,
            exclude={"self"}
        )
        scene_kwargs = {
            key: value for key, value in kwargs.items()
            if key in scene_params
        }

        frame_params = get_valid_keyword_parameters(
            WorldScene.add_frame,
            exclude={"self", "frame"}
        )
        frame_kwargs = {
            key: value for key, value in kwargs.items()
            if key in frame_params
        }

        tool_visual_params = get_valid_keyword_parameters(
            self._add_tool_visual,
            exclude={"self", "ws"}
        )
        tool_visual_kwargs = {
            key: value for key, value in kwargs.items()
            if key in tool_visual_params
        }

        unknown_kwargs = set(kwargs) - scene_params - frame_params - tool_visual_params
        if unknown_kwargs:
            raise TypeError(
                f"Unknown plot_frames keyword argument(s): "
                f"{', '.join(sorted(unknown_kwargs))}"
            )

        scene_kwargs.update({"world_frame_scale": world_frame_scale})

        scene_kwargs.update({
            k: v
            for k, v in self._plot_scene_kwargs.items()
            if k not in scene_kwargs.keys()
        })
        frame_kwargs.update({
            k: v
            for k, v in self._plot_frame_kwargs.items()
            if k not in frame_kwargs.keys()
        })
        tool_visual_kwargs.update({
            k: v
            for k, v in self._plot_tool_visual_kwargs.items()
            if k not in tool_visual_kwargs.keys()
        })
        return scene_kwargs, frame_kwargs, tool_visual_kwargs

    def set_plot_options(self, **kwargs) -> None:
        """
        Set default options used by subsequent plot calls.

        Parameters
        ----------
        **kwargs
            Keyword arguments accepted by the plot scene, frame, and tool/TCP
            visual helpers.
        """
        tup = self._plot_kwargs_dispatcher(**kwargs)
        self._plot_scene_kwargs = tup[0]
        self._plot_frame_kwargs = tup[1]
        self._plot_tool_visual_kwargs = tup[2]

    def _anim_kwargs_dispatcher(self, **kwargs) -> tuple[dict[str, Any], ...]:
        world_frame_scale = kwargs.pop(
            "world_frame_scale",
            self._anim_scene_kwargs.get("world_frame_scale", 1.0)
        )

        scene_params = get_valid_keyword_parameters(
            WorldScene.__init__,
            exclude={"self"}
        )
        scene_kwargs = {
            key: value for key, value in kwargs.items()
            if key in scene_params
        }

        anim_params = get_valid_keyword_parameters(
            KinematicChainAnimator.animate_chain_sequence,
            exclude={"self"}
        )
        anim_kwargs = {
            key: value for key, value in kwargs.items()
            if key in anim_params
        }

        unknown_kwargs = set(kwargs) - scene_params - anim_params
        if unknown_kwargs:
            raise TypeError(
                f"Unknown keyword argument(s): "
                f"{', '.join(sorted(unknown_kwargs))}"
            )

        scene_kwargs.update({"world_frame_scale": world_frame_scale})

        scene_kwargs.update({
            k: v
            for k, v in self._anim_scene_kwargs.items()
            if k not in scene_kwargs.keys()
        })
        anim_kwargs.update({
            k: v
            for k, v in self._anim_anim_kwargs.items()
            if k not in anim_kwargs.keys()
        })
        return scene_kwargs, anim_kwargs

    def set_animation_options(self, **kwargs) -> None:
        """
        Set default options used by subsequent animation calls.

        Parameters
        ----------
        **kwargs
            Keyword arguments accepted by the animation scene and animator.
        """
        tup = self._anim_kwargs_dispatcher(**kwargs)
        self._anim_scene_kwargs = tup[0]
        self._anim_anim_kwargs = tup[1]

    @staticmethod
    def _create_scene(**kwargs) -> WorldScene:
        world_frame_scale = kwargs.pop("world_frame_scale", 1.0)

        ws = WorldScene(**kwargs)
        ws.camera.enable_view_shortcuts()
        ws.add_plane_grid()
        ws.add_world_frame(frame_scale=world_frame_scale)

        return ws

    def _plot(self, **kwargs) -> WorldScene:
        """
        Create a scene for the current kinematic-chain configuration.

        This method builds the scene but does not show it. It is mainly used by
        `plot()` and `plot_async()`, but can also be useful when the caller wants
        direct access to the underlying `WorldScene`.

        Parameters
        ----------

        **kwargs
            Additional keyword arguments for `WorldScene.__init__()` and
            `WorldScene.add_frame()`.

        Returns
        -------
        WorldScene
            Scene containing the current manipulator visualization.
        """
        scene_kwargs, frame_kwargs, tool_visual_kwargs = self._plot_kwargs_dispatcher(**kwargs)

        ws = self._create_scene(**scene_kwargs)

        for frame in self._get_link_frames():
            ws.add_frame(frame, **frame_kwargs)

        for p1, p2 in self._get_link_endpoints():
            ws.add_link(p1=p1, p2=p2)

        self._add_tool_visual(
            ws=ws,
            **tool_visual_kwargs
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

    def _animate(
        self,
        **kwargs
    ) -> tuple[KinematicChainAnimator, dict[str, Any]]:
        scene_kwargs, anim_kwargs = self._anim_kwargs_dispatcher(**kwargs)
        ws = self._create_scene(**scene_kwargs)
        animator = KinematicChainAnimator(ws)
        return animator, anim_kwargs

    def animate(
        self,
        joint_coords: Sequence[Sequence[float]],
        **kwargs
    ) -> None:
        """
        Animate a sequence of manipulator joint configurations.

        Parameters
        ----------
        joint_coords : Sequence[Sequence[float]]
            Sequence of joint-coordinate vectors. Each item is one full
            manipulator configuration, with joints ordered from the base toward
            the tool end of the chain.
        **kwargs
            Additional keyword arguments for `WorldScene.__init__()` and
            `KinematicChainAnimator.animate_chain_sequence()`.
        """
        animator, anim_kwargs = self._animate(**kwargs)
        animator.animate_chain_sequence(
            chain=self.kinematic_chain,
            joint_coord_sets=joint_coords,
            **anim_kwargs
        )

    async def animate_async(
        self,
        joint_coords: Sequence[Sequence[float]],
        **kwargs
    ) -> None:
        """
        Animate a sequence of manipulator joint configurations asynchronously.

        This is the asynchronous counterpart of `animate()` and is intended
        for Jupyter notebooks and other async contexts. Call it with `await`.
        """
        animator, anim_kwargs = self._animate(**kwargs)
        await animator.animate_chain_sequence_async(
            chain=self.kinematic_chain,
            joint_coord_sets=joint_coords,
            **anim_kwargs
        )

from typing import Sequence

import numpy as np

from python_robot.base.types import ArrayLike3
from python_robot.base import Frame
from python_robot.visualisation import WorldScene, KinematicChainAnimator

from .kinematic_chain import KinematicChain

__all__ = ["KinematicChainViewer"]


class KinematicChainViewer:
    """Class for plotting and animating a kinematic chain in 3D-space."""

    def __init__(self, kinematic_chain: KinematicChain) -> None:
        self.kinematic_chain = kinematic_chain
        self.min_link_length = min([l.link_length for l in kinematic_chain if l.link_length])

    def _get_link_frames(self) -> list[Frame]:
        return [
            self.kinematic_chain.pose(i)
            for i in self.kinematic_chain.iter_indices()
        ]

    def _get_link_endpoints(self) -> list[tuple[ArrayLike3, ArrayLike3]]:
        origins = [np.zeros(3)] + [frame.origin for frame in self._get_link_frames()]
        return list(zip(origins[:-1], origins[1:]))

    @staticmethod
    def _create_scene(**kwargs) -> WorldScene:
        ws = WorldScene(**kwargs)
        ws.camera.enable_view_shortcuts()
        ws.add_plane_grid()
        ws.add_world_frame()
        return ws

    def _plot(self, **kwargs) -> WorldScene:
        """
        Plots the current joint-and-links configuration of the kinematic chain in
        3D-space.

        Returns
        -------
        WorldScene
        """
        ws = self._create_scene(**kwargs)

        for frame in self._get_link_frames():
            ws.add_frame(frame, scale=kwargs.get("frame_scale", 1 / self.min_link_length))

        for p1, p2 in self._get_link_endpoints():
            ws.add_link(p1=p1, p2=p2)

        return ws

    def plot(self, **kwargs) -> None:
        """
        Plots the current joint-and-links configuration of the kinematic chain in
        3D-space.

        Returns
        -------
        None
        """
        ws = self._plot(**kwargs)
        ws.show()

    async def plot_async(self, **kwargs) -> None:
        """
        Plots the current joint-and-links configuration of the kinematic chain in
        3D-space.

        This is an asynchronous version of the plot method that can be used in
        Jupyter notebooks. (When calling this function, you need keyword await
        in front of the method call.)

        Returns
        -------
        None
        """
        ws = self._plot(**kwargs)
        await ws.show_async()

    def animate(
        self,
        joint_coord_sets: Sequence[Sequence[float]],
        fps: int = 20,
        step: int = 1,
        gif_path: str | None = None,
        mp4_path: str | None = None,
        show: bool = True,
        **kwargs
    ) -> None:
        """
        Animate a sequence of manipulator configurations.

        Parameters
        ----------
        joint_coord_sets : Sequence[Sequence[float]]
            Sequence of joint coordinate time-sets that define the successive
            kinematic configurations of the chain in the course of time from
            start to end. Each row contains the coordinates of a joint at
            sampled time moments from start to finish of the motion. Joints are
            ordered from the base toward the tool-end of the chain.
        fps : int, default=20
            Playback rate.
        step : int, default=1
            Use every ``step``-th configuration.
        gif_path, mp4_path : str | Path | None
            Optional output file.
        show : bool, default=True
            Whether to show the render window.
        **kwargs
            Additional keyword arguments for configuration of the scene (see
            docstring of class WorldScene in scene.py).
        """
        ws = self._create_scene(**kwargs)

        animator = KinematicChainAnimator(ws)
        animator.animate_chain_sequence(
            chain=self.kinematic_chain,
            joint_coord_sets=joint_coord_sets,
            frame_scale=kwargs.get("frame_scale", 1 / self.min_link_length),
            show_frames=True,
            frame_names=None,
            fps=fps,
            step=step,
            gif_path=gif_path,
            mp4_path=mp4_path,
            show=show
        )

from __future__ import annotations
from typing import Sequence, TYPE_CHECKING

from pathlib import Path
import time

import numpy as np
from spatialmath import SE3, SO3
from spatialmath.base.types import ArrayLike3

from .scene import WorldScene
from .artists import FrameArtist, LinkArtist

if TYPE_CHECKING:
    from ..base.frame import Frame
    from ..manipulator.kinematic_chain import KinematicChain


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
    ) -> None:
        """
        Finalize animation writing.
        """
        if gif_path is not None or mp4_path is not None:
            self.scene.plotter.close()
        elif close_plotter:
            self.scene.plotter.close()

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
    ) -> None:
        """
        Animate a sequence of frames.

        This is the most general animation method. Each frame can have a different
        origin and a different orientation, so both translational and rotational
        motion are supported.

        Parameters
        ----------
        frames : list[Frame]
            Sequence of frame poses.
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
        gif_path : str | Path | None, default=None
            If given, write the animation to a GIF file.
        mp4_path : str | Path | None, default=None
            If given, write the animation to an MP4 file.
        show : bool, default=True
            Whether to show the PyVista window.
        interactive_update : bool, default=True
            Passed to ``plotter.show(...)``.
        close_plotter : bool, default=False
            If True, close the plotter when the animation ends and no output
            file is being written.
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
            self.scene.plotter.show(
                auto_close=False,
                interactive_update=interactive_update,
            )

        if gif_file is not None or mp4_file is not None:
            self.scene.plotter.write_frame()

        dt = 1.0 / fps

        for frame in sampled_frames[1:]:
            self.scene.update_frame_artist(artist, frame=frame, scale=scale, render=False)

            if gif_file is not None or mp4_file is not None:
                self.scene.plotter.render()
                self.scene.plotter.write_frame()
            else:
                self.scene.plotter.render()
                if interactive_update:
                    self.scene.plotter.update()
                time.sleep(dt)

        self._finalize_animation_output(gif_file, mp4_file, close_plotter)

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
    ) -> None:
        """
        Animate a sequence of SE3 matrices.
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
    ) -> None:
        """
        Animate a sequence of SO3 orientations for a frame with fixed origin.
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
        )



class KinematicChainAnimator(FrameAnimator):
    """
    Animate a kinematic chain in an existing world scene.
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
        origins = [np.zeros(3)] + [np.asarray(frame.origin, dtype=float) for frame in frames]
        return list(zip(origins[:-1], origins[1:]))

    def animate_chain_sequence(
        self,
        chain: KinematicChain,
        joint_coord_sets: Sequence[Sequence[float]],
        frame_scale: float = 1.0,
        frame_line_width: float = 2.0,
        link_line_width: float = 5.0,
        show_frames: bool = True,
        frame_names: Sequence[str] | None = None,
        show_world_frame: bool = False,
        fps: int = 20,
        step: int = 1,
        gif_path: str | Path | None = None,
        mp4_path: str | Path | None = None,
        show: bool = True,
        interactive_update: bool = True,
        close_plotter: bool = False,
    ) -> None:
        """
        Animate a sequence of joint configurations for a kinematic chain.

        Parameters
        ----------
        chain : KinematicChain
            The chain to animate. Its state will be restored afterward.
        joint_coord_sets : Sequence[Sequence[float]]
            Sequence of joint coordinate vectors.
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
        show_world_frame : bool, default=False
            If True, add the world frame before the animation starts.
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
        """
        sampled_joint_sets = list(joint_coord_sets[::step])
        if not sampled_joint_sets:
            raise ValueError("The joint-coordinate sequence is empty.")

        original_joint_coords = list(chain.joint_coords)

        gif_file, mp4_file = self._prepare_animation_output(gif_path, mp4_path, fps)

        try:
            chain.joint_coords = sampled_joint_sets[0]

            if show_world_frame:
                self.scene.add_world_frame()

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
                        frame=frame,  #type: ignore
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

            if show:
                self.scene.plotter.show(
                    auto_close=False,
                    interactive_update=interactive_update,
                )

            if gif_file is not None or mp4_file is not None:
                self.scene.plotter.write_frame()

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

                self.scene.plotter.render()

                if gif_file is not None or mp4_file is not None:
                    self.scene.plotter.write_frame()
                else:
                    if interactive_update:
                        self.scene.plotter.update()
                    time.sleep(dt)

        finally:
            chain.joint_coords = original_joint_coords
            self._finalize_animation_output(gif_file, mp4_file, close_plotter)

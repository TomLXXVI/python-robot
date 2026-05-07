from __future__ import annotations

import numpy as np
import pyvista as pv
from spatialmath.base.types import ArrayLike3

from ..base.frame import Frame, WREF_FRAME
from .artists import FrameArtist, LinkArtist, PointArtist


class Camera:
    """
    Convenience wrapper around PyVista camera controls.
    """
    def __init__(self, plotter: pv.Plotter) -> None:
        self.plotter = plotter

    def set_top_view(self) -> None:
        """Set the camera to top view."""
        self.plotter.view_xy()

    def set_bottom_view(self) -> None:
        """Set the camera to bottom view."""
        self.plotter.view_xy(negative=True)

    def set_right_view(self) -> None:
        """Set the camera to right view."""
        self.plotter.view_xz(negative=True)

    def set_left_view(self) -> None:
        """Set the camera to left view."""
        self.plotter.view_xz()

    def set_front_view(self) -> None:
        """Set the camera to front view."""
        self.plotter.view_yz()

    def set_back_view(self) -> None:
        """Set the camera to back view."""
        self.plotter.view_yz(negative=True)

    def set_isometric_view(self) -> None:
        """Set the camera to a default isometric view."""
        self.plotter.view_isometric()

    def reset_camera_to_scene(self) -> None:
        """Reset the camera so that all visible actors fit in the view."""
        self.plotter.reset_camera()

    def pan(self, dx: float = 0.0, dy: float = 0.0, amount: float = 0.1) -> None:
        cam = self.plotter.camera

        position = np.asarray(cam.position, dtype=float)
        focal_point = np.asarray(cam.focal_point, dtype=float)
        up = np.asarray(cam.up, dtype=float)

        direction = focal_point - position
        right = np.cross(direction, up)

        right /= np.linalg.norm(right)
        up /= np.linalg.norm(up)

        delta = amount * (dx * right + dy * up)

        cam.position = tuple(position + delta)
        cam.focal_point = tuple(focal_point + delta)

        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def enable_view_shortcuts(self) -> None:
        """
        Enable keyboard shortcuts for standard orthographic views.

        Shortcuts
        ---------
        t : top view
        b : bottom view
        r : right view
        l : left view
        f : front view
        k : back view
        i : isometric view
        c : reset camera
        a : pan right
        d : pan left
        w : pan down
        s : pan up
        """
        self.plotter.add_key_event("t", self.set_top_view)
        self.plotter.add_key_event("b", self.set_bottom_view)
        self.plotter.add_key_event("r", self.set_right_view)
        self.plotter.add_key_event("l", self.set_left_view)
        self.plotter.add_key_event("f", self.set_front_view)
        self.plotter.add_key_event("k", self.set_back_view)
        self.plotter.add_key_event("i", self.set_isometric_view)
        self.plotter.add_key_event("c", self.reset_camera_to_scene)
        self.plotter.add_key_event("a", lambda : self.pan(dx=-1.0))
        self.plotter.add_key_event("d", lambda : self.pan(dx=1.0))
        self.plotter.add_key_event("w", lambda : self.pan(dy=1.0))
        self.plotter.add_key_event("s", lambda : self.pan(dy=-1.0))


class WorldScene:
    """
    3D world scene for static visualization of frames, links, and points.

    The class also provides low-level support for updateable artists that can be
    used by an external animator.
    """

    # noinspection PyUnusedLocal
    def __init__(
        self,
        extent: float = 4.0,
        spacing: float = 1.0,
        grid_color: str = "lightgray",
        axis_color: str = "black",
        background_color: str = "white",
        off_screen: bool = False,
        window_size: tuple[int, int] = (800, 600),  #type: ignore
    ) -> None:
        """
        Create the scene.

        Parameters
        ----------
        extent : float, default=4.0
            Default half-size used by the grid helper methods.
        spacing : float, default=1.0
            Default spacing used by the grid helper methods.
        grid_color : str, default="lightgray"
            Default color of ordinary grid lines.
        axis_color : str, default="black"
            Default color of the main axes in a grid.
        background_color : str, default="white"
            Background color of the render window.
        off_screen : bool, default=False
            Whether the plotter renders off-screen.
        window_size : tuple[int, int], default=(1200, 900)
            Window size of the plotter.
        """
        self.extent = float(extent)
        self.spacing = float(spacing)
        self.grid_color = grid_color
        self.axis_color = axis_color

        self.plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)  #type:ignore
        self.plotter.set_background(background_color)

        self.camera = Camera(self.plotter)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _as_point(coords: ArrayLike3) -> np.ndarray:
        """
        Convert an input coordinate-like object to a 3-element NumPy array.
        """
        return np.asarray(coords, dtype=float).reshape(3)

    @staticmethod
    def _frame_axis_endpoints(
        frame: Frame,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the origin and the endpoints of the three frame axes.

        Parameters
        ----------
        frame : Frame
            The frame pose.
        scale : float
            Axis length scaling factor.

        Returns
        -------
        origin, px, py, pz : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Origin and axis endpoints.
        """
        origin = np.asarray(frame.origin, dtype=float).reshape(3)
        rotation_matrix = np.asarray(frame.orient_mat.A, dtype=float).reshape(3, 3)

        px = origin + scale * rotation_matrix[:, 0]
        py = origin + scale * rotation_matrix[:, 1]
        pz = origin + scale * rotation_matrix[:, 2]

        return origin, px, py, pz

    @staticmethod
    def _set_line_points(mesh: pv.PolyData, p1: ArrayLike3, p2: ArrayLike3) -> None:
        """
        Update the endpoints of an existing PyVista line mesh.
        """
        p1_arr = np.asarray(p1, dtype=float).reshape(3)
        p2_arr = np.asarray(p2, dtype=float).reshape(3)
        mesh.points = np.vstack([p1_arr, p2_arr])

    def _make_label_actor(
        self,
        point: ArrayLike3,
        text: str,
        font_size: int,
        shape_opacity: float,
        always_visible: bool,
    ) -> object:
        """
        Create a point-label actor and return the handle.
        """
        return self.plotter.add_point_labels(
            [np.asarray(point, dtype=float).reshape(3)],
            [text],
            font_size=font_size,
            point_size=0,
            show_points=False,
            shape_opacity=shape_opacity,
            always_visible=always_visible,
        )

    def _remove_actor_if_needed(self, actor: object | None) -> None:
        """
        Remove a PyVista actor from the scene if it exists.
        """
        if actor is not None:
            self.plotter.remove_actor(actor, render=False)

    # -------------------------------------------------------------------------
    # Grid helpers
    # -------------------------------------------------------------------------

    def add_plane_grid(
        self,
        plane: str = "xy",
        offset: float = 0.0,
        extent: float | None = None,
        spacing: float | None = None,
        color: str | None = None,
        axis_color: str | None = None,
        line_width: float = 1.0,
        axis_width: float = 1.0,
    ) -> None:
        """
        Add a planar grid in one of the principal planes.

        Parameters
        ----------
        plane : str, default="xy"
            The plane in which the grid is drawn: "xy", "xz", or "yz".
        offset : float, default=0.0
            Offset of the plane along the remaining axis.
        extent : float | None, default=None
            Half-size of the grid in both in-plane directions.
        spacing : float | None, default=None
            Distance between neighboring grid lines.
        color : str | None, default=None
            Color of ordinary grid lines.
        axis_color : str | None, default=None
            Color of the main axes within the plane.
        line_width : float, default=1.0
            Line width for ordinary grid lines.
        axis_width : float, default=1.0
            Line width for the main axes.
        """
        plane = plane.lower()

        if plane not in {"xy", "xz", "yz"}:
            raise ValueError("plane must be one of: 'xy', 'xz', or 'yz'")

        extent: float = self.extent if extent is None else float(extent)  #type: ignore
        spacing = self.spacing if spacing is None else float(spacing)
        color = self.grid_color if color is None else color
        axis_color = self.axis_color if axis_color is None else axis_color

        values = np.arange(-extent, extent + 0.5 * spacing, spacing)

        if plane == "xy":
            for y in values:
                is_axis = np.isclose(y, 0.0)
                self.plotter.add_mesh(
                    pv.Line((-extent, y, offset), (extent, y, offset)),
                    color=axis_color if is_axis else color,
                    line_width=axis_width if is_axis else line_width,
                )

            for x in values:
                is_axis = np.isclose(x, 0.0)
                self.plotter.add_mesh(
                    pv.Line((x, -extent, offset), (x, extent, offset)),
                    color=axis_color if is_axis else color,
                    line_width=axis_width if is_axis else line_width,
                )

        elif plane == "xz":
            for z in values:
                is_axis = np.isclose(z, 0.0)
                self.plotter.add_mesh(
                    pv.Line((-extent, offset, z), (extent, offset, z)),
                    color=axis_color if is_axis else color,
                    line_width=axis_width if is_axis else line_width,
                )

            for x in values:
                is_axis = np.isclose(x, 0.0)
                self.plotter.add_mesh(
                    pv.Line((x, offset, -extent), (x, offset, extent)),
                    color=axis_color if is_axis else color,
                    line_width=axis_width if is_axis else line_width,
                )

        else:  # yz
            for z in values:
                is_axis = np.isclose(z, 0.0)
                self.plotter.add_mesh(
                    pv.Line((offset, -extent, z), (offset, extent, z)),
                    color=axis_color if is_axis else color,
                    line_width=axis_width if is_axis else line_width,
                )

            for y in values:
                is_axis = np.isclose(y, 0.0)
                self.plotter.add_mesh(
                    pv.Line((offset, y, -extent), (offset, y, extent)),
                    color=axis_color if is_axis else color,
                    line_width=axis_width if is_axis else line_width,
                )

    # -------------------------------------------------------------------------
    # Static scene primitives
    # -------------------------------------------------------------------------

    def add_world_frame(
        self,
        scale: float = 1.0,
        line_width: float = 3.0,
        label: bool = True,
        label_offset: float = 0.1,
    ) -> None:
        """
        Add the world reference frame.
        """
        self.add_frame(
            frame=WREF_FRAME,
            scale=scale,
            line_width=line_width,
            name="W",
            show_label=label,
            label_offset=label_offset,
        )

    def add_frame(
        self,
        frame: Frame,
        scale: float = 1.0,
        line_width: float = 2.0,
        name: str | None = None,
        show_label: bool = True,
        label_offset: float = 0.1,
        label_font_size: int = 14,
    ) -> None:
        """
        Add a static coordinate frame to the scene.
        """
        origin, px, py, pz = self._frame_axis_endpoints(frame, scale)

        self.plotter.add_mesh(pv.Line(origin, px), color="red", line_width=line_width)
        self.plotter.add_mesh(pv.Line(origin, py), color="green", line_width=line_width)
        self.plotter.add_mesh(pv.Line(origin, pz), color="blue", line_width=line_width)

        if show_label and name is not None:
            label_point = pz.copy()
            label_point[2] += label_offset
            self._make_label_actor(
                point=label_point,
                text=name,
                font_size=label_font_size,
                shape_opacity=0.25,
                always_visible=True,
            )

    def add_link(
        self,
        p1: ArrayLike3,
        p2: ArrayLike3,
        color: str = "dimgray",
        line_width: float = 5.0,
    ) -> None:
        """
        Add a static line segment between two points.
        """
        p1_arr = self._as_point(p1)
        p2_arr = self._as_point(p2)

        self.plotter.add_mesh(
            pv.Line(p1_arr, p2_arr),
            color=color,
            line_width=line_width,
        )

    def add_point(
        self,
        point: ArrayLike3,
        color: str = "black",
        size: float = 12.0,
        name: str | None = None,
        label_font_size: int = 12,
    ) -> None:
        """
        Add a static point marker.
        """
        point_arr = self._as_point(point)

        mesh = pv.PolyData(point_arr[np.newaxis, :])
        self.plotter.add_mesh(
            mesh,
            color=color,
            point_size=size,
            render_points_as_spheres=True,
        )

        if name is not None:
            self._make_label_actor(
                point=point_arr,
                text=name,
                font_size=label_font_size,
                shape_opacity=0.10,
                always_visible=True,
            )

    # -------------------------------------------------------------------------
    # Updateable artists
    # -------------------------------------------------------------------------

    def create_frame_artist(
        self,
        frame: Frame,
        scale: float = 1.0,
        line_width: float = 2.0,
        name: str | None = None,
        show_label: bool = True,
        label_offset: float = 0.1,
        label_font_size: int = 14,
        label_shape_opacity: float = 0.25,
        label_always_visible: bool = True,
    ) -> FrameArtist:
        """
        Create a frame artist that can later be updated.
        """
        origin, px, py, pz = self._frame_axis_endpoints(frame, scale)

        x_mesh = pv.Line(origin, px)
        y_mesh = pv.Line(origin, py)
        z_mesh = pv.Line(origin, pz)

        self.plotter.add_mesh(x_mesh, color="red", line_width=line_width)
        self.plotter.add_mesh(y_mesh, color="green", line_width=line_width)
        self.plotter.add_mesh(z_mesh, color="blue", line_width=line_width)

        label_actor = None
        if show_label and name is not None:
            label_point = pz.copy()
            label_point[2] += label_offset
            label_actor = self._make_label_actor(
                point=label_point,
                text=name,
                font_size=label_font_size,
                shape_opacity=label_shape_opacity,
                always_visible=label_always_visible,
            )

        return FrameArtist(
            x_mesh=x_mesh,
            y_mesh=y_mesh,
            z_mesh=z_mesh,
            label_actor=label_actor,
            label_text=name if show_label else None,
            label_offset=label_offset,
            label_font_size=label_font_size,
            label_shape_opacity=label_shape_opacity,
            label_always_visible=label_always_visible,
        )

    def update_frame_artist(
        self,
        artist: FrameArtist,
        frame: Frame,
        scale: float = 1.0,
        render: bool = False,
    ) -> None:
        """
        Update an existing frame artist.

        Both translational and rotational motion are supported because the full
        pose is recomputed from the given frame.
        """
        origin, px, py, pz = self._frame_axis_endpoints(frame, scale)

        self._set_line_points(artist.x_mesh, origin, px)
        self._set_line_points(artist.y_mesh, origin, py)
        self._set_line_points(artist.z_mesh, origin, pz)

        if artist.label_text is not None:
            self._remove_actor_if_needed(artist.label_actor)

            label_point = pz.copy()
            label_point[2] += artist.label_offset

            artist.label_actor = self._make_label_actor(
                point=label_point,
                text=artist.label_text,
                font_size=artist.label_font_size,
                shape_opacity=artist.label_shape_opacity,
                always_visible=artist.label_always_visible,
            )

        if render:
            self.plotter.render()

    def create_link_artist(
        self,
        p1: ArrayLike3,
        p2: ArrayLike3,
        color: str = "dimgray",
        line_width: float = 5.0,
    ) -> LinkArtist:
        """
        Create an updateable link artist.
        """
        p1_arr = self._as_point(p1)
        p2_arr = self._as_point(p2)

        mesh = pv.Line(p1_arr, p2_arr)
        self.plotter.add_mesh(mesh, color=color, line_width=line_width)
        return LinkArtist(mesh=mesh)

    def update_link_artist(
        self,
        artist: LinkArtist,
        p1: ArrayLike3,
        p2: ArrayLike3,
        render: bool = False,
    ) -> None:
        """
        Update an existing link artist.
        """
        self._set_line_points(artist.mesh, p1, p2)

        if render:
            self.plotter.render()

    def create_point_artist(
        self,
        point: ArrayLike3,
        color: str = "black",
        size: float = 12.0,
        name: str | None = None,
        label_font_size: int = 12,
        label_shape_opacity: float = 0.10,
        label_always_visible: bool = True,
    ) -> PointArtist:
        """
        Create an updateable point artist.
        """
        point_arr = self._as_point(point)

        mesh = pv.PolyData(point_arr[np.newaxis, :])
        self.plotter.add_mesh(
            mesh,
            color=color,
            point_size=size,
            render_points_as_spheres=True,
        )

        label_actor = None
        if name is not None:
            label_actor = self._make_label_actor(
                point=point_arr,
                text=name,
                font_size=label_font_size,
                shape_opacity=label_shape_opacity,
                always_visible=label_always_visible,
            )

        return PointArtist(
            mesh=mesh,
            label_actor=label_actor,
            label_text=name,
            label_font_size=label_font_size,
            label_shape_opacity=label_shape_opacity,
            label_always_visible=label_always_visible,
        )

    def update_point_artist(
        self,
        artist: PointArtist,
        point: ArrayLike3,
        render: bool = False,
    ) -> None:
        """
        Update an existing point artist.
        """
        point_arr = self._as_point(point)
        artist.mesh.points = point_arr[np.newaxis, :]

        if artist.label_text is not None:
            self._remove_actor_if_needed(artist.label_actor)
            artist.label_actor = self._make_label_actor(
                point=point_arr,
                text=artist.label_text,
                font_size=artist.label_font_size,
                shape_opacity=artist.label_shape_opacity,
                always_visible=artist.label_always_visible,
            )

        if render:
            self.plotter.render()

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    def show(self, **kwargs) -> None:
        """
        Show the scene.

        Any keyword arguments are passed to ``pyvista.Plotter.show``.
        """
        self.plotter.show(**kwargs)

    async def show_async(self, **kwargs) -> None:
        """
        Render the scene asynchronously.

        This is mainly useful when using a PyVista backend that relies on a
        Trame/Jupyter server.
        """
        self.plotter.show(jupyter_backend="server", **kwargs)
        await pv.trame.jupyter.launch_server(wslink_backend="jupyter").ready

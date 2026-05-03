from __future__ import annotations

from dataclasses import dataclass

import pyvista as pv


@dataclass
class FrameArtist:
    """
    Container for the PyVista objects that represent a coordinate frame.

    The line meshes are updated in place during animation. The optional label actor
    is recreated whenever the frame pose changes.
    """
    x_mesh: pv.PolyData
    y_mesh: pv.PolyData
    z_mesh: pv.PolyData
    label_actor: object | None = None
    label_text: str | None = None
    label_offset: float = 0.1
    label_font_size: int = 14
    label_shape_opacity: float = 0.25
    label_always_visible: bool = True


@dataclass
class LinkArtist:
    """
    Container for the PyVista object that represents a links.
    """
    mesh: pv.PolyData


@dataclass
class PointArtist:
    """
    Container for the PyVista objects that represent a point marker.
    """
    mesh: pv.PolyData
    label_actor: object | None = None
    label_text: str | None = None
    label_font_size: int = 12
    label_shape_opacity: float = 0.10
    label_always_visible: bool = True
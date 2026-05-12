from __future__ import annotations

from copy import deepcopy

import numpy as np
from roboticstoolbox import Link as RTBLink

from ..link import AbstractLink, LinkDynamicParams

__all__ = ["URDFLink"]


class URDFLink(AbstractLink):
    """
    Adapter around a Robotics Toolbox Link imported from URDF.
    """
    def __init__(self, rtb_link: RTBLink) -> None:
        link = deepcopy(rtb_link)
        self._detach_from_urdf_tree(link)

        super().__init__(
            link_length=self._estimate_link_length(link),
            rtb_link=link,
            dynamics=self._extract_dynamics(link),
        )

    @staticmethod
    def _detach_from_urdf_tree(link: RTBLink) -> None:
        link._parent = None
        link._parent_name = None
        link._children = []

    @staticmethod
    def _estimate_link_length(link: RTBLink) -> float:
        pose = link.A(0.0)
        matrix = pose.A if hasattr(pose, "A") else np.asarray(pose, dtype=float)
        return float(np.linalg.norm(matrix[:3, 3]))

    @staticmethod
    def _extract_dynamics(link: RTBLink) -> LinkDynamicParams | None:
        mass = float(link.m)
        center_of_mass = np.asarray(link.r, dtype=float)
        inertia = np.asarray(link.I, dtype=float)

        if (
            np.isclose(mass, 0.0)
            and np.allclose(center_of_mass, 0.0)
            and np.allclose(inertia, 0.0)
        ):
            return None

        return LinkDynamicParams(
            mass=mass,
            center_of_mass=center_of_mass,
            inertia=inertia,
            motor_inertia=float(link.Jm),
            viscous_friction=float(link.B),
            coulomb_friction=np.asarray(link.Tc, dtype=float),
            gear_ratio=float(link.G),
        )

    @property
    def is_revolute(self) -> bool:
        return bool(self.rtb_link.isrevolute)

    @property
    def is_prismatic(self) -> bool:
        return bool(self.rtb_link.isprismatic)

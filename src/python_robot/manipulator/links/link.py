"""
Shared link abstractions for serial manipulators.

The module defines dynamic link parameters and the abstract link interfaces used
by Denavit-Hartenberg, ETS, URDF, and PoE link implementations.
"""

from typing import Sequence, Any

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from roboticstoolbox import Link as RTBLink
from roboticstoolbox import ETS

from ...base.types import NumpyArray
from ...base import Frame

__all__ = ["LinkDynamicParams", "AbstractLink"]


@dataclass(frozen=True)
class LinkDynamicParams:
    """
    Dynamic parameters of a robot link.

    Parameters
    ----------
    mass : float, default = 0.0
        Link mass.
    center_of_mass : Sequence[float], default = (0.0, 0.0, 0.0)
        Center of mass expressed in the link frame.
    inertia : NumpyArray | Sequence[float] | None, optional
        Link inertia tensor expressed in the link frame.
    motor_inertia : float, default = 0.0
        Reflected motor inertia.
    viscous_friction : float, default = 0.0
        Viscous friction coefficient.
    coulomb_friction : float | Sequence[float], default = 0.0
        Coulomb friction value or positive/negative friction pair.
    gear_ratio : float, default = 1.0
        Gear ratio used by the underlying Robotics Toolbox link.
    """
    mass: float = 0.0
    center_of_mass: Sequence[float] = (0.0, 0.0, 0.0)
    inertia: NumpyArray | Sequence[float] | None = None

    motor_inertia: float = 0.0
    viscous_friction: float = 0.0
    coulomb_friction: float | Sequence[float] = 0.0
    gear_ratio: float = 1.0

    def inertia_matrix(self) -> NumpyArray:
        """
        Return the link inertia tensor as a 3-by-3 matrix.
        """
        if self.inertia is None:
            return np.zeros((3, 3))
        return np.asarray(self.inertia, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        """
        Return the dynamic parameters as a plain dictionary.
        """
        return {
            "mass": self.mass,
            "center_of_mass": self.center_of_mass,
            "inertia": self.inertia,
            "motor_inertia": self.motor_inertia,
            "viscous_friction": self.viscous_friction,
            "coulomb_friction": self.coulomb_friction,
            "gear_ratio": self.gear_ratio
        }

    def __str__(self) -> str:
        lines = [f"{k}: {v}" for k, v in self.to_dict().items()]
        lines[2] = f"inertia:\n{self.inertia}"
        return "\n".join(lines)


class AbstractLink(ABC):
    """
    Abstract base class for a single serial-manipulator link.

    Parameters
    ----------
    link_length : float
        Geometric length used by plotting and visualisation helpers.
    rtb_link : RTBLink
        Robotics Toolbox link that stores the kinematic and dynamic model.
    dynamics : LinkDynamicParams, optional
        Dynamic parameters copied to the Robotics Toolbox link.
    """

    def __init__(
        self,
        link_length: float,
        rtb_link: RTBLink,
        dynamics: LinkDynamicParams | None = None
    ) -> None:
        """
        Create a link wrapper around a Robotics Toolbox link.
        """
        self._link_length = link_length
        self._rtb_link = rtb_link
        self._dynamics = dynamics

        self._variable: float = 0.0

        self._add_dynamics_to_rtb_link(self._dynamics)

    def _add_dynamics_to_rtb_link(
        self,
        dynamics: LinkDynamicParams | None
    ) -> None:
        if dynamics is None:
            return

        self._rtb_link.m = dynamics.mass
        self._rtb_link.r = np.asarray(dynamics.center_of_mass, dtype=float)
        self._rtb_link.I = dynamics.inertia_matrix()
        self._rtb_link.Jm = dynamics.motor_inertia
        self._rtb_link.B = dynamics.viscous_friction
        self._rtb_link.Tc = dynamics.coulomb_friction
        self._rtb_link.G = dynamics.gear_ratio

    @property
    def rtb_link(self) -> RTBLink:
        """
        Return the wrapped Robotics Toolbox link.
        """
        return self._rtb_link

    @property
    def variable(self) -> float:
        """
        Return the current joint variable of the link.
        """
        return self._variable

    @variable.setter
    def variable(self, v: float) -> None:
        """
        Set the current joint variable of the link.
        """
        self._variable = v

    @property
    def frame(self) -> Frame:
        """
        Return this link frame relative to the previous link frame.
        """
        SE3_mat = self._rtb_link.A(self._variable)
        return Frame.from_matrix(SE3_mat)

    @property
    def ets(self) -> ETS:
        """
        Return the Elementary Transform Sequence of this link.
        """
        return self._rtb_link.ets

    @property
    def link_length(self) -> float:
        """
        Return the nominal link length used for visualisation.
        """
        return self._link_length

    @property
    def dynamics(self) -> LinkDynamicParams | None:
        """
        Return the dynamic parameters assigned to the link, if any.
        """
        return self._dynamics

    @property
    def q_lim(self) -> NumpyArray | None:
        """
        Return the lower and upper joint limits, if configured.
        """
        return self._rtb_link.qlim

    @q_lim.setter
    def q_lim(self, v: tuple[float, float]) -> None:
        """
        Set the lower and upper joint limits.
        """
        self._rtb_link.qlim = v

    @property
    @abstractmethod
    def is_revolute(self) -> bool:
        """
        Return True when this link has a revolute joint.
        """
        ...

    @property
    @abstractmethod
    def is_prismatic(self) -> bool:
        """
        Return True when this link has a prismatic joint.
        """
        ...


class AbstractRevoluteLink(AbstractLink, ABC):
    """
    Base class for links whose joint variable is a rotation angle.
    """

    @property
    def joint_angle(self) -> float:
        """
        Return the current revolute joint angle.
        """
        return self._variable

    @joint_angle.setter
    def joint_angle(self, v: float) -> None:
        """
        Set the current revolute joint angle.
        """
        self._variable = v

    @property
    def is_revolute(self) -> bool:
        """
        Return True for revolute links.
        """
        return True

    @property
    def is_prismatic(self) -> bool:
        """
        Return False for revolute links.
        """
        return False


class AbstractPrismaticLink(AbstractLink, ABC):
    """
    Base class for links whose joint variable is a linear displacement.
    """

    @property
    def link_offset(self) -> float:
        """
        Return the current prismatic link offset.
        """
        return self._variable

    @link_offset.setter
    def link_offset(self, v: float) -> None:
        """
        Set the current prismatic link offset.
        """
        self._variable = v

    @property
    def is_prismatic(self) -> bool:
        """
        Return True for prismatic links.
        """
        return True

    @property
    def is_revolute(self) -> bool:
        """
        Return False for prismatic links.
        """
        return False

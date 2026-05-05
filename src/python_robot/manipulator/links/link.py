from typing import Sequence

from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
from roboticstoolbox import ETS
from roboticstoolbox import Link as RTBLink

from python_robot.base.types import NumpyArray, AngleUnit
from python_robot.base import Frame


__all__ = ["LinkDynamicParams", "AbstractLink"]


@dataclass(frozen=True)
class LinkDynamicParams:
    """
    Dynamic parameters of a robot link.

    Parameters follow the conventions of Robotics Toolbox for Python.
    """
    mass: float = 0.0
    center_of_mass: Sequence[float] = (0.0, 0.0, 0.0)
    inertia: NumpyArray | Sequence[float] | None = None

    motor_inertia: float = 0.0
    viscous_friction: float = 0.0
    coulomb_friction: float | Sequence[float] = 0.0
    gear_ratio: float = 1.0

    def inertia_matrix(self) -> NumpyArray:
        if self.inertia is None:
            return np.zeros((3, 3))
        return np.asarray(self.inertia, dtype=float)


class AbstractLink(ABC):
    """
    Abstract base class for a robot link.

    Attributes
    ----------
    name : str
        Name to identify the link in the kinematic chain.
    angle_unit : AngleUnit
        Angle unit used for the angles used in configuring the link.
    dynamics : LinkDynamicParams
        Dynamic parameters of the link.
    """
    @dataclass(frozen=True)
    class defaults:
        angle_unit: AngleUnit = "rad"  # same unit for all angles present in the links

    def __init__(
        self,
        angle_unit: AngleUnit | None = None,
        dynamics: LinkDynamicParams | None = None,
        q_lim: Sequence[float] | NumpyArray | None = None,
    ) -> None:
        """
        Base initializer for classes derived from AbstractLink.
        """
        self.angle_unit: AngleUnit = self.defaults.angle_unit if angle_unit is None else angle_unit  #type: ignore
        self._variable: float | None = None
        self._q_lim = self._set_q_lim(q_lim)
        self.name: str = ""
        self.dynamics = dynamics

    def _set_q_lim(self, q_lim: Sequence[float] | NumpyArray | None) -> NumpyArray | None:
        if q_lim is None:
            return None

        limits = np.asarray(q_lim, dtype=float)
        if limits.shape != (2,):
            raise ValueError("q_lim must contain exactly two values: lower and upper joint limit.")

        if self.is_revolute and self.angle_unit == "deg":
            return np.deg2rad(limits)
        return limits

    @property
    def q_lim(self) -> NumpyArray | None:
        if self._q_lim is None:
            return None
        if self.is_revolute and self.angle_unit == "deg":
            return np.rad2deg(self._q_lim)
        return self._q_lim.copy()

    @property
    def q_lim_internal(self) -> NumpyArray | None:
        if self._q_lim is None:
            return None
        return self._q_lim.copy()

    def _rtb_q_lim_kwargs(self) -> dict[str, NumpyArray]:
        return {} if self._q_lim is None else {"qlim": self._q_lim}

    @property
    @abstractmethod
    def variable(self) -> float | None:
        return self._variable

    @variable.setter
    @abstractmethod
    def variable(self, v: float) -> None:
        ...

    def is_configured(self) -> bool:
        """
        Returns True if the links is fully configured (i.e., the joint variable
        of the links is set), False otherwise.
        """
        return bool(self._variable is not None)

    @abstractmethod
    def _get_frame(self) -> Frame:
        ...

    @property
    def frame(self) -> Frame:
        """
        Returns the links frame, if the joint variable is set. If the joint
        variable is not set (None), a ValueError is raised that the links has
        not been configured yet.

        Returns
        -------
        Frame
        """
        if self.is_configured():
            return self._get_frame()
        raise ValueError("links is not configured.")

    @property
    @abstractmethod
    def ets(self) -> ETS:
        """
        Returns the underlying Robotics Toolbox ETS object.
        """
        ...

    @property
    @abstractmethod
    def is_revolute(self) -> bool:
        """
        Returns True if the links has a revolute joint.
        """
        ...

    @property
    @abstractmethod
    def is_prismatic(self) -> bool:
        """
        Returns True if the links has a prismatic joint.
        """
        ...

    @property
    @abstractmethod
    def link_length(self) -> float:
        """
        Returns the length of the links.
        """
        ...

    @property
    @abstractmethod
    def rtb_link(self) -> RTBLink:
        """
        Returns the underlying Robotics Toolbox links object.
        """
        ...

    def _apply_dynamics_to_rtb_link(self, link: RTBLink) -> None:
        """
        Copies the stored dynamic parameters to a Robotics Toolbox Link object.
        """
        if self.dynamics is None:
            return

        dyn = self.dynamics
        link.m = dyn.mass
        link.r = np.asarray(dyn.center_of_mass, dtype=float)
        link.I = dyn.inertia_matrix()
        link.Jm = dyn.motor_inertia
        link.B = dyn.viscous_friction
        link.Tc = dyn.coulomb_friction
        link.G = dyn.gear_ratio

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

    def to_dict(self) -> dict[str, Any]:
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

    def __init__(
        self,
        link_length: float,
        rtb_link: RTBLink,
        dynamics: LinkDynamicParams | None = None
    ) -> None:
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
        return self._rtb_link

    @property
    def variable(self) -> float:
        return self._variable

    @variable.setter
    def variable(self, v: float) -> None:
        self._variable = v

    @property
    def frame(self) -> Frame:
        SE3_mat = self._rtb_link.A(self._variable)
        return Frame.from_matrix(SE3_mat)

    @property
    def ets(self) -> ETS:
        return self._rtb_link.ets

    @property
    def link_length(self) -> float:
        return self._link_length

    @property
    def dynamics(self) -> LinkDynamicParams | None:
        return self._dynamics

    @property
    def q_lim(self) -> NumpyArray | None:
        return self._rtb_link.qlim

    @q_lim.setter
    def q_lim(self, v: tuple[float, float]) -> None:
        self._rtb_link.qlim = v

    @property
    @abstractmethod
    def is_revolute(self) -> bool:
        ...

    @property
    @abstractmethod
    def is_prismatic(self) -> bool:
        ...


class AbstractRevoluteLink(AbstractLink, ABC):

    @property
    def joint_angle(self) -> float:
        return self._variable

    @joint_angle.setter
    def joint_angle(self, v: float) -> None:
        self._variable = v

    @property
    def is_revolute(self) -> bool:
        return True

    @property
    def is_prismatic(self) -> bool:
        return False


class AbstractPrismaticLink(AbstractLink, ABC):

    @property
    def link_offset(self) -> float:
        return self._variable

    @link_offset.setter
    def link_offset(self, v: float) -> None:
        self._variable = v

    @property
    def is_prismatic(self) -> bool:
        return True

    @property
    def is_revolute(self) -> bool:
        return False

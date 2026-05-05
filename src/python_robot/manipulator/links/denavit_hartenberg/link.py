from abc import ABC
from typing import Sequence

import numpy as np
from roboticstoolbox import RevoluteDH, PrismaticDH, RevoluteMDH, PrismaticMDH, ETS
from roboticstoolbox import Link as RTBLink

from python_robot.base.types import AngleUnit, NumpyArray
from python_robot.base import Frame

from ..link import AbstractLink, LinkDynamicParams


DHLink = RevoluteDH | PrismaticDH | RevoluteMDH | PrismaticMDH | None


class AbstractDHLink(AbstractLink, ABC):

    def __init__(
        self,
        length: float,
        twist: float,
        offset: float | None,
        angle: float | None,
        angle_unit: AngleUnit | None = None,
        dynamics: LinkDynamicParams | None = None,
        q_lim: Sequence[float] | NumpyArray | None = None,
    ) -> None:
        super().__init__(angle_unit, dynamics, q_lim)

        self.length = length
        self.twist = self._set_twist(twist)
        self._offset = offset
        self._angle = angle
        self._rtb_link: DHLink = None

    @property
    def rtb_link(self) -> RTBLink:
        """Returns the underlying Link object from Robotics Toolbox."""
        if self._rtb_link is None:
            raise ValueError("The underlying Robotics Toolbox link are not initialized.")
        return self._rtb_link

    @property
    def ets(self) -> ETS:
        """Returns the underlying ETS object from Robotics Toolbox."""
        if self._rtb_link is None:
            raise ValueError("The underlying Robotics Toolbox link are not initialized.")
        return self._rtb_link.ets

    def _set_twist(self, twist: float) -> float:
        return twist if self.angle_unit == "rad" else float(np.deg2rad(twist))

    def _get_frame(self) -> Frame:
        if self._rtb_link is not None and self._variable is not None:
            SE3_mat = self._rtb_link.A(self._variable)
            return Frame.from_matrix(SE3_mat, angle_unit="rad")
        raise ValueError("Link is not configured.")

    @property
    def link_length(self) -> float:
        return self.length


class AbstractRevoluteDHLink(AbstractDHLink, ABC):

    def __init__(
        self,
        length: float,
        twist: float,
        offset: float,
        angle_unit: AngleUnit | None = None,
        dynamics: LinkDynamicParams | None = None,
        q_lim: Sequence[float] | NumpyArray | None = None,
    ) -> None:
        super().__init__(length, twist, offset, None, angle_unit, dynamics, q_lim)

    @property
    def angle(self) -> float | None:
        """
        Returns the rotation angle of the link.
        If self.angle_unit == "deg", the angle is returned in degrees.
        """
        if self._angle is not None:
            if self.angle_unit == "rad":
                return self._angle
            else:
                return float(np.rad2deg(self._angle))
        return None

    @angle.setter
    def angle(self, v: float) -> None:
        """
        Sets the rotation angle of the link.
        If self.angle_unit == "deg", the angle must be set in degrees.
        """
        self._angle = v if self.angle_unit == "rad" else float(np.deg2rad(v))
        self._variable = self._angle

    @property
    def variable(self) -> float | None:
        return super().variable

    @variable.setter
    def variable(self, v: float) -> None:
        self.angle = v

    @property
    def is_revolute(self) -> bool:
        return True

    @property
    def is_prismatic(self) -> bool:
        return False


class AbstractPrismaticDHLink(AbstractDHLink, ABC):

    def __init__(
        self,
        length: float,
        twist: float,
        angle: float,
        angle_unit: AngleUnit | None = None,
        dynamics: LinkDynamicParams | None = None,
        q_lim: Sequence[float] | NumpyArray | None = None,
    ) -> None:
        resolved_angle_unit = self.defaults.angle_unit if angle_unit is None else angle_unit
        angle = np.deg2rad(angle) if resolved_angle_unit == "deg" else angle
        super().__init__(length, twist, None, angle, angle_unit, dynamics, q_lim)

    @property
    def offset(self) -> float:
        """Returns the linear offset of the link."""
        return self._offset

    @offset.setter
    def offset(self, v: float) -> None:
        """Sets the linear offset of the link."""
        self._offset = v
        self._variable = self._offset

    @property
    def variable(self) -> float | None:
        return super().variable

    @variable.setter
    def variable(self, v: float) -> None:
        self.offset = v

    @property
    def angle(self) -> float:
        """
        Returns the fixed joint angle of the prismatic link.
        If self.angle_unit == "deg", the angle is returned in degrees.
        """
        return self._angle if self.angle_unit == "rad" else float(np.rad2deg(self._angle))  #type: ignore

    @property
    def is_revolute(self) -> bool:
        return False

    @property
    def is_prismatic(self) -> bool:
        return True

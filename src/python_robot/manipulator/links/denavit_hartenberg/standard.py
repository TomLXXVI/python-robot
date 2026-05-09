"""
Standard Denavit-Hartenberg notation.

Link frames are placed at the far (distal) end of each link.
"""
from typing import Sequence

import numpy as np
from roboticstoolbox import RevoluteDH, PrismaticDH

from python_robot.base.types import AngleUnit, NumpyArray

from .link import AbstractRevoluteDHLink, AbstractPrismaticDHLink, LinkDynamicParams


__all__ = ["StandardRevoluteDHLink", "StandardPrismaticDHLink", "Standard"]


class StandardRevoluteDHLink(AbstractRevoluteDHLink):

    def __init__(
        self,
        length: float,
        twist: float,
        offset: float,
        zero_joint_angle: float = 0.0,
        angle_unit: AngleUnit | None = None,
        dynamics: LinkDynamicParams | None = None,
        q_lim: Sequence[float] | NumpyArray | None = None,
    ) -> None:
        """
        Creates a revolute link.

        Parameters
        ----------
        length: float
            Link length of this link.
        twist: float
            Twist angle of this link.
        offset: float
            Link offset with respect to the next link.
        zero_joint_angle: float, default = 0.0
            Joint angle when the manipulator is in the "zero-joint" shape.
        angle_unit: AngleUnit | None, optional
            The unit used for *all* type of angles (twist angle and joint angle)
            entered by the user (either "deg" or "rad").
            If None, uses the default unit set at the class level (which is
            "rad", but this can be modified through class attribute
            defaults.angle_unit).
            Note that internally all angles will be converted to radians.
        dynamics: LinkDynamics, optional
            Adds dynamic properties to this link.
        q_lim: Sequence[float] | NumpyArray | None, optional
            Mechanical joint limits. Revolute limits use angle_unit and are
            converted to radians internally.
        """
        super().__init__(length, twist, offset, angle_unit, dynamics, q_lim)
        self._rbt_link: RevoluteDH = RevoluteDH(
            d=self._offset,  # type: ignore
            a=self.length,
            alpha=self.twist,
            offset=np.deg2rad(zero_joint_angle) if angle_unit == "deg" else zero_joint_angle,
            **self._rbt_q_lim_kwargs()
        )
        self._apply_dynamics_to_rbt_link(self._rbt_link)


class StandardPrismaticDHLink(AbstractPrismaticDHLink):

    def __init__(
        self,
        length: float,
        twist: float,
        angle: float,
        zero_joint_offset: float = 0.0,
        angle_unit: AngleUnit | None = None,
        dynamics: LinkDynamicParams | None = None,
        q_lim: Sequence[float] | NumpyArray | None = None,
    ) -> None:
        """
        Creates a prismatic link.

        Parameters
        ----------
        length: float
            Link length of this link.
        twist: float
            Twist angle of this link.
        angle: float
            Fixed joint angle with respect to the next link.
        zero_joint_offset: float, default = 0.0
            The offset distance when the manipulator is in the "zero-joint" 
            shape.
        angle_unit: AngleUnit | None, optional
            The unit used for *all* type of angles (twist angle and joint angle)
            entered by the user (either "deg" or "rad").
            If None, uses the default unit set at the class level (which is
            "rad", but this can be modified through class attribute
            defaults.angle_unit).
            Note that internally all angles will be converted to radians.
        dynamics: LinkDynamics, optional
            Adds dynamic properties to this link.
        q_lim: Sequence[float] | NumpyArray | None, optional
            Mechanical joint limits.
        """
        super().__init__(length, twist, angle, angle_unit, dynamics, q_lim)
        self._rbt_link: PrismaticDH = PrismaticDH(
            theta=self._angle,  # type: ignore
            a=self.length,
            alpha=self.twist,
            offset=zero_joint_offset,
            **self._rbt_q_lim_kwargs()
        )
        self._apply_dynamics_to_rbt_link(self._rbt_link)


class Standard:
    """
    Convenience class that groups the link definitions according to the standard
    Denavit-Hartenberg notation.
    """
    RevoluteLink = RLink = StandardRevoluteDHLink
    PrismaticLink = PLink = StandardPrismaticDHLink

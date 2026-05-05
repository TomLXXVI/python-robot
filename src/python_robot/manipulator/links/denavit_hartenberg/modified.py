"""
Modified Denavit-Hartenberg notation.

Link frames are placed at the near (proximal) end of each link.
"""
from typing import Sequence

from roboticstoolbox import RevoluteMDH, PrismaticMDH

from python_robot.base.types import AngleUnit, NumpyArray

from .link import AbstractRevoluteDHLink, AbstractPrismaticDHLink, LinkDynamicParams

__all__ = ["ModifiedLinkRevolute", "ModifiedLinkPrismatic", "Modified"]


class ModifiedLinkRevolute(AbstractRevoluteDHLink):

    def __init__(
        self,
        length: float,
        twist: float,
        offset: float,
        angle_unit: AngleUnit | None = None,
        dynamics: LinkDynamicParams | None = None,
        q_lim: Sequence[float] | NumpyArray | None = None,
    ) -> None:
        """
        Creates a revolute link.

        Parameters
        ----------
        length: float
            Link length of previous link.
        twist: float
            Twist angle of previous link.
        offset: float
            Link offset with respect to the previous link.
        angle_unit: AngleUnit | None, optional
            The unit used for *all* type of angles (twist angle and joint angle)
            entered by the user (either "deg" or "rad").
            If None, uses the default unit set at the class level (which is
            "rad", but this can be modified through class attribute
            defaults.angle_unit).
            Note that internally all angles will be converted to radians.
        dynamics: LinkDynamics, optional
            Adds dynamic properties to the link.
        q_lim: Sequence[float] | NumpyArray | None, optional
            Mechanical joint limits. Revolute limits use angle_unit and are
            converted to radians internally.
        """
        super().__init__(length, twist, offset, angle_unit, dynamics, q_lim)
        self._rtb_link: RevoluteMDH = RevoluteMDH(
            self._offset,  #type: ignore
            self.length,
            self.twist,
            **self._rtb_q_lim_kwargs()
        )
        self._apply_dynamics_to_rtb_link(self._rtb_link)


class ModifiedLinkPrismatic(AbstractPrismaticDHLink):

    def __init__(
        self,
        length: float,
        twist: float,
        angle: float,
        angle_unit: AngleUnit | None = None,
        dynamics: LinkDynamicParams | None = None,
        q_lim: Sequence[float] | NumpyArray | None = None,
    ) -> None:
        """
        Creates a prismatic link.

        Parameters
        ----------
        length: float
            Link length of previous link.
        twist: float
            Twist angle of previous link.
        angle: float
            Fixed joint angle with respect to the previous link.
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
        self._rtb_link: PrismaticMDH = PrismaticMDH(
            self._angle,  #type: ignore
            self.length,
            self.twist,
            **self._rtb_q_lim_kwargs()
        )
        self._apply_dynamics_to_rtb_link(self._rtb_link)


class Modified:
    """
    Convenience class that groups the link definitions according to the modified
    Denavit-Hartenberg notation.
    """
    RevoluteLink = RLink = ModifiedLinkRevolute
    PrismaticLink = PLink = ModifiedLinkPrismatic

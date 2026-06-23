"""
Abstract Denavit-Hartenberg link classes.

Concrete subclasses select the standard or modified Robotics Toolbox DH link
constructor while sharing parameter handling for revolute and prismatic joints.
"""

from typing import Type

from abc import ABC

from roboticstoolbox import (
    RevoluteDH, RevoluteMDH,
    PrismaticDH, PrismaticMDH
)

from ..link import (
    AbstractLink, LinkDynamicParams,
    AbstractRevoluteLink, AbstractPrismaticLink
)


class AbstractDHLink(AbstractLink, ABC):
    """
    Abstract base class for links defined with DH parameters.
    """

    def __init__(
        self,
        link_length: float,
        rtb_link: RevoluteDH | RevoluteMDH | PrismaticDH | PrismaticMDH,
        dynamics: LinkDynamicParams | None = None,
    ) -> None:
        """
        Create a DH link wrapper.
        """
        super().__init__(link_length, rtb_link, dynamics)


class AbstractRevoluteDHLink(AbstractDHLink, AbstractRevoluteLink):
    """
    Abstract base class for standard or modified revolute DH links.
    """

    _rtb_link_constructor: Type[RevoluteDH | RevoluteMDH]

    def __init__(
        self,
        link_length: float,
        twist_angle: float,
        link_offset: float,
        zero_joint_angle: float = 0.0,
        limits_joint_angle: tuple[float, float] | None = None,
        dynamics: LinkDynamicParams | None = None,
    ) -> None:
        """
        Creates a revolute link based on the Denavit-Hartenberg notation.

        Two different notations are possible: standard notation and modified
        notation (see class RevoluteSDHLink in standard.py c.q. class
        RevoluteMDHLink in modified.py).

        Parameters
        ----------
        link_length: float
            Length of the link.
        twist_angle: float
            Twist angle of the link.
        link_offset: float
            Offset distance between this link and its neighboring link (depends
            on which type of Denavit-Hartenberg notation is used).
        zero_joint_angle: float
            The joint angle when the manipulator is in its "zero-shape"
            configuration.
        limits_joint_angle: tuple[float, float], optional
            Lower and upper mechanical limit of the revolute joint.
        dynamics: LinkDynamicParams, optional
            Data-object containing dynamical parameters of the revolute link.
        """
        rtb_link: RevoluteDH | RevoluteMDH = self._rtb_link_constructor(
            d=link_offset,
            a=link_length,
            alpha=twist_angle,
            offset=zero_joint_angle,
            qlim=limits_joint_angle
        )
        super().__init__(link_length, rtb_link, dynamics)


class AbstractPrismaticDHLink(AbstractDHLink, AbstractPrismaticLink):
    """
    Abstract base class for standard or modified prismatic DH links.
    """

    _rtb_link_constructor: Type[PrismaticDH | PrismaticMDH]

    def __init__(
        self,
        link_length: float,
        twist_angle: float,
        joint_angle: float,
        zero_link_offset: float = 0.0,
        limits_link_offset: tuple[float, float] | None = None,
        dynamics: LinkDynamicParams | None = None,
    ) -> None:
        """
        Creates a prismatic link based on the Denavit-Hartenberg notation.

        Two different notations are possible: standard notation and modified
        notation (see class PrismaticDHLink in standard.py c.q. class
        PrismaticMDH in modified.py).

        Parameters
        ----------
        link_length: float
            Length of the link.
        twist_angle: float
            Twist angle of the link.
        joint_angle: float
            Fixed angle between this link and its neighboring link (depends on
            which type of Denavit-Hartenberg notation is used).
        zero_link_offset: float, default=0.0
            Link offset distance between this link and its neighboring link
            (depends on which type of Denavit-Hartenberg notation is used).
        limits_link_offset: tuple[float, float], optional
            Lower and upper mechanical limit of the prismatic link.
        dynamics: LinkDynamicParams, optional
            Data-object containing dynamical parameters of the prismatic link.
        """
        rtb_link: PrismaticDH | PrismaticMDH = self._rtb_link_constructor(
            theta=joint_angle,
            a=link_length,
            alpha=twist_angle,
            offset=zero_link_offset,
            qlim=limits_link_offset
        )
        super().__init__(link_length, rtb_link, dynamics)

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

    def __init__(
        self,
        link_length: float,
        twist_angle: float,
        rtb_link: RevoluteDH | RevoluteMDH | PrismaticDH | PrismaticMDH,
        dynamics: LinkDynamicParams = None,
    ) -> None:
        self._twist_angle = twist_angle
        super().__init__(link_length, rtb_link, dynamics)


class AbstractRevoluteDHLink(AbstractDHLink, AbstractRevoluteLink):
    _rtb_constructor: Type[RevoluteDH | RevoluteMDH]

    def __init__(
        self,
        link_length: float,
        twist_angle: float,
        link_offset: float,
        zero_joint_angle: float = 0.0,
        limits_joint_angle: tuple[float, float] | None = None,
        dynamics: LinkDynamicParams = None,
    ) -> None:
        self._link_offset = link_offset
        self._zero_joint_angle = zero_joint_angle
        self._limits_joint_angle = limits_joint_angle

        rtb_link: RevoluteDH | RevoluteMDH = self._rtb_constructor(
            d=self._link_offset,
            a=link_length,
            alpha=twist_angle,
            offset=self._zero_joint_angle,
            qlim=self._limits_joint_angle
        )

        super().__init__(link_length, twist_angle, rtb_link, dynamics)


class AbstractPrismaticDHLink(AbstractDHLink, AbstractPrismaticLink):
    _rtb_constructor: Type[PrismaticDH | PrismaticMDH]

    def __init__(
        self,
        link_length: float,
        twist_angle: float,
        joint_angle: float,
        zero_link_offset: float = 0.0,
        limits_link_offset: tuple[float, float] | None = None,
        dynamics: LinkDynamicParams = None,
    ) -> None:
        self._joint_angle = joint_angle
        self._zero_link_offset = zero_link_offset
        self._limits_link_offset = limits_link_offset

        rtb_link: PrismaticDH | PrismaticMDH = self._rtb_constructor(
            theta=self._joint_angle,
            a=link_length,
            alpha=twist_angle,
            offset=self._zero_link_offset,
            qlim=self._limits_link_offset
        )

        super().__init__(link_length, twist_angle, rtb_link, dynamics)

from roboticstoolbox import RevoluteMDH, PrismaticMDH

from .link import AbstractRevoluteDHLink, AbstractPrismaticDHLink

__all__ = ["RevoluteMDHLink", "PrismaticMDHLink"]


class RevoluteMDHLink(AbstractRevoluteDHLink):
    _rtb_constructor = RevoluteMDH


class PrismaticMDHLink(AbstractPrismaticDHLink):
    _rtb_constructor = PrismaticMDH

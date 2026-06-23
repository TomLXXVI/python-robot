"""
Modified Denavit-Hartenberg link classes.
"""

from roboticstoolbox import RevoluteMDH, PrismaticMDH

from .link import AbstractRevoluteDHLink, AbstractPrismaticDHLink

__all__ = ["RevoluteMDHLink", "PrismaticMDHLink"]


class RevoluteMDHLink(AbstractRevoluteDHLink):
    """
    Defines a revolute link according to the modified Denavit-Hartenberg
    notation.
    """
    _rtb_link_constructor = RevoluteMDH


class PrismaticMDHLink(AbstractPrismaticDHLink):
    """
    Defines a prismatic link according to the modified Denavit-Hartenberg
    notation.
    """
    _rtb_link_constructor = PrismaticMDH

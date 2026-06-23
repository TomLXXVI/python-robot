"""
Standard Denavit-Hartenberg link classes.
"""

from roboticstoolbox import RevoluteDH, PrismaticDH

from .link import AbstractRevoluteDHLink, AbstractPrismaticDHLink

__all__ = ["RevoluteSDHLink", "PrismaticSDHLink"]


class RevoluteSDHLink(AbstractRevoluteDHLink):
    """
    Defines a revolute link according to the standard Denavit-Hartenberg
    notation.
    """
    _rtb_link_constructor = RevoluteDH


class PrismaticSDHLink(AbstractPrismaticDHLink):
    """
    Defines a prismatic link according to the standard Denavit-Hartenberg
    notation.
    """
    _rtb_link_constructor = PrismaticDH

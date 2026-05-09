from roboticstoolbox import RevoluteDH, PrismaticDH

from .link import AbstractRevoluteDHLink, AbstractPrismaticDHLink

__all__ = ["RevoluteSDHLink", "PrismaticSDHLink"]


class RevoluteSDHLink(AbstractRevoluteDHLink):
    _rtb_constructor = RevoluteDH


class PrismaticSDHLink(AbstractPrismaticDHLink):
    _rtb_constructor = PrismaticDH

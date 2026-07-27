"""Import Robotics Toolbox without changing global Matplotlib settings."""

from collections.abc import Iterator
from contextlib import contextmanager

from matplotlib import rcParams


@contextmanager
def _preserve_matplotlib_settings() -> Iterator[None]:
    """Preserve the active global Matplotlib configuration.

    Yields
    ------
    None
        Control to imports that may modify :data:`matplotlib.rcParams`.

    Notes
    -----
    Use this context around third-party imports that mutate Matplotlib's
    process-wide configuration. All settings are restored on exit, including
    when the import raises an exception.
    """
    saved_settings = rcParams.copy()
    try:
        yield
    finally:
        rcParams.update(saved_settings)


with _preserve_matplotlib_settings():
    from roboticstoolbox import (
        DHLink,
        ERobot,
        ET,
        ETS,
        Link,
        PrismaticDH,
        PrismaticMDH,
        RevoluteDH,
        RevoluteMDH,
    )
    from roboticstoolbox.tools import URDF

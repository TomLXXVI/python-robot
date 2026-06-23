"""
3D visualisation tools for frames, kinematic chains, and robot trajectories.

The package exposes PyVista-based scene primitives and animators. Swift support
is imported when the optional Swift dependency is available.
"""

from .core import *

try:
    from .swift import *
except ModuleNotFoundError as exc:
    if exc.name != "swift":
        raise

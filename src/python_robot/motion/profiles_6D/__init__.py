"""
Vector-valued motion profiles for six-dimensional Cartesian pose paths.

The profiles in this package work on pose vectors of the form
``(x, y, z, rx, ry, rz)`` and provide position, velocity, acceleration,
spatial-velocity, and spatial-acceleration samples.
"""

from ._profile_abc import MultiPointVectorMotionProfile
from .parabolic_blend import MultiLinearVectorPath

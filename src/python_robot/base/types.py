"""
Shared type aliases for robotics geometry and numerical arrays.

The aliases in this module keep public signatures compact while preserving the
distinction between angle units, angular speed units, and NumPy-backed array
values.
"""

from typing import Literal, TypeAlias, Any

import numpy as np

from spatialmath.base import ArrayLike3, ArrayLike6

__all__ = [
    "AngleUnit",
    "AngularSpeedUnit",
    "NumpyArray",
    "ArrayLike3",
    "ArrayLike6"
]

AngleUnit = Literal["deg", "rad"]
AngularSpeedUnit = Literal["deg/s", "rad/s"]
NumpyArray: TypeAlias = np.ndarray[Any, np.dtype[Any]]

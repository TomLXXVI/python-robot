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

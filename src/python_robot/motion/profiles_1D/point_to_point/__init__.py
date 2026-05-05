"""
Point-to-point motion profiles.

Implementation of straight-line motion profiles either based on polynomials or
motion profiles composed of three phases (acceleration, constant velocity, and
deceleration).
"""
from ._profile_abc import MotionProfile
from .polynomials import *
from .triphase import *
from .poly import *

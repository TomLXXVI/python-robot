"""
Motion profiles and motion-planning helpers for robot trajectories.

This package re-exports one-dimensional motion profiles from
``automation_motion`` together with vector-valued 6D Cartesian profiles and
planning utilities for joint-space and Cartesian-space schemes.
"""

from automation_motion.profiles_1D import *
from .profiles_6D import *
from .planning import *

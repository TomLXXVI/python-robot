"""
Motion profiles and trajectory helpers for robot motion planning.

This package re-exports one-dimensional motion profiles from
``automation_motion`` together with joint-space and Cartesian-space trajectory
planning utilities.
"""

from automation_motion.profiles_1D import *
from .joint_space import *
from .cartesian_space import *

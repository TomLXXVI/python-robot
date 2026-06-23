"""
Exceptions raised by manipulator and kinematic-chain operations.
"""

__all__ = [
    "InvalidArgument",
    "ConfigurationError",
    "SingularityError",
    "IKSolverError",
]


class InvalidArgument(Exception):
    """
    Raised when a manipulator method receives an unsupported argument value.
    """

    pass


class ConfigurationError(Exception):
    """
    Raised when a manipulator, link, or joint configuration is inconsistent.
    """

    pass


class SingularityError(Exception):
    """
    Raised when a manipulator operation cannot continue at a singular pose.
    """

    pass


class IKSolverError(Exception):
    """
    Raised when an inverse-kinematics solver cannot find a valid solution.
    """

    pass

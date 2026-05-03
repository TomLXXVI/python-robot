__all__ = [
    "InvalidArgument",
    "ConfigurationError",
    "SingularityError",
    "IKSolverError",
]


class InvalidArgument(Exception):
    pass


class ConfigurationError(Exception):
    pass


class SingularityError(Exception):
    pass


class IKSolverError(Exception):
    pass

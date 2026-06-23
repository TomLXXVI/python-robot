"""
Callable introspection helpers used by plotting and animation dispatchers.
"""

from typing import Callable, Any

import inspect

__all__ = ["get_valid_keyword_parameters"]


def get_valid_keyword_parameters(
    func: Callable[..., Any],
    *,
    exclude: set[str] | None = None
) -> set[str]:
    """
    Returns a set of valid keyword parameter names that can be used to call a
    given function.

    Parameters
    ----------
    func:
        A callable whose valid keyword parameters must be retrieved.
    exclude: set[str], optional
        Set of keyword parameter names that need to be excluded from the
        introspection.

    Returns
    -------
    set[str]
        Set of valid keyword parameter names.
    """
    exclude = exclude or set()
    signature = inspect.signature(func)

    return {
        name
        for name, param in signature.parameters.items()
        if name not in exclude
        and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
